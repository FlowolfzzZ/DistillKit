# Copyright 2024 Charles O. Goddard

import torch
import torch.distributed as dist
from transformers import (
    PreTrainedModel,
)
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.utils import pad
from typing import Any

from distillkit.configuration import DistillationRunConfig, LossFunctionConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs import ALL_LOSS_CLASSES, LossFunctionBase
from distillkit.signals import OnlineSignalSource, OfflineSignalSource, SignalSource, TeacherSignal, DenseSignal, SparseSignal


def create_loss_func(cfg: LossFunctionConfig) -> LossFunctionBase:
    for cls in ALL_LOSS_CLASSES:
        if cfg.function.value == cls.name():
            return cls(
                **cfg.model_dump(exclude=["function", "weight"], exclude_none=True)
            )
    raise RuntimeError(f"Unknown loss function '{cfg.function}'")


class DistillationTrainer(SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        config: DistillationRunConfig,
        multi_signal_sources: list[SignalSource],
        true_vocab_size: int,
        *args,
        multi_hidden_state_mappings: list[HiddenStateMapping] = [],
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.true_vocab_size = true_vocab_size
        self.config = config
        # 用于累积多步的 distillation loss，避免梯度累计时重复上报同一个 global_step
        self._distill_accum_losses = None
        self._distill_micro_step = 0

        if self.config.compute_device is None:
            self.config.compute_device = self.model.device

        self.loss_functions = [create_loss_func(lfc) for lfc in config.loss_functions]
        self.need_hidden_states = any(
            lf.requires_hidden_states() for lf in self.loss_functions
        )

        self.multi_signal_sources = multi_signal_sources
        self.multi_hidden_state_mappings = multi_hidden_state_mappings
        self.teacher_names = []
        for idx, ss in enumerate(self.multi_signal_sources):
            name = getattr(ss, "name", None)
            if not name:
                name = f"teacher{idx + 1}"
            self.teacher_names.append(name)
        # 记录离线多教师逐教师的 loss，用于日志
        self._per_teacher_accum = {}  # key: (loss_fn, teacher) -> {"loss_sum": float, "count": int}
        self._last_per_teacher_logs = []

        if self.need_hidden_states and not any(
            ss.supports_hidden_states() for ss in self.multi_signal_sources
        ):
            raise ValueError(
                "Configuration requests hidden state loss, but the provided Teacher "
                "(Offline/Dataset) does not support hidden states."
            )

        if any(hsm is None for hsm in self.multi_hidden_state_mappings) and self.need_hidden_states:
            raise ValueError(
                "Must define a hidden state mapping to use hidden state losses."
            )

        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"]
        if self.config.dataset.eos_label_token_ids:
            inputs["labels"] = inputs["labels"].clone()
            for tok_id in self.config.dataset.eos_label_token_ids:
                inputs["labels"][inputs["labels"] == tok_id] = (
                    self.model.config.eos_token_id
                )
        student_model = model.module if hasattr(model, "module") else model
        student_outputs = student_model(
            **{
                k: inputs[k]
                for k in ["input_ids", "attention_mask", "labels"]
                if k in inputs
            },
            return_dict=True,
            output_hidden_states=self.need_hidden_states,
            **kwargs,
        )
        if student_outputs.logits.shape[-1] != self.true_vocab_size:
            # truncate any extra logits from padding
            student_outputs.logits = student_outputs.logits[..., : self.true_vocab_size]

        total_loss = self.total_distillation_loss(
            student_outputs,
            inputs,
            num_items_in_batch=None,
        ).to(student_model.device)
        # recover device
        student_outputs.logits = student_outputs.logits.to(student_model.device)
        student_outputs.loss = student_outputs.loss.to(student_model.device)
        if student_outputs.hidden_states is not None:
            student_outputs.hidden_states = tuple(hs.to(student_model.device) for hs in student_outputs.hidden_states)
        torch.cuda.empty_cache()
        
        # 保存本步逐教师原始 loss（未乘顶层 weight）以便日志
        step_teacher_logs = getattr(self, "_last_per_teacher_logs", [])
        for loss_fn, teacher_name, mean_val, count in step_teacher_logs:
            key = (loss_fn, teacher_name)
            entry = self._per_teacher_accum.get(key, {"loss_sum": 0.0, "count": 0})
            entry["loss_sum"] += mean_val * count
            entry["count"] += count
            self._per_teacher_accum[key] = entry

        return (total_loss, student_outputs) if return_outputs else total_loss

    def total_distillation_loss(
        self, student_outputs, inputs, num_items_in_batch: int | None = None
    ):
        valid_mask = (inputs["labels"] >= 0).unsqueeze(-1).to(self.config.compute_device)
        # attentions, past_key_values are not used for loss computation
        student_outputs.logits = student_outputs.logits.to(self.config.compute_device)
        student_outputs.loss = student_outputs.loss.to(self.config.compute_device)
        if student_outputs.hidden_states is not None:
            student_outputs.hidden_states = tuple(hs.to(self.config.compute_device) for hs in student_outputs.hidden_states)

        signals = []
        for signal_source in self.multi_signal_sources:
            signal: TeacherSignal = signal_source.get_signal(
                inputs,
                return_hidden_states=self.need_hidden_states,
            )
            signals.append(signal)
        # 存储逐教师原始 loss（未乘顶层 weight），用于日志
        per_teacher_logs: list[tuple[str, str, float, int]] = []

        # vllm and hf inference produce different lengths of logits
        if isinstance(signals[0], SparseSignal):
            student_outputs.logits = student_outputs.logits[:, :-1, :]
            valid_mask = valid_mask[:, :-1, :]

        losses = []
        loss_fns = []
        weights = []
        loss_teacher_indices = []
        for i, signal in enumerate(signals):
            if signal.hidden_states is not None:
                signal.hidden_states = tuple(hs.to(self.config.compute_device) for hs in signal.hidden_states)
            if isinstance(signal, DenseSignal):
                signal.logits = signal.logits.to(self.config.compute_device)
            else:
                signal.sparse_ids = signal.sparse_ids.to(self.config.compute_device)
                signal.sparse_values = signal.sparse_values.to(self.config.compute_device)
            for j, loss_fn in enumerate(self.loss_functions):
                cfg = self.config.loss_functions[j]
                # teacher-specific weights
                weight_dict = {}
                # multi-teacher offline distillation
                if isinstance(cfg.weight, list):
                    for w in cfg.weight:
                        weight_dict.update(w.to_dict())
                    # offline distillation: weights decided by teacher column
                    if "teacher" in inputs:
                        batch_size = inputs['input_ids'].shape[0]
                        loss = torch.zeros((), device=self.config.compute_device)
                        per_teacher_sums = {}
                        per_teacher_counts = {}
                        for k in range(batch_size):
                            teacher_name = inputs['teacher'][k]
                            if teacher_name not in weight_dict:
                                raise ValueError(f"Teacher name {teacher_name} not found in weight dict {weight_dict.keys()}")
                            partial_student_outputs = student_outputs.__class__(
                                logits=student_outputs.logits[k:k+1],
                                loss=student_outputs.loss,
                                hidden_states=student_outputs.hidden_states,
                            )
                            partial_signal = SparseSignal(
                                sparse_ids=signal.sparse_ids[k:k+1],
                                sparse_values=signal.sparse_values[k:k+1],
                                log_values=signal.log_values,
                                generation_temperature=signal.generation_temperature,
                                hidden_states=signal.hidden_states,
                                vocab_size=signal.vocab_size,
                            )
                            partial_loss = loss_fn(
                                partial_student_outputs,
                                partial_signal,
                                mask=valid_mask[k:k+1],
                                hidden_state_mapping=self.multi_hidden_state_mappings[i],
                                num_items_in_batch=None,
                            )
                            loss += partial_loss * weight_dict[teacher_name]
                            # 逐教师累加原始 loss
                            per_teacher_sums[teacher_name] = per_teacher_sums.get(teacher_name, 0.0) + partial_loss.detach()
                            per_teacher_counts[teacher_name] = per_teacher_counts.get(teacher_name, 0) + 1
                        # 为了简化流程，这里假设每个教师的loss_function的weight总和都是1，没有考虑weight总和非1的情况
                        loss = loss / batch_size
                        weight_value = sum(weight_dict[inputs['teacher'][k]] for k in range(batch_size)) / batch_size
                        # 记录逐教师原始 loss（未乘顶层 weight），用于日志
                        for t, s in per_teacher_sums.items():
                            cnt = per_teacher_counts[t]
                            per_teacher_logs.append((cfg.function.value, t, float(s / cnt), cnt))
                    else:
                        # online distillation: fall back to teacher index ordering
                        teacher_weights = [w.weight for w in cfg.weight]
                        weight_value = teacher_weights[i] if i < len(teacher_weights) else sum(teacher_weights) / len(teacher_weights)
                        loss = loss_fn(
                            student_outputs,
                            signal,
                            mask=valid_mask,
                            hidden_state_mapping=self.multi_hidden_state_mappings[i],
                            num_items_in_batch=num_items_in_batch,
                        )
                else:
                    loss = loss_fn(
                        student_outputs,
                        signal,
                        mask=valid_mask,
                        hidden_state_mapping=self.multi_hidden_state_mappings[i],
                        num_items_in_batch=num_items_in_batch,
                    )
                    weight_value = cfg.weight
                losses.append(loss)
                loss_fns.append(cfg.function.value)
                if isinstance(cfg.weight, list):
                    weights.append(weight_value)
                else:
                    weights.append(cfg.weight)
                loss_teacher_indices.append(i)
            # recover
            if signal.hidden_states is not None:
                signal.hidden_states = tuple(hs.to(self.model.device) for hs in signal.hidden_states)
            if isinstance(signal, DenseSignal):
                signal.logits = signal.logits.to(self.model.device)
            else:
                signal.sparse_ids = signal.sparse_ids.to(self.model.device)
                signal.sparse_values = signal.sparse_values.to(self.model.device)
            torch.cuda.empty_cache()

        # 暂存本步逐教师原始 loss（未乘顶层 weight），供 compute_loss 日志使用
        self._last_per_teacher_logs = per_teacher_logs

        total_loss = 0.0
        for loss, weight in zip(losses, weights):
            total_loss += loss * weight
        total_loss = total_loss / sum(weights) / len(self.multi_signal_sources)

        # 累计若干 micro-batch 后再在主进程上报一次，避免 wandb 同一 global_step 记录多值
        device_for_logs = student_outputs.logits.device
        log_losses = torch.tensor([loss.item() for loss in losses], device=device_for_logs)
        if self._distill_accum_losses is None:
            self._distill_accum_losses = torch.zeros_like(log_losses)
            self._distill_micro_step = 0
        self._distill_accum_losses += log_losses
        self._distill_micro_step += 1

        should_log = self._distill_micro_step >= self.args.gradient_accumulation_steps
        if should_log:
            # 当前进程的平均
            avg_losses = self._distill_accum_losses / float(self._distill_micro_step)

            # 多机/多卡同步平均
            is_main = True
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(avg_losses, op=dist.ReduceOp.SUM)
                avg_losses = avg_losses / dist.get_world_size()
                is_main = dist.get_rank() == 0

            if is_main:
                log_dict = {}
                # 逐教师原始 loss（未乘顶层 weight）
                for (loss_fn, teacher_name), entry in list(getattr(self, "_per_teacher_accum", {}).items()):
                    if entry["count"] > 0:
                        log_dict[f"distillation_loss_raw/{loss_fn}/{teacher_name}"] = entry["loss_sum"] / entry["count"]
                # 加权后（顶层权重之前）的多教师 loss，按信号源 index 记录
                for idx, (val, loss_fn, teacher_idx) in enumerate(zip(avg_losses, loss_fns, loss_teacher_indices)):
                    log_key = f"distillation_loss/{loss_fn}/total"
                    log_dict[log_key] = val.item()
                if log_dict:
                    self.log(log_dict)

            # 重置累计
            self._distill_accum_losses.zero_()
            self._distill_micro_step = 0
            self._per_teacher_accum = {}
        return total_loss


class DistillationDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        if "id" in examples[0]:
            batch["id"] = [example["id"] for example in examples]
        if "teacher" in examples[0]:
            batch["teacher"] = [example["teacher"] for example in examples]
        if "packed_indices" in examples[0]:
            packed_indices = [torch.tensor(example["packed_indices"]) for example in examples]
            batch["packed_indices"] = pad(packed_indices,
                                          padding_value=self.pad_token_id,
                                          padding_side="right")
        if "exact_values" in examples[0]:
            exact_values = [torch.tensor(example["exact_values"]) for example in examples]
            batch["exact_values"] = pad(exact_values,
                                        padding_value=self.pad_token_id,
                                        padding_side="right")
        # Current implementation assumes using exact logits, so coeffs are absent
        if "coeffs" in examples[0]:
            batch["coeffs"] = torch.tensor([example["coeffs"] for example in examples])

        return batch
