# Copyright 2024 Charles O. Goddard

import torch
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

        if self.config.compute_device is None:
            self.config.compute_device = self.model.device

        self.loss_functions = [create_loss_func(lfc) for lfc in config.loss_functions]
        self.need_hidden_states = any(
            lf.requires_hidden_states() for lf in self.loss_functions
        )

        self.multi_signal_sources = multi_signal_sources
        self.multi_hidden_state_mappings = multi_hidden_state_mappings

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

        # vllm and hf inference produce different lengths of logits
        if isinstance(signals[0], SparseSignal):
            student_outputs.logits = student_outputs.logits[:, :-1, :]
            valid_mask = valid_mask[:, :-1, :]

        losses = []
        loss_fns = []
        weights = []
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
                    batch_size = inputs['input_ids'].shape[0]
                    loss = torch.zeros((), device=self.config.compute_device)
                    for k in range(batch_size):
                        if inputs['teacher'][k] not in weight_dict:
                            raise ValueError(f"Teacher name {inputs['teacher'][k]} not found in weight dict {weight_dict.keys()}")
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
                        loss += partial_loss * weight_dict[inputs['teacher'][k]]
                    # 为了简化流程，这里假设每个教师的loss_function的weight总和都是1，没有考虑weight总和非1的情况
                    loss = loss / batch_size
                else:
                    loss = loss_fn(
                        student_outputs,
                        signal,
                        mask=valid_mask,
                        hidden_state_mapping=self.multi_hidden_state_mappings[i],
                        num_items_in_batch=num_items_in_batch,
                    )
                losses.append(loss)
                loss_fns.append(cfg.function.value)
                if isinstance(cfg.weight, list):
                    batch_size = inputs['input_ids'].shape[0]
                    for k in range(batch_size):
                        weights.append(weight_dict[inputs['teacher'][k]] / batch_size)
                else:
                    weights.append(cfg.weight)
            # recover
            if signal.hidden_states is not None:
                signal.hidden_states = tuple(hs.to(self.model.device) for hs in signal.hidden_states)
            if isinstance(signal, DenseSignal):
                signal.logits = signal.logits.to(self.model.device)
            else:
                signal.sparse_ids = signal.sparse_ids.to(self.model.device)
                signal.sparse_values = signal.sparse_values.to(self.model.device)
            torch.cuda.empty_cache()

        total_loss = 0.0
        for loss, weight in zip(losses, weights):
            if isinstance(weight, list):
                # teacher-specific weights
                for w in weight:
                    teacher_name = w.teacher_name
                    teacher_weight = w.weight
                    if inputs['teacher'][0] == teacher_name:
                        total_loss += loss * teacher_weight
            else:
                total_loss += loss * weight
        total_loss = total_loss / sum(weights) / len(self.multi_signal_sources)
        self.log(
            {
                f"distillation_loss/{idx + 1}_{loss_fn}": loss.item()
                for idx, (loss, loss_fn) in enumerate(zip(losses, loss_fns))
            }
        )
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