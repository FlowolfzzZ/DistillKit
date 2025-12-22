# Copyright 2024 Charles O. Goddard

import torch
from transformers import (
    PreTrainedModel,
)
from trl import SFTTrainer

from distillkit.configuration import DistillationRunConfig, LossFunctionConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs import ALL_LOSS_CLASSES, LossFunctionBase
from distillkit.signals import OnlineSignalSource, SignalSource, TeacherSignal
import copy


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
            if signal.hidden_states is not None:
                signal.hidden_states = tuple(hs.to(self.config.compute_device) for hs in signal.hidden_states)
            signal.logits = signal.logits.to(self.config.compute_device)
            signals.append(signal)

        losses = []
        loss_fns = []
        weights = []
        for signal in signals:
            for idx, loss_fn in enumerate(self.loss_functions):
                cfg = self.config.loss_functions[idx]
                loss = loss_fn(
                    student_outputs,
                    signal,
                    mask=valid_mask,
                    hidden_state_mapping=self.multi_hidden_state_mappings[idx],
                    num_items_in_batch=num_items_in_batch,
                )
                losses.append(loss)
                loss_fns.append(cfg.function.value)
                weights.append(cfg.weight)

        total_loss = 0.0
        for loss, weight in zip(losses, weights):
            total_loss += loss * weight
        total_loss = total_loss / sum(weights)
        self.log(
            {
                f"distillation_loss/{idx + 1}_{loss_fn}": loss.item()
                for idx, (loss, loss_fn) in enumerate(zip(losses, loss_fns))
            }
        )
        return total_loss
