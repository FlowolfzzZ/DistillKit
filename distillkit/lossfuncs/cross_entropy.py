import torch
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import (
    LossFunctionBase,
)
from distillkit.signals import TeacherSignal


class CrossEntropyLoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "cross_entropy"

    @override
    def __init__(self): ...

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        return student_outputs.loss
