import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import (
    LossFunctionBase,
)
from distillkit.signals import TeacherSignal


def compute_hs_loss(
    kind: str,
    student_outputs: CausalLMOutput,
    signal: TeacherSignal,
    mask: torch.Tensor | None = None,
    hidden_state_mapping: HiddenStateMapping | None = None,
):
    assert hidden_state_mapping is not None, (
        "Hidden state losses require HiddenStateMapping"
    )
    assert len(hidden_state_mapping.layer_mapping) > 0, (
        "No layers specified in hidden state mapping"
    )
    assert student_outputs.hidden_states is not None
    assert signal.hidden_states is not None

    if mask is None:
        mask = torch.ones(
            student_outputs.hidden_states[0].shape[:-1],
            dtype=torch.bool,
            device=student_outputs.hidden_states[0].device,
        )

    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    total_loss = torch.tensor(0.0, device=student_outputs.hidden_states[0].device)
    for i, (student_layer_idx, teacher_layer_idx) in enumerate(
        hidden_state_mapping.layer_mapping
    ):
        student_h = student_outputs.hidden_states[student_layer_idx]
        teacher_h = signal.hidden_states[teacher_layer_idx]

        if hidden_state_mapping.projections is not None:
            student_h = hidden_state_mapping.projections[i](student_h)

        if kind == "mse":
            squared_error = (student_h - teacher_h) ** 2
            masked_error = squared_error * mask
            layer_loss = masked_error.sum() / (mask.sum() * student_h.shape[-1])
        elif kind == "cosine":
            cosine_sim = F.cosine_similarity(student_h, teacher_h, dim=-1)
            cosine_distance = (1 - cosine_sim) * mask.squeeze(-1)
            layer_loss = cosine_distance.sum() / mask.sum()
        else:
            raise RuntimeError(f"Unimplemented hidden state loss type {repr(kind)}")

        total_loss += layer_loss

    return total_loss / len(hidden_state_mapping.layer_mapping)


class HiddenStateCosineLoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "hs_cosine"

    @override
    def requires_hidden_states(self) -> bool:
        return True

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
        return compute_hs_loss(
            "cosine", student_outputs, signal, mask, hidden_state_mapping
        )


class HiddenStateMSELoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "hs_mse"

    @override
    def requires_hidden_states(self) -> bool:
        return True

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
        return compute_hs_loss(
            "mse", student_outputs, signal, mask, hidden_state_mapping
        )
