import torch
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import LossFunctionBase
from distillkit.signals import DenseSignal, TeacherSignal


def sparse_hinge_loss(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    log_target: bool = True,
    margin: float | None = None,
) -> torch.Tensor:
    # Validate input shapes
    assert logits.size()[:2] == target_ids.size()[:2] == target_values.size()[:2], (
        "Batch and sequence length must match"
    )
    B, S, K = target_ids.shape
    assert target_values.shape == (B, S, K), "Target values shape must match target_ids"

    # Gather the logits for the target_ids
    student_probs = torch.softmax(logits, dim=-1)  # Shape: [B, S, V]
    student_target_probs = student_probs.gather(-1, target_ids)  # Shape: [B, S, K]

    if log_target:
        teacher_probs = torch.exp(target_values)
    else:
        teacher_probs = target_values

    prob_diff = student_target_probs.unsqueeze(-1) - student_target_probs.unsqueeze(
        -2
    )  # Shape: [B, S, K, K]
    if margin is None:
        margin_values = teacher_probs.unsqueeze(-1) - teacher_probs.unsqueeze(-2)
    else:
        margin_values = margin * torch.ones_like(prob_diff)
    loss_terms = margin_values - prob_diff
    max_terms = torch.relu(loss_terms)  # Shape: [B, S, K, K]

    actually_supported_mask = teacher_probs > eps
    supported_k = actually_supported_mask.unsqueeze(-1)  # [B, S, K, 1]
    supported_l = actually_supported_mask.unsqueeze(-2)  # [B, S, 1, K]
    pair_is_genuinely_supported_mask = supported_k & supported_l  # [B, S, K, K]

    preference_mask = teacher_probs.unsqueeze(-1) > (teacher_probs.unsqueeze(-2) + eps)
    valid_mask = preference_mask & pair_is_genuinely_supported_mask
    active_terms = max_terms * valid_mask.float()

    num_contributing_pairs = valid_mask.float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).float()
        active_terms = active_terms * mask_expanded
        num_contributing_pairs = num_contributing_pairs * mask_expanded

    total_loss = active_terms.sum() / (num_contributing_pairs.sum() + eps)
    return total_loss


class HingeLoss(LossFunctionBase):
    margin: float

    @override
    @classmethod
    def name(cls) -> str:
        return "hinge"

    @override
    def __init__(self, margin: float) -> None:
        self.margin = margin

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if isinstance(signal, DenseSignal):
            raise RuntimeError("Hinge loss is not supported for dense predictions")

        return sparse_hinge_loss(
            logits=student_outputs.logits,
            target_ids=signal.sparse_ids,
            target_values=signal.sparse_values,
            mask=mask,
            log_target=signal.log_values,
            margin=self.margin,
        )
