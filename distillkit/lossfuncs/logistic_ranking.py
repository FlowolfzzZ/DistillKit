import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import LossFunctionBase
from distillkit.signals import DenseSignal, TeacherSignal


def sparse_logistic_ranking_loss(
    student_logits: torch.Tensor,
    teacher_target_ids: torch.LongTensor,
    teacher_target_values: torch.Tensor,
    log_target: bool = True,
    sequence_mask: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Computes a logistic ranking loss between student logits and sparse teacher probabilities.
    The loss encourages the student to rank the teacher's *actually supported* tokens
    (those with probability > eps) in the same relative order as the teacher.

    Args:
        student_logits: Logits predicted by the student model.
                        Shape: [batch_size, seq_len, vocab_size]
        teacher_target_ids: Indices of the tokens in the teacher's sparse set (up to k_sparse).
                            0 <= teacher_target_ids < vocab_size.
                            Shape: [batch_size, seq_len, k_sparse]
        teacher_target_values: Probabilities or log-probabilities for the tokens in
                               teacher_target_ids. Shape: [batch_size, seq_len, k_sparse]
                               It's assumed that if k_sparse is a max capacity and fewer items
                               are truly supported, the corresponding values here reflect that
                               (e.g., 0 for probability, or very small/negative logprob).
        log_target: Boolean, True if teacher_target_values are log-probabilities.
        sequence_mask: Optional boolean mask for sequence positions. True indicates active,
                       False indicates padded/ignored. Shape: [batch_size, seq_len]
        eps: Small value to avoid numerical issues. Default is 1e-9.

    Returns:
        A scalar tensor representing the mean logistic ranking loss over active sequence positions.
    """
    if log_target:
        teacher_probs = torch.exp(teacher_target_values)
    else:
        teacher_probs = teacher_target_values

    student_logits_at_targets = torch.gather(
        student_logits,
        dim=2,
        index=teacher_target_ids,
    )  # [B, S, k_sparse]

    # Mask for tokens that are *actually* supported by the teacher (prob > eps)
    actually_supported_mask = teacher_probs > eps  # [B, S, k_sparse]

    # Teacher preferences: teacher_probs_k > teacher_probs_l
    # teacher_probs_k.shape [B, S, k_sparse, 1], teacher_probs_l.shape [B, S, 1, k_sparse]
    teacher_prefers_k_over_l_mask = teacher_probs.unsqueeze(-1) > (
        teacher_probs.unsqueeze(-2) + eps
    )  # [B, S, k_sparse, k_sparse]
    student_logit_diff_k_minus_l = student_logits_at_targets.unsqueeze(
        -1
    ) - student_logits_at_targets.unsqueeze(-2)  # [B, S, k_sparse, k_sparse]

    # Pair activity mask: both tokens in the pair must be actually supported
    supported_k = actually_supported_mask.unsqueeze(-1)  # [B, S, k_sparse, 1]
    supported_l = actually_supported_mask.unsqueeze(-2)  # [B, S, 1, k_sparse]
    pair_is_supported_mask = supported_k & supported_l  # [B, S, k_sparse, k_sparse]

    # Valid preference pairs: teacher prefers k over l and both k and l are supported
    valid_preference_pair_mask = teacher_prefers_k_over_l_mask & pair_is_supported_mask

    pair_loss = F.softplus(-student_logit_diff_k_minus_l)  # [B, S, k_sparse, k_sparse]
    masked_pair_loss = pair_loss * valid_preference_pair_mask.float()
    sum_pair_loss_per_pos = masked_pair_loss.sum(dim=(2, 3))  # [B, S]

    if sequence_mask is None:
        active_sequence_mask = torch.ones_like(
            sum_pair_loss_per_pos, dtype=torch.bool, device=student_logits.device
        )
    else:
        active_sequence_mask = sequence_mask.bool()  # Ensure boolean

    final_summed_loss = (sum_pair_loss_per_pos * active_sequence_mask.float()).sum()

    num_contributing_pairs = (
        valid_preference_pair_mask.float()
        * active_sequence_mask.float().unsqueeze(-1).unsqueeze(-1)
    ).sum()
    loss = final_summed_loss / (num_contributing_pairs + eps)

    return loss


class LogisticRankingLoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "logistic_ranking"

    @override
    def __init__(self) -> None:
        pass

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
            raise RuntimeError(
                "Logistic ranking loss is not supported for dense predictions"
            )

        return sparse_logistic_ranking_loss(
            student_logits=student_outputs.logits,
            teacher_target_ids=signal.sparse_ids,
            teacher_target_values=signal.sparse_values,
            sequence_mask=mask,
            log_target=signal.log_values,
        )
