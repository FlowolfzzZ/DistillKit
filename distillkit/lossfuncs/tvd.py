import torch
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import (
    LossFunctionBase,
    MissingProbabilityHandling,
    accumulate_over_chunks,
    get_logprobs,
)
from distillkit.signals import DenseSignal, TeacherSignal


def sparse_tvd_inner(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the Total Variation Distance (TVD) between dense student predictions and sparse teacher targets.

    See `sparse_tvd_div` for details.
    """
    batch_size, seq_len, vocab_size = logits.shape
    sparse_student_logprobs, sparse_target_logprobs = get_logprobs(
        logits,
        target_ids,
        target_values,
        eps=eps,
        missing=missing,
        log_target=log_target,
        distillation_temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )
    sparse_student_probs = torch.exp(sparse_student_logprobs)
    del sparse_student_logprobs
    sparse_teacher_probs = torch.exp(sparse_target_logprobs)
    del sparse_target_logprobs

    # --- 3. Compute TVD for sparse indices ---
    # sum_i |P_i - Q_i| for i in target_ids
    tvd_sparse_terms_sum = torch.sum(
        torch.abs(sparse_teacher_probs - sparse_student_probs), dim=-1
    )  # Shape: (B, S)

    # --- 4. Compute TVD contribution from missing indices ---
    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        # In this case, get_target_logprobs has normalized P such that P_sparse + P_miss = 1
        # (where P_miss is the (k+1)th synthetic category).
        # So, P_miss = 1 - sum(P_sparse)
        teacher_prob_sum_sparse = sparse_teacher_probs.sum(dim=-1)  # B, S
        # Clamp to avoid small numerical errors making this < 0 or > 1 due to float precision
        teacher_missing_prob_mass = (1.0 - teacher_prob_sum_sparse).clamp(
            min=0.0, max=1.0
        )

        # Similarly for student Q, Q_miss = 1 - sum(Q_sparse)
        student_prob_sum_sparse = sparse_student_probs.sum(dim=-1)  # B, S
        student_missing_prob_mass = (1.0 - student_prob_sum_sparse).clamp(
            min=0.0, max=1.0
        )

        tvd_missing_contrib = torch.abs(
            teacher_missing_prob_mass - student_missing_prob_mass
        )  # B, S

    else:
        # Teacher P_j = 0 for j not in target_ids.
        # So, sum_{j not in target_ids} |P_j - Q_j| = sum_{j not in target_ids} |0 - Q_j|
        # = sum_{j not in target_ids} Q_j
        # This is the total probability mass of the student for tokens NOT in target_ids.
        student_prob_sum_sparse = sparse_student_probs.sum(dim=-1)  # B, S
        student_total_missing_prob_mass = (1.0 - student_prob_sum_sparse).clamp(
            min=0.0
        )  # B, S
        tvd_missing_contrib = student_total_missing_prob_mass

    del sparse_teacher_probs, sparse_student_probs
    tvd_token_level = 0.5 * (tvd_sparse_terms_sum + tvd_missing_contrib)

    # --- 6. Masking and Aggregation ---
    if mask is not None:
        if mask.dim() == 2:  # B, S
            mask_squozed = mask
        else:  # B, S, 1
            mask_squozed = mask.squeeze(-1)
        tvd_token_level *= mask_squozed

    return torch.sum(tvd_token_level)


def sparse_tvd(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
    chunk_length: int | None = None,
) -> torch.Tensor:
    """Compute the Total Variation Distance (TVD) between a dense set of student predictions
    and a sparse set of teacher targets.

    Uses a chunked approach to avoid memory issues with large sequences.

    TVD = 0.5 * sum_i |P_i - Q_i|

    Args:
        logits: Dense tensor of student predictions (batch_size, seq_len, vocab_size).
        target_ids: Tensor of indices for teacher target probabilities/logits
                    (batch_size, seq_len, num_sparse_targets).
        target_values: Tensor of values for teacher target probabilities/logits
                       (batch_size, seq_len, num_sparse_targets).
        mask: Optional boolean mask tensor (batch_size, seq_len). True indicates tokens to include.
        eps: Small value to avoid numerical issues. Default is 1e-8.
        missing: How to handle missing probabilities in the target distribution.
                 ZERO: Missing teacher probabilities are zero. Student's missing mass contributes to TVD.
                 SYMMETRIC_UNIFORM: Missing mass in P and Q is treated as a (k+1)th category.
        log_target: Whether the target_values are log probabilities (True) or logits (False).
        temperature: Distillation temperature to apply to both student and teacher.
        target_generation_temperature: Temperature originally used to generate teacher targets.
        student_generation_temperature: Temperature originally used to generate student logits.
        chunk_length: Number of tokens per chunk. If None, entire sequence is processed at once.
    """
    return accumulate_over_chunks(
        logits,
        target_ids,
        target_values,
        mask,
        chunk_length,
        sparse_tvd_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


def dense_tvd(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the Total Variation Distance (TVD) between dense student predictions
    and dense teacher targets.

    TVD = 0.5 * sum_i |P_i - Q_i|

    Args:
        logits: Student logits (batch_size, seq_len, vocab_size).
        target_logits: Teacher logits (batch_size, seq_len, vocab_size).
        mask: Optional mask (batch_size, seq_len) or (batch_size, seq_len, 1).
        temperature: Distillation temperature.
    """
    # 1. Apply temperature and compute probabilities
    student_probs = torch.softmax(logits.float() / temperature, dim=-1)
    teacher_probs = torch.softmax(target_logits.float() / temperature, dim=-1)

    # 2. Compute TVD per token
    # Sum over the vocabulary dimension (dim=-1)
    # Shape becomes: (Batch, Seq_Len)
    tvd_token_level = 0.5 * torch.sum(torch.abs(teacher_probs - student_probs), dim=-1)

    # 3. Apply Mask
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        tvd_token_level = tvd_token_level * mask

    return torch.sum(tvd_token_level)


class TVDLoss(LossFunctionBase):
    temperature: float
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO
    chunk_length: int | None = None

    @override
    @classmethod
    def name(cls) -> str:
        return "tvd"

    @override
    def __init__(
        self,
        temperature: float,
        missing_probability_handling: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
        sparse_chunk_length: int | None = None,
    ) -> None:
        self.temperature = temperature
        self.missing = missing_probability_handling
        self.chunk_length = sparse_chunk_length

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if num_items_in_batch is None:
            if mask is not None:
                num_items_in_batch = mask.float().sum()
            else:
                num_items_in_batch = (
                    student_outputs.logits.shape[0] * student_outputs.logits.shape[1]
                )
        if isinstance(signal, DenseSignal):
            res = dense_tvd(
                student_outputs.logits,
                signal.logits,
                mask=mask,
                temperature=self.temperature,
            )
        else:
            res = sparse_tvd(
                logits=student_outputs.logits,
                target_ids=signal.sparse_ids,
                target_values=signal.sparse_values,
                mask=mask,
                missing=self.missing,
                log_target=signal.log_values,
                temperature=self.temperature,
                target_generation_temperature=signal.generation_temperature,
                chunk_length=self.chunk_length,
            )
        return res * (self.temperature**2) / num_items_in_batch
