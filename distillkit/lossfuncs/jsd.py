import math

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


def sparse_jsd_inner(
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
    batch_size, seq_len, vocab_size = logits.shape
    out_dtype = logits.dtype
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
    sparse_teacher_probs = torch.exp(sparse_target_logprobs)

    # --- 3. Compute Mixture M (log M) ---
    # Common sparse part for M
    # Ensure P_i + Q_i is not zero before log. Add eps to P+Q or use M.clamp(min=eps)
    # log(0.5 * (P_i + Q_i)) = log(0.5) + log(P_i + Q_i)
    # Using .clamp for safety if P_i and Q_i can both be zero for some index.
    M_sparse_probs = 0.5 * (sparse_teacher_probs + sparse_student_probs)
    log_M_sparse = torch.log(M_sparse_probs.clamp(min=eps))

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        teacher_prob_sum_sparse = sparse_teacher_probs.to(torch.float32).sum(
            dim=-1, keepdim=True
        )
        # log1p for log(1-x) is good
        log_teacher_missing_prob = torch.log1p(
            -teacher_prob_sum_sparse.clamp(min=eps, max=1.0 - eps)
        )
        teacher_missing_prob = torch.exp(log_teacher_missing_prob)

        student_prob_sum_sparse = sparse_student_probs.to(torch.float32).sum(
            dim=-1, keepdim=True
        )
        log_student_missing_prob = torch.log1p(
            -student_prob_sum_sparse.clamp(min=eps, max=1.0 - eps)
        )
        student_missing_prob = torch.exp(log_student_missing_prob)

        M_missing_prob = 0.5 * (teacher_missing_prob + student_missing_prob)
        log_M_missing = torch.log(M_missing_prob.clamp(min=eps))

    # --- 4. Compute KL(P || M) ---
    # P_i * (log P_i - log M_i)
    # Handle P_i = 0 case: 0 * log(0/M_i) = 0. This is implicitly handled if P_i is small.
    # If sparse_teacher_probs can be exactly zero, ensure 0 * -inf is 0.
    # PyTorch's 0 * -inf is nan. Use torch.where(sparse_teacher_probs > eps, ..., 0.0)
    kl_P_M_sparse_terms = sparse_teacher_probs * (sparse_target_logprobs - log_M_sparse)
    kl_P_M_sparse_sum = torch.sum(
        torch.where(
            sparse_teacher_probs > eps,
            kl_P_M_sparse_terms,
            torch.zeros_like(kl_P_M_sparse_terms),
        ),
        dim=-1,
    )

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        # teacher_missing_prob * (log_teacher_missing_prob - log_M_missing)
        kl_P_M_missing_term = teacher_missing_prob * (
            log_teacher_missing_prob - log_M_missing
        )
        kl_P_M = kl_P_M_sparse_sum + torch.where(
            teacher_missing_prob.squeeze(-1) > eps,
            kl_P_M_missing_term.squeeze(-1),
            torch.zeros_like(kl_P_M_missing_term.squeeze(-1)),
        )
    else:  # ZERO
        kl_P_M = kl_P_M_sparse_sum
    del sparse_target_logprobs  # No longer needed for P

    # --- 5. Compute KL(Q || M) ---
    # Q_i * (log Q_i - log M_i)
    kl_Q_M_sparse_terms = sparse_student_probs * (
        sparse_student_logprobs - log_M_sparse
    )
    kl_Q_M_sparse_sum = torch.sum(
        torch.where(
            sparse_student_probs > eps,
            kl_Q_M_sparse_terms,
            torch.zeros_like(kl_Q_M_sparse_terms),
        ),
        dim=-1,
    )

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        kl_Q_M_missing_term = student_missing_prob * (
            log_student_missing_prob - log_M_missing
        )
        kl_Q_M = kl_Q_M_sparse_sum + torch.where(
            student_missing_prob.squeeze(-1) > eps,
            kl_Q_M_missing_term.squeeze(-1),
            torch.zeros_like(kl_Q_M_missing_term.squeeze(-1)),
        )
        del (
            student_missing_prob,
            log_student_missing_prob,
            M_missing_prob,
            log_M_missing,
        )  # Free memory
    else:  # ZERO
        # Contribution from Q_missing_i * log(2)
        # We need total prob mass of student for tokens NOT in target_ids
        student_prob_sum_sparse = sparse_student_probs.sum(dim=-1)  # B, S
        # total probability is 1, so mass for missing is 1 - sum_sparse
        student_total_missing_prob_mass = (1.0 - student_prob_sum_sparse).clamp(
            min=0.0
        )  # B, S
        kl_Q_M_missing_contrib = student_total_missing_prob_mass * math.log(2.0)
        kl_Q_M = kl_Q_M_sparse_sum + kl_Q_M_missing_contrib

    del (
        sparse_student_logprobs,
        sparse_student_probs,
        M_sparse_probs,
        log_M_sparse,
    )

    # --- 6. Combine for JSD ---
    jsd_terms = 0.5 * (kl_P_M + kl_Q_M)

    # --- 7. Masking and Aggregation ---
    if mask is not None:
        if mask.dim() == 2:  # B, S
            mask_squozed = mask
        else:  # B, S, 1
            mask_squozed = mask.squeeze(-1)
        jsd_terms *= mask_squozed

    return torch.sum(jsd_terms).to(out_dtype)


def sparse_js_div(
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
    """Compute the Jensen-Shannon Divergence (JSD) between a dense set of predictions and a sparse set of target logits.

    Uses a chunked approach to avoid memory issues with large sequences.

    Args:
        logits: Dense tensor of predictions.
        target_ids: Tensor of indices for target logits.
        target_values: Tensor of values for target logits or log probabilities.
        mask: Optional boolean mask tensor. True indicates tokens to include, False to exclude.
        eps: Small value to prevent numerical instability.
        missing: How to handle missing probabilities in the target distribution. If ZERO, missing
            probabilities are assumed to be zero. If SYMMETRIC_UNIFORM, missing probabilities are
            assumed to be distributed uniformly over the missing tokens in both the teacher and
            student distributions.
        log_target: Whether the target values are already log probabilities.
        temperature: Temperature to apply to the distributions.
        target_generation_temperature: Temperature already applied to the target logits/logprobs.
        student_generation_temperature: Temperature already applied to the student logits.
        chunk_length: Number of tokens per chunk. If None, the entire sequence is processed at once.
    """
    return accumulate_over_chunks(
        logits,
        target_ids,
        target_values,
        mask,
        chunk_length,
        sparse_jsd_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


def dense_js_div(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute the Jensen-Shannon Divergence (JSD) between dense student predictions and dense target logits.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        logits: Student logits (Batch, Seq, Vocab).
        target_logits: Teacher/Target logits (Batch, Seq, Vocab).
        mask: Optional boolean mask tensor. True indicates tokens to include.
        temperature: Temperature to apply to both distributions.

    Returns:
        torch.Tensor: Scalar JSD loss averaged over the batch.
    """
    out_dtype = logits.dtype

    # 1. Apply temperature scaling
    student_logits = logits / temperature
    teacher_logits = target_logits / temperature

    # 2. Compute log probabilities
    # Using log_softmax ensures numerical stability compared to softmax -> log
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)

    # 3. Compute probabilities
    student_probs = torch.exp(student_log_probs)
    teacher_probs = torch.exp(teacher_log_probs)

    # 4. Compute Mixture Distribution M (in log space)
    # M = 0.5 * (P + Q)
    # log(M) = log(0.5) + log(exp(log_P) + exp(log_Q))
    # We use logaddexp for numerical stability to avoid underflow/overflow
    mixture_log_probs = torch.logaddexp(
        student_log_probs, teacher_log_probs
    ) - math.log(2.0)

    # 5. Compute KL Divergences
    # KL(P || M) = sum( P * (log_P - log_M) )
    # Note: We compute manually rather than using F.kl_div to utilize precomputed log_probs
    kl_student_mixture = torch.sum(
        student_probs * (student_log_probs - mixture_log_probs), dim=-1
    )
    kl_teacher_mixture = torch.sum(
        teacher_probs * (teacher_log_probs - mixture_log_probs), dim=-1
    )

    # 6. Combine for JSD
    jsd_terms = 0.5 * (kl_student_mixture + kl_teacher_mixture)

    # 7. Apply Masking
    if mask is not None:
        if mask.dim() == 2:  # B, S
            mask_squeezed = mask
        else:  # B, S, 1
            mask_squeezed = mask.squeeze(-1)
        jsd_terms = jsd_terms * mask_squeezed

    return torch.sum(jsd_terms).to(out_dtype)


class JSDLoss(LossFunctionBase):
    temperature: float
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO
    chunk_length: int | None = None

    @override
    @classmethod
    def name(cls) -> str:
        return "jsd"

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
            res = dense_js_div(
                student_outputs.logits,
                signal.logits,
                mask=mask,
                temperature=self.temperature,
            )
        else:
            res = sparse_js_div(
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
