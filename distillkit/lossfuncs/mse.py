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


def sparse_mse_inner(
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
    """Compute a sparse MSE between student and teacher distributions on teacher-provided indices.

    Notes:
    - The core term is computed on the teacher-provided sparse indices.
    - If `missing == SYMMETRIC_UNIFORM`, the remaining probability mass is treated as an extra
      (k+1)th category (matching the approach used by other sparse distribution losses).
    - If `missing == ZERO`, the contribution from tokens missing in `target_ids` is ignored.
    """
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

    student_probs = sparse_student_logprobs.exp()
    teacher_probs = sparse_target_logprobs.exp()
    del sparse_student_logprobs, sparse_target_logprobs

    mse_per_token = torch.sum((teacher_probs - student_probs) ** 2, dim=-1)

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        teacher_sum = teacher_probs.to(torch.float32).sum(dim=-1)
        student_sum = student_probs.to(torch.float32).sum(dim=-1)
        teacher_missing = (1.0 - teacher_sum).clamp(min=0.0, max=1.0)
        student_missing = (1.0 - student_sum).clamp(min=0.0, max=1.0)
        mse_per_token = mse_per_token + (teacher_missing - student_missing) ** 2

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mse_per_token = mse_per_token * mask

    return torch.sum(mse_per_token).to(out_dtype)


def sparse_mse(
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
    """Compute a sparse MSE loss between a dense student distribution and sparse teacher targets.

    Uses a chunked approach to avoid memory issues with large sequences.
    """
    return accumulate_over_chunks(
        logits,
        target_ids,
        target_values,
        mask,
        chunk_length,
        sparse_mse_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


def dense_mse(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the MSE between dense student and teacher distributions."""
    out_dtype = logits.dtype

    student_probs = torch.softmax(logits.float() / temperature, dim=-1)
    teacher_probs = torch.softmax(target_logits.float() / temperature, dim=-1)

    mse_per_token = torch.sum((teacher_probs - student_probs) ** 2, dim=-1)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mse_per_token = mse_per_token * mask

    return torch.sum(mse_per_token).to(out_dtype)


class MSELoss(LossFunctionBase):
    temperature: float
    missing: MissingProbabilityHandling
    chunk_length: int | None

    @override
    @classmethod
    def name(cls) -> str:
        return "mse"

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
            res = dense_mse(
                student_outputs.logits,
                signal.logits,
                mask=mask,
                temperature=self.temperature,
            )
        else:
            res = sparse_mse(
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

