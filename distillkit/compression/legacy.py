# Copyright 2024 Charles O. Goddard

import torch

from distillkit.compression.config import (
    LegacyLogitCompressionConfig as LogitCompressionConfig,
)
from distillkit.compression.densify import densify


class LogitCompressor:
    """Compresses and decompresses logits using polynomial approximation."""

    def __init__(self, config: LogitCompressionConfig):
        self.config = config
        self.vocab_index_bits = (
            torch.log2(torch.tensor(self.config.vocab_size, dtype=torch.float32))
            .ceil()
            .item()
        )
        self._validate_config()
        self._setup_polynomial_terms()

    def _validate_config(self):
        assert self.config.exact_k <= self.config.k, (
            "exact_k must be less than or equal to k"
        )
        assert self.config.k > 0, "k must be greater than 0"

    def _setup_polynomial_terms(self):
        a = torch.arange(self.config.k - self.config.exact_k, dtype=torch.float32) + 1
        exponents = [
            -i if self.config.invert_polynomial else i
            for i in range(self.config.polynomial_degree + 1)
        ]
        terms = [a**exp for exp in exponents]
        if self.config.with_sqrt_term:
            terms.append(a.sqrt())
        self.X = torch.stack(terms, dim=-1).unsqueeze(0).unsqueeze(0)

    def compress_from_sparse(
        self, top_indices: torch.LongTensor, top_values: torch.Tensor
    ):
        exact_values = top_values[..., : self.config.exact_k]
        approx_values = top_values[..., self.config.exact_k : self.config.k]

        if self.config.exact_k < self.config.k:
            X = self.X.to(top_values.device, top_values.dtype)
            y = approx_values.unsqueeze(-1)
            coeffs = self._solve_least_squares(X, y).squeeze(-1)

            if self.config.term_dtype != "float32":
                coeffs = coeffs.to(self._str_to_dtype(self.config.term_dtype))
        else:
            coeffs = torch.zeros(
                top_values.shape[:-1] + (0,),
                device=top_values.device,
                dtype=self._str_to_dtype(self.config.term_dtype),
            )

        packed_indices = pack_tensor(top_indices, int(self.vocab_index_bits))
        return packed_indices, exact_values.to(dtype=torch.float16), coeffs

    def compress(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = logits.mean(dim=-1, keepdim=True)
        centered_logits = logits - mean
        top_values, top_indices = torch.topk(centered_logits, self.config.k, dim=-1)
        return self.compress_from_sparse(top_indices, top_values)

    def decompress_to_sparse(
        self,
        packed_indices: torch.Tensor,
        exact_values: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        batch_size, seq_len = packed_indices.shape[:2]
        top_indices = unpack_tensor(
            packed_indices,
            torch.Size([batch_size, seq_len, self.config.k]),
            int(self.vocab_index_bits),
        )
        if self.config.exact_k < self.config.k:
            approx_logits = torch.sum(
                self.X.to(coeffs.device, coeffs.dtype)
                * coeffs.to(dtype=self.X.dtype).unsqueeze(-2),
                dim=-1,
            )
            top_values = torch.cat([exact_values, approx_logits], dim=-1)
        else:
            top_values = exact_values
        return top_indices, top_values

    def decompress(
        self,
        packed_indices: torch.Tensor,
        exact_values: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> torch.Tensor:
        top_indices, top_values = self.decompress_to_sparse(
            packed_indices, exact_values, coeffs
        )
        return densify(
            top_indices,
            top_values,
            self.config.vocab_size,
            renormalize=True,
        )

    @staticmethod
    def _solve_least_squares(A, B):
        # because for some reason torch.linalg.lstsq only works for full-rank matrices on GPU
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        tol = 1e-5
        Spinv = torch.zeros_like(S)
        Spinv[S > tol] = 1 / S[S > tol]
        UhB = U.transpose(-1, -2) @ B
        SpinvUhB = Spinv.unsqueeze(-1) * UhB
        return Vh.transpose(-1, -2) @ SpinvUhB

    @staticmethod
    def _str_to_dtype(dtype_str: str):
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }.get(dtype_str, torch.float32)

    def bytes_per_token(self) -> int:
        index_bits = self.vocab_index_bits * self.config.k
        index_longs = (index_bits + 63) // 64
        index_bytes = index_longs * 8
        return (
            self.config.exact_k * 2
            + self.config.polynomial_degree
            * (self._str_to_dtype(self.config.term_dtype).itemsize)
            + index_bytes
        )


def pack_tensor(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-packs a tensor of integers into a tensor of longs.

    Args:
        x (torch.Tensor): Input tensor of integers
        bits (int): Number of bits to use for each element
    """
    # written by Claude Sonnet 3.5, thanx bro
    assert x.dtype == torch.long, "Input tensor must be of type torch.long"
    assert 1 <= bits <= 63, "Number of bits must be between 1 and 63"

    device = x.device
    max_value = 2**bits - 1
    assert torch.all(x >= 0) and torch.all(x <= max_value), (
        f"All values must be between 0 and {max_value}"
    )

    # Calculate the number of elements that can fit in 64 bits
    elements_per_64bits = 64 // bits

    # Pad the last dimension to be a multiple of elements_per_64bits
    pad_size = (
        elements_per_64bits - x.shape[-1] % elements_per_64bits
    ) % elements_per_64bits
    x_padded = torch.nn.functional.pad(x, (0, pad_size))

    # Reshape the tensor to group elements that will be packed together
    x_reshaped = x_padded.reshape(*x_padded.shape[:-1], -1, elements_per_64bits)

    packed = torch.zeros(*x_reshaped.shape[:-1], dtype=torch.int64, device=device)

    for i in range(elements_per_64bits):
        packed |= x_reshaped[..., i] << (bits * i)

    return packed


def unpack_tensor(
    packed: torch.Tensor, original_size: torch.Size, bits: int
) -> torch.Tensor:
    """Unpacks a bit-packed tensor of longs into a tensor of integers.

    Inverse operation of pack_tensor.

    Args:
        packed (torch.Tensor): Input tensor of longs
        original_size (torch.Size): Original size of the unpacked tensor
        bits (int): Number of bits used for each element
    """
    assert packed.dtype == torch.long, "Packed tensor must be of type torch.long"
    assert 1 <= bits <= 63, "Number of bits must be between 1 and 63"

    device = packed.device
    elements_per_64bits = 64 // bits
    mask = (1 << bits) - 1

    unpacked = torch.zeros(
        *packed.shape[:-1],
        packed.shape[-1] * elements_per_64bits,
        dtype=torch.long,
        device=device,
    )

    for i in range(elements_per_64bits):
        unpacked[..., i::elements_per_64bits] = (packed >> (bits * i)) & mask

    # Trim the unpacked tensor to the original size
    return unpacked[..., : original_size[-1]]
