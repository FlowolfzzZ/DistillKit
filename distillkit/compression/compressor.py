import logging
from typing import Any

import torch

from distillkit.compression.bitpack import (
    pack_to_bytes,
    unpack_from_bytes,
)
from distillkit.compression.config import (
    DistributionQuantizationConfig,
    LegacyLogitCompressionConfig,
)
from distillkit.compression.legacy import LogitCompressor as LegacyLogitCompressor
from distillkit.compression.monotonic_logprobs import (
    compress_monotonic_logprobs,
    decompress_monotonic_logprobs,
)

LOG = logging.getLogger(__name__)


class LogprobCompressor:
    config: DistributionQuantizationConfig | None
    legacy_compressor: LegacyLogitCompressor | None

    def __init__(
        self,
        config: DistributionQuantizationConfig | None = None,
        legacy_config: LegacyLogitCompressionConfig | None = None,
    ):
        if config is not None and legacy_config is not None:
            raise ValueError(
                "At most one of `config` or `legacy_config` should be provided."
            )
        self.config = config
        if legacy_config is not None:
            self.legacy_compressor = LegacyLogitCompressor(legacy_config)
            self.vocab_index_bits = int(self.legacy_compressor.vocab_index_bits)
        else:
            self.legacy_compressor = None
            self.vocab_index_bits = int(
                torch.log2(torch.tensor(self.config.d, dtype=torch.float32))
                .ceil()
                .item()
            )

    def compress_from_sparse(
        self, indices: torch.LongTensor, logprobs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.legacy_compressor is not None:
            packed_indices, exact_values, coeffs = (
                self.legacy_compressor.compress_from_sparse(indices, logprobs)
            )
            return {
                "packed_indices": packed_indices,
                "exact_values": exact_values,
                "coeffs": coeffs,
            }
        elif self.config is None:
            raise ValueError("No config provided for compression.")

        # enforce monotonicity
        _, sorted_indices = torch.sort(logprobs, descending=True, dim=-1)
        sorted_values = logprobs.gather(-1, sorted_indices)
        sorted_indices = indices.gather(-1, sorted_indices)

        logprob_bytes = compress_monotonic_logprobs(
            sorted_values,
            self.config,
        )
        indices_bytes = pack_to_bytes(
            sorted_indices,
            self.vocab_index_bits,
        )
        return {
            "compressed_logprobs": logprob_bytes,
            "bytepacked_indices": indices_bytes,
        }

    def decompress_to_sparse(
        self,
        row: dict[str, Any],
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        if "top_values" in row and "token_ids" in row:
            return row["token_ids"], row["top_values"]
        elif "packed_indices" in row and "exact_values" in row and "coeffs" in row:
            if self.legacy_compressor is None:
                raise ValueError("Row is in legacy format, but compressor is not.")
            return self.legacy_compressor.decompress_to_sparse(
                row["packed_indices"], row["exact_values"], row["coeffs"]
            )
        elif "compressed_logprobs" in row and "bytepacked_indices" in row:
            if self.config is None:
                raise ValueError("Row is in new format, but compressor is not.")
            logprobs = decompress_monotonic_logprobs(
                row["compressed_logprobs"].to(torch.uint8),
                self.config,
            )
            indices = unpack_from_bytes(
                row["bytepacked_indices"].to(torch.uint8),
                self.vocab_index_bits,
                original_num_elements=logprobs.shape[-1],
            )
            return indices, logprobs
        else:
            raise ValueError(
                "Unknown row format. Expected either raw top-k, legacy compressed, or new compressed format."
            )

    def compress(
        self,
        logprobs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.legacy_compressor is not None:
            k = self.legacy_compressor.config.k
        else:
            k = self.config.k

        sparse_logprobs, sparse_indices = torch.topk(logprobs, k, dim=-1)
        return self.compress_from_sparse(sparse_indices, sparse_logprobs)
