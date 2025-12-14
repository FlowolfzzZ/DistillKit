import torch

from distillkit.compression.bitpack import pack_to_bytes, unpack_from_bytes
from distillkit.compression.config import (
    DistributionQuantizationConfig,
    SpecialTerm,
)


def _work_dtype(*inputs: torch.Tensor | None) -> torch.dtype:
    for x in inputs:
        if x is not None and x.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _solve_least_squares(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    work_dtype = _work_dtype(A, B)

    # because for some reason torch.linalg.lstsq only works for full-rank matrices on GPU
    U, S, Vh = torch.linalg.svd(A.to(work_dtype), full_matrices=False)
    tol = 1e-5
    Spinv = torch.zeros_like(S)
    Spinv[S > tol] = 1 / S[S > tol]
    UhB = U.transpose(-1, -2) @ B.to(work_dtype)
    SpinvUhB = Spinv.unsqueeze(-1) * UhB
    return Vh.transpose(-1, -2) @ SpinvUhB


def polynomial_terms(
    terms: list[SpecialTerm | int],
    t: int,
    dtype: torch.dtype,
    device: torch.device,
    normalize_t: bool,
) -> torch.Tensor:
    assert all(isinstance(i, (int, SpecialTerm)) for i in terms), (
        "terms must be a list of integers or SpecialTerm instances"
    )
    if normalize_t:
        pts = torch.linspace(0, 1, steps=t, dtype=dtype, device=device)
    else:
        pts = torch.arange(t, dtype=dtype, device=device)
    X = torch.stack(
        [pts**i if isinstance(i, int) else getattr(torch, i.value)(pts) for i in terms],
        dim=-1,
    )
    return X


def fit_polynomial(
    values: torch.Tensor,
    terms: list[SpecialTerm | int],
    dtype: torch.dtype,
    normalize_t: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = _work_dtype(values)
    X = polynomial_terms(
        terms,
        values.shape[-1],
        dtype=work_dtype,
        device=values.device,
        normalize_t=normalize_t,
    )
    while len(X.shape) < len(values.shape):
        X = X.unsqueeze(0)

    y = values.unsqueeze(-1)
    coeffs = _solve_least_squares(X, y).squeeze(-1)

    coeffs_final = coeffs.to(dtype)
    approx = torch.sum(
        X.to(dtype) * coeffs_final.unsqueeze(-2),
        dim=-1,
    ).to(work_dtype)
    residual = values - approx.squeeze(-1)
    return coeffs_final, residual


def _get_quantize_range(element_bits: int):
    if element_bits == 1:
        quant_min = 0
        quant_max = 1
    else:
        quant_min = -(2 ** (element_bits - 1))
        quant_max = (2 ** (element_bits - 1)) - 1.0

    return quant_min, quant_max


def _get_quantize_scale_factors(values: torch.Tensor, element_bits: int):
    # Compute max absolute value for each group (..., 1)
    max_abs_val = torch.amax(torch.abs(values), dim=-1, keepdim=True)

    # Determine the maximum quantized absolute value based on element_bits
    if element_bits == 1:
        max_quant_abs = 1.0
    else:
        max_quant_abs = 2 ** (element_bits - 1)

    return max_abs_val, max_quant_abs


def error_diffuse_and_quantize(
    values: torch.Tensor,
    element_bits: int,
    scale_dtype: torch.dtype,
    error_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """
    Quantize the input tensor to the specified number of bits per element using error diffusion.
    """
    original_shape = values.shape
    n = original_shape[-1]

    (max_abs_val, max_quant_abs) = _get_quantize_scale_factors(values, element_bits)

    # Calculate scale factor, avoiding division by zero
    scale_factor = torch.where(
        max_abs_val == 0, torch.ones_like(max_abs_val), max_abs_val / max_quant_abs
    )

    # Simulate output precision
    scale_factor = scale_factor.to(scale_dtype).to(values.dtype)

    # Scale the input values
    scaled_vals = values / scale_factor

    # Initialize error buffer and quantized tensor
    if error_buffer is None:
        error_buffer = torch.zeros_like(scaled_vals[..., 0])
    quantized_values = torch.zeros_like(scaled_vals, dtype=torch.long)

    quant_min, quant_max = _get_quantize_range(element_bits)

    # Process each element along the last dimension with error diffusion
    for i in range(n):
        current = scaled_vals[..., i] + error_buffer
        quantized_i = torch.round(current)
        quantized_i = torch.clamp(quantized_i, quant_min, quant_max)
        error = current - quantized_i
        error_buffer = error
        quantized_values[..., i] = (quantized_i - quant_min).to(torch.long)

    return scale_factor.to(scale_dtype), quantized_values, error_buffer


def error_diffuse_float(
    values: torch.Tensor,
    out_dtype: torch.dtype,
    error_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = _work_dtype(values, error_buffer)
    values = values.to(work_dtype)
    if error_buffer is None:
        error_buffer = torch.zeros_like(values[..., 0])

    out_values = torch.zeros_like(values, dtype=out_dtype)
    n = values.shape[-1]
    for i in range(n):
        current = values[..., i] + error_buffer
        q = current.to(out_dtype)
        error = current - q.to(work_dtype)
        error_buffer = error
        out_values[..., i] = q

    return out_values, error_buffer


def quantize_naive(
    values: torch.Tensor,
    element_bits: int,
    scale_dtype: torch.dtype,
    error_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """
    Naive quantization of the input tensor to the specified number of bits per element.
    """
    work_dtype = _work_dtype(values, error_buffer)
    values = values.to(work_dtype)

    (max_abs_val, max_quant_abs) = _get_quantize_scale_factors(values, element_bits)

    # Calculate scale factor, avoiding division by zero
    scale_factor = torch.where(
        max_abs_val == 0, torch.ones_like(max_abs_val), max_abs_val / max_quant_abs
    )

    # Simulate output precision
    scale_factor = scale_factor.to(scale_dtype).to(values.dtype)

    # Scale the input values
    scaled_vals = values / scale_factor

    # Quantize the scaled values
    quant_min, quant_max = _get_quantize_range(element_bits)
    quantized_values = (
        torch.round(scaled_vals).clamp(quant_min, quant_max) - quant_min
    ).to(torch.long)

    return (
        scale_factor,
        quantized_values,
        torch.zeros_like(scaled_vals[..., 0]),
    )


def dequantize(
    quantized_values: torch.LongTensor,
    scale: torch.Tensor,
    element_bits: int,
) -> torch.Tensor:
    """
    Dequantize the input tensor using the specified scale and number of bits per element.
    """
    if element_bits == 1:
        return (quantized_values.to(torch.float32) * 2.0 - 1.0) * scale
    quant_min, quant_max = _get_quantize_range(element_bits)
    dequantized_values = (quantized_values + quant_min).to(torch.float32) * scale
    return dequantized_values


def compress_monotonic_logprobs(
    logprobs: torch.Tensor,
    config: DistributionQuantizationConfig,
) -> torch.ByteTensor:
    """
    Compresses logprobs using the specified configuration.
    Args:
        logprobs (torch.Tensor): Log probabilities to compress, shape (batch_size, seq_len, k). Must be
            monotonically decreasing along the last dimension.
        config (DistributionQuantizationConfig): Configuration for compression.
    Returns:
        torch.ByteTensor: Compressed logprobs. Shape (batch_size, seq_len, ceil(config.total_bits() / 8)).
    """
    work_dtype = _work_dtype(logprobs)
    if config.delta_encoding:
        # Apply delta encoding
        # Logprobs are all <= 0 and monotonically decreasing
        # replace logprobs with their deltas with respect to previous value
        # the first value is unchanged
        if config.error_diffusion:
            work_dtype = torch.float64
        logprobs_work = logprobs.to(work_dtype)
        deltas = logprobs_work[..., 1:] - logprobs_work[..., :-1]
        deltas = torch.cat(
            [logprobs[..., :1], deltas],
            dim=-1,
        )
        if config.error_diffusion:
            logprobs, _ = error_diffuse_float(deltas, logprobs.dtype, error_buffer=None)
        else:
            logprobs = deltas.to(logprobs.dtype)

    chunks = []

    error_buffer = None
    if config.exact_k > 0:
        exact_values = logprobs[..., : config.exact_k].to(config.exact_dtype.dtype())
        chunks.append(
            exact_values.view(torch.uint8).reshape(
                *logprobs.shape[:-1],
                -1,
            ),
        )

    if config.polynomial_terms:
        approx_values = logprobs[..., config.exact_k : config.k]
        coeffs, residual = fit_polynomial(
            approx_values,
            config.polynomial_terms,
            dtype=config.term_dtype.dtype(),
            normalize_t=config.normalize_t,
        )

        coeffs = coeffs.to(config.term_dtype.dtype())
        coeff_bytes = coeffs.view(torch.uint8).reshape(
            *logprobs.shape[:-1],
            -1,
        )
        chunks.append(coeff_bytes)
    else:
        residual = logprobs[..., config.exact_k : config.k]

    cur_index = 0
    for bin in config.residual_bins:
        values = residual[..., cur_index : cur_index + bin.num_elements]
        if config.error_diffusion:
            scale, scaled, error_buffer = error_diffuse_and_quantize(
                values, bin.element_bits, bin.scale_dtype.dtype(), error_buffer
            )
        else:
            scale, scaled, error_buffer = quantize_naive(
                values, bin.element_bits, bin.scale_dtype.dtype(), error_buffer
            )

        scale = scale.to(bin.scale_dtype.dtype())
        scale_bytes = scale.view(torch.uint8).reshape(
            *logprobs.shape[:-1],
            -1,
        )
        chunks.append(scale_bytes)

        packed = pack_to_bytes(scaled, bin.element_bits)
        packed = packed.reshape(
            *logprobs.shape[:-1],
            -1,
        )
        chunks.append(packed)

        cur_index += bin.num_elements

    # return byte tensor
    return torch.cat(chunks, dim=-1)


def decompress_monotonic_logprobs(
    bytes: torch.ByteTensor,
    config: DistributionQuantizationConfig,
    out_dtype: torch.dtype | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """
    Decompresses logprobs using the specified configuration.

    Args:
        bytes (torch.ByteTensor): Compressed logprobs, shape (batch_size, seq_len, num_bytes).
        config (DistributionQuantizationConfig): Configuration for decompression.
        out_dtype (torch.dtype | None): Data type for the output tensor. If None, uses the dtype for the
            exact values.
    Returns:
        torch.Tensor: Decompressed logprobs, shape (batch_size, seq_len, k).
    """
    device = bytes.device
    if out_dtype is None:
        out_dtype = config.exact_dtype.dtype()

    # Extract exact values
    if config.exact_k > 0:
        exact_dtype_torch = config.exact_dtype.dtype()
        bytes_per_exact = config.exact_dtype.bit_width() // 8
        exact_bytes = config.exact_k * bytes_per_exact
        exact_part = bytes[..., :exact_bytes].contiguous()
        exact_values = exact_part.view(dtype=exact_dtype_torch).reshape(
            *bytes.shape[:-1], config.exact_k
        )
        remaining_bytes = bytes[..., exact_bytes:]
    else:
        exact_values = torch.empty(
            (*bytes.shape[:-1], 0), dtype=out_dtype, device=device
        )
        remaining_bytes = bytes

    # Extract polynomial coefficients if applicable
    if config.polynomial_terms and len(config.polynomial_terms) > 0:
        term_dtype_torch = config.term_dtype.dtype()
        terms_count = len(config.polynomial_terms)
        coeff_bytes = terms_count * (config.term_dtype.bit_width() // 8)
        coeff_part = remaining_bytes[..., :coeff_bytes].contiguous()
        coeffs = coeff_part.view(dtype=term_dtype_torch).reshape(
            *remaining_bytes.shape[:-1], terms_count
        )
        remaining_bytes = remaining_bytes[..., coeff_bytes:]
    else:
        coeffs = None
        terms_count = 0

    # Process residual bins
    residuals = []
    for bin in config.residual_bins:
        # Extract scale
        scale_dtype_torch = bin.scale_dtype.dtype()
        scale_bytes = bin.scale_dtype.bit_width() // 8
        scale_part = remaining_bytes[..., :scale_bytes].contiguous()
        scale = scale_part.view(dtype=scale_dtype_torch).reshape(
            *remaining_bytes.shape[:-1], 1
        )
        remaining_bytes = remaining_bytes[..., scale_bytes:]

        # Extract and unpack elements
        num_elements = bin.num_elements
        element_bits = bin.element_bits
        packed_bits = num_elements * element_bits
        packed_bytes = (packed_bits + 7) // 8
        packed_part = remaining_bytes[..., :packed_bytes].contiguous()
        remaining_bytes = remaining_bytes[..., packed_bytes:]

        elements = unpack_from_bytes(packed_part, element_bits, num_elements)
        residual_bin = dequantize(
            elements,
            scale,
            element_bits,
        )
        residuals.append(residual_bin)

    # Combine residuals and pad if necessary
    approx_terms = config.k - config.exact_k
    sum_bin_elems = sum(bin.num_elements for bin in config.residual_bins)
    if use_residual and residuals:
        residual = torch.cat(residuals, dim=-1)
        if sum_bin_elems < approx_terms:
            residual = torch.nn.functional.pad(
                residual, (0, approx_terms - sum_bin_elems)
            )
    else:
        residual = torch.zeros(
            (*remaining_bytes.shape[:-1], approx_terms), dtype=out_dtype, device=device
        )

    # Compute polynomial approximation
    if (
        coeffs is not None
        and config.polynomial_terms
        and len(config.polynomial_terms) > 0
    ):
        X = polynomial_terms(
            terms=config.polynomial_terms,
            t=approx_terms,
            dtype=config.term_dtype.dtype(),
            device=device,
            normalize_t=config.normalize_t,
        )
        fit = torch.sum(
            X.to(coeffs.device, coeffs.dtype) * coeffs.unsqueeze(-2),
            dim=-1,
        )
        approx_values = fit + residual.to(out_dtype)
    else:
        approx_values = residual.to(out_dtype)

    logprobs = torch.cat([exact_values.to(out_dtype), approx_values], dim=-1)
    if config.delta_encoding:
        # Apply inverse delta encoding
        logprobs = torch.cumsum(logprobs.float(), dim=-1).to(out_dtype)
    return logprobs
