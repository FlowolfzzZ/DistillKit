import torch


def pack_to_bytes(
    x: torch.LongTensor,
    elem_bits: int,
) -> torch.ByteTensor:
    """
    Pack a tensor of integers into a byte tensor.

    Args:
        x (torch.LongTensor): The input tensor of integers, with shape (..., N).
        elem_bits (int): The number of bits per element. Must be between 1 and 64.

    Returns:
        torch.ByteTensor: The packed byte tensor, with shape (..., ceil(N * elem_bits / 8)).
    """
    assert 1 <= elem_bits <= 64, "elem_bits must be between 1 and 64"

    # Mask the input to elem_bits
    mask = (1 << elem_bits) - 1
    x = x & mask

    # Generate positions of each bit, from MSB to LSB within each element
    bit_positions = torch.arange(elem_bits - 1, -1, -1, device=x.device)

    # Expand each element into its constituent bits as uint8 (..., N, elem_bits)
    bits = ((x.unsqueeze(-1) >> bit_positions) & 1).to(torch.uint8)

    # Flatten the bits into a single bit stream (..., N * elem_bits)
    original_shape = x.shape
    bits = bits.view(*original_shape[:-1], -1)

    # Calculate padding needed to make total bits a multiple of 8
    total_bits = bits.size(-1)
    pad_length = (8 - (total_bits % 8)) % 8
    if pad_length > 0:
        bits = torch.nn.functional.pad(bits, (0, pad_length))
    bits = bits.contiguous()  # Ensure contiguous for efficient reshaping

    # Reshape into groups of 8 bits and convert to bytes
    bits = bits.view(*original_shape[:-1], -1, 8)
    # Precompute power as uint8 for efficiency
    power = torch.tensor(
        [128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=x.device
    )
    bytes = (bits * power).sum(dim=-1).to(torch.uint8)

    return bytes


def unpack_from_bytes(
    bytes_tensor: torch.ByteTensor,
    elem_bits: int,
    original_num_elements: int,
) -> torch.LongTensor:
    """
    Unpack a byte tensor back into the original integers.

    Args:
        bytes_tensor (torch.ByteTensor): The packed byte tensor, with shape (..., ceil(N * elem_bits / 8)).
        elem_bits (int): The number of bits per element used during packing. Must be between 1 and 64.
        original_num_elements (int): The number of elements in the original tensor along the last dimension (N).

    Returns:
        torch.LongTensor: The unpacked tensor of integers, with shape (..., N).
    """
    assert 1 <= elem_bits <= 64, "elem_bits must be between 1 and 64"
    assert original_num_elements >= 0, "original_num_elements must be non-negative"

    total_bits_needed = original_num_elements * elem_bits
    original_shape = bytes_tensor.shape
    M = original_shape[-1]
    total_bits_available = M * 8

    if total_bits_needed > total_bits_available:
        raise ValueError(
            f"original_num_elements {original_num_elements} with elem_bits {elem_bits} "
            f"requires {total_bits_needed} bits, but only {total_bits_available} available"
        )

    # Convert each byte to 8 bits (MSB to LSB) as uint8
    bit_positions = torch.arange(7, -1, -1, device=bytes_tensor.device)  # 7, 6, ..., 0
    bits = ((bytes_tensor.unsqueeze(-1) >> bit_positions) & 1).to(
        torch.uint8
    )  # (..., M, 8)
    bits_flat = bits.view(*original_shape[:-1], -1)  # (..., M*8)

    # Slice to get the needed bits and discard padding
    bits_needed = bits_flat[..., :total_bits_needed]

    # Reshape into (..., original_num_elements, elem_bits) ensuring contiguous
    new_shape = list(original_shape[:-1]) + [original_num_elements, elem_bits]
    bits_needed = bits_needed.contiguous().view(*new_shape)

    # Convert bits to integers using appropriate power tensor
    powers = 2 ** torch.arange(elem_bits - 1, -1, -1, device=bits_needed.device)
    result = (bits_needed * powers).sum(dim=-1).long()

    return result
