import pytest
import torch

from distillkit.compression.bitpack import pack_to_bytes, unpack_from_bytes


def test_pack_unpack_elem_bits_8():
    x = torch.tensor([255, 128, 64, 0], dtype=torch.long)
    packed = pack_to_bytes(x, 8)
    unpacked = unpack_from_bytes(packed, 8, x.size(-1))
    assert (x == unpacked).all()


def test_pack_unpack_elem_bits_3():
    x = torch.tensor([5, 3], dtype=torch.long)  # 0b101, 0b011
    packed = pack_to_bytes(x, 3)
    assert packed.size(-1) == 1
    unpacked = unpack_from_bytes(packed, 3, x.size(-1))
    assert (x == unpacked).all()


def test_pack_unpack_elem_bits_1():
    x = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.long)  # 9 bits → 2 bytes
    packed = pack_to_bytes(x, 1)
    assert packed.size(-1) == 2
    unpacked = unpack_from_bytes(packed, 1, x.size(-1))
    assert (x == unpacked).all()


def test_pack_unpack_elem_bits_16():
    x = torch.tensor([65535, 32768], dtype=torch.long)
    packed = pack_to_bytes(x, 16)
    unpacked = unpack_from_bytes(packed, 16, x.size(-1))
    assert (x == unpacked).all()


@pytest.mark.parametrize("elem_bits", [5, 9, 17, 53])
def test_pack_unpack_random(elem_bits):
    for _ in range(10):
        original_num_elements = torch.randint(100, 128, (1,)).item()
        x = torch.randint(0, 2**elem_bits, (original_num_elements,), dtype=torch.long)
        packed = pack_to_bytes(x, elem_bits)
        unpacked = unpack_from_bytes(packed, elem_bits, original_num_elements)
        assert x.shape == unpacked.shape, (
            f"Shape mismatch for elem_bits={elem_bits}, original_num_elements={original_num_elements}"
        )
        assert (x == unpacked).all(), (
            f"Random test failed for elem_bits={elem_bits}, original_num_elements={original_num_elements}"
        )
