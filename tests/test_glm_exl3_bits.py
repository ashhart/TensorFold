"""GLM-5.3's CUDA config reads one bit width a checkpoint and refuses a mixed-bit EXL3 encode by name (#226)."""

import pytest

pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda.weights import bits_of  # noqa: E402


def test_one_bit_width_reads_and_a_mixed_encode_is_refused_by_name():
    assert bits_of({"bits": 4}) == 4 and bits_of({"bits": "3"}) == 3 and bits_of({}) == 4
    with pytest.raises(ValueError, match="mixed-bit EXL3 encodes are not supported yet"):
        bits_of({"bits": "mixed_k34_per_tensor"})
