"""GLM-5.3's CUDA config reads MLX's one bit width, and an EXL3 encode's stated average or label (#226)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")         # the CUDA weights module imports the latent kernels

from tensorfold.families.glm5_next.cuda.weights import bits_of  # noqa: E402


def test_one_mlx_width_reads_and_an_exl3_mix_is_its_average_or_a_label():
    assert bits_of({"bits": 4}) == 4 and bits_of({"bits": "3"}) == 3 and bits_of({}) == 4
    assert bits_of({"quant_method": "exl3", "bits": 3.3333}) == 3.3333
    assert bits_of({"quant_method": "exl3", "bits": "mixed_k34_per_tensor"}) == 0
    with pytest.raises(ValueError, match="one bit width"):
        bits_of({"bits": "mixed_k34_per_tensor"})
