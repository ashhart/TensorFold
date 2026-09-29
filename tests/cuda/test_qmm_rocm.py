"""ROCm's decode matmul (qmm_rocm.cu): a row's bits never depend on how many rows share the call."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import hip  # noqa: E402

if not hip():
    pytest.skip("the HIP decode matmul is ROCm's", allow_module_level=True)

from tensorfold.cuda.kernels import qmm_groups  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm import dequantize  # noqa: E402


def _weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    return (words, scales, biases), gen


@pytest.mark.parametrize("wmma", [False, True])
@pytest.mark.parametrize("n,k", [(1000, 1024), (48, 5120), (5120, 17408), (17408, 5120), (130, 64), (1, 128)])
def test_rows_do_not_depend_on_the_row_count(n, k, wmma):
    mlx, gen = _weight(n, k, n + k)
    g = qmm_groups.to_groups(*mlx)
    x = torch.randn(40, k, generator=gen, device="cuda").bfloat16()
    whole = qmm_groups.gemv(x, *g, n, wmma=wmma)
    for m in (1, 12, 16, 17, 33):
        assert torch.equal(qmm_groups.gemv(x[:m].contiguous(), *g, n, wmma=wmma), whole[:m]), m
    assert torch.equal(qmm_groups.gemv(x, *g, n, f32=True, wmma=wmma).bfloat16(), whole)
    ref = x.double() @ dequantize(*mlx).double().t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 5e-3
