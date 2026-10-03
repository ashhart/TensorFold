"""The Volta matmuls: a row never depends on the row count, on the decode kernel or the prompt GEMM, and both match an fp32 reference."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("Volta only", allow_module_level=True)

from tensorfold.cuda.kernels import qmm_volta  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm import dequantize  # noqa: E402

SHAPES = [(48, 5120), (1024, 5120), (5120, 6144), (10240, 5120), (5120, 17408), (17408, 5120), (1000, 256)]
ROWS = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200]


def _weights(n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return words.to(torch.int32), scales, biases


@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(n, k):
    weight, scales, biases = _weights(n, k, n + k)
    x = torch.randn((200, k), generator=torch.Generator(device="cuda").manual_seed(7), device="cuda").to(torch.bfloat16)
    alone = torch.cat([qmm_volta.lane_matmul(x[r:r + 1], weight, scales, biases) for r in range(200)])
    for m in ROWS:
        assert torch.equal(qmm_volta.lane_matmul(x[:m], weight, scales, biases), alone[:m]), f"{n}x{k}: M={m}"
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(qmm_volta.lane_matmul(x[perm], weight, scales, biases), alone[perm])


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("big", [False, True])
def test_accuracy_matches_fp32_reference(n, k, big):
    weight, scales, biases = _weights(n, k, 11 * n + k)
    x = torch.randn((16, k), device="cuda")
    if big:                                   # values far past fp16's range (massive activations) still work
        x[3] *= 3e5
        x[5, :7] = 2e6
    x = x.to(torch.bfloat16)
    y = qmm_volta.lane_matmul(x, weight, scales, biases).float()
    ref = x.float() @ dequantize(weight, scales, biases).T
    for r in range(16):
        err = (y[r] - ref[r]).abs().max().item()
        scale = ref[r].abs().max().item()
        assert err <= scale * 2 ** -7, (r, err, scale)


def test_f32_and_strided_rows():
    weight, scales, biases = _weights(512, 1024, 1)
    big = torch.randn((8, 2048), device="cuda").to(torch.bfloat16)
    x = big[:, :1024]                         # strided rows
    a = qmm_volta.lane_matmul(x, weight, scales, biases, f32=True)
    b = qmm_volta.lane_matmul(x.contiguous(), weight, scales, biases, f32=True)
    assert a.dtype == torch.float32 and torch.equal(a, b)


@pytest.mark.parametrize("n,k", SHAPES)
def test_tiled_layout_round_trips_and_gives_the_same_bits(n, k):
    weight, scales, biases = _weights(n, k, 5 * n + k)
    t = qmm_volta.Tiled(weight, scales, biases)
    assert all(torch.equal(a, b) for a, b in zip(t.untile(), (weight, scales, biases)))
    x = torch.randn((200, k), device="cuda").to(torch.bfloat16)
    alone = torch.cat([qmm_volta.tiled_matmul(x[r:r + 1], t) for r in range(200)])
    for m in ROWS:
        y = qmm_volta.tiled_matmul(x[:m], t)
        assert torch.equal(y, qmm_volta.lane_matmul(x[:m], weight, scales, biases)), (n, k, m)
        assert torch.equal(y, alone[:m]), (n, k, m)


@pytest.mark.parametrize("n,k", SHAPES)
def test_prompt_rows_do_not_depend_on_row_count_or_offset(n, k):
    """prompt_matmul (dense fp16 + one pinned cuBLASLt algorithm a shape): any chunking gives the same bits."""

    weight, scales, biases = _weights(n, k, 3 * n + k)
    t = qmm_volta.Tiled(weight, scales, biases)
    x = torch.randn((1100, k), generator=torch.Generator(device="cuda").manual_seed(5), device="cuda").to(torch.bfloat16)
    whole = qmm_volta.prompt_matmul(x, t)
    for size in (1, 7, 64, 129, 333, 1000):
        parts = torch.cat([qmm_volta.prompt_matmul(x[a:a + size], t) for a in range(0, 1100, size)])
        assert torch.equal(parts, whole), f"{n}x{k}: chunks of {size}"
    ref = x.float() @ qmm_volta.dequant(t).float().t()
    lane = qmm_volta.tiled_matmul(x[:16], t, f32=True)
    assert ((qmm_volta.prompt_matmul(x, t, f32=True) - ref).norm() / ref.norm()).item() < 1e-5
    assert ((lane - ref[:16]).norm() / ref[:16].norm()).item() < 1e-5      # the dense values are the lane kernel's
