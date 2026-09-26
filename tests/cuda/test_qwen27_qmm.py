"""The CUDA lane matmul: rows never depend on the row count, and accuracy matches an fp32 reference."""

import pytest
import torch

cuda = pytest.importorskip("torch").cuda
if not cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import qmm  # noqa: E402

SHAPES = [(48, 5120), (1024, 5120), (5120, 6144), (10240, 5120), (5120, 17408)]
ROWS = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128]


def _weights(n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    weight = words.to(torch.int32)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return weight, scales, biases


@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(n, k):
    weight, scales, biases = _weights(n, k, n + k)
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((128, k), generator=g, device="cuda").to(torch.bfloat16)
    alone = torch.cat([qmm.lane_matmul(x[r:r + 1], weight, scales, biases) for r in range(128)])
    for m in ROWS:
        batch = qmm.lane_matmul(x[:m], weight, scales, biases)
        assert torch.equal(batch, alone[:m]), f"{n}x{k}: rows differ at M={m}"
    # a row placed anywhere in a window gives the same bits
    perm = torch.randperm(128, generator=torch.Generator().manual_seed(3)).cuda()
    shuffled = qmm.lane_matmul(x[perm], weight, scales, biases)
    assert torch.equal(shuffled, alone[perm])


@pytest.mark.parametrize("n,k", SHAPES)
def test_accuracy_matches_fp32_reference(n, k):
    weight, scales, biases = _weights(n, k, 11 * n + k)
    x = torch.randn((16, k), device="cuda").to(torch.bfloat16)
    y = qmm.lane_matmul(x, weight, scales, biases).float()
    ref = x.float() @ qmm.dequantize(weight, scales, biases).T
    err = (y - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= scale * 2 ** -7, (err, scale)


def test_split_depends_only_on_shape():
    for n, k in SHAPES:
        assert qmm.split_k(n, k) == qmm.split_k(n, k)
        assert (k // 64) % qmm.split_k(n, k) == 0


@pytest.mark.parametrize("n,k", SHAPES)
def test_tiled_layout_gives_the_same_bits(n, k):
    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    weight, scales, biases = _weights(n, k, 5 * n + k)
    q = QLinear(weight, scales, biases)
    t = qmm_fast.tile(q)
    back = qmm_fast.untile(t)
    assert torch.equal(back.weight, weight) and torch.equal(back.scales, scales)
    x = torch.randn((128, k), device="cuda").to(torch.bfloat16)
    for m in (1, 7, 16, 17, 32, 33, 64, 100, 128):
        assert torch.equal(qmm_fast.matmul(x[:m], t), qmm.lane_matmul(x[:m], weight, scales, biases)), (n, k, m)
