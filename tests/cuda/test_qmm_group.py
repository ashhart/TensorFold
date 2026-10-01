"""The grouped sm_12x lane matmul: each projection in a group keeps the serial reference's bits at any row count."""

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("grouped launches run on sm_12x (GB10, RTX 50, RTX PRO 6000) only", allow_module_level=True)

from tensorfold.cuda.kernels import qmm  # noqa: E402
from tensorfold.families.qwen3_5.cuda import qmm as triton_qmm  # noqa: E402

ROWS = [1, 2, 7, 8, 9, 16, 17, 33, 64, 65, 100, 128, 129]
TILES = list(range(13))                              # 0 picks by rows and chip; 6 and 7 are the swapped 8-row tiles
GROUPS = {"gdn": [(10240, 5120), (6144, 5120), (48, 5120), (48, 5120)],      # K splits 2, 2, 8, 8 in one launch
          "attention": [(12288, 5120), (1024, 5120), (1024, 5120)],          # 1, 8, 8
          "mlp": [(17408, 5120), (17408, 5120)],                             # 1, 1
          "out": [(5120, 6144)], "down": [(5120, 17408)],                    # 4; 4
          "drafter_attention": [(4096, 5120), (1024, 5120), (1024, 5120)], "drafter_o": [(5120, 4096)]}
MODEL = Path(os.environ.get("TF_QWEN27_MODEL", "/models/Qwen3.8-27B-MLX-4bit"))


def _weights(n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return words.to(torch.int32), scales, biases


def _check(ws, rows) -> None:
    qs = [qmm.pack(*w, 64) for w in ws]
    k = qs[0].k
    x = torch.randn((max(rows), k), generator=torch.Generator(device="cuda").manual_seed(k), device="cuda").bfloat16()
    for m in rows:
        want = [triton_qmm.lane_matmul(x[:m], *w) for w in ws]
        for tile in TILES:
            for early in (0, 1):                          # launches overlapped with the previous kernel or not
                got = qmm.matmul_group(x[:m], qs, tile=tile, early=early)
                assert all(torch.equal(a, b) for a, b in zip(got, want)), (m, tile, early)
        assert all(torch.equal(qmm.matmul(x[:m], q), b) for q, b in zip(qs, want)), m


@pytest.mark.parametrize("name", list(GROUPS))
def test_parts_keep_the_serial_reference_bits(name):
    _check([_weights(n, k, 3 * n + i) for i, (n, k) in enumerate(GROUPS[name])], ROWS)


@pytest.mark.parametrize("name", list(GROUPS))
def test_every_row_count_to_256_keeps_the_reference_bits(name):
    """matmul_group's own block at every row count from 1 to 256 (128-row blocks on a wide SM 12.0): the bits."""

    ws = [_weights(n, k, 5 * n + i) for i, (n, k) in enumerate(GROUPS[name])]
    qs = [qmm.pack(*w, 64) for w in ws]
    k = qs[0].k
    x = torch.randn((256, k), generator=torch.Generator(device="cuda").manual_seed(k + 1), device="cuda").bfloat16()
    want = [triton_qmm.lane_matmul(x, *w) for w in ws]                    # rows alone: a prefix is its own rows
    bad = [m for m in range(1, 257)
           if not all(torch.equal(a, b[:m]) for a, b in zip(qmm.matmul_group(x[:m], qs), want))]
    assert not bad, bad[:10]


def test_rows_past_the_tile_and_strided_rows():
    """A strided input and fp32 sums give the bits of a contiguous one, part by part."""

    ws = [_weights(n, k, n) for n, k in GROUPS["gdn"]]
    qs = [qmm.pack(*w, 64) for w in ws]
    wide = torch.randn((24, 5120 + 64), device="cuda").bfloat16()
    for f32 in (False, True):
        got = qmm.matmul_group(wide[:, :5120], qs, f32=f32)
        want = [qmm.matmul(wide[:, :5120].contiguous(), q, f32=f32) for q in qs]
        assert all(torch.equal(a, b) for a, b in zip(got, want)), f32


@pytest.mark.skipif(not MODEL.exists(), reason=f"needs the 27B checkpoint at {MODEL} (set TF_QWEN27_MODEL)")
def test_real_27b_projections_keep_their_bits():
    from tensorfold.families.qwen3_5.cuda.weights import _Tensors

    t = _Tensors(MODEL, "cuda")
    prefix = "language_model." if any(name.startswith("language_model.") for name in t) else ""

    def get(name: str):
        name = prefix + name
        return t.pop(name + ".weight").view(torch.int32), t.pop(name + ".scales"), t.pop(name + ".biases")

    groups = [[f"model.layers.0.linear_attn.in_proj_{p}" for p in ("qkv", "z", "b", "a")],
              [f"model.layers.3.self_attn.{p}_proj" for p in ("q", "k", "v")],
              [f"model.layers.0.mlp.{p}_proj" for p in ("gate", "up")],
              ["model.layers.0.linear_attn.out_proj"], ["model.layers.3.self_attn.o_proj"],
              ["model.layers.0.mlp.down_proj"]]
    try:
        for names in groups:
            _check([get(name) for name in names], [1, 2, 16, 64, 128])
    finally:
        t.close()
