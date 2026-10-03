"""The 8-bit lane matmul: every byte exact, a row's bits independent of the row count, tile and kernel, fp32 sums."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import affine, qmm  # noqa: E402
from tensorfold.families.qwen3_5.cuda import qmm_fast  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import QLinear  # noqa: E402

ROWS = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 256]
SHAPES = [(32, 2048), (512, 2048), (1000, 2048), (8192, 2048), (2048, 4096), (5120, 17408)]
GROUPS = {"gdn": [(8192, 2048), (4096, 2048), (32, 2048), (32, 2048)], "attention": [(8192, 2048), (512, 2048),
          (512, 2048)], "out": [(2048, 4096)], "down": [(5120, 17408)]}
TILES = list(range(13))                              # 0 picks by rows and chip; 6 and 7 are the swapped 8-row tiles
MODEL = Path(os.environ.get("TF_AFFINE8_MODEL", "/models/Qwen3.6-35B-A3B-MLX-8bit"))
grouped_chip = torch.cuda.get_device_capability()[0] == 12


def _weights(n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 4), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.002 + 0.0001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return words.to(torch.int32), scales, biases


def _dequant(words, scales, biases) -> torch.Tensor:
    n = words.shape[0]
    w = words.to(torch.int64) & 0xFFFFFFFF
    q = ((w[:, :, None] >> (torch.arange(4, device=w.device) * 8)) & 0xFF).reshape(n, -1).double()
    return q * scales.double().repeat_interleave(64, 1) + biases.double().repeat_interleave(64, 1)


@pytest.fixture(params=["group", "lane"] if grouped_chip else ["lane"])
def kernel(request, monkeypatch):
    """``group``: qmm_group.cu (sm_12x); ``lane``: qmm.cu, which every other GPU runs."""

    if request.param == "lane":
        monkeypatch.setattr(qmm, "grouped", lambda device: False)
    return request.param


@pytest.mark.parametrize("n,k", SHAPES)
def test_pack_round_trip(n, k):
    w = _weights(n, k, n + k)
    q = qmm.pack(*w, 64, bits=8)
    assert q.bits == 8 and q.weight.shape == (-(-n // 128) * 2, k // 64, 8, 32, 4)
    assert all(torch.equal(a, b) for a, b in zip(qmm.unpack(q), w))


@pytest.mark.parametrize("f32", [False, True])
def test_every_byte_decodes_exactly(kernel, f32):
    """One-hot rows read back each stored byte (scale 1, bias 0) through every K split: q itself, 0-255."""

    n, k = 128, 4096
    words = _weights(n, k, 29)[0]
    ones = torch.ones((n, k // 64), device="cuda").bfloat16()
    q = qmm.pack(words, ones, torch.zeros_like(ones), 64, bits=8)
    want = _dequant(words, ones, torch.zeros_like(ones)).T.float()
    assert qmm.split_k(n, k, bits=8) > 1 and set(want.unique().tolist()) == set(range(256))
    assert torch.equal(qmm.matmul(torch.eye(k, device="cuda").bfloat16(), q, f32=f32).float(), want)


@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(kernel, n, k):
    q = qmm.pack(*_weights(n, k, 3 * n + k), 64, bits=8)
    x = torch.randn((max(ROWS), k), generator=torch.Generator(device="cuda").manual_seed(7), device="cuda").bfloat16()
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), f"{n}x{k}: rows differ at M={m}"
    perm = torch.randperm(max(ROWS), generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])
    f32 = qmm.matmul(x[:40], q, f32=True)
    assert torch.equal(torch.cat([qmm.matmul(x[r:r + 1], q, f32=True) for r in range(40)]), f32)


@pytest.mark.skipif(not grouped_chip, reason="grouped launches run on sm_12x only")
@pytest.mark.parametrize("name", list(GROUPS))
def test_every_group_tile_keeps_qmm_cu_bits(name, monkeypatch):
    """The grouped kernel's tiles, overlapped or not, give each part qmm.cu's bits for the same rows."""

    ws = [_weights(n, k, 5 * n + i) for i, (n, k) in enumerate(GROUPS[name])]
    qs = [qmm.pack(*w, 64, bits=8) for w in ws]
    x = torch.randn((129, qs[0].k), generator=torch.Generator(device="cuda").manual_seed(1), device="cuda").bfloat16()
    with monkeypatch.context() as m:
        m.setattr(qmm, "grouped", lambda device: False)
        want = [qmm.matmul(x, q) for q in qs]
    for rows in (1, 2, 8, 9, 16, 17, 33, 64, 65, 129):
        for tile in TILES:
            for early in (0, 1):
                got = qmm.matmul_group(x[:rows], qs, tile=tile, early=early)
                assert all(torch.equal(a, b[:rows]) for a, b in zip(got, want)), (rows, tile, early)


def _as_eight(words4: torch.Tensor, shift: int) -> torch.Tensor:
    """The 4-bit codes of MLX words as 8-bit MLX words holding q << shift."""

    w = words4.to(torch.int64) & 0xFFFFFFFF
    q = ((w[:, :, None] >> (torch.arange(8, device=w.device) * 4)) & 0xF) << shift       # (n, k/8, 8)
    q = q.reshape(w.shape[0], -1, 4)
    packed = (q << (torch.arange(4, device=w.device) * 8)).sum(-1)
    return torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)


@pytest.mark.parametrize("shift", [0, 4])
def test_8bit_words_holding_4bit_codes_give_the_4bit_bits(kernel, shift):
    """q8 = q4 (or 16 q4 with scales / 16, exact) at the same K split: the same products in the same order as the
    4-bit kernel, so the same bits, which test_qmm_group ties to the 27B's Triton reference."""

    n, k = 2048, 4096
    g = torch.Generator(device="cuda").manual_seed(13)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    four = qmm.pack(words.to(torch.int32), scales, biases, 64)
    eight = qmm.pack(_as_eight(words.to(torch.int32), shift), scales / 2 ** shift, biases, 64, bits=8)
    x = torch.randn((129, k), generator=torch.Generator(device="cuda").manual_seed(5), device="cuda").bfloat16()
    for sk in (1, 2, 4, 8):
        for m in (1, 2, 8, 9, 16, 17, 33, 64, 65, 129):
            for f32 in (False, True):
                assert torch.equal(qmm.matmul(x[:m], eight, sk=sk, f32=f32),
                                   qmm.matmul(x[:m], four, sk=sk, f32=f32)), (sk, m, f32)


def test_strided_rows_and_unreduced_slices(kernel):
    n, k = 5120, 17408
    q = qmm.pack(*_weights(n, k, 17), 64, bits=8)
    x = torch.randn((24, k + 64), device="cuda").bfloat16()[:, :k]          # rows 16-byte aligned, stride k + 64
    want = qmm.matmul(x.contiguous(), q, f32=True)
    assert torch.equal(qmm.matmul(x, q, f32=True), want)
    slices = qmm.matmul(x, q, reduce=False, f32=True)
    total = slices[0]
    for s in range(1, slices.shape[0]):
        total = total + slices[s]
    assert slices.shape[0] == qmm.split_k(n, k, bits=8) > 1 and torch.equal(total, want)


@pytest.mark.parametrize("n,k", [(8192, 2048), (2048, 4096), (5120, 17408)])
def test_sums_match_the_fp64_reference_as_closely_as_the_generic_kernel(kernel, n, k):
    """Same products, other fp32 order: within 2^-20 of the summed |x w|, like the generic kernel; bf16 rounds once."""

    w = _weights(n, k, 11 * n + k)
    q = qmm.pack(*w, 64, bits=8)
    x = torch.randn((32, k), device="cuda").bfloat16()
    dense = _dequant(*w)
    ref = x.double() @ dense.T
    bound = (x.double().abs() @ dense.abs().T) * 2 ** -20
    lane = qmm.matmul(x, q, f32=True)
    generic = affine.matmul(x, QLinear(*w, bits=8), f32=True).double()
    assert ((lane.double() - ref).abs() <= bound).all() and ((generic - ref).abs() <= bound).all()
    assert torch.equal(qmm.matmul(x, q), lane.to(torch.bfloat16))


def test_tiled_qlinear_keeps_its_width_in_every_helper():
    """tile/untile, rows() views and copies, and a group of mixed widths: each projection keeps its own bits."""

    eight = QLinear(*_weights(1024, 2048, 41), bits=8)
    four = QLinear(*[t.contiguous() for t in _four(1024, 2048)])
    t8, t4 = qmm_fast.tile(eight), qmm_fast.tile(four)
    assert (t8.layout, t8.bits, t8.n, t8.k) == ("tiled", 8, 1024, 2048) and t4.bits == 4
    back = qmm_fast.untile(t8)
    assert back.bits == 8 and all(torch.equal(a, b) for a, b in zip((back.weight, back.scales, back.biases),
                                                                      (eight.weight, eight.scales, eight.biases)))
    x = torch.randn((9, 2048), device="cuda").bfloat16()
    full = qmm_fast.matmul(x, t8)
    assert torch.equal(full, qmm.matmul(x, qmm.Q4(t8.weight, t8.scales, t8.biases, 1024, 2048, 64, 8)))
    partial = qmm.matmul(x, qmm.pack(*_weights(1024, 2048, 41), 64, bits=8), f32=True)
    assert torch.equal(qmm_fast.matmul_partial(x, t8), partial)
    from tensorfold.families.qwen3_5.cuda.distributed import row_partial

    assert torch.equal(row_partial(x, t8), partial)                     # a row-parallel rank's share
    got8, got4 = qmm_fast.matmul_group(x, [t8, t4])
    assert torch.equal(got8, full) and torch.equal(got4, qmm_fast.matmul(x, t4))
    assert torch.equal(qmm_fast.matmul_group(x, [t8, t8])[1], full)
    for a, b in ((0, 512), (64, 320), (100, 900)):                       # a view, then copies off the tile edges
        part = qmm_fast.rows(t8, a, b)
        assert part.bits == 8 and part.n == b - a
        sub = QLinear(eight.weight[a:b].contiguous(), eight.scales[a:b].contiguous(), eight.biases[a:b].contiguous(),
                      bits=8)
        assert torch.equal(qmm_fast.matmul(x, part), qmm_fast.matmul(x, qmm_fast.tile(sub)))


def _four(n: int, k: int):
    g = torch.Generator(device="cuda").manual_seed(n + k)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    return words.to(torch.int32), scales, (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).bfloat16()


@pytest.mark.skipif(not MODEL.exists(), reason=f"needs an 8-bit g64 MLX checkpoint at {MODEL} (set TF_AFFINE8_MODEL)")
def test_real_8bit_projections():
    """A checkpoint's 8-bit projections: rows independent of the row count, sums within the reference bound."""

    from tensorfold.families.qwen3_5.cuda.weights import _Tensors

    t = _Tensors(MODEL, "cuda")
    prefix = "language_model." if any(name.startswith("language_model.") for name in t) else ""
    names = [f"model.layers.0.linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj")]
    names += [f"model.layers.3.self_attn.{p}_proj" for p in ("q", "k", "o")]
    try:
        for name in names:
            w = [t.pop(prefix + name + s) for s in (".weight", ".scales", ".biases")]
            w[0] = w[0].view(torch.int32)
            if w[0].shape[1] * 4 != w[1].shape[1] * 64:
                continue                                 # not 8 bits in this checkpoint
            q = qmm.pack(*w, 64, bits=8)
            x = torch.randn((64, q.k), device="cuda").bfloat16()
            alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(64)])
            assert all(torch.equal(qmm.matmul(x[:m], q), alone[:m]) for m in (2, 16, 17, 64)), name
            dense = _dequant(*w)
            err = (qmm.matmul(x, q, f32=True).double() - x.double() @ dense.T).abs()
            assert (err <= (x.double().abs() @ dense.abs().T) * 2 ** -20).all(), name
    finally:
        t.close()
