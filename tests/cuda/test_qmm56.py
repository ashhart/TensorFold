"""The 5- and 6-bit lane matmul: every code exact, a row's bits independent of the row count, tile and kernel, the
4-bit kernel's bits for 4-bit codes, fp32 sums within the reference bound."""

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

BITS = [5, 6]
ROWS = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 256]
SHAPES = [(32, 2048), (512, 2048), (1000, 2048), (8192, 2048), (2048, 4096), (5120, 17408), (6144, 5120)]
GROUPS = {"gdn": [(10240, 5120), (6144, 5120), (48, 5120), (48, 5120)], "attention": [(8192, 2048), (512, 2048),
          (512, 2048)], "out": [(5120, 6144)], "down": [(5120, 17408)]}
TILES = list(range(13))                              # 0 picks by rows and chip; 6 and 7 are the swapped 8-row tiles
MODEL = Path(os.environ.get("TF_AFFINE56_MODEL", "/models/Qwen3.8-27B-MLX-5bit"))
grouped_chip = torch.cuda.get_device_capability()[0] == 12


def _weights(n: int, k: int, bits: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k * bits // 32), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.002 + 0.0001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return words.to(torch.int32), scales, biases


def _codes(words: torch.Tensor, bits: int) -> torch.Tensor:
    """The MLX bit stream read code by code (independent of ``qmm``): code i at bits [i * bits, (i + 1) * bits)."""

    w = words.to(torch.int64) & 0xFFFFFFFF
    bit = torch.arange(w.shape[1] * 32 // bits, device=w.device) * bits
    lo, sh = w[:, bit // 32], bit % 32
    hi = torch.where(sh + bits > 32, w[:, torch.clamp(bit // 32 + 1, max=w.shape[1] - 1)] << (32 - sh), 0)
    return ((lo >> sh) | hi) & ((1 << bits) - 1)


def _dequant(words, scales, biases, bits) -> torch.Tensor:
    q = _codes(words, bits).double()
    return q * scales.double().repeat_interleave(64, 1) + biases.double().repeat_interleave(64, 1)


def _words(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """(n, k) codes -> MLX words: the stream's bits gathered 32 at a time."""

    n, k = codes.shape
    stream = (codes[:, :, None].to(torch.int64) >> torch.arange(bits, device=codes.device)) & 1
    words = (stream.reshape(n, -1, 32) << torch.arange(32, device=codes.device)).sum(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


@pytest.fixture(params=["group", "lane"] if grouped_chip else ["lane"])
def kernel(request, monkeypatch):
    """``group``: qmm_group.cu (sm_12x); ``lane``: qmm.cu, which every other GPU runs."""

    if request.param == "lane":
        monkeypatch.setattr(qmm, "grouped", lambda device: False)
    return request.param


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_pack_round_trip(bits, n, k):
    w = _weights(n, k, bits, n + k + bits)
    q = qmm.pack(*w, 64, bits=bits)
    assert q.bits == bits and q.weight.shape == (-(-n // 128) * 2, k // 64, 128 * bits)
    assert all(torch.equal(a, b) for a, b in zip(qmm.unpack(q), w))


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("f32", [False, True])
def test_every_code_decodes_exactly(kernel, bits, f32):
    """One-hot rows read back each stored code (scale 1, bias 0) through every K split: q itself, 0 to 2^bits - 1,
    as the generic kernel reads the same words."""

    n, k = 128, 4096
    words = _weights(n, k, bits, 29)[0]
    ones = torch.ones((n, k // 64), device="cuda").bfloat16()
    q = qmm.pack(words, ones, torch.zeros_like(ones), 64, bits=bits)
    want = _dequant(words, ones, torch.zeros_like(ones), bits).T.float()
    eye = torch.eye(k, device="cuda").bfloat16()
    assert qmm.split_k(n, k, bits=bits) > 1 and set(want.unique().tolist()) == set(range(2 ** bits))
    assert torch.equal(affine.matmul(eye, QLinear(words, ones, torch.zeros_like(ones), bits=bits), f32=True), want)
    assert torch.equal(qmm.matmul(eye, q, f32=f32).float(), want)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(kernel, bits, n, k):
    q = qmm.pack(*_weights(n, k, bits, 3 * n + k), 64, bits=bits)
    x = torch.randn((max(ROWS), k), generator=torch.Generator(device="cuda").manual_seed(7), device="cuda").bfloat16()
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), f"{n}x{k}: rows differ at M={m}"
    perm = torch.randperm(max(ROWS), generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])
    f32 = qmm.matmul(x[:40], q, f32=True)
    assert torch.equal(torch.cat([qmm.matmul(x[r:r + 1], q, f32=True) for r in range(40)]), f32)


@pytest.mark.skipif(not grouped_chip, reason="grouped launches run on sm_12x only")
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("name", list(GROUPS))
def test_every_group_tile_keeps_qmm_cu_bits(bits, name, monkeypatch):
    """The grouped kernel's tiles, overlapped or not, give each part qmm.cu's bits for the same rows."""

    qs = [qmm.pack(*_weights(n, k, bits, 5 * n + i), 64, bits=bits) for i, (n, k) in enumerate(GROUPS[name])]
    x = torch.randn((129, qs[0].k), generator=torch.Generator(device="cuda").manual_seed(1), device="cuda").bfloat16()
    with monkeypatch.context() as m:
        m.setattr(qmm, "grouped", lambda device: False)
        want = [qmm.matmul(x, q) for q in qs]
    for rows in (1, 2, 8, 9, 16, 17, 33, 64, 65, 129):
        for tile in TILES:
            for early in (0, 1):
                got = qmm.matmul_group(x[:rows], qs, tile=tile, early=early)
                assert all(torch.equal(a, b[:rows]) for a, b in zip(got, want)), (rows, tile, early)


@pytest.mark.parametrize("bits", BITS)
def test_4bit_codes_in_wider_words_give_the_4bit_bits(kernel, bits):
    """q = 2^s q4 with scales / 2^s (exact, s from 0 until the top bit) at the same K split: the same products in the
    same order as the 4-bit kernel, so the same bits, which test_qmm_group ties to the 27B's Triton reference."""

    n, k = 2048, 4096
    g = torch.Generator(device="cuda").manual_seed(13)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // 64), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    four = qmm.pack(words.to(torch.int32), scales, biases, 64)
    codes = _codes(words.to(torch.int32), 4)
    x = torch.randn((129, k), generator=torch.Generator(device="cuda").manual_seed(5), device="cuda").bfloat16()
    for shift in range(bits - 3):
        wide = qmm.pack(_words(codes << shift, bits), scales / 2 ** shift, biases, 64, bits=bits)
        for sk in (1, 2, 4, 8):
            for m in (1, 2, 8, 9, 16, 17, 33, 64, 65, 129):
                for f32 in (False, True):
                    assert torch.equal(qmm.matmul(x[:m], wide, sk=sk, f32=f32),
                                       qmm.matmul(x[:m], four, sk=sk, f32=f32)), (shift, sk, m, f32)


@pytest.mark.parametrize("bits", BITS)
def test_strided_rows_and_unreduced_slices(kernel, bits):
    n, k = 1024, 5120
    q = qmm.pack(*_weights(n, k, bits, 17), 64, bits=bits)
    x = torch.randn((24, k + 64), device="cuda").bfloat16()[:, :k]          # rows 16-byte aligned, stride k + 64
    want = qmm.matmul(x.contiguous(), q, f32=True)
    assert torch.equal(qmm.matmul(x, q, f32=True), want)
    slices = qmm.matmul(x, q, reduce=False, f32=True)
    total = slices[0]
    for s in range(1, slices.shape[0]):
        total = total + slices[s]
    assert slices.shape[0] == qmm.split_k(n, k, bits=bits) > 1 and torch.equal(total, want)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("n,k", [(8192, 2048), (2048, 4096), (5120, 17408), (17408, 5120)])
def test_sums_match_the_fp64_reference_as_closely_as_the_generic_kernel(kernel, bits, n, k):
    """Same products, other fp32 order: within 2^-20 of the summed |x w|, like the generic kernel; bf16 rounds once."""

    w = _weights(n, k, bits, 11 * n + k)
    q = qmm.pack(*w, 64, bits=bits)
    x = torch.randn((32, k), device="cuda").bfloat16()
    dense = _dequant(*w, bits)
    ref = x.double() @ dense.T
    bound = (x.double().abs() @ dense.abs().T) * 2 ** -20
    lane = qmm.matmul(x, q, f32=True)
    generic = affine.matmul(x, QLinear(*w, bits=bits), f32=True).double()
    assert ((lane.double() - ref).abs() <= bound).all() and ((generic - ref).abs() <= bound).all()
    assert torch.equal(qmm.matmul(x, q), lane.to(torch.bfloat16))


@pytest.mark.skipif(not MODEL.exists(), reason=f"needs a 5- or 6-bit g64 MLX checkpoint at {MODEL} "
                                               "(set TF_AFFINE56_MODEL)")
def test_real_5_and_6bit_projections():
    """A checkpoint's 5- or 6-bit projections: rows independent of the row count, sums within the reference bound."""

    from tensorfold.families.qwen3_5.cuda.weights import _Tensors

    t = _Tensors(MODEL, "cuda")
    prefix = "language_model." if any(name.startswith("language_model.") for name in t) else ""
    names = [f"model.layers.0.linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj")]
    names += [f"model.layers.3.self_attn.{p}_proj" for p in ("q", "k", "o")]
    names += [f"model.layers.1.mlp.{p}_proj" for p in ("gate", "down")]
    seen = 0
    try:
        for name in names:
            if prefix + name + ".weight" not in t:
                continue
            w = [t.pop(prefix + name + s) for s in (".weight", ".scales", ".biases")]
            w[0] = w[0].view(torch.int32)
            bits = w[0].shape[1] * 32 // (w[1].shape[1] * 64)
            if bits not in BITS:
                continue
            seen += 1
            q = qmm.pack(*w, 64, bits=bits)
            x = torch.randn((64, q.k), device="cuda").bfloat16()
            alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(64)])
            assert all(torch.equal(qmm.matmul(x[:m], q), alone[:m]) for m in (2, 16, 17, 64)), name
            dense = _dequant(*w, bits)
            err = (qmm.matmul(x, q, f32=True).double() - x.double() @ dense.T).abs()
            assert (err <= (x.double().abs() @ dense.abs().T) * 2 ** -20).all(), name
    finally:
        t.close()
    assert seen, "no 5- or 6-bit projection in the checkpoint"
