"""5-, 6- and 8-bit lane matmul in groups of 32: every code exact, a row's bits independent of the row count, the
4-bit kernel's bits for 4-bit codes, fp32 sums within the reference bound."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import affine, qmm  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import QLinear  # noqa: E402

BITS = [5, 6, 8]
GS = 32
ROWS = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 256]
SHAPES = [(32, 2048), (512, 2048), (1000, 2048), (8192, 2048), (2048, 4096), (5120, 17408), (6144, 5120)]


def _weights(n: int, k: int, bits: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k * bits // 32), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // GS), generator=g, device="cuda") * 0.002 + 0.0001).to(torch.bfloat16)
    biases = (torch.randn((n, k // GS), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
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
    return q * scales.double().repeat_interleave(GS, 1) + biases.double().repeat_interleave(GS, 1)


def _words(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """(n, k) codes -> MLX words: the stream's bits gathered 32 at a time."""

    n, k = codes.shape
    stream = (codes[:, :, None].to(torch.int64) >> torch.arange(bits, device=codes.device)) & 1
    words = (stream.reshape(n, -1, 32) << torch.arange(32, device=codes.device)).sum(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_pack_round_trip(bits, n, k):
    w = _weights(n, k, bits, n + k + bits)
    q = qmm.pack(*w, GS, bits=bits)
    tiles = -(-n // 128) * 2
    shape = (tiles, k // GS, 8, 32, 2) if bits == 8 else (tiles, k // GS, 64 * bits)
    assert q.bits == bits and q.gs == GS and q.weight.shape == shape
    assert all(torch.equal(a, b) for a, b in zip(qmm.unpack(q), w))


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("f32", [False, True])
def test_every_code_decodes_exactly(bits, f32):
    """One-hot rows read back each stored code (scale 1, bias 0) through every K split, as the generic kernel does."""

    n, k = 128, 4096
    words = _weights(n, k, bits, 29)[0]
    ones = torch.ones((n, k // GS), device="cuda").bfloat16()
    q = qmm.pack(words, ones, torch.zeros_like(ones), GS, bits=bits)
    want = _dequant(words, ones, torch.zeros_like(ones), bits).T.float()
    eye = torch.eye(k, device="cuda").bfloat16()
    assert qmm.split_k(n, k, GS, bits=bits) > 1 and set(want.unique().tolist()) == set(range(2 ** bits))
    generic = QLinear(words, ones, torch.zeros_like(ones), gs=GS, bits=bits)
    assert torch.equal(affine.matmul(eye, generic, f32=True), want)
    assert torch.equal(qmm.matmul(eye, q, f32=f32).float(), want)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(bits, n, k):
    q = qmm.pack(*_weights(n, k, bits, 3 * n + k), GS, bits=bits)
    x = torch.randn((max(ROWS), k), generator=torch.Generator(device="cuda").manual_seed(7), device="cuda").bfloat16()
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), f"{n}x{k}: rows differ at M={m}"
    perm = torch.randperm(max(ROWS), generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])
    f32 = qmm.matmul(x[:40], q, f32=True)
    assert torch.equal(torch.cat([qmm.matmul(x[r:r + 1], q, f32=True) for r in range(40)]), f32)


def test_prompt_rows_do_not_depend_on_the_chunk():
    """Prompt-sized row counts (64-row tiles in bands): each row's bits as in a one-row launch."""

    for bits in BITS:
        q = qmm.pack(*_weights(5120, 17408, bits, 9 + bits), GS, bits=bits)
        x = torch.randn((1500, q.k), generator=torch.Generator(device="cuda").manual_seed(2), device="cuda").bfloat16()
        full = qmm.matmul(x, q)
        assert torch.equal(qmm.matmul(x[:700], q), full[:700])
        assert all(torch.equal(qmm.matmul(x[r:r + 1], q), full[r:r + 1]) for r in (0, 63, 64, 699, 1499))


@pytest.mark.parametrize("bits", BITS)
def test_4bit_codes_in_wider_words_give_the_4bit_bits(bits):
    """q = 2^s q4 with scales / 2^s (exact) at the same K split: the same products in the same order as the 4-bit
    kernel in groups of 32, so the same bits."""

    n, k = 2048, 4096
    g = torch.Generator(device="cuda").manual_seed(13)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64)
    scales = (torch.rand((n, k // GS), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // GS), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    four = qmm.pack(words.to(torch.int32), scales, biases, GS)
    codes = _codes(words.to(torch.int32), 4)
    x = torch.randn((129, k), generator=torch.Generator(device="cuda").manual_seed(5), device="cuda").bfloat16()
    for shift in range(bits - 3):
        wide = qmm.pack(_words(codes << shift, bits), scales / 2 ** shift, biases, GS, bits=bits)
        for sk in (1, 2, 4, 8):
            for m in (1, 2, 9, 16, 17, 33, 64, 65, 129):
                for f32 in (False, True):
                    assert torch.equal(qmm.matmul(x[:m], wide, sk=sk, f32=f32),
                                       qmm.matmul(x[:m], four, sk=sk, f32=f32)), (shift, sk, m, f32)


@pytest.mark.parametrize("bits", BITS)
def test_strided_rows_and_unreduced_slices(bits):
    n, k = 1024, 5120
    q = qmm.pack(*_weights(n, k, bits, 17), GS, bits=bits)
    x = torch.randn((24, k + 64), device="cuda").bfloat16()[:, :k]          # rows 16-byte aligned, stride k + 64
    want = qmm.matmul(x.contiguous(), q, f32=True)
    assert torch.equal(qmm.matmul(x, q, f32=True), want)
    slices = qmm.matmul(x, q, reduce=False, f32=True)
    total = slices[0]
    for s in range(1, slices.shape[0]):
        total = total + slices[s]
    assert slices.shape[0] == qmm.split_k(n, k, GS, bits=bits) > 1 and torch.equal(total, want)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("n,k", [(8192, 2048), (2048, 4096), (5120, 17408), (17408, 5120)])
def test_sums_match_the_fp64_reference_as_closely_as_the_generic_kernel(bits, n, k):
    """Same products, other fp32 order: within 2^-20 of the summed |x w|, like the generic kernel; bf16 rounds once."""

    w = _weights(n, k, bits, 11 * n + k)
    q = qmm.pack(*w, GS, bits=bits)
    x = torch.randn((32, k), device="cuda").bfloat16()
    dense = _dequant(*w, bits)
    ref = x.double() @ dense.T
    bound = (x.double().abs() @ dense.abs().T) * 2 ** -20
    lane = qmm.matmul(x, q, f32=True)
    generic = affine.matmul(x, QLinear(*w, gs=GS, bits=bits), f32=True).double()
    assert ((lane.double() - ref).abs() <= bound).all() and ((generic - ref).abs() <= bound).all()
    assert torch.equal(qmm.matmul(x, q), lane.to(torch.bfloat16))
