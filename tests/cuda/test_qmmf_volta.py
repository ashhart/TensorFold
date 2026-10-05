"""The sm_70 NVFP4 / FP8 / 16-bit matmuls: row-count invariant, stored values exact, prompts chunk invariant."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("Volta only", allow_module_level=True)

from tensorfold.cuda.kernels.qmmf_volta import DENSE_PROMPT_LIMIT, FP4, FP8, F16, VoltaLinear  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
SHAPES = [(48, 5120), (1024, 5120), (5120, 6144), (10240, 5120), (5120, 17408), (17408, 5120), (1000, 256),
          (2560, 160), (1000, 96), (320, 2592)]                     # K ending in a 32-input half group
ROWS = list(range(1, 34)) + [63, 64, 65, 100, 127, 128, 129, 200]


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _make(fmt: int, n: int, k: int, seed: int):
    """A random weight in ``fmt`` and its fp32 reference values (N, K)."""

    g = _gen(seed)
    if fmt == FP4:
        codes = torch.randint(0, 256, (n, k // 2), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        sc = torch.randint(0, 0x7F, (n, k // 16), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        sc[0, :3] = torch.tensor([1, 2, 7], dtype=torch.uint8)            # e4m3 subnormal scales
        gs = 2.0 ** -7
        lin = VoltaLinear.from_nvfp4(codes, sc.view(torch.float8_e4m3fn), gs)
        lo, hi = (codes & 0xF).long(), (codes >> 4).long()
        vals = torch.stack([E2M1.cuda()[lo], E2M1.cuda()[hi]], -1).view(n, k)
        ref = vals * sc.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * gs
        return lin, ref
    if fmt == FP8:
        w = torch.randint(0, 256, (n, k), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        w[(w & 0x7F) == 0x7F] = 0x11                                        # no NaN codes
        lin = VoltaLinear.from_fp8(w.view(torch.float8_e4m3fn), 2.0 ** -8)
        return lin, w.view(torch.float8_e4m3fn).float() * 2.0 ** -8
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w[1] *= 1e-3                                                            # columns of very different ranges
    w[2] *= 300.0
    return VoltaLinear.from_bf16(w), w.float()


@pytest.mark.parametrize("fmt", [FP4, FP8, F16])
@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(fmt, n, k):
    lin, _ = _make(fmt, n, k, n + k + fmt)
    x = torch.randn((200, k), generator=_gen(7), device="cuda").to(torch.bfloat16)
    alone = torch.cat([lin(x[r:r + 1]) for r in range(200)])
    for m in ROWS:
        assert torch.equal(lin(x[:m]), alone[:m]), f"fmt {fmt} {n}x{k}: M={m}"
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(lin(x[perm]), alone[perm])


@pytest.mark.parametrize("fmt", [FP4, FP8, F16])
@pytest.mark.parametrize("n,k", [(1024, 5120), (1000, 256), (5120, 17408), (2560, 160), (100, 96)])
def test_expanded_weights_are_the_stored_values(fmt, n, k):
    """The fp16 values the kernels multiply, times the column factors, are the checkpoint's values bit for bit."""

    lin, ref = _make(fmt, n, k, 3 * n + k + fmt)
    got = lin.dense().double() * lin.alpha[:n, None].double()
    assert torch.equal(got.float(), ref)
    assert torch.equal(got, ref.double())


@pytest.mark.parametrize("fmt", [FP4, FP8, F16])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("big", [False, True])
def test_decode_and_prompt_match_fp32_reference(fmt, n, k, big):
    lin, ref_w = _make(fmt, n, k, 11 * n + k + fmt)
    x = torch.randn((40, k), generator=_gen(1), device="cuda")
    if big:                                   # values far past fp16's range (massive activations) still work
        x[3] *= 3e5
        x[5, :7] = 2e6
    x = x.to(torch.bfloat16)
    ref = x.float() @ ref_w.T
    for y in (lin.matmul(x, f32=True), lin.prefill(x, f32=True)):
        assert torch.isfinite(y).all()
        for r in range(40):
            err = (y[r] - ref[r]).abs().max().item()
            assert err <= ref[r].abs().max().item() * 2 ** -12 + 1e-30, (fmt, r, err)


@pytest.mark.parametrize("fmt", [FP4, FP8, F16])
@pytest.mark.parametrize("n,k", [(1024, 5120), (5120, 17408), (10240, 5120)])
def test_prompt_rows_do_not_depend_on_chunking(fmt, n, k):
    lin, _ = _make(fmt, n, k, 5 * n + k + fmt)
    x = torch.randn((1100, k), generator=_gen(5), device="cuda").to(torch.bfloat16)
    whole = lin.prefill(x)
    for a, b in ((0, 1), (0, 7), (3, 40), (100, 613), (613, 1100), (1, 1100)):
        assert torch.equal(lin.prefill(x[a:b]), whole[a:b]), (fmt, a, b)


def test_huge_weight_prompts_on_the_decode_kernel():
    """A weight past ``DENSE_PROMPT_LIMIT`` (the bf16 head) prompts with decode's bits: still chunk invariant."""

    n, k = DENSE_PROMPT_LIMIT // 5120 + 64, 5120
    lin, _ = _make(F16, n, k, 2)
    x = torch.randn((20, k), generator=_gen(2), device="cuda").to(torch.bfloat16)
    assert torch.equal(lin.prefill(x), lin(x))


@pytest.mark.parametrize("fmt", [FP4, FP8, F16])
def test_tile_views_are_the_columns(fmt):
    lin, _ = _make(fmt, 1000, 512, 9 + fmt)
    x = torch.randn((5, 512), generator=_gen(4), device="cuda").to(torch.bfloat16)
    full = lin(x)
    for t0, t1 in ((0, 1), (2, 5), (14, 16)):
        part = lin.tiles(t0, t1)
        assert torch.equal(part(x), full[:, 64 * t0:min(1000, 64 * t1)])


def test_f32_and_strided_rows():
    lin, _ = _make(FP4, 512, 1024, 1)
    big = torch.randn((8, 2048), device="cuda").to(torch.bfloat16)
    a = lin.matmul(big[:, :1024], f32=True)
    b = lin.matmul(big[:, :1024].contiguous(), f32=True)
    assert a.dtype == torch.float32 and torch.equal(a, b)


@pytest.mark.parametrize("cols", [slice(0, 160), slice(160, 320), slice(480, 640), slice(64, 128), slice(32, 64)])
def test_half_group_shards_are_the_rows_slices(cols):
    """A 16-bit shard of whole 32-input halves: its inputs' stored values with the whole rows' column factors."""

    g = _gen(cols.start + 1)
    w = (torch.randn((2560, 640), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w[3] *= 1e-3
    full = VoltaLinear.from_bf16(w)
    shard = VoltaLinear.from_bf16(w, cols=cols)
    assert shard.k == cols.stop - cols.start
    assert torch.equal(shard.alpha, full.alpha)
    assert torch.equal(shard.dense().double() * shard.alpha[:shard.n, None].double(), w[:, cols].double())
    if cols.start % 64 == 0 and cols.stop % 64 == 0:
        part = full.groups(cols.start // 64, cols.stop // 64)
        assert torch.equal(shard.words, part.words)
    x = torch.randn((9, shard.k), generator=g, device="cuda").to(torch.bfloat16)
    ref = x.float() @ w[:, cols].float().T
    got = shard.matmul(x, f32=True)
    assert (got - ref).abs().max() <= ref.abs().max() * 2 ** -12
    assert torch.equal(torch.cat([shard.matmul(x[r:r + 1], f32=True) for r in range(9)]), got)
