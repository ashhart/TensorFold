"""The CUDA chunk pass of DeepSeek-V4.1 decode attention over NVFP4 compressed entries (mqa_fp4.cu): decode tables,
a float64 reference, row invariance (alone == batched == graph replay), masking, and the prompt staging ring."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("sm_12x only", allow_module_level=True)

from tensorfold.families.deepseek_v41.cuda import kernels as K
from tensorfold.families.deepseek_v41.cuda import mqa_fp4

H, D, W, E = 32, 512, 128, 3000
FP4_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture
def cuda_mqa(monkeypatch):
    monkeypatch.setattr(K, "CUDA_MQA", True)


def _tables(n=1 << 16):
    freqs = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    return K.rope_tables(freqs, n)


def _cache(seed, n=E):
    """Fp4Rows of rows with per-row magnitudes 2^-3 .. 2^1 (NaN scale bytes in the last 100: never selected)."""

    cos, sin = _tables()
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, D), generator=g, device="cuda") * torch.exp2(
        torch.randint(-3, 2, (n, 1), generator=g, device="cuda").float())
    rows = K.Fp4Rows(n, D, device="cuda")
    rows.store(torch.arange(n, device="cuda"), x.to(torch.bfloat16), torch.arange(n, device="cuda") * 3, cos, sin)
    rows.s[n - 100:] = 0x7F
    return rows


def _case(R, seed, *, p0=5000, streams=1, comp=True, ring=None, n_idx=512):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn((R, H, D), generator=g, device="cuda") * 0.6).to(torch.bfloat16)
    pos = p0 + torch.randint(0, 1000, (R,), generator=g, device="cuda")
    sink = torch.randn((H,), generator=g, device="cuda")
    idx = None
    if comp:
        idx = torch.stack([torch.randperm(E - 100, generator=g, device="cuda")[:n_idx] for _ in range(R)]).int()
    kw = {}
    if streams == 1:
        swa = (torch.randn((ring or 4096, D), generator=g, device="cuda") * 2).to(torch.bfloat16)
    else:
        swa = (torch.randn((streams * 256, D), generator=g, device="cuda") * 2).to(torch.bfloat16)
        kw = {"sbase": (torch.arange(R, device="cuda") % streams) * 256, "ring": 256}
    return q, idx, swa, pos, sink, kw


def _ref(q, rows, idx, swa, pos, sink, cos=None, sin=None, sbase=None, ring=None):
    """float64 attention: the compressed entries (dequantized exactly) + the window + the sink, then inverse RoPE."""

    comp = rows.dequant().double() if rows is not None else None
    ring = ring or swa.shape[0]
    outs = []
    for r in range(q.shape[0]):
        p = int(pos[r])
        base = int(sbase[r]) if sbase is not None else 0
        keys = [swa[[base + s % ring for s in range(max(0, p - W + 1), p + 1)]].double()]
        if idx is not None:
            keys.insert(0, comp[[int(i) for i in idx[r] if int(i) >= 0]])
        k = torch.cat(keys)
        s = q[r].double() @ k.T * D ** -0.5
        full = torch.cat([s, sink[:, None].double()], dim=1)
        o = torch.softmax(full, -1)[:, :-1] @ k
        if cos is not None:
            c, sn = cos[p].double(), sin[p].double()
            e, od = o[:, D - 64::2].clone(), o[:, D - 63::2].clone()
            o[:, D - 64::2], o[:, D - 63::2] = e * c + od * sn, od * c - e * sn
        outs.append(o)
    return torch.stack(outs)


def _mqa(q, rows, idx, swa, pos, sink, kw, rope=False, buf=None):
    cos, sin = _tables() if rope else (None, None)
    buf = buf or K.AttnBuffers(32, H, D, 512 + W)
    return K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin, **kw)


def test_decode_tables_are_exact():
    """Every nibble pair and every e4m3 byte decodes to its exact bf16, in the build the container's nvcc chose and in
    the table build (bit-identical either way)."""

    mags = torch.tensor(FP4_MAGS)
    val = torch.cat([mags, -mags]).to(torch.bfloat16).view(torch.int16).long() & 0xFFFF
    b = torch.arange(256)
    want = val[b & 15] | (val[b >> 4] << 16)
    e4 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(e4)
    want4 = e4.to(torch.bfloat16).view(torch.int16)
    for lut in (False, True):
        e2m1, e4m3, cvt = mqa_fp4.ext(lut).decode_table(torch.empty(1, device="cuda"))
        assert torch.equal(e2m1.cpu().to(torch.int64) & 0xFFFFFFFF, want), (lut, cvt)
        assert torch.equal(e4m3.cpu()[finite], want4[finite]), (lut, cvt)
        assert not (lut and cvt)


@pytest.mark.parametrize("R,streams,comp", [(1, 1, True), (3, 1, True), (16, 2, True), (32, 4, True), (5, 1, False)])
def test_matches_float64_like_triton(cuda_mqa, monkeypatch, R, streams, comp):
    """fp32 output (no inverse RoPE) and the bf16 inverse-rotated output against float64: errors at the Triton fp4
    path's level (both round P to bf16 for the PV product)."""

    rows = _cache(1)
    q, idx, swa, pos, sink, kw = _case(R, 10 + R, streams=streams, comp=comp)
    rows_ = rows if comp else None
    ref = _ref(q, rows_, idx, swa, pos, sink, **kw)
    got = _mqa(q, rows_, idx, swa, pos, sink, kw)
    monkeypatch.setattr(K, "CUDA_MQA", False)
    tri = _mqa(q, rows_, idx, swa, pos, sink, kw)
    monkeypatch.setattr(K, "CUDA_MQA", True)
    e_cuda = (got.double() - ref).abs().max().item()
    e_tri = (tri.double() - ref).abs().max().item()
    assert torch.isfinite(got).all()
    assert e_cuda <= 1.25 * e_tri + 1e-6 and e_cuda < 4e-3 * ref.abs().max().item(), (e_cuda, e_tri)
    cos, sin = _tables()
    ref_r = _ref(q, rows_, idx, swa, pos, sink, cos, sin, **kw)
    got_r = _mqa(q, rows_, idx, swa, pos, sink, kw, rope=True)
    assert got_r.dtype == torch.bfloat16
    assert (got_r.double() - ref_r).abs().max().item() < 4e-3 * ref_r.abs().max().item() + 2 * e_cuda


def test_rows_alone_equal_rows_batched(cuda_mqa):
    """Each row alone == the same row among random others (R = 2..32), and mixed stream bases."""

    rows = _cache(2)
    q, idx, swa, pos, sink, kw = _case(32, 7, streams=4)
    full = _mqa(q, rows, idx, swa, pos, sink, kw, rope=True)
    sb = kw["sbase"]
    for r in (0, 5, 31):
        one = _mqa(q[r:r + 1], rows, idx[r:r + 1], swa, pos[r:r + 1], sink, {"sbase": sb[r:r + 1], "ring": 256},
                   rope=True)
        assert torch.equal(one[0], full[r])
    g = torch.Generator(device="cuda").manual_seed(3)
    for n in (2, 7, 19):
        pick = torch.randperm(32, generator=g, device="cuda")[:n]
        part = _mqa(q[pick], rows, idx[pick], swa, pos[pick], sink, {"sbase": sb[pick], "ring": 256}, rope=True)
        assert torch.equal(part, full[pick])


def test_verify_rows_equal_stepping(cuda_mqa):
    """A verify batch (consecutive positions of one stream, ring rows written causally) == one row at a time."""

    rows = _cache(3)
    q, idx, swa, pos, sink, _ = _case(6, 4, streams=1, ring=256)
    pos = 777 + torch.arange(6, device="cuda")
    kw = {"sbase": torch.zeros(6, dtype=torch.long, device="cuda"), "ring": 256}
    batch = _mqa(q, rows, idx, swa, pos, sink, kw, rope=True)
    for r in range(6):
        one = _mqa(q[r:r + 1], rows, idx[r:r + 1], swa, pos[r:r + 1], sink,
                   {"sbase": kw["sbase"][:1], "ring": 256}, rope=True)
        assert torch.equal(one[0], batch[r])


@pytest.mark.parametrize("R", [1, 7, 32])
def test_graph_replay_equals_eager(cuda_mqa, R):
    rows = _cache(4)
    q, idx, swa, pos, sink, kw = _case(R, 20 + R, streams=2)
    buf = K.AttnBuffers(32, H, D, 512 + W)
    cos, sin = _tables()
    _mqa(q, rows, idx, swa, pos, sink, kw, rope=True, buf=buf)            # warm-up (attributes, compile)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin, **kw)
    g = torch.Generator(device="cuda").manual_seed(R)
    for _ in range(3):
        q.copy_((torch.randn(q.shape, generator=g, device="cuda") * 0.6).to(torch.bfloat16))
        pos.copy_(5000 + torch.randint(0, 3000, (R,), generator=g, device="cuda"))
        idx.copy_(torch.stack([torch.randperm(E - 100, generator=g, device="cuda")[:512] for _ in range(R)]).int())
        idx[:, -7:] = -1
        swa.copy_((torch.randn(swa.shape, generator=g, device="cuda") * 2).to(torch.bfloat16))
        graph.replay()
        eager = _mqa(q, rows, idx, swa, pos, sink, kw, rope=True)
        assert torch.equal(out, eager)


@pytest.mark.parametrize("case", ["all_empty", "edge63", "edge64", "edge65", "early", "window_only"])
def test_masking(cuda_mqa, case):
    """-1 indices (whole splits empty, padding at a split edge), early positions (window slots < 0), no compressed
    entries; NaN scale bytes sit in rows nobody selects."""

    rows = _cache(5)
    q, idx, swa, pos, sink, kw = _case(4, 30, streams=1)
    comp = rows
    if case == "all_empty":
        idx[:] = -1
    elif case.startswith("edge"):
        n = int(case[4:])
        idx[:, n:] = -1
        idx[1, :] = torch.where(torch.arange(512, device="cuda") < 64 + n, idx[1], -1)
    elif case == "early":
        pos = torch.tensor([0, 1, 50, 126], device="cuda")
        idx[:, 3:] = -1
        idx[0] = -1
    else:
        comp = idx = None
    ref = _ref(q, comp, idx, swa, pos, sink, **kw)
    got = _mqa(q, comp, idx, swa, pos, sink, kw)
    assert torch.isfinite(got).all()
    assert (got.double() - ref).abs().max().item() < 4e-3 * ref.abs().max().item()


def test_prompt_staging_ring(cuda_mqa):
    """Small prompt chunks: the 4096-row staging ring, no stream base."""

    rows = _cache(6)
    q, idx, swa, pos, sink, kw = _case(9, 40, streams=1, ring=4096, p0=10000)
    ref = _ref(q, rows, idx, swa, pos, sink)
    got = _mqa(q, rows, idx, swa, pos, sink, {})
    assert (got.double() - ref).abs().max().item() < 4e-3 * ref.abs().max().item()
