"""Engram on the device: file rows dequantize bit-exactly, two ranks' exchange and ``wkv`` gather equal one rank's
concatenation, the gate tracks the fp32 reference port in mirror mode (the op-level bound), staging never races its
copy, rows do not depend on the window and graph replays equal eager runs."""

from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_reference as ref
from dsv41_pair import run_pair

from tensorfold.cuda.nvfp4.linear import Mx8Linear
from tensorfold.families.deepseek_v41.cuda import engram as E

MODEL = os.environ.get("TF_DSV41_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")
D, HC, HEAD, LAYERS, COLS = 5120, 4, 256, 2, 24
REC = HEAD + HEAD // 32                     # a file row: e4m3 bytes, then an E8M0 byte per 32
KV = (HC + 1) * D
ROWS = 6
EPS = 1e-20
POW2 = torch.tensor([math.ldexp(1.0, u - 127) for u in range(255)] + [math.nan])
# the first row of hash column 12 (rank 1's first), and the reference's own ceil(rows/2) split, per layer
COLUMN_12 = {1: 192_001_740, 14: 192_007_016}


def _cpu_rows(raw: torch.Tensor) -> torch.Tensor:
    """u8 [..., REC] -> bf16 [..., HEAD]: e4m3 times 2^(e - 127) per 32 in fp32, then bf16 (M:312-320)."""

    v = raw[..., :HEAD].contiguous().view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
    return (v * POW2[raw[..., HEAD:].long()][..., None]).flatten(-2).to(torch.bfloat16)


def _raw(R: int, cols: int, seed: int) -> torch.Tensor:
    """Random file rows: finite e4m3 bytes, scale bytes 2^-9 .. 2^3."""

    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (R, LAYERS, cols, REC), generator=g, dtype=torch.uint8)
    w = raw[..., :HEAD]
    w[(w & 0x7F) == 0x7F] = 0x38
    raw[..., HEAD:] = torch.randint(118, 131, (R, LAYERS, cols, HEAD // 32), generator=g, dtype=torch.uint8)
    return raw


def _mx8(n: int, seed: int) -> Mx8Linear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(0, 256, (n, COLS * HEAD), generator=g, dtype=torch.uint8, device="cuda")
    w[(w & 0x7F) >= 0x70] = 0x30
    s = torch.randint(116, 124, (n, COLS * HEAD // 32), generator=g, dtype=torch.uint8, device="cuda")
    return Mx8Linear.from_checkpoint(w.view(torch.float8_e4m3fn), s)


def _streams(R: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((R, HC, D), generator=g) * torch.tensor([1.0, 0.3, 2.5, 0.8])[None, :, None]
    return x.to(torch.bfloat16).cuda()


def _wqk(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (1 + 0.3 * torch.randn((HC, D), generator=g)).cuda()


def _bufs(world: int, rows: int = ROWS) -> SimpleNamespace:
    """The Engram scratch of ``buffers.Buffers`` for one rank of ``world``."""

    cols, bf = COLS // world, torch.bfloat16

    def t(*shape, dtype=bf):
        return torch.empty(shape, dtype=dtype, device="cuda")

    return SimpleNamespace(eraw=t(rows, LAYERS, cols, REC, dtype=torch.uint8), eloc=t(rows, LAYERS, cols * HEAD),
                           egat=t(world, rows, LAYERS, cols * HEAD), eng=t(rows, LAYERS, COLS * HEAD),
                           ekv=t(rows, KV // world), ekv_gat=t(world, rows, KV // world))


def _chain(b, raw, X, wkvs, wqk, comm=None, *, prompt=False):
    """Dequant, exchange, then each layer's ``wkv`` and gate into its own copy of X -> (e, [kv], [X after])."""

    R = raw.shape[0]
    b.eraw[:R].copy_(raw)
    eloc = E.dequant_rows(b.eraw[:R], b.eloc[:R])
    e = E.exchange(eloc, comm, b.egat, b.eng)
    kvs, outs = [], []
    for layer, wkv in enumerate(wkvs):
        kv = E.kv(e[:, layer], wkv, comm, b.ekv, b.ekv_gat, prompt=prompt)
        kvs.append(kv.clone())
        outs.append(E.inject(X.clone(), kv, wqk))
    return e.clone(), kvs, outs


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(x):
        bits = x.contiguous().view(torch.int16).cpu().to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _t1(out: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Every element within 1 bf16 ulp of the reference, at most 0.5% of them differing."""

    u = _ulps(out, want.to(torch.bfloat16))
    off = float((u > 0).double().mean())
    print(f"{what}: max {int(u.max())} ulp, {off:.4%} differ")
    assert int(u.max()) <= 1, f"{what}: {int(u.max())} ulps"
    assert off <= 0.005, f"{what}: {off:.4%} differ"


def _t1_sum(got: torch.Tensor, x: torch.Tensor, w: torch.Tensor, what: str) -> None:
    """The op-level bound for a bf16 projection against the fp64 product: within 1 ulp with <= 0.5% off, except a sum
    that cancels to far below its terms, which stays within the fp32 sum bound plus the half ulp of its rounding."""

    want = x.double() @ w.double().T
    bound = 8 * math.sqrt(x.shape[1]) * 2.0**-24 * (x.double().abs() @ w.double().abs().T)
    d = (got.cpu().double() - want).abs()
    top = torch.maximum(got.cpu().double().abs(), want.abs())
    ulp = torch.ldexp(torch.ones_like(top), torch.frexp(top).exponent - 8).clamp_min(2.0**-133)
    off = float((got.cpu() != want.float().to(torch.bfloat16)).double().mean())
    cancels = (d > ulp) & (d <= bound + 0.5 * ulp)
    worst = float((d / ulp)[~cancels].max())
    print(f"{what}: max {worst:.3g} ulp, {off:.4%} differ, {int(cancels.sum())} cancelled")
    assert worst <= 1, f"{what}: {worst} ulps"
    assert off <= 0.005, f"{what}: {off:.4%} differ"


def test_dequant_is_exact_for_every_byte_and_exponent():
    raw = torch.empty((16, LAYERS, 8, REC), dtype=torch.uint8)
    flat = raw.view(256, REC)
    flat[:, :HEAD] = torch.arange(256, dtype=torch.uint8)[None, :]          # every e4m3 byte, NaNs included
    flat[:, HEAD:] = torch.arange(256, dtype=torch.uint8)[:, None]          # every E8M0 byte: 2^-127 .. NaN
    out = torch.empty((16, LAYERS, 8 * HEAD), dtype=torch.bfloat16, device="cuda")
    got = E.dequant_rows(raw.cuda(), out).cpu().view(256, HEAD)
    want = _cpu_rows(flat)
    nan = want.isnan()
    assert torch.equal(got.isnan(), nan)
    assert torch.equal(got[~nan].view(torch.int16), want[~nan].view(torch.int16))


def test_gate_takes_copysign_at_a_zero_dot():
    X = torch.full((1, HC, D), 512.0)
    X[0, 3] = 0.0                                       # a silent copy: eps keeps its norm finite
    kv = torch.zeros((1, KV))
    kv[0, HC * D:] = -1024.0                            # the value
    kv[0, 1 * D] = -1.0                                 # copy 1's key: one channel against a tiny weight
    kv[0, 3 * D:4 * D] = 0.5
    wqk = torch.ones((HC, D))
    wqk[1, 0] = 1e-12
    got = E.inject(X.to(torch.bfloat16).cuda(), kv.to(torch.bfloat16).cuda(), wqk.cuda()).cpu()
    h, key = X, kv[:, :HC * D].view(1, HC, D)
    rstd = torch.rsqrt(h.square().mean(-1) + EPS) * torch.rsqrt(key.square().mean(-1) + EPS)
    dot = (h * wqk * key).sum(-1) * rstd * D**-0.5
    assert dot[0, 0].item() == 0 and not math.copysign(1, dot[0, 0].item()) < 0 and -1e-6 < dot[0, 1].item() < 0
    g = torch.sigmoid(torch.copysign(dot.abs().clamp_min(E.CLAMP).sqrt(), dot))
    want = (h + g[..., None] * kv[:, None, HC * D:]).to(torch.bfloat16)
    assert torch.equal(got, want)
    signed = (h + torch.sigmoid(torch.sign(dot) * dot.abs().clamp_min(E.CLAMP).sqrt())[..., None]
              * kv[:, None, HC * D:]).to(torch.bfloat16)
    assert not torch.equal(got[0, 0], signed[0, 0])     # sign() would close copy 0's gate to one half


def test_two_ranks_equal_one_ranks_concatenation():
    raw = _raw(ROWS, COLS, 1)
    halves = [_mx8(KV // 2, 10 + r) for r in range(2)]
    X, wqk = _streams(ROWS, 2), _wqk(3)
    one = _bufs(1)
    e1 = E.exchange(E.dequant_rows(raw.cuda(), one.eloc), None, one.egat, one.eng)
    assert torch.equal(e1.cpu(), _cpu_rows(raw).flatten(-2))
    kv1 = [torch.cat([E.kv(e1[:, layer], h, None, torch.empty((ROWS, KV // 2), dtype=torch.bfloat16, device="cuda"),
                           torch.empty((1, ROWS, KV // 2), dtype=torch.bfloat16, device="cuda"))[0] for h in halves],
                     dim=1) for layer in range(LAYERS)]
    X1 = [E.inject(X.clone(), kv, wqk) for kv in kv1]

    def rank(r, R):
        cut = slice(12 * r, 12 * (r + 1))
        return lambda comm: _chain(_bufs(2), raw[:R, :, cut].cuda(), X[:R], [halves[r]] * LAYERS, wqk, comm)

    for R in (ROWS, 1, 4):
        for e, kvs, outs in run_pair(rank(0, R), rank(1, R)):
            assert torch.equal(e, e1[:R])
            for layer in range(LAYERS):
                assert torch.equal(kvs[layer].permute(1, 0, 2).flatten(1), kv1[layer][:R]), (R, layer)
                assert torch.equal(outs[layer], X1[layer][:R]), (R, layer)


def test_rows_do_not_depend_on_the_window_and_graphs_equal_eager():
    raw = _raw(ROWS, COLS, 4)
    wkv, wqk, X = _mx8(KV, 20), _wqk(5), _streams(ROWS, 6)
    full = _chain(_bufs(1), raw.cuda(), X, [wkv, wkv], wqk)
    for R in range(1, ROWS + 1):
        for a in (0, ROWS - R):
            e, kvs, outs = _chain(_bufs(1), raw[a:a + R].cuda(), X[a:a + R], [wkv, wkv], wqk)
            assert torch.equal(e, full[0][a:a + R]), (R, a)
            for layer in range(LAYERS):
                assert torch.equal(kvs[layer], full[1][layer][:, a:a + R]), (R, a)
                assert torch.equal(outs[layer], full[2][layer][a:a + R]), (R, a)
    for R in range(1, ROWS + 1):
        b = _bufs(1)
        Xs = X[:R].clone()

        def step(b=b, Xs=Xs, R=R):
            e = E.exchange(E.dequant_rows(b.eraw[:R], b.eloc[:R]), None, b.egat, b.eng)
            for layer in range(LAYERS):
                E.inject(Xs, E.kv(e[:, layer], wkv, None, b.ekv, b.ekv_gat), wqk)

        b.eraw[:R].copy_(raw[:R])
        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for seed in (7, 8):
            fresh, x0 = _raw(R, COLS, seed), _streams(R, seed)
            b.eraw[:R].copy_(fresh)
            Xs.copy_(x0)
            graph.replay()
            torch.cuda.synchronize()
            eager = x0.clone()
            c = _bufs(1)
            c.eraw[:R].copy_(fresh)
            e = E.exchange(E.dequant_rows(c.eraw[:R], c.eloc[:R]), None, c.egat, c.eng)
            for layer in range(LAYERS):
                E.inject(eager, E.kv(e[:, layer], wkv, None, c.ekv, c.ekv_gat), wqk)
            assert torch.equal(Xs, eager), R


def test_prompt_rows_do_not_depend_on_the_chunk():
    raw = _raw(129, COLS, 9)
    wkv, wqk, X = _mx8(KV, 21), _wqk(10), _streams(129, 11)
    full = _chain(_bufs(1, 129), raw.cuda(), X, [wkv], wqk, prompt=True)
    for a, n in ((0, 1), (5, 7), (128, 1), (0, 129)):
        _, kvs, outs = _chain(_bufs(1, 129), raw[a:a + n].cuda(), X[a:a + n], [wkv], wqk, prompt=True)
        assert torch.equal(kvs[0], full[1][0][:, a:a + n]) and torch.equal(outs[0], full[2][0][a:a + n]), (a, n)


class _SlowReader:
    """Rows whose bytes follow from their id, written a few at a time with pauses (a cold disk)."""

    wrow, srow = HEAD, HEAD // 32
    layout = SimpleNamespace(starts=(0, 1 << 20))

    def __init__(self, pause: float) -> None:
        self.pause = pause

    @staticmethod
    def bytes(ids: np.ndarray) -> np.ndarray:
        return ((ids[:, None] * 131 + np.arange(REC)[None, :]) % 251).astype(np.uint8)

    def gather(self, ids, out_w, out_s) -> None:
        rows = self.bytes(np.asarray(ids))
        for i in range(0, len(rows), 16):
            out_w[i:i + 16], out_s[i:i + 16] = rows[i:i + 16, :HEAD], rows[i:i + 16, HEAD:]
            time.sleep(self.pause)


def test_staging_never_overwrites_a_half_before_its_copy():
    reader, steps, cols = _SlowReader(5e-4), 8, COLS // 2
    halves = [torch.zeros((ROWS, LAYERS, cols, REC), dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    done = [torch.cuda.Event() for _ in range(2)]
    eraw = torch.empty((ROWS, LAYERS, cols, REC), dtype=torch.uint8, device="cuda")
    seen = torch.empty((steps, *eraw.shape), dtype=torch.uint8, device="cuda")
    rng = np.random.default_rng(12)
    ids = [rng.integers(0, 1 << 20, size=(ROWS, LAYERS, cols)) for _ in range(steps)]
    for i in range(steps):
        torch.cuda._sleep(50_000_000)                   # the chunk's compute, ahead of its rows' copy
        assert E.stage_rows(ids[i], reader, halves[i % 2], done[i % 2], eraw) == ROWS
        seen[i].copy_(eraw)
    torch.cuda.synchronize()
    for i in range(steps):
        flat = ids[i] + np.array(reader.layout.starts)[None, :, None]
        want = torch.from_numpy(reader.bytes(flat.reshape(-1))).view(eraw.shape)
        assert torch.equal(seen[i].cpu(), want), i


def _real():
    """(config, Layout, Reader) of the checkpoint's Engram tables."""

    from tensorfold.families.deepseek_v41 import engram_hash
    from tensorfold.families.deepseek_v41.config import Config
    from tensorfold.families.deepseek_v41.engram_table import Layout, Reader

    cfg = Config.read(MODEL)
    primes = engram_hash.primes(cfg.engram_layer_ids, cfg.engram_vocab_size, cfg.engram_n_heads,
                                cfg.engram_max_ngram_size).reshape(len(cfg.engram_layer_ids), -1)
    layout = Layout.read(MODEL, cfg.engram_layer_ids, primes.tolist())
    return cfg, layout, Reader(layout)


def _ids(layout, cfg, seed: int) -> np.ndarray:
    """Layer-local ids [ROWS, layers, 24]: random in each column, plus both sides of the column-12 boundary, the
    reference's own split row, and each table's first and last rows."""

    rng = np.random.default_rng(seed)
    out = np.empty((ROWS, len(cfg.engram_layer_ids), COLS), dtype=np.int64)
    for i, layer in enumerate(cfg.engram_layer_ids):
        t = layout.table(layer)
        lo, hi = np.array(t.bounds[:-1]), np.array(t.bounds[1:])
        out[:, i] = lo[None] + (rng.random((ROWS, COLS)) * (hi - lo)[None]).astype(np.int64)
        assert t.bounds[12] == COLUMN_12[layer]
        out[0, i, 11], out[0, i, 12] = t.bounds[12] - 1, t.bounds[12]
        out[1, i, 0], out[1, i, 23] = 0, t.rows - 1
        half = -(-t.rows // 2)
        col = int(np.searchsorted(t.bounds, half, side="right")) - 1
        out[2, i, col] = half
    return out


@needs_model
def test_real_rows_dequantize_bit_exactly_from_the_file_bytes():
    pytest.importorskip("safetensors")
    from dsv41_ref_pack import RefPack

    cfg, layout, reader = _real()
    pack = RefPack(MODEL)
    ids = _ids(layout, cfg, 13)
    want = torch.empty((ROWS, LAYERS, COLS, HEAD), dtype=torch.bfloat16)
    for i, layer in enumerate(cfg.engram_layer_ids):
        name = f"layers.{layer}.engram.embed"
        for r in range(ROWS):
            for c in range(COLS):
                take = slice(int(ids[r, i, c]), int(ids[r, i, c]) + 1)
                w = pack.tensor(f"{name}.weight", take).view(torch.uint8)
                s = pack.tensor(f"{name}.scale", take).view(torch.uint8)
                want[r, i, c] = _cpu_rows(torch.cat([w, s], dim=1))[0]
    for rank in range(2):
        b = _bufs(2)
        host = torch.zeros(b.eraw.shape, dtype=torch.uint8, pin_memory=True)
        cut = slice(12 * rank, 12 * (rank + 1))
        E.stage_rows(ids[:, :, cut], reader, host, torch.cuda.Event(), b.eraw)
        got = E.dequant_rows(b.eraw, b.eloc).cpu().view(ROWS, LAYERS, 12, HEAD)
        assert torch.equal(got.view(torch.int16), want[:, :, cut].view(torch.int16)), rank
    reader.close()


@needs_model
@pytest.mark.parametrize("world", [1, 2])
def test_real_layers_track_the_reference_in_mirror_mode(world, monkeypatch):
    """kv against the fp64 product (the op-level bound); the gate against the reference port's Engram fed this kv, so
    its inputs are identical."""

    pytest.importorskip("safetensors")
    from dsv41_layouts import mx8_from_block
    from dsv41_ref_weights import RefWeights

    from tensorfold.families.deepseek_v41.cuda.split import slice_for

    cfg, layout, reader = _real()
    rw = RefWeights(MODEL)
    rw.reader = reader                                  # rows by id, no tokenizer needed
    reference = ref.Reference(rw.cfg, rw, engram_rows=rw.engram_rows, mode=ref.MIRROR)
    ids, X = _ids(layout, cfg, 14), _streams(ROWS, 15)
    host = [torch.zeros((ROWS, LAYERS, COLS // world, REC), dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    for i, layer in enumerate(cfg.engram_layer_ids):
        p = f"layers.{layer}.engram"
        name, scale = f"{p}.wkv.weight", f"{p}.wkv.scale"
        wqk = (rw(f"{p}.q_weight") * rw(f"{p}.k_weight")).cuda()
        lins = []
        for r in range(world):
            rows = slice_for(r, world, name, rw.pack.shape(name))[0] if world > 1 else None
            srows = slice_for(r, world, scale, rw.pack.shape(scale))[0] if world > 1 else None
            lins.append(mx8_from_block(rw.pack.tensor(name, rows).cuda(), rw.pack.tensor(scale, srows).cuda()))
        rows = rw.engram_rows(layer, torch.as_tensor(ids[:, i])).flatten(1)

        def rank(r, i=i, lins=lins, wqk=wqk):
            def go(comm):
                b = _bufs(world)
                cut = slice(r * COLS // world, (r + 1) * COLS // world)
                E.stage_rows(ids[:, :, cut], reader, host[r], torch.cuda.Event(), b.eraw)
                e = E.exchange(E.dequant_rows(b.eraw, b.eloc), comm, b.egat, b.eng)
                kv = E.kv(e[:, i], lins[r], comm, b.ekv, b.ekv_gat)
                return e[:, i].clone(), kv.permute(1, 0, 2).flatten(1), E.inject(X.clone(), kv, wqk)
            return go

        results = [rank(0)(None)] if world == 1 else run_pair(rank(0), rank(1))
        for e, kv, out in results:
            assert torch.equal(e.cpu(), rows)
            _t1_sum(kv, rows.float(), rw(name), f"layer {layer} world {world} kv")
            with monkeypatch.context() as patch:
                patch.setattr(ref, "linear", lambda *args, kv=kv, **kwargs: kv.float().cpu())
                want = reference.engram(layer, X.float().cpu(), torch.as_tensor(ids[:, i]))
            _t1(out, want, f"layer {layer} world {world} gate")
        if world == 2:
            assert torch.equal(results[0][2], results[1][2])
        del lins
        torch.cuda.empty_cache()
    reader.close()
