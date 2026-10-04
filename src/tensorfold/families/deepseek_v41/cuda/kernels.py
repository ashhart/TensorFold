"""DeepSeek-V4.1 decode/prefill kernels: RMSNorm, table RoPE, and MQA attention over compressed + window keys.

Attention: one 512-wide latent per key is both key and value for every head (MQA). A row attends to the first
``(p + 1) // ratio`` entries of its kv source's compressed cache and to its own window slots p - 127 .. p, plus
a per-head sink logit in the denominator. Heads fill the tile rows (16 a program), so a key block is loaded once
per 16 heads; chunks of keys run as separate programs and a merge adds the sink and normalizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_TILE = 16              # heads a chunk program (decode and verify rows; prompt chunks use FULL_HT)
KEY_TILE = 64
# above this many rows (prompt chunks) with RoPE: _mqa_full, one program a row (= the decode rows at most)
FULL_ROWS = int(__import__("os").environ.get("TF_DSV41_DECODE_ROWS") or 32)
FULL_HT, FULL_KT, FULL_WARPS, FULL_STAGES = 32, 32, 8, 1
CHUNK = int(__import__("os").environ.get("TF_MQA_CHUNK") or 128)   # keys a chunk program takes (multiple of 32)


@triton.jit
def _rmsnorm(X, W, OUT, x_stride, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    ok = o < N
    x = tl.load(X + r * x_stride + o, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + o, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * N + o, (x * rinv * w).to(tl.bfloat16), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 RMSNorm of each row of x [R, N] (any float dtype, fp32 math)."""

    R, N = x.shape
    out = torch.empty((R, N), dtype=torch.bfloat16, device=x.device)
    _rmsnorm[(R,)](x, w, out, x.stride(0), eps, N=N, BLOCK=triton.next_power_of_2(N), num_warps=4)
    return out


@triton.jit
def _rope(X, POS, COS, SIN, OUT, heads, D: tl.constexpr, HALF: tl.constexpr, SIGN: tl.constexpr):
    """GPT-J rotation of the last 2*HALF dims of each [D] head vector at the row's position; others copied."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    base = (r * heads + h) * D
    d = tl.arange(0, D)
    x = tl.load(X + base + d).to(tl.float32)
    tl.store(OUT + base + d, x.to(OUT.dtype.element_ty), mask=d < D - 2 * HALF)
    p = tl.load(POS + r)
    i = tl.arange(0, HALF)
    c = tl.load(COS + p * HALF + i)
    s = tl.load(SIN + p * HALF + i) * SIGN
    e = tl.load(X + base + D - 2 * HALF + 2 * i).to(tl.float32)
    od = tl.load(X + base + D - 2 * HALF + 2 * i + 1).to(tl.float32)
    tl.store(OUT + base + D - 2 * HALF + 2 * i, (e * c - od * s).to(OUT.dtype.element_ty))
    tl.store(OUT + base + D - 2 * HALF + 2 * i + 1, (od * c + e * s).to(OUT.dtype.element_ty))


def rope(x: torch.Tensor, pos: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool = False,
         out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [R, D] or [R, H, D]; tables [max_pos, 32]; returns the rotated copy (fp32 math)."""

    shape = x.shape
    x3 = x.reshape(shape[0], -1, shape[-1]).contiguous()
    out = torch.empty(x3.shape, dtype=out_dtype or x.dtype, device=x.device)
    _rope[(x3.shape[0], x3.shape[1])](x3, pos, cos, sin, out, x3.shape[1], D=shape[-1], HALF=cos.shape[1],
                                      SIGN=-1.0 if inverse else 1.0, num_warps=4)
    return out.reshape(shape)


def rope_tables(freqs: torch.Tensor, max_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    ang = torch.arange(max_pos, dtype=torch.float64, device=freqs.device)[:, None] * freqs.double()[None, :]
    return ang.cos().float().contiguous(), ang.sin().float().contiguous()


FP8_MAX = 448.0


class QRows:
    """Quantized cache rows as byte planes (``planes``: attribute names, row-major [n, ...] tensors) plus metadata.
    Supports what the caches use: slicing (views) / clone / copy_ / zero_ / take / put_rows / shape, all as byte
    copies of every plane (rows are never re-quantized); kernels read the planes."""

    planes: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return (getattr(self, self.planes[0]).shape[0], self.dim)

    def _parts(self, ts) -> QRows:
        out = object.__new__(type(self))
        out.__dict__.update(self.__dict__)
        for p, t in zip(self.planes, ts):
            setattr(out, p, t)
        return out

    def _bytes(self) -> list[torch.Tensor]:
        return [getattr(self, p).view(torch.uint8) for p in self.planes]

    def __getitem__(self, key) -> QRows:
        return self._parts([getattr(self, p)[key] for p in self.planes])

    def clone(self) -> QRows:
        return self._parts([getattr(self, p).clone() for p in self.planes])

    def copy_(self, other: QRows) -> QRows:
        for a, b in zip(self._bytes(), other._bytes()):
            a.copy_(b)
        return self

    def zero_(self) -> QRows:
        for a in self._bytes():
            a.zero_()
        return self

    def take(self, index: torch.Tensor) -> QRows:
        """Copies of rows ``index`` (through uint8: no fp8 gather kernel)."""

        return self._parts([b[index].view(getattr(self, p).dtype) for p, b in zip(self.planes, self._bytes())])

    def put_rows(self, index: torch.Tensor, rows: QRows) -> None:
        """Rows ``index`` set to ``rows`` as stored (no re-quantization)."""

        for a, b in zip(self._bytes(), rows._bytes()):
            a.index_copy_(0, index, b)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._bytes())


class Fp8Rows(QRows):
    """Rows stored as fp8 e4m3 with an fp32 scale per ``group`` values, the last ``plain`` values kept bf16 (the
    compressed KV keeps its 64 RoPE dims bf16, as DeepSeek's V4 fp8 KV cache does); kernels read q, r and s."""

    planes = ("q", "r", "s")

    def __init__(self, n: int = 0, dim: int = 0, *, plain: int = 0, group: int = 64, device="cuda",
                 alloc=None) -> None:
        self.dim, self.plain, self.group = dim, plain, group
        f = dim - plain
        alloc = alloc or (lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device))
        self.q = alloc((n, f), torch.uint8).view(torch.float8_e4m3fn)
        self.r = alloc((n, max(plain, 1)), torch.bfloat16)
        self.s = alloc((n, f // group), torch.float32)

    @staticmethod
    def row_bytes(dim: int, plain: int, group: int) -> int:
        return (dim - plain) + max(plain, 1) * 2 + (dim - plain) // group * 4

    def quantize(self, x: torch.Tensor):
        """(q, r, s) of rows x [G, dim] (fp32 or bf16)."""

        G = x.shape[0]
        f = self.dim - self.plain
        body = x[:, :f].float().view(G, f // self.group, self.group)
        scale = body.abs().amax(-1).clamp(min=1e-12) / FP8_MAX
        q = (body / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(G, f)
        r = x[:, f:].to(torch.bfloat16) if self.plain else torch.zeros((G, 1), dtype=torch.bfloat16, device=x.device)
        return q, r, scale

    def index_copy_(self, dim: int, index: torch.Tensor, x: torch.Tensor) -> Fp8Rows:
        q, r, sc = self.quantize(x)
        self.q.view(torch.uint8).index_copy_(0, index, q.view(torch.uint8))     # (no fp8 index_copy kernel)
        self.r.index_copy_(0, index, r)
        self.s.index_copy_(0, index, sc)
        return self

    def dequant(self) -> torch.Tensor:
        """bf16 [n, dim] (tests)."""

        n, f = self.q.shape[0], self.dim - self.plain
        body = (self.q.float().view(n, f // self.group, self.group) * self.s[..., None]).view(n, f)
        return torch.cat([body, self.r[:, :self.plain].float()], dim=1).to(torch.bfloat16) if self.plain else \
            body.to(torch.bfloat16)


def cache_nbytes(t) -> int:
    return t.nbytes() if isinstance(t, QRows) else t.numel() * t.element_size()


@triton.jit
def _comp_rows(COMP, CR, CS, kidx, ok_c, d, D: tl.constexpr, FP8: tl.constexpr, F: tl.constexpr, G: tl.constexpr):
    """Compressed entries ``kidx`` [KT] as fp32/bf16 [KT, D]: bf16 rows, or fp8 rows (F values, a scale per G) with
    the last D - F values bf16."""

    row = tl.maximum(kidx, 0)[:, None].to(tl.int64)
    if FP8:
        body = d[None, :] < F
        q = tl.load(COMP + row * F + d[None, :], mask=ok_c[:, None] & body).to(tl.float32)
        sc = tl.load(CS + row * (F // G) + d[None, :] // G, mask=ok_c[:, None] & body, other=0.0)
        r = tl.load(CR + row * (D - F) + (d[None, :] - F), mask=ok_c[:, None] & (d[None, :] >= F),
                    other=0.0).to(tl.float32)
        out = tl.where(body, tl.where(ok_c[:, None], q, 0.0) * sc, r)
    else:
        out = tl.load(COMP + row * D + d[None, :], mask=ok_c[:, None], other=0.0).to(tl.float32)
    return out


@triton.jit
def _mqa_chunks(Q, COMP, IDX, SWA, POS, PO, PM, PL, n_idx, idx_stride, SBASE, CR, CS, H: tl.constexpr,
                D: tl.constexpr, W: tl.constexpr, RING: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr,
                NCH: tl.constexpr, HT: tl.constexpr, KT: tl.constexpr, HAS_BASE: tl.constexpr = False,
                FP8: tl.constexpr = False, F: tl.constexpr = 448, G: tl.constexpr = 64):
    """Keys: the first ``n_idx`` slots are compressed entries named by IDX (-1: none), then the row's window
    positions p - W + 1 .. p read from the SWA ring at pos % RING."""

    r = tl.program_id(0)
    hg = tl.program_id(1)
    c = tl.program_id(2)
    p = tl.load(POS + r)
    sbase = tl.load(SBASE + r) if HAS_BASE else 0          # the row's stream: its window ring's first row
    hh = hg * HT + tl.arange(0, HT)
    d = tl.arange(0, D)
    m = tl.full((HT,), float("-inf"), tl.float32)
    l = tl.zeros((HT,), tl.float32)
    o = tl.zeros((HT, D), tl.float32)
    total = n_idx + W
    base = (r * NCH + c) * H + hh
    q = tl.load(Q + (r * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)
    for t in range(CH // KT):
        k = c * CH + t * KT + tl.arange(0, KT)
        is_comp = k < n_idx
        kidx = tl.load(IDX + r * idx_stride + k, mask=is_comp, other=-1)
        slot = p - (W - 1) + (k - n_idx)                             # window position of a window key
        ok_c = is_comp & (kidx >= 0)
        ok_w = (k >= n_idx) & (k < total) & (slot >= 0)
        kc = _comp_rows(COMP, CR, CS, kidx, ok_c, d, D, FP8, F, G)
        kw = tl.load(SWA + (sbase + tl.maximum(slot, 0) % RING)[:, None].to(tl.int64) * D + d[None, :],
                     mask=ok_w[:, None], other=0.0)
        kk = tl.where(ok_c[:, None], kc.to(tl.bfloat16), kw.to(tl.bfloat16))
        ok = ok_c | ok_w
        scores = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        scores = tl.where(ok[None, :], scores, float("-inf"))
        tile_m = tl.max(scores, 1)
        active = tile_m != float("-inf")
        next_m = tl.where(active, tl.maximum(m, tile_m), m)
        alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        pr = tl.where(ok[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
        o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kk)
        l = l * alpha + tl.sum(pr, 1)
        m = next_m
    tl.store(PO + base[:, None] * D + d[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)


@triton.jit
def _mqa_full(Q, COMP, IDX, SWA, POS, SINK, OUT, COS, SIN, n_idx, idx_stride, CR, CS, H: tl.constexpr,
              D: tl.constexpr, W: tl.constexpr, RING: tl.constexpr, SCALE: tl.constexpr, HT: tl.constexpr,
              KT: tl.constexpr, HALF: tl.constexpr, FP8: tl.constexpr = False, F: tl.constexpr = 448,
              G: tl.constexpr = 64):
    """Prompt rows: one program takes a row's every key (as _mqa_chunks) and finishes it (sink, normalize, inverse
    RoPE of the last 2 * HALF dims), writing bf16 [R, H, D]; no per-chunk partials."""

    r = tl.program_id(0)
    hg = tl.program_id(1)
    p = tl.load(POS + r)
    hh = hg * HT + tl.arange(0, HT)
    d = tl.arange(0, D)
    m = tl.full((HT,), float("-inf"), tl.float32)
    l = tl.zeros((HT,), tl.float32)
    o = tl.zeros((HT, D), tl.float32)
    total = n_idx + W
    q = tl.load(Q + (r * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)
    for k0 in range(0, total, KT):
        k = k0 + tl.arange(0, KT)
        is_comp = k < n_idx
        kidx = tl.load(IDX + r * idx_stride + k, mask=is_comp, other=-1)
        slot = p - (W - 1) + (k - n_idx)
        ok_c = is_comp & (kidx >= 0)
        ok_w = (k >= n_idx) & (k < total) & (slot >= 0)
        kc = _comp_rows(COMP, CR, CS, kidx, ok_c, d, D, FP8, F, G)
        kw = tl.load(SWA + (tl.maximum(slot, 0) % RING)[:, None].to(tl.int64) * D + d[None, :], mask=ok_w[:, None],
                     other=0.0)
        kk = tl.where(ok_c[:, None], kc.to(tl.bfloat16), kw.to(tl.bfloat16))
        ok = ok_c | ok_w
        scores = tl.dot(q, tl.trans(kk)).to(tl.float32) * SCALE
        scores = tl.where(ok[None, :], scores, float("-inf"))
        tile_m = tl.max(scores, 1)
        active = tile_m != float("-inf")
        next_m = tl.where(active, tl.maximum(m, tile_m), m)
        alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        pr = tl.where(ok[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
        o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kk)
        l = l * alpha + tl.sum(pr, 1)
        m = next_m
    sink = tl.load(SINK + hh)                  # a logit with a zero value vector
    top = tl.maximum(m, sink)
    a = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))
    o = o * (a / (l * a + tl.exp(sink - top)))[:, None]
    rot = d >= D - 2 * HALF
    i = tl.maximum(d - (D - 2 * HALF), 0) // 2
    c_ = tl.load(COS + p * HALF + i, mask=rot, other=1.0)
    s_ = tl.load(SIN + p * HALF + i, mask=rot, other=0.0)
    ev, od = tl.split(tl.reshape(o, (HT, D // 2, 2)))
    cev, _ = tl.split(tl.reshape(c_, (D // 2, 2)))
    sev, _ = tl.split(tl.reshape(s_, (D // 2, 2)))
    ne = ev * cev[None, :] + od * sev[None, :]
    no = od * cev[None, :] - ev * sev[None, :]
    o = tl.reshape(tl.join(ne, no), (HT, D))
    tl.store(OUT + (r * H + hh[:, None]) * D + d[None, :], o.to(tl.bfloat16))


@triton.jit
def _mqa_merge(PO, PM, PL, SINK, OUT, POS, COS, SIN, H: tl.constexpr, D: tl.constexpr, NCH: tl.constexpr,
               HALF: tl.constexpr, ROPE: tl.constexpr):
    """Combine the chunks with the sink; with ROPE, rotate the last 2 * HALF dims back (inverse RoPE) and write
    bf16 (the output projection's input), else fp32."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    m = tl.load(SINK + h)                   # the sink is a logit with a zero value vector
    l = 1.0
    o = tl.zeros((D,), tl.float32)
    for c in range(NCH):
        base = (r * NCH + c) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        active = cl > 0.0
        co = tl.load(PO + base * D + d, mask=(d < D) & active, other=0.0)
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.exp(m - next_m)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    o = o / l
    if ROPE:
        p = tl.load(POS + r)
        rot = d >= D - 2 * HALF
        i = tl.maximum(d - (D - 2 * HALF), 0) // 2
        c_ = tl.load(COS + p * HALF + i, mask=rot, other=1.0)
        s_ = tl.load(SIN + p * HALF + i, mask=rot, other=0.0)
        # pair values come from the normalized o itself: even dims pair with the next odd dim
        oe = tl.reshape(o, (D // 2, 2))
        ev, od = tl.split(oe)
        ce = tl.reshape(c_, (D // 2, 2))
        cev, _ = tl.split(ce)
        se = tl.reshape(s_, (D // 2, 2))
        sev, _ = tl.split(se)
        # inverse rotation: e' = e c + o s, o' = o c - e s (identity where c = 1, s = 0)
        ne = ev * cev + od * sev
        no = od * cev - ev * sev
        o = tl.reshape(tl.join(ne, no), (D,))
        tl.store(OUT + (r * H + h) * D + d, o.to(tl.bfloat16))
    else:
        tl.store(OUT + (r * H + h) * D + d, o)


class AttnBuffers:
    def __init__(self, rows: int, heads: int, dims: int, max_keys: int, device="cuda") -> None:
        nch = triton.cdiv(max_keys, CHUNK)
        self.po = torch.empty((rows * nch * heads * dims,), dtype=torch.float32, device=device)
        self.pm = torch.empty((rows * nch * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((rows * nch * heads,), dtype=torch.float32, device=device)
        self.max_keys = max_keys


def mqa(q: torch.Tensor, comp: torch.Tensor | None, idx: torch.Tensor | None, swa: torch.Tensor, pos: torch.Tensor,
        sink: torch.Tensor, window: int, buf: AttnBuffers, scale: float, cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None, sbase: torch.Tensor | None = None, ring: int | None = None) -> torch.Tensor:
    """q [R, H, D] (RoPE'd) -> o [R, H, D] over the compressed entries ``idx`` [R, n] of ``comp`` and the window
    (``swa`` a ring of window rows addressed by position modulo its length): fp32, or with RoPE tables the
    inverse-rotated bf16 the output projection takes."""

    R, H, D = q.shape
    assert H % HEAD_TILE == 0
    n_idx = 0 if idx is None else idx.shape[1]
    keys = n_idx + window
    nch = triton.cdiv(keys, CHUNK)
    assert keys <= buf.max_keys
    rope = cos is not None
    out = torch.empty((R, H, D), dtype=torch.bfloat16 if rope else torch.float32, device=q.device)
    idx_t = idx if idx is not None else pos
    fp8 = isinstance(comp, Fp8Rows)
    cq = comp.q if fp8 else (comp if comp is not None else swa)
    cr, cs = (comp.r, comp.s) if fp8 else (swa, swa)
    fkw = {"FP8": True, "F": comp.dim - comp.plain, "G": comp.group} if fp8 else {}
    if sbase is not None and R > FULL_ROWS:
        raise ValueError("stream window bases are for decode rows (the chunk path)")
    if rope and R > FULL_ROWS:
        _mqa_full[(R, H // FULL_HT)](q.contiguous(), cq, idx_t, swa, pos, sink, out, cos, sin, n_idx,
                                     idx_t.stride(0) if idx is not None else 0, cr, cs, H=H, D=D, W=window,
                                     RING=swa.shape[0], SCALE=scale, HT=FULL_HT, KT=FULL_KT, HALF=cos.shape[1],
                                     num_warps=FULL_WARPS, num_stages=FULL_STAGES, **fkw)
        return out
    _mqa_chunks[(R, H // HEAD_TILE, nch)](q.contiguous(), cq, idx_t, swa, pos, buf.po, buf.pm, buf.pl, n_idx,
                                          idx_t.stride(0) if idx is not None else 0,
                                          sbase if sbase is not None else pos, cr, cs, H=H, D=D, W=window,
                                          RING=ring or swa.shape[0], CH=CHUNK, SCALE=scale, NCH=nch,
                                          HT=HEAD_TILE, KT=32, HAS_BASE=sbase is not None, num_warps=8, num_stages=1,
                                          **fkw)
    _mqa_merge[(R, H)](buf.po, buf.pm, buf.pl, sink, out, pos, cos if rope else sink, sin if rope else sink, H=H, D=D,
                       NCH=nch, HALF=cos.shape[1] if rope else 1, ROPE=rope, num_warps=4)
    return out


@triton.jit
def _index_scores(IQ, WTS, KEYS, POS, OUT, n_keys, ratio, KBASE, KS, HI: tl.constexpr, DI: tl.constexpr,
                  BS: tl.constexpr, HAS_BASE: tl.constexpr = False, FP8: tl.constexpr = False):
    """I[r, s] = sum_h w[r, h] * relu(iq[r, h] . k[s]) for visible s < (p + 1) // ratio, -inf elsewhere."""

    r = tl.program_id(0)
    sb = tl.program_id(1)
    p = tl.load(POS + r)
    n_vis = (p + 1) // ratio
    h = tl.arange(0, HI)
    d = tl.arange(0, DI)
    sidx = sb * BS + tl.arange(0, BS)
    kbase = tl.load(KBASE + r) if HAS_BASE else 0          # the row's stream: its first key
    q = tl.load(IQ + (r * HI + h[:, None]) * DI + d[None, :])
    # only visible keys are read: a slot sized for a long window (600K) holds ~150K entries of which a decode row
    # sees (p + 1) // ratio; reading the rest cost ~20 MB a layer a step (their scores are -inf either way)
    live = (sidx < n_keys) & (sidx < n_vis)
    k = tl.load(KEYS + (kbase + sidx)[:, None].to(tl.int64) * DI + d[None, :], mask=live[:, None], other=0.0)
    if FP8:                                                    # e4m3 -> bf16 is exact; the key's scale after the dot
        k = k.to(tl.bfloat16)
    dots = tl.dot(q, tl.trans(k)).to(tl.float32)                     # [HI, BS]
    if FP8:
        dots = dots * tl.load(KS + kbase + sidx, mask=live, other=0.0)[None, :]
    w = tl.load(WTS + r * HI + h)
    score = tl.sum(w[:, None] * tl.maximum(dots, 0.0), axis=0)
    score = tl.where(sidx < n_vis, score, float("-inf"))
    tl.store(OUT + r * n_keys + sidx, score, mask=sidx < n_keys)


def index_scores(iq: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor, ratio: int,
                 kbase: torch.Tensor | None = None, n_keys: int | None = None) -> torch.Tensor:
    """fp32 [R, S] indexer scores over every compressed entry, -inf where not yet visible. ``kbase`` (decode rows of
    several streams): each row's first key in ``keys``, its stream's ``n_keys`` following."""

    R, HI, DI = iq.shape
    S = keys.shape[0] if n_keys is None else n_keys
    scores = torch.empty((R, S), dtype=torch.float32, device=iq.device)
    BS = 64
    fp8 = isinstance(keys, Fp8Rows)
    kq, ks = (keys.q, keys.s) if fp8 else (keys, pos)
    _index_scores[(R, triton.cdiv(S, BS))](iq.contiguous(), wts.contiguous(), kq, pos, scores, S, ratio,
                                           kbase if kbase is not None else pos, ks, HI=HI, DI=DI, BS=BS,
                                           HAS_BASE=kbase is not None, FP8=fp8, num_warps=4)
    return scores


def untie(scores: torch.Tensor, first: int = 0) -> torch.Tensor:
    """Exact-zero scores (every head's ReLU closed) as tiny negatives ordered by index (lowest first), so a top-k
    among tied zeros keeps the same entries whatever the rows a call holds (torch.topk leaves ties unspecified)."""

    idx = torch.arange(first, first + scores.shape[1], device=scores.device, dtype=torch.float32)
    return torch.where(scores == 0, -1e-30 * (1.0 + idx * 2.0 ** -21), scores)


def top_entries(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """int32 [R, topk]: the best visible entries ascending, -1 padded (every visible one when <= topk)."""

    R, S = scores.shape
    k = min(topk, S)
    vals, idx = torch.topk(untie(scores), k, dim=1, sorted=False)
    idx = torch.where(torch.isinf(vals) & (vals < 0), torch.full_like(idx, S), idx)   # invisible sort last, dropped
    idx = torch.sort(idx, dim=1).values
    idx = torch.where(idx >= S, torch.full_like(idx, -1), idx).int()
    if k < topk:
        idx = torch.cat([idx, torch.full((R, topk - k), -1, dtype=idx.dtype, device=idx.device)], dim=1)
    return idx.contiguous()


def candidate_blocks(scores: torch.Tensor, pos: torch.Tensor, ratio: int, block: int, keep: int) -> torch.Tensor:
    """Layer 20's blocks of ``block`` entries scored by their best entry, the newest block pinned, the ``keep`` best
    kept: int64 [R, keep], -1 padded."""

    R, S = scores.shape
    nb = -(-S // block)
    padded = torch.full((R, nb * block), float("-inf"), dtype=scores.dtype, device=scores.device)
    padded[:, :S] = scores
    best = untie(padded.view(R, nb, block).amax(-1))
    newest = ((pos + 1) // ratio - 1).clamp(min=0) // block
    best.scatter_(1, newest[:, None].long(), float("inf"))
    vals, idx = torch.topk(best, min(keep, nb), dim=1)
    idx = torch.where(torch.isinf(vals) & (vals < 0), torch.full_like(idx, -1), idx)
    if idx.shape[1] < keep:
        idx = torch.cat([idx, torch.full((R, keep - idx.shape[1]), -1, dtype=idx.dtype, device=idx.device)], dim=1)
    return idx


def mask_to_blocks(scores: torch.Tensor, blocks: torch.Tensor, block: int) -> torch.Tensor:
    """Scores outside the chosen blocks set to -inf."""

    R, S = scores.shape
    nb = -(-S // block)
    flags = torch.zeros((R, nb + 1), dtype=torch.bool, device=scores.device)
    flags.scatter_(1, torch.where(blocks >= 0, blocks, nb), True)
    keep = flags[:, :nb].repeat_interleave(block, dim=1)[:, :S]
    return scores.masked_fill(~keep, float("-inf"))


@triton.jit
def _index_scores_seg(IQ, WTS, KEYS, POS, OUT, n_keys, off, seg, out_stride, ratio, KS, HI: tl.constexpr,
                      DI: tl.constexpr, BS: tl.constexpr, FP8: tl.constexpr = False):
    """_index_scores over the key segment [off, off + seg): OUT[r, j] for key off + j (-inf past n_keys or not yet
    visible), the same arithmetic per key."""

    r = tl.program_id(0)
    sb = tl.program_id(1)
    p = tl.load(POS + r)
    n_vis = (p + 1) // ratio
    h = tl.arange(0, HI)
    d = tl.arange(0, DI)
    j = sb * BS + tl.arange(0, BS)
    sidx = off + j
    q = tl.load(IQ + (r * HI + h[:, None]) * DI + d[None, :])
    k = tl.load(KEYS + sidx[:, None].to(tl.int64) * DI + d[None, :], mask=(sidx < n_keys)[:, None], other=0.0)
    if FP8:
        k = k.to(tl.bfloat16)
    dots = tl.dot(q, tl.trans(k)).to(tl.float32)
    if FP8:
        dots = dots * tl.load(KS + sidx, mask=sidx < n_keys, other=0.0)[None, :]
    w = tl.load(WTS + r * HI + h)
    score = tl.sum(w[:, None] * tl.maximum(dots, 0.0), axis=0)
    score = tl.where((sidx < n_vis) & (sidx < n_keys), score, float("-inf"))
    tl.store(OUT + r * out_stride + j, score, mask=j < seg)


SELECT_ROWS = 512          # prompt rows a blocked selection pass takes
SELECT_SEG = 16384         # keys a segment scores at once (a multiple of every candidate block size)


def index_select_blocked(iq: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor, ratio: int,
                         topk: int, *, blocks: torch.Tensor | None = None, block: int = 8,
                         candidates: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prompt rows: top_entries(index_scores(...)) (masked to ``blocks`` when given) without the [rows, keys]
    matrix: rows in passes of SELECT_ROWS, keys in segments of SELECT_SEG, a running top-k merged per segment.
    With ``candidates`` > 0 also returns candidate_blocks(...) of the unmasked scores (the source layer's)."""

    R, HI, DI = iq.shape
    S = keys.shape[0]
    dev = iq.device
    iq, wts = iq.contiguous(), wts.contiguous()
    k = min(topk, S)
    seg = min(SELECT_SEG, -(-S // block) * block)
    idx_out = torch.full((R, topk), -1, dtype=torch.int32, device=dev)
    cand_out = torch.full((R, candidates), -1, dtype=torch.int64, device=dev) if candidates else None
    nb = -(-S // block)
    buf = torch.empty((min(R, SELECT_ROWS), seg), dtype=torch.float32, device=dev)
    fp8 = isinstance(keys, Fp8Rows)
    kq, ks = (keys.q, keys.s) if fp8 else (keys, pos)
    for r0 in range(0, R, SELECT_ROWS):
        r1 = min(R, r0 + SELECT_ROWS)
        n = r1 - r0
        vals = torch.full((n, k), float("-inf"), dtype=torch.float32, device=dev)
        ids = torch.full((n, k), S, dtype=torch.int64, device=dev)
        best = torch.full((n, nb), float("-inf"), dtype=torch.float32, device=dev) if candidates else None
        flags = None
        if blocks is not None:
            flags = torch.zeros((n, nb + 1), dtype=torch.bool, device=dev)
            b = blocks[r0:r1]
            flags.scatter_(1, torch.where(b >= 0, b, nb), True)
        for off in range(0, S, seg):
            length = min(seg, S - off)
            sc = buf[:n, :length]
            _index_scores_seg[(n, triton.cdiv(length, 64))](iq[r0:r1], wts[r0:r1], kq, pos[r0:r1], sc, S, off,
                                                           length, buf.stride(0), ratio, ks, HI=HI, DI=DI, BS=64,
                                                           FP8=fp8, num_warps=4)
            if best is not None:                               # the source layer's block maxima (unmasked)
                padded = sc if length % block == 0 else torch.nn.functional.pad(sc, (0, block - length % block),
                                                                                value=float("-inf"))
                best[:, off // block: off // block + padded.shape[1] // block] = padded.view(n, -1, block).amax(-1)
            if flags is not None:                              # the later layers: only the candidate blocks
                keep = flags[:, off // block: off // block + -(-length // block)].repeat_interleave(block, 1)
                sc = sc.masked_fill(~keep[:, :length], float("-inf"))
            kk = min(k, length)
            v, i = torch.topk(untie(sc, off), kk, dim=1, sorted=False)
            vals, pick = torch.topk(torch.cat([vals, v], dim=1), k, dim=1, sorted=False)
            ids = torch.gather(torch.cat([ids, i + off], dim=1), 1, pick)
        ids = torch.where(torch.isinf(vals) & (vals < 0), torch.full_like(ids, S), ids)   # invisible: dropped
        ids = torch.sort(ids, dim=1).values
        idx_out[r0:r1, :k] = torch.where(ids >= S, torch.full_like(ids, -1), ids).int()
        if best is not None:
            best = untie(best)
            newest = ((pos[r0:r1] + 1) // ratio - 1).clamp(min=0) // block
            best.scatter_(1, newest[:, None].long(), float("inf"))
            v, i = torch.topk(best, min(candidates, nb), dim=1)
            cand_out[r0:r1, :i.shape[1]] = torch.where(torch.isinf(v) & (v < 0), torch.full_like(i, -1), i)
    return idx_out, cand_out


def index_select(iq: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor, ratio: int,
                 topk: int) -> torch.Tensor:
    """The compressed entries each row attends to: int32 [R, topk], ascending, -1 padded (all visible when <= topk)."""

    return top_entries(index_scores(iq, wts, keys, pos, ratio), topk)

@triton.jit
def _route(L, BIAS, PICK, WTS, scale, E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr):
    """sqrt(softplus) scores; the K best of score + bias (lowest id on ties); weights = scores renormalized x scale."""

    r = tl.program_id(0)
    e = tl.arange(0, EP)
    ok = e < E
    x = tl.load(L + r * E + e, mask=ok, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
    sc = tl.sqrt(sp)
    choice = tl.where(ok, sc + tl.load(BIAS + e, mask=ok, other=0.0), float("-inf"))
    total = 0.0
    for k in tl.static_range(K):
        best = tl.max(choice, axis=0)
        idx = tl.min(tl.where(choice == best, e, EP), axis=0)
        w = tl.sum(tl.where(e == idx, sc, 0.0), axis=0)
        tl.store(PICK + r * K + k, idx)
        tl.store(WTS + r * K + k, w)
        total += w
        choice = tl.where(e == idx, float("-inf"), choice)
    kk = tl.arange(0, KP)
    w = tl.load(WTS + r * K + kk, mask=kk < K, other=0.0)
    tl.store(WTS + r * K + kk, w / total * scale, mask=kk < K)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    R, E = logits.shape
    pick = torch.empty((R, k), dtype=torch.int32, device=logits.device)
    wts = torch.empty((R, k), dtype=torch.float32, device=logits.device)
    _route[(R,)](logits.contiguous(), bias, pick, wts, scale, E=E, EP=triton.next_power_of_2(E), K=k,
                     KP=triton.next_power_of_2(k), num_warps=4)
    return pick, wts


@triton.jit
def _router_logits(X, W, OUT, R, E: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BE: tl.constexpr,
                   BK: tl.constexpr, KS: tl.constexpr):
    """OUT [KS, R, E] fp32: slice ks of X [R, D] fp16 @ W [E, D]^T fp16 over D / KS inputs; a row's K order within
    a slice is fixed and MMA rows are independent."""

    rb = tl.program_id(0)
    eb = tl.program_id(1)
    ks = tl.program_id(2)
    r = rb * BR + tl.arange(0, BR)
    e = eb * BE + tl.arange(0, BE)
    k = tl.arange(0, BK)
    acc = tl.zeros((BR, BE), dtype=tl.float32)
    for k0 in range(ks * (D // KS), (ks + 1) * (D // KS), BK):
        x = tl.load(X + r[:, None] * D + k0 + k[None, :], mask=(r < R)[:, None], other=0.0)
        w = tl.load(W + e[:, None] * D + k0 + k[None, :], mask=(e < E)[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + (ks * R + r[:, None]) * E + e[None, :], acc, mask=(r < R)[:, None] & (e < E)[None, :])


@triton.jit
def _sum_slices(P, OUT, n, KS: tl.constexpr, B: tl.constexpr):
    """OUT [n] = P [KS, n] summed over the slices in order (fixed per element)."""

    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    acc = tl.load(P + i, mask=m, other=0.0)
    for s in range(1, KS):
        acc += tl.load(P + s * n + i, mask=m, other=0.0)
    tl.store(OUT + i, acc, mask=m)


# K slices of the router / indexer-weight matmuls: one row's 384 (or 64) outputs alone are a dozen programs, too few
# to stream the weights; slices fill the SMs and are added in a fixed order (the same for every row count)
ROUTER_SLICES = int(__import__("os").environ.get("TF_ROUTER_SLICES") or 8)


def router_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 router logits; x bf16/fp16 [R, D] (bf16 -> fp16 exact for normed rows), w fp16 [E, D]."""

    R, D = x.shape
    E = w.shape[0]
    BR, BE, BK = 16, 32, 64
    KS = ROUTER_SLICES if D % (ROUTER_SLICES * BK) == 0 else 1
    part = torch.empty((KS, R, E), dtype=torch.float32, device=x.device)
    _router_logits[(triton.cdiv(R, BR), triton.cdiv(E, BE), KS)](x.half().contiguous(), w, part, R, E=E, D=D, BR=BR,
                                                                 BE=BE, BK=BK, KS=KS, num_warps=4)
    if KS == 1:
        return part[0]
    out = torch.empty((R, E), dtype=torch.float32, device=x.device)
    _sum_slices[(triton.cdiv(R * E, 1024),)](part, out, R * E, KS=KS, B=1024, num_warps=4)
    return out


@triton.jit
def _l2_prefetch(ADDR, LINES, n, B: tl.constexpr):
    """Prefetch regions into L2 (evict-last): region j is LINES[j] 128-byte lines from address ADDR[j]."""

    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for j in range(n):
        base = tl.load(ADDR + j)
        lines = tl.load(LINES + j)
        for i0 in range(pid * B, lines, npg * B):
            i = i0 + tl.arange(0, B)
            a = base + tl.where(i < lines, i, 0).to(tl.int64) * 128
            tl.inline_asm_elementwise("prefetch.global.L2::evict_last [$1]; mov.u32 $0, 0;", "=r,l", [a],
                                      dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _l2_load(ADDR, LINES, SINK, n, B: tl.constexpr):
    """Read regions (ADDR[j], LINES[j] 128-byte lines) with evict-last loads, so they stay in L2 for the next kernel
    (GB10 drops prefetch hints; real loads stay). One int32 a 32-byte sector, folded into a sink the kernel writes."""

    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    acc = tl.zeros((B,), dtype=tl.int32)
    for j in range(n):
        base = tl.load(ADDR + j).to(tl.pointer_type(tl.int32))
        sectors = tl.load(LINES + j) * 4                     # L2 fills 32-byte sectors: one load each
        for i0 in range(pid * B, sectors, npg * B):
            i = i0 + tl.arange(0, B)
            m = i < sectors
            acc ^= tl.load(base + i.to(tl.int64) * 8, mask=m, other=0, eviction_policy="evict_last")
    tl.store(SINK + pid * B + tl.arange(0, B), acc)


def prefetch_table(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(addresses, 128-byte line counts) of tensors' storage, for ``l2_prefetch``."""

    dev = tensors[0].device
    addr = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=dev)
    lines = torch.tensor([(t.numel() * t.element_size() + 127) // 128 for t in tensors], dtype=torch.int64, device=dev)
    return addr, lines


_SINK: dict = {}


def l2_prefetch(table: tuple[torch.Tensor, torch.Tensor], programs: int = 48) -> None:
    """Warm L2 with weights a later step reads: one evict-last load a 128-byte line (prefetch hints are dropped)."""

    addr, lines = table
    sink = _SINK.get(addr.device)
    if sink is None or sink.numel() < programs * 256:
        sink = _SINK[addr.device] = torch.zeros((max(programs, 64) * 256,), dtype=torch.int32, device=addr.device)
    _l2_load[(programs,)](addr, lines, sink, addr.numel(), B=256, num_warps=8)


@triton.jit
def _l2_bulk(ADDR, BYTES, n, CHUNK: tl.constexpr, B: tl.constexpr):
    """One cp.async.bulk.prefetch.L2 a CHUNK-byte piece of regions (ADDR[j], BYTES[j]): the SM's bulk-copy unit streams
    them into L2 with no registers or data returned (measured on GB10: a 12-20 MB read 1.4-1.5x faster right after;
    prefetch.global.L2 lines are dropped). Writes nothing. After Jay Leaton's l2pf.cu (MIT)."""

    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for j in range(n):
        base = tl.load(ADDR + j)
        size = tl.load(BYTES + j)
        pieces = (size + CHUNK - 1) // CHUNK
        for i0 in range(pid * B, pieces, npg * B):
            i = i0 + tl.arange(0, B)
            off = i.to(tl.int64) * CHUNK
            left = tl.minimum(size - off, CHUNK).to(tl.int32)
            ok = i < pieces
            a = tl.where(ok, base + off, base)
            m = tl.where(ok, left, 16)
            tl.inline_asm_elementwise("cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;", "=r,l,r", [a, m],
                                      dtype=tl.int32, is_pure=False, pack=1)


def bulk_table(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(addresses, bytes) of tensors' storage for ``l2_bulk``: 16-byte aligned starts, sizes a multiple of 16."""

    dev = tensors[0].device
    addr, size = [], []
    for t in tensors:
        a = t.data_ptr()
        lead = -a % 16
        n = (t.numel() * t.element_size() - lead) // 16 * 16
        if n > 0:
            addr.append(a + lead)
            size.append(n)
    return (torch.tensor(addr, dtype=torch.int64, device=dev), torch.tensor(size, dtype=torch.int64, device=dev))


def l2_bulk(table: tuple[torch.Tensor, torch.Tensor], programs: int = 2) -> None:
    addr, size = table
    _l2_bulk[(programs,)](addr, size, addr.numel(), CHUNK=32768, B=128, num_warps=4)


@triton.jit
def _await_rows(FLAG, SEEN, SRC, DST, ERR, n, B: tl.constexpr, SPIN: tl.constexpr):
    """Wait until the host's flag passes this graph's counter (``SEEN`` + 1), then copy ``n`` int32 words of rows from
    pinned host memory into the graph's buffer. ``ERR`` (pinned) <- 1 when the host never published (a bounded wait)."""

    target = tl.load(SEEN) + 1
    f = tl.load(FLAG, volatile=True)
    it = 0
    while (f < target) & (it < SPIN):
        f = tl.load(FLAG, volatile=True)
        it += 1
    if f < target:
        tl.store(ERR, 1)
    tl.inline_asm_elementwise("fence.acq_rel.sys; mov.b64 $0, $1;", "=l,l", [f], dtype=tl.int64, is_pure=False,
                              pack=1)
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for i0 in range(pid * B, n, npg * B):
        i = i0 + tl.arange(0, B)
        m = i < n
        tl.store(DST + i, tl.load(SRC + i, mask=m, volatile=True), mask=m)


@triton.jit
def _bump(SEEN):
    tl.store(SEEN, tl.load(SEEN) + 1)


def await_rows(flag: torch.Tensor, seen: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, err: torch.Tensor) -> None:
    """In a decode graph: wait for the host's rows (``flag`` > ``seen``), copy them in, count the wait in ``seen``."""

    s32, d32 = src.view(torch.int32).reshape(-1), dst.view(torch.int32).reshape(-1)
    _await_rows[(8,)](flag, seen, s32, d32, err, s32.numel(), B=1024, SPIN=20_000_000, num_warps=4)
    _bump[(1,)](seen)


@triton.jit
def _engram_gate(X, KV, QW, KW, OUT, eps, clamp, D: tl.constexpr, S: tl.constexpr, CH: tl.constexpr):
    """One (row, stream): RMS-cosine gate of the stream against its key, then stream + gate * value (fixed order)."""

    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, CH)
    hh = 0.0
    kk = 0.0
    dot = 0.0
    for c in range(D // CH):
        o = c * CH + d
        h = tl.load(X + (r * S + s) * D + o).to(tl.float32)
        key = tl.load(KV + r * (S + 1) * D + s * D + o).to(tl.float32)
        q = tl.load(QW + s * D + o)
        k = tl.load(KW + s * D + o)
        hh += tl.sum(h * h, axis=0)
        kk += tl.sum(key * key, axis=0)
        dot += tl.sum(h * q * k * key, axis=0)
    dot = dot * (1.0 / tl.sqrt(hh / D + eps)) * (1.0 / tl.sqrt(kk / D + eps)) / tl.sqrt(D * 1.0)
    g = tl.sqrt(tl.maximum(tl.abs(dot), clamp))
    g = tl.where(dot < 0.0, -g, g)
    gate = 1.0 / (1.0 + tl.exp(-g))
    for c in range(D // CH):
        o = c * CH + d
        h = tl.load(X + (r * S + s) * D + o).to(tl.float32)
        val = tl.load(KV + r * (S + 1) * D + S * D + o).to(tl.float32)
        tl.store(OUT + (r * S + s) * D + o, (h + gate * val).to(tl.bfloat16))


def engram_gate(X: torch.Tensor, kv: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, eps: float,
                clamp: float = 1e-6) -> torch.Tensor:
    """X bf16 [R, S, D], kv [R, (S + 1) D] (S keys then the value), q/k weights fp32 [S, D] -> bf16 [R, S, D]."""

    R, S, D = X.shape
    out = torch.empty_like(X)
    _engram_gate[(R, S)](X.contiguous(), kv.contiguous(), qw, kw, out, eps, clamp, D=D, S=S, CH=1024, num_warps=4)
    return out
