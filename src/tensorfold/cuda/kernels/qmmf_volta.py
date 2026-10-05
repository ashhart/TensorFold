"""sm_70 linears for an NVFP4 checkpoint: NVFP4 (W4A16), FP8 (W8A16) and 16-bit weights as exact fp16 values."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.build import VOLTA, load

FP4, FP8, F16 = 0, 1, 2
WORDS = {FP4: 8, FP8: 16, F16: 32}             # 32-bit words a column's 64-input group takes
NAMES = {FP4: "volta-nvfp4", FP8: "volta-fp8", F16: "volta-f16"}
DENSE_PROMPT_LIMIT = 1 << 28                   # weights of more inputs x outputs prompt on the decode kernel (the head)


@lru_cache(maxsize=1)
def _ext():
    here = Path(__file__).parent
    return load(name="tensorfold_qmmf_volta_v4", sources=[str(here / "qmmf_volta.cu")], need=VOLTA,
                extra_cuda_cflags=["-O3"], verbose=False)


_COUNTERS: dict = {}


def _counters(device) -> torch.Tensor:
    """int32 zeros the kernel counts finished K slices in, one buffer a (device, CUDA stream)."""

    key = (str(device), torch.cuda.current_stream(device).cuda_stream)
    c = _COUNTERS.get(key)
    if c is None:
        c = _COUNTERS[key] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return c


def prep(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 rows -> prep884's fp16 rows in fragment order and row scales, shared by one input's projections."""

    from tensorfold.cuda.kernels import qmm_volta

    if x.stride(1) != 1:
        x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return tuple(qmm_volta._ext().prep884(x, 1))


def _tile(words: torch.Tensor, wpg: int) -> torch.Tensor:
    """int32 (N, K/64 * wpg) -> [N/32][K/64][32][wpg], N padded to 32 and a half group's K to 64 with zeros."""

    n, kw = words.shape
    if kw % wpg:
        words = torch.cat([words, words.new_zeros((n, -kw % wpg))], dim=1)
        kw = words.shape[1]
    kg = kw // wpg
    pad = -n % 32
    if pad:
        words = torch.cat([words, words.new_zeros((pad, kw))])
    return words.view((n + pad) // 32, 32, kg, wpg).permute(0, 2, 1, 3).contiguous()


def _bytes_as_words(b: torch.Tensor) -> torch.Tensor:
    """uint8 (..., B) -> int32 (..., ceil(B / 4)), zero bytes past B."""

    if b.shape[-1] % 4:
        b = torch.cat([b, b.new_zeros((*b.shape[:-1], -b.shape[-1] % 4))], dim=-1)
    return b.contiguous().view(torch.int32)


def _alpha(values: torch.Tensor | float, n: int, device) -> torch.Tensor:
    """fp32 column factors padded to a multiple of 32 (``tiles`` slices them with the words)."""

    npad = -(-n // 32) * 32
    a = torch.ones(npad, dtype=torch.float32, device=device)
    a[:n] = values if torch.is_tensor(values) else float(values)
    return a


class VoltaLinear:
    """One projection in a Volta format: Tiled words, NVFP4 block-scale words, fp32 column factors."""

    def __init__(self, fmt: int, words: torch.Tensor, scales: torch.Tensor | None, alpha: torch.Tensor, n: int, k: int):
        self.fmt, self.words, self.alpha, self.n, self.k = fmt, words, alpha, n, k
        self.scales = scales if scales is not None else words.new_zeros(1)
        self.layout = NAMES[fmt]

    # ---- construction ------------------------------------------------------------------------------------------

    @classmethod
    def from_nvfp4(cls, weight: torch.Tensor, weight_scale: torch.Tensor, global_scale: float) -> "VoltaLinear":
        """``weight`` uint8 [N, K/2] (code k at nibble k % 2 of byte k // 2), e4m3 ``weight_scale`` [N, K/16]."""

        n, k = weight.shape[0], weight.shape[1] * 2
        if k % 32:
            raise ValueError(f"NVFP4 weight [{n}, {k}]: K must be a multiple of 32")
        words = _tile(weight.contiguous().view(torch.int32), 8)
        scales = _tile(_bytes_as_words(weight_scale.contiguous().view(torch.uint8)), 1).view(words.shape[:3])
        return cls(FP4, words, scales.contiguous(), _alpha(float(global_scale) * 16384.0, n, weight.device), n, k)

    @classmethod
    def from_fp8(cls, weight: torch.Tensor, scale: float) -> "VoltaLinear":
        """``weight`` e4m3 [N, K] with one tensor scale."""

        n, k = weight.shape
        if k % 32:
            raise ValueError(f"FP8 weight [{n}, {k}]: K must be a multiple of 32")
        words = _tile(weight.contiguous().view(torch.uint8).view(torch.int32), 16)
        return cls(FP8, words, None, _alpha(float(scale) * 256.0, n, weight.device), n, k)

    @classmethod
    def from_bf16(cls, weight: torch.Tensor, chunk: int = 8192, cols: slice | None = None) -> "VoltaLinear":
        """A 16-bit [N, K] weight as fp16, each column scaled by a power of two into [2^14, 2^15), a chunk at a time."""

        n, k = weight.shape
        lo, hi = (0, k) if cols is None else (cols.start, cols.stop)    # ``cols``: whole 32-input halves of each row
        if lo % 32 or hi % 32 or not 0 <= lo < hi <= k:
            raise ValueError(f"16-bit weight [{n}, {k}]: inputs [{lo}, {hi}) are not whole 32-input halves")
        n32, kg = -(-n // 32), -(-(hi - lo) // 64)
        words = torch.empty((n32, kg, 32, 32), dtype=torch.int32, device=weight.device)
        exps = torch.empty(n, dtype=torch.float32, device=weight.device)
        for a in range(0, n, chunk):                            # chunk is a multiple of 32
            part = weight[a:a + chunk].float()
            amax = part.abs().amax(1)                           # the whole row's maximum, whatever ``cols`` keeps
            e = torch.where(amax > 0, torch.frexp(amax).exponent.float() - 15.0, torch.zeros_like(amax))
            w16 = torch.ldexp(part[:, lo:hi], -e[:, None]).to(torch.float16)
            del part
            words[a // 32:a // 32 + -(-w16.shape[0] // 32)] = _tile(w16.view(torch.int32), 32)
            exps[a:a + chunk] = torch.ldexp(torch.ones_like(e), e)
            del w16
        return cls(F16, words, None, _alpha(exps, n, weight.device), n, hi - lo)

    # ---- views -------------------------------------------------------------------------------------------------

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.words, self.scales, self.alpha))

    def tiles(self, t0: int, t1: int) -> "VoltaLinear":
        """Outputs [64 t0, 64 t1) as views, no copy (the drafter's head rows)."""

        a, b = 2 * t0, min(2 * t1, self.words.shape[0])
        scales = self.scales[a:b] if self.fmt == FP4 else self.scales
        return VoltaLinear(self.fmt, self.words[a:b], scales, self.alpha[32 * a:32 * b], min(self.n, 64 * t1) - 64 * t0,
                           self.k)

    def copy(self) -> "VoltaLinear":
        """Its own tensors (``tiles`` gives views that keep the whole weight alive)."""

        return VoltaLinear(self.fmt, self.words.clone(), self.scales.clone() if self.fmt == FP4 else None,
                           self.alpha.clone(), self.n, self.k)

    def dense(self) -> torch.Tensor:
        """The weight as dense fp16 (N, K): the decode kernel's values, before the column factors."""

        return _ext().dequantf(self.words, self.scales, self.fmt, self.n, self.k)

    # ---- two ranks: shards of the stored words, never re-encoded ---------------------------------------------------

    def outputs(self, rows: torch.Tensor, chunk: int = 8192) -> "VoltaLinear":
        """Column-parallel shard: output columns ``rows`` with every stored word, re-tiled ``chunk`` outputs at a time."""

        rows = rows.to(self.words.device, torch.int64)
        n = int(rows.numel())
        if n == 0 or int(rows.min()) < 0 or int(rows.max()) >= self.n:
            raise ValueError(f"{self.layout}: output rows outside [0, {self.n})")
        kg, wpg = -(-self.k // 64), WORDS[self.fmt]
        words = torch.empty((-(-n // 32), kg, 32, wpg), dtype=torch.int32, device=self.words.device)
        scales = (torch.empty((-(-n // 32), kg, 32), dtype=torch.int32, device=self.words.device)
                  if self.fmt == FP4 else None)
        for a in range(0, n, chunk):
            part = rows[a:a + chunk]
            tiles, where = torch.unique(part // 32, return_inverse=True)
            local = where * 32 + part % 32                    # the row inside the touched tiles, untiled
            got = self.words.index_select(0, tiles).permute(0, 2, 1, 3).reshape(-1, kg * wpg).index_select(0, local)
            words[a // 32:a // 32 + -(-got.shape[0] // 32)] = _tile(got, wpg)
            if scales is not None:
                got = self.scales.index_select(0, tiles).permute(0, 2, 1).reshape(-1, kg).index_select(0, local)
                scales[a // 32:a // 32 + -(-got.shape[0] // 32)] = _tile(got, 1).view(-1, kg, 32)
        return VoltaLinear(self.fmt, words, scales, _alpha(self.alpha.index_select(0, rows), n, rows.device), n, self.k)

    def inputs(self, rank: int, world: int = 2) -> "VoltaLinear":
        """Row-parallel shard: whole 64-input groups with their block scales; the column factors kept."""

        if world not in (2, 4) or rank not in range(world) or self.k % (64 * world):
            raise ValueError(f"{self.layout}: K {self.k} does not split into {world} parts of whole 64-input groups")
        half = self.k // 64 // world
        return self.groups(rank * half, (rank + 1) * half)

    def groups(self, g0: int, g1: int) -> "VoltaLinear":
        """Row-parallel shard of the 64-input groups [g0, g1): stored words and block scales, column factors kept."""

        kg = -(-self.k // 64)
        if not 0 <= g0 < g1 <= kg:
            raise ValueError(f"{self.layout}: input groups [{g0}, {g1}) outside [0, {kg})")
        return VoltaLinear(self.fmt, self.words[:, g0:g1].contiguous(),
                           self.scales[:, g0:g1].contiguous() if self.fmt == FP4 else None, self.alpha.clone(),
                           self.n, min(self.k, 64 * g1) - 64 * g0)

    def partial(self, x: torch.Tensor) -> torch.Tensor:
        """A row-parallel rank's fp32 (M, n) product, unrounded, for the rank-ordered sum."""

        return self.matmul(x, f32=True)

    # ---- matmuls -----------------------------------------------------------------------------------------------

    def _check(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.k:
            raise ValueError(f"{self.layout}: rows must be (M, {self.k}), not {tuple(x.shape)}")
        return x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)

    def matmul(self, x: torch.Tensor, *, f32: bool = False, xs=None) -> torch.Tensor:
        """Decode rows (any count): the same bits for a row alone or among others."""

        from tensorfold.cuda.kernels import qmm_volta

        x = self._check(x)
        m = x.shape[0]
        sk = qmm_volta.split_k884(self.n, self.k)
        out = torch.empty((m, self.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
        part = torch.empty((sk, m, self.n), dtype=torch.float32, device=x.device) if sk > 1 else None
        x16, rs = xs if isinstance(xs, (tuple, list)) and xs[0].dim() == 4 else prep(x)
        _ext().qmmf884s(x16, rs, self.words, self.scales, self.alpha, out, part, _counters(x.device), self.fmt, sk,
                        qmm_volta._row_tiles(m), f32, self.n)
        return out

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        y = self.matmul(x)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def prefill(self, x: torch.Tensor, *, f32: bool = False) -> torch.Tensor:
        """Prompt rows via a dense fp16 copy and cuBLASLt (chunk invariant); a weight past the limit uses decode's."""

        from tensorfold.cuda.kernels import qmm_volta

        x = self._check(x)
        if self.n * self.k > DENSE_PROMPT_LIMIT:
            return self.matmul(x, f32=f32)
        x16, rs = qmm_volta.prep(x if x.stride(1) == 1 else x.contiguous())
        y = qmm_volta._ext().gemm(x16, self.dense())
        return _ext().unscale2(y, rs, self.alpha, f32)


class HostRows:
    """A 16-bit table (the embedding) in page-locked host memory, gathered by the GPU as stored (``host-b16``)."""

    layout = "host-b16"

    def __init__(self, weight: torch.Tensor):
        self.weight = (weight if weight.device.type == "cpu" else weight.cpu()).contiguous().pin_memory()

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def nbytes(self) -> int:
        return 0                                       # no device memory

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        return _ext().gather_host_rows(self.weight, ids.to(torch.int64).contiguous())


EX_ROWS = 8               # pairs an expert item takes (``qmmf_volta.cu``)


def _tile_experts(words: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 [E, N, K/2] codes and e4m3 [E, N, K/16] scales -> [E, N/32, K/64, 32, 8] words, [E, N/32, K/64, 32]."""

    e, n, k2 = words.shape
    k = k2 * 2
    kg = k // 64
    w32 = words.contiguous().view(torch.int32)
    s8 = scales.contiguous().view(torch.uint8)
    if k % 64 == 0:
        w = w32.view(e, n // 32, 32, kg, 8).permute(0, 1, 3, 2, 4).contiguous()
        s = s8.view(torch.int32).view(e, n // 32, 32, kg).permute(0, 1, 3, 2).contiguous()
        return w, s
    if k % 32:
        raise ValueError(f"experts [{e}, {n}, {k}]: K must be a multiple of 32")
    # a 32-input tail: each column tile's whole groups, then the tail packed ([32][4] words, 16-bit scale pairs)
    wf = w32[..., :kg * 8].reshape(e, n // 32, 32, kg, 8).permute(0, 1, 3, 2, 4).reshape(e, n // 32, kg * 256)
    wt = w32[..., kg * 8:].reshape(e, n // 32, 128)
    sf = s8[..., :kg * 4].contiguous().view(torch.int32).view(e, n // 32, 32, kg).permute(0, 1, 3, 2)
    st = s8[..., kg * 4:].contiguous().view(torch.int16).view(e, n // 32, 32)
    s = torch.cat([sf.reshape(e, n // 32, kg * 32), st.contiguous().view(torch.int32).view(e, n // 32, 16)], dim=2)
    return torch.cat([wf, wt], dim=2).contiguous(), s.contiguous()


class VoltaExperts:
    """A layer's routed NVFP4 experts (shared last) for the sm_70 grouped kernel, gate and up side by side a column."""

    def __init__(self, up, up_s, up_a, down, down_s, down_a, width: int, dims: int, limit: float = 0.0):
        self.up, self.up_s, self.up_a = up, up_s, up_a
        self.down, self.down_s, self.down_a = down, down_s, down_a
        self.width, self.dims, self.limit = width, dims, float(limit)
        self.router: VoltaLinear | None = None

    @property
    def count(self) -> int:
        return int(self.up.shape[0])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.up, self.up_s, self.up_a, self.down, self.down_s,
                                                          self.down_a))

    @classmethod
    def make(cls, gate: tuple, up: tuple, down: tuple, *, limit: float = 0.0, chunk: int = 32) -> "VoltaExperts":
        """Each of gate, up, down: (words [E, N, K/2] uint8, e4m3 scales [E, N, K/16], per-expert scales [E])."""

        e, width, dims = int(gate[0].shape[0]), int(gate[0].shape[1]), int(gate[0].shape[2]) * 2
        if width % 32 or dims % 64:
            raise ValueError(f"experts [{e}, {width}, {dims}]: the width must be whole 32s, the hidden size whole 64s")
        dev = gate[0].device
        uw = torch.empty((e, width // 32, dims // 64, 32, 2, 8), dtype=torch.int32, device=dev)
        us = torch.empty((e, width // 32, dims // 64, 32, 2), dtype=torch.int32, device=dev)
        for e0 in range(0, e, chunk):                          # one copy of a chunk of experts at a time
            for m, (wds, scs, _) in enumerate((gate, up)):
                w, s = _tile_experts(wds[e0:e0 + chunk], scs[e0:e0 + chunk])
                uw[e0:e0 + chunk, :, :, :, m] = w
                us[e0:e0 + chunk, :, :, :, m] = s
        dw0, ds0 = _tile_experts(down[0][:1], down[1][:1])          # one expert gives the (packed) shapes
        dw = torch.empty((e, *dw0.shape[1:]), dtype=torch.int32, device=dev)
        ds = torch.empty((e, *ds0.shape[1:]), dtype=torch.int32, device=dev)
        del dw0, ds0
        for e0 in range(0, e, chunk):
            dw[e0:e0 + chunk], ds[e0:e0 + chunk] = _tile_experts(down[0][e0:e0 + chunk], down[1][e0:e0 + chunk])
        ua = torch.stack([gate[2], up[2]], dim=1).to(torch.float32).mul(16384.0).contiguous().to(dev)
        da = down[2].to(torch.float32).reshape(-1, 1).mul(16384.0).contiguous().to(dev)
        return cls(uw, us, ua, dw, ds, da, width, dims, limit)

    def run(self, x: torch.Tensor, picks: torch.Tensor, act: torch.Tensor, y: torch.Tensor) -> None:
        """x [R, D] bf16, picks [R, slots] -> act [R * slots, NI] (SwiGLU), y [R * slots, D]; a pair.s bits are its own."""

        from tensorfold.cuda.kernels import qmm_volta

        rows, slots = picks.shape
        pairs = rows * slots
        experts = self.count
        top = min(pairs, experts) + pairs // EX_ROWS
        sc = _expert_scratch(pairs, x.device)
        _ext().plan_experts(picks.contiguous(), experts, sc["members"], sc["items"], sc["counts"])
        x16, rs = qmm_volta.prep(x if x.stride(1) == 1 else x.contiguous())
        cnt = _counters(x.device)
        sk = 4 if self.dims // 64 >= 32 else 1
        part = sc["part"] if sk > 1 else sc["none"]
        need = sk * pairs * 2 * self.width
        if sk > 1 and part.numel() < need:
            if torch.cuda.is_current_stream_capturing():
                _RETIRED.append(part)
            part = sc["part"] = torch.empty((need,), dtype=torch.float32, device=x.device)
        _ext().experts884(2, x16, rs, self.up, self.up_s, self.up_a, sc["members"], sc["items"], sc["counts"], slots,
                          act, part[:need] if sk > 1 else part, cnt, self.width, sk, top, self.limit)
        a16, ars = qmm_volta.prep(act[:pairs])
        kg = -(-self.width // 64)
        sk = 4 if kg >= 32 and kg % 4 == 0 else 1
        _ext().experts884(0 if y.dtype == torch.float32 else 3, a16, ars, self.down, self.down_s, self.down_a,
                          sc["members"], sc["items"], sc["counts"], 0, y, sc["none"], cnt, self.dims, sk, top, 0.0)


_EXPERT_SCRATCH: dict = {}
# Scratch a larger capture replaced: a graph captured with it still writes there, so it is never freed.
_RETIRED: list = []


def _expert_scratch(pairs: int, device) -> dict:
    """Plan buffers for up to ``pairs`` pairs, grown by powers of two (CUDA-graph safe once grown)."""

    key = (str(device), torch.cuda.current_stream(device).cuda_stream)
    sc = _EXPERT_SCRATCH.get(key)
    if sc is None or sc["pairs"] < pairs:
        if sc is not None and torch.cuda.is_current_stream_capturing():
            _RETIRED.append(sc)
        cap = 1 << max(8, (pairs - 1).bit_length())
        sc = _EXPERT_SCRATCH[key] = {
            "pairs": cap, "members": torch.zeros(cap, dtype=torch.int32, device=device),
            "items": torch.zeros((cap + cap // EX_ROWS + 1) * 3, dtype=torch.int32, device=device),
            "counts": torch.zeros(2, dtype=torch.int32, device=device),
            "part": torch.empty(0, dtype=torch.float32, device=device),
            "none": torch.empty(0, dtype=torch.float32, device=device)}
    return sc
