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
    return load(name="tensorfold_qmmf_volta_v2", sources=[str(here / "qmmf_volta.cu")], need=VOLTA,
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
    """int32 (N, K/64 * wpg) -> [N/32][K/64][32][wpg], N padded to 32 with zeros."""

    n, kw = words.shape
    kg = kw // wpg
    pad = -n % 32
    if pad:
        words = torch.cat([words, words.new_zeros((pad, kw))])
    return words.view((n + pad) // 32, 32, kg, wpg).permute(0, 2, 1, 3).contiguous()


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
        if k % 64:
            raise ValueError(f"NVFP4 weight [{n}, {k}]: K must be a multiple of 64")
        words = _tile(weight.contiguous().view(torch.int32), 8)
        scales = _tile(weight_scale.contiguous().view(torch.uint8).view(torch.int32), 1).view(words.shape[:3])
        return cls(FP4, words, scales.contiguous(), _alpha(float(global_scale) * 16384.0, n, weight.device), n, k)

    @classmethod
    def from_fp8(cls, weight: torch.Tensor, scale: float) -> "VoltaLinear":
        """``weight`` e4m3 [N, K] with one tensor scale."""

        n, k = weight.shape
        if k % 64:
            raise ValueError(f"FP8 weight [{n}, {k}]: K must be a multiple of 64")
        words = _tile(weight.contiguous().view(torch.uint8).view(torch.int32), 16)
        return cls(FP8, words, None, _alpha(float(scale) * 256.0, n, weight.device), n, k)

    @classmethod
    def from_bf16(cls, weight: torch.Tensor, chunk: int = 8192) -> "VoltaLinear":
        """A 16-bit [N, K] weight as fp16, each column scaled by a power of two into [2^14, 2^15), a chunk at a time."""

        n, k = weight.shape
        if k % 64:
            raise ValueError(f"16-bit weight [{n}, {k}]: K must be a multiple of 64")
        n32, kg = -(-n // 32), k // 64
        words = torch.empty((n32, kg, 32, 32), dtype=torch.int32, device=weight.device)
        exps = torch.empty(n, dtype=torch.float32, device=weight.device)
        for a in range(0, n, chunk):                            # chunk is a multiple of 32
            part = weight[a:a + chunk].float()
            amax = part.abs().amax(1)
            e = torch.where(amax > 0, torch.frexp(amax).exponent.float() - 15.0, torch.zeros_like(amax))
            w16 = torch.ldexp(part, -e[:, None]).to(torch.float16)
            del part
            words[a // 32:a // 32 + -(-w16.shape[0] // 32)] = _tile(w16.view(torch.int32), 32)
            exps[a:a + chunk] = torch.ldexp(torch.ones_like(e), e)
            del w16
        return cls(F16, words, None, _alpha(exps, n, weight.device), n, k)

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
        kg, wpg = self.k // 64, WORDS[self.fmt]
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

        if not 0 <= g0 < g1 <= self.k // 64:
            raise ValueError(f"{self.layout}: input groups [{g0}, {g1}) outside [0, {self.k // 64})")
        return VoltaLinear(self.fmt, self.words[:, g0:g1].contiguous(),
                           self.scales[:, g0:g1].contiguous() if self.fmt == FP4 else None, self.alpha.clone(),
                           self.n, 64 * (g1 - g0))

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
