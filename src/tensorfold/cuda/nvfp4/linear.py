"""NVFP4 and FP8 linears on CUDA: exact W4A16 / W8A16 lane matmuls for decode, the FP8 prefill GEMM for prompts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

FP4, FP8, MXFP8, FP8G = 0, 1, 2, 3


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_nvfp4_v3", sources=[str(here / "qmmf.cpp"), str(here / "qmmf.cu"),
                                                      str(here / "experts.cu")],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3"], verbose=False)


def fragment_index(k: int, npad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(k, n) of every byte of ``qmm_prefill8w``'s weight order [npad/64][K/64][8][32][2][8], as index tensors."""

    lane, byte = torch.arange(32, device=device), torch.arange(8, device=device)
    p = 16 * (byte // 4)[None, :] + 4 * (lane % 4)[:, None] + (byte % 4)[None, :]           # position in a k32 step
    kin = 16 * (p // 16) + 2 * ((p % 16) // 4) + (p % 2) + 8 * ((p % 4) // 2)             # ``quantize_rows``' order
    kg = k // 64
    kk = (torch.arange(kg, device=device).view(1, kg, 1, 1, 1, 1) * 64
          + torch.arange(2, device=device).view(1, 1, 1, 1, 2, 1) * 32 + kin.view(1, 1, 1, 32, 1, 8))
    nn = (torch.arange(npad // 64, device=device).view(-1, 1, 1, 1, 1, 1) * 64
          + torch.arange(8, device=device).view(1, 1, 8, 1, 1, 1) * 8 + (lane // 4).view(1, 1, 1, 32, 1, 1))
    shape = (npad // 64, kg, 8, 32, 2, 8)
    return kk.expand(shape), nn.expand(shape)


def _fragment_order(codes: torch.Tensor, npad: int) -> torch.Tensor:
    """e4m3 bytes [N, K] -> ``qmm_prefill8w``'s order [npad/64][K/64][8][32][2][8] (fragment lanes, k32 halves)."""

    n, k = codes.shape
    padded = torch.zeros((npad, k), dtype=torch.uint8, device=codes.device)
    padded[:n] = codes
    kk, nn = fragment_index(k, npad, codes.device)
    return padded.t()[kk, nn].contiguous().view(-1)


_ONES: dict[str, torch.Tensor] = {}


def _ones(kg: int, npad: int, device) -> torch.Tensor:
    """Unit bf16 scales [kg, npad] for weights with one tensor scale: views of one buffer per device."""

    have = _ONES.get(str(device))
    if have is None or have.numel() < kg * npad:
        have = _ONES[str(device)] = torch.ones((kg * npad,), dtype=torch.bfloat16, device=device)
    return have[:kg * npad].view(kg, npad)


def _round_up_bf16(v: torch.Tensor) -> torch.Tensor:
    """Positive fp32 values rounded up to bf16 (a scale that never pushes a code past e4m3's 448)."""

    b = v.to(torch.bfloat16)
    low = b.float() < v
    return torch.where(low, (b.view(torch.int16) + 1).view(torch.bfloat16), b)


class Staging:
    """The e4m3 bytes and scales one NVFP4 projection is staged into for the prompt GEMM, grown to the largest."""

    def __init__(self) -> None:
        self.w8: torch.Tensor | None = None
        self.s8: torch.Tensor | None = None

    def take(self, npad: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.w8 is None or self.w8.numel() < npad * k:
            self.w8 = torch.empty((npad * k,), dtype=torch.uint8, device=device)
        if self.s8 is None or self.s8.numel() < k // 64 * npad:
            self.s8 = torch.empty((k // 64 * npad,), dtype=torch.bfloat16, device=device)
        return self.w8[:npad * k], self.s8[:k // 64 * npad].view(k // 64, npad)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w8, self.s8) if t is not None)


@dataclass
class Fp4Linear:
    """An NVFP4 projection: e2m1 codes tiled for the lane matmul, e4m3 scales per 16 inputs, one fp32 global scale."""

    words: torch.Tensor           # int32 [npad/64, K/64, 8, 32, 2] (``qmm.pack``'s tiling of the codes)
    bs: torch.Tensor              # uint8 [npad/64, K/64, 64, 4]: a 64-column tile's scales together, like its words
    scale: float
    n: int
    k: int
    layout: str = "nvfp4"
    staging: Staging | None = None   # shared by the model's NVFP4 projections (prompts)

    @property
    def npad(self) -> int:
        return int(self.words.shape[0]) * 64

    @classmethod
    def from_checkpoint(cls, weight: torch.Tensor, weight_scale: torch.Tensor, global_scale: float) -> "Fp4Linear":
        """``weight`` uint8 [N, K/2] (low nibble first), ``weight_scale`` e4m3 bytes [N, K/16], the product's global."""

        from tensorfold.cuda.kernels import qmm

        n, k = weight.shape[0], weight.shape[1] * 2
        if k % 64:
            raise ValueError(f"NVFP4 weight [{n}, {k}]: K must be a multiple of 64")
        words = weight.contiguous().view(torch.int32)
        dummy = torch.zeros((n, k // 64), dtype=torch.bfloat16, device=weight.device)
        packed = qmm.pack(words, dummy, dummy, 64).weight
        npad = packed.shape[0] * 64
        bs = torch.zeros((npad, k // 16), dtype=torch.uint8, device=weight.device)
        bs[:n] = weight_scale.contiguous().view(torch.uint8)
        bs = bs.view(npad // 64, 64, k // 64, 4).permute(0, 2, 1, 3).contiguous()
        return cls(packed, bs, float(global_scale), n, k)

    def nbytes(self) -> int:
        return self.words.numel() * 4 + self.bs.numel()

    def tiles(self, t0: int, t1: int) -> "Fp4Linear":
        """Outputs [64 t0, 64 t1) as views, no copy (decode only: the prompt GEMM wants 128-column multiples)."""

        return Fp4Linear(self.words[t0:t1], self.bs[t0:t1], self.scale, min(self.n, 64 * t1) - 64 * t0, self.k)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return _matmul(FP4, self.words, self.bs, self.scale, self.n, self.k, self.npad, x, out)

    def prefill(self, xq) -> torch.Tensor:
        """Prompt rows (FP8 from the glue) times this weight staged as e4m3 per (64 inputs, column), once a call."""

        from tensorfold.cuda.kernels import qmm

        if self.staging is None:
            self.staging = Staging()
        w8, s8 = self.staging.take(self.npad, self.k, self.words.device)
        _ext().stage_fp4(self.words, self.bs, self.scale, w8, s8)
        out = torch.empty((xq[0].shape[0], self.npad), dtype=torch.bfloat16, device=xq[0].device)
        qmm._ext().qmm_prefill8w(xq[0], xq[2], w8, s8, out, self.npad, 64, False, 0, False)
        return out if self.npad == self.n else out[:, :self.n].contiguous()


@dataclass
class Fp8Linear:
    """An FP8 projection: e4m3 bytes in the FP8 GEMM's fragment order (which the W8A16 decode reads too), one scale."""

    w8: torch.Tensor              # uint8, [npad/64][K/64][8][32][2][8]
    scale: float
    n: int
    k: int
    npad: int
    layout: str = "fp8"
    groups: torch.Tensor | None = None   # bf16 [K/64, npad] of a copy made from bf16 (prompts only); else unit

    @classmethod
    def from_checkpoint(cls, weight: torch.Tensor, scale: float) -> "Fp8Linear":
        """``weight`` e4m3 [N, K] with one fp32 scale (ModelOpt's per-tensor FP8)."""

        n, k = weight.shape
        if k % 64:
            raise ValueError(f"FP8 weight [{n}, {k}]: K must be a multiple of 64")
        npad = -(-n // 128) * 128
        return cls(_fragment_order(weight.contiguous().view(torch.uint8), npad), float(scale), n, k, npad)

    @classmethod
    def from_bf16(cls, weight: torch.Tensor) -> "Fp8Linear":
        """A bf16 [N, K] weight as e4m3 with a bf16 scale per (64 inputs, output), rounded up: a prompt GEMM copy."""

        n, k = weight.shape
        if k % 64:
            raise ValueError(f"bf16 weight [{n}, {k}]: K must be a multiple of 64")
        npad = -(-n // 128) * 128
        w = weight.float().view(n, k // 64, 64)
        top = w.abs().amax(-1)
        s = _round_up_bf16(torch.where(top > 0, top / 448.0, torch.ones_like(top)))
        codes = (w / s.float()[..., None]).view(n, k).to(torch.float8_e4m3fn).view(torch.uint8)
        groups = torch.ones((k // 64, npad), dtype=torch.bfloat16, device=weight.device)
        groups[:, :n] = s.t()
        return cls(_fragment_order(codes, npad), 1.0, n, k, npad, groups=groups)

    def nbytes(self) -> int:
        return self.w8.numel() + (self.groups.numel() * 2 if self.groups is not None else 0)

    def tiles(self, t0: int, t1: int) -> "Fp8Linear":
        """Outputs [64 t0, 64 t1) as views, no copy (decode only)."""

        per = 64 * self.k
        return Fp8Linear(self.w8[t0 * per:t1 * per], self.scale, min(self.n, 64 * t1) - 64 * t0, self.k,
                         64 * (t1 - t0))

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return _matmul(FP8, self.w8, None, self.scale, self.n, self.k, self.npad, x, out)

    def prefill(self, xq: tuple) -> torch.Tensor:
        """Prompt rows (FP8 from the glue) times the stored bytes; the tensor scale joins each row's scale."""

        from tensorfold.cuda.kernels import qmm

        groups = self.groups if self.groups is not None else _ones(self.k // 64, self.npad, self.w8.device)
        out = torch.empty((xq[0].shape[0], self.npad), dtype=torch.bfloat16, device=xq[0].device)
        qmm._ext().qmm_prefill8w(xq[0], xq[2] * self.scale, self.w8, groups, out, self.npad, 64, False, 0, False)
        return out if self.npad == self.n else out[:, :self.n].contiguous()


@dataclass
class Mx8Linear:
    """An MXFP8 projection: e4m3 bytes in the FP8 GEMM's fragment order, an e8m0 scale per 32 inputs."""

    w8: torch.Tensor              # uint8, [npad/64][K/64][8][32][2][8]
    bs: torch.Tensor              # uint8 [npad/64, K/64, 64, 2]: e8m0 exponents, a tile's together
    n: int
    k: int
    npad: int
    layout: str = "mxfp8"
    kernel: str = "mx8"
    groups: torch.Tensor | None = None   # bf16 [K/32, npad], the scales as powers of two (prompts, made on first use)

    @classmethod
    def from_checkpoint(cls, weight: torch.Tensor, scale: torch.Tensor) -> "Mx8Linear":
        """``weight`` e4m3 [N, K] and ``scale`` e8m0 bytes [N, K/32] (compressed-tensors / ModelOpt MXFP8)."""

        n, k = weight.shape
        if k % 64:
            raise ValueError(f"MXFP8 weight [{n}, {k}]: K must be a multiple of 64")
        npad = -(-n // 128) * 128
        bs = torch.full((npad, k // 32), 127, dtype=torch.uint8, device=weight.device)
        bs[:n] = scale.contiguous().view(torch.uint8)
        bs = bs.view(npad // 64, 64, k // 64, 2).permute(0, 2, 1, 3).contiguous()
        return cls(_fragment_order(weight.contiguous().view(torch.uint8), npad), bs, n, k, npad)

    @classmethod
    def stack(cls, parts: list["Mx8Linear"]) -> "Mx8Linear":
        """Projections of one input as one: outputs in order (each part's padding dropped, the stack's own added)."""

        dense = [(p.w8_rows(), p.scale_rows()) for p in parts]
        return cls.from_checkpoint(torch.cat([w for w, _ in dense]), torch.cat([s for _, s in dense]))

    def w8_rows(self) -> torch.Tensor:
        """The stored e4m3 bytes back to [n, K] (the fragment order undone)."""

        kk, nn = fragment_index(self.k, self.npad, self.w8.device)
        rows = torch.empty((self.npad, self.k), dtype=torch.uint8, device=self.w8.device)
        rows.t()[kk, nn] = self.w8.view(kk.shape)
        return rows[:self.n].view(torch.float8_e4m3fn)

    def scale_rows(self) -> torch.Tensor:
        return self.bs.permute(0, 2, 1, 3).reshape(self.npad, self.k // 32)[:self.n]

    def nbytes(self) -> int:
        return self.w8.numel() + self.bs.numel() + (self.groups.numel() * 2 if self.groups is not None else 0)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return _matmul(MXFP8, self.w8, self.bs, 1.0, self.n, self.k, self.npad, x, out)

    def prefill(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """bf16 prompt rows through FP8 rows and the stored bytes, each 32 inputs' power of two as a bf16 scale."""

        from tensorfold.cuda.kernels import qmm

        if self.groups is None:
            e = self.bs.permute(1, 3, 0, 2).reshape(self.k // 32, self.npad).to(torch.int32)
            self.groups = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127).to(torch.bfloat16)
        xq = qmm.quantize_rows(x if x.stride(-1) == 1 else x.contiguous())
        y = torch.empty((x.shape[0], self.npad), dtype=torch.bfloat16, device=x.device)
        qmm._ext().qmm_prefill8w(xq[0], xq[2], self.w8, self.groups, y, self.npad, 32, False, 0, True)
        y = y if self.npad == self.n else y[:, :self.n]
        if out is None:
            return y.contiguous()
        out.copy_(y)
        return out


@dataclass
class Fp8BlockLinear:
    """A block-scaled FP8 projection (DeepSeek / ModelOpt ``FP8_PB_WO``: e4m3 bytes, an fp32 ``weight_scale_inv`` per
    128x128 block): the e4m3 bytes in the FP8 GEMM's fragment order and the block scales expanded to one fp32 per
    (64 inputs, column), which the lane matmul applies after each 64-input stage (mode FP8G) -- the stored weight
    exactly. Prompts run the FP8 prompt GEMM with those scales as bf16 group scales (rounded to nearest)."""

    w8: torch.Tensor              # uint8, [npad/64][K/64][8][32][2][8]
    bs: torch.Tensor              # uint8 view of fp32 [npad/64, K/64, 64]: a tile's column scales together
    n: int
    k: int
    npad: int
    layout: str = "fp8block"
    groups: torch.Tensor | None = None   # bf16 [K/64, npad] (prompts, made on first use)

    @staticmethod
    def column_scales(scale_inv: torch.Tensor, n: int, k: int, block=(128, 128)) -> torch.Tensor:
        """fp32 [n, K/64]: each (row, 64-input group)'s block scale."""

        bn, bk = block
        if bk % 64:
            raise ValueError(f"FP8 block {block}: the input block must be a multiple of 64")
        s = scale_inv.float()
        if s.shape != (-(-n // bn), k // bk):
            raise ValueError(f"weight_scale_inv {tuple(s.shape)} does not tile [{n}, {k}] in {block} blocks")
        return s.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk // 64, dim=1).contiguous()

    @classmethod
    def from_rows(cls, weight: torch.Tensor, cols: torch.Tensor) -> "Fp8BlockLinear":
        """``weight`` e4m3 [N, K] and its fp32 scales per (row, 64 inputs) [N, K/64]."""

        n, k = weight.shape
        if k % 64:
            raise ValueError(f"FP8 weight [{n}, {k}]: K must be a multiple of 64")
        npad = -(-n // 128) * 128
        full = torch.ones((npad, k // 64), dtype=torch.float32, device=weight.device)
        full[:n] = cols
        bs = full.view(npad // 64, 64, k // 64).permute(0, 2, 1).contiguous().view(torch.uint8)
        return cls(_fragment_order(weight.contiguous().view(torch.uint8), npad), bs, n, k, npad)

    @classmethod
    def from_checkpoint(cls, weight: torch.Tensor, scale_inv: torch.Tensor, block=(128, 128)) -> "Fp8BlockLinear":
        n, k = weight.shape
        return cls.from_rows(weight, cls.column_scales(scale_inv, n, k, block))

    def scale_rows(self) -> torch.Tensor:
        """fp32 [n, K/64] back from the tiled scales."""

        return self.bs.view(torch.float32).view(self.npad // 64, self.k // 64, 64).permute(0, 2, 1).reshape(
            self.npad, self.k // 64)[:self.n]

    def nbytes(self) -> int:
        return self.w8.numel() + self.bs.numel() + (self.groups.numel() * 2 if self.groups is not None else 0)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return _matmul(FP8G, self.w8, self.bs, 1.0, self.n, self.k, self.npad, x, out)

    def prefill(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """bf16 prompt rows through FP8 rows and the stored bytes, each (64 inputs, column)'s scale as bf16."""

        from tensorfold.cuda.kernels import qmm

        if self.groups is None:
            self.groups = self.bs.view(torch.float32).view(self.npad // 64, self.k // 64, 64).permute(1, 0, 2).reshape(
                self.k // 64, self.npad).to(torch.bfloat16).contiguous()
        xq = qmm.quantize_rows(x if x.stride(-1) == 1 else x.contiguous())
        y = torch.empty((x.shape[0], self.npad), dtype=torch.bfloat16, device=x.device)
        qmm._ext().qmm_prefill8w(xq[0], xq[2], self.w8, self.groups, y, self.npad, 64, False, 0, False)
        y = y if self.npad == self.n else y[:, :self.n]
        if out is None:
            return y.contiguous()
        out.copy_(y)
        return out


class Concat:
    """Linears of one input whose storage differs (block FP8 beside bf16), outputs side by side: each part runs on its
    own kernel into its columns of ``out`` (the face ``forward._mm`` calls for a stack it cannot join)."""

    def __init__(self, parts: list) -> None:
        self.parts = parts
        self.n = sum(p.n for p in parts)
        self.k = parts[0].k

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self.parts)

    def _run(self, x: torch.Tensor, out: torch.Tensor | None, prefill: bool) -> torch.Tensor:
        from tensorfold.families.qwen4_exp.cuda import bf16 as b16

        y = out if out is not None else torch.empty((x.shape[0], self.n), dtype=torch.bfloat16, device=x.device)
        c = 0
        for p in self.parts:
            if getattr(p, "kernel", "") == "b16":
                r = b16.matmul(x, p)
            else:
                r = p.prefill(x) if prefill else p(x)
            y[:, c:c + p.n].copy_(r)
            c += p.n
        return y

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self._run(x, out, False)

    def prefill(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self._run(x, out, True)


def _matmul(mode: int, w: torch.Tensor, bs: torch.Tensor | None, scale: float, n: int, k: int, npad: int,
            x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """x (M, K) bf16 -> (M, n) bf16; K slices from the shape alone, so a row's bits never depend on M."""

    from tensorfold.cuda.kernels import qmm

    if x.dtype != torch.bfloat16 or x.stride(-1) != 1:
        x = x.to(torch.bfloat16).contiguous()
    m = x.shape[0]
    y = out if out is not None and out.is_contiguous() and out.dtype == torch.bfloat16 else \
        torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    sk = qmm.split_k(n, k)
    part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device) if sk > 8 else None
    _ext().qmmf(x, w, bs, scale, y, part, mode, n, sk, npad, qmm.bucket(m), False)
    if out is not None and y is not out:
        out.copy_(y)
    return y
