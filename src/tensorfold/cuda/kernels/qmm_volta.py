"""Volta (sm_70) matmuls for 4-bit affine weights: fp16 tensor cores with fp32 sums for decode rows, a dense fp16 copy
through cuBLASLt for prompt chunks (``qmm_volta.cu``). Rows never depend on the row count on either path."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.build import VOLTA, load, volta

# Decode rows go to the kernel in fragment order from this many rows: below it, row-major rows share cache lines
# across k-steps (the head at one row: 864 against 1,053 us on a V100). Wide inputs (the MLP down projection, split
# 16 ways over K) gain from the fragment order at any row count.
FRAG_MIN = 7
FRAG_ALWAYS_FROM_K = 16384


@lru_cache(maxsize=1)
def _ext():
    here = Path(__file__).parent
    return load(name="tensorfold_qmm_volta_v1", sources=[str(here / "qmm_volta.cu")], need=VOLTA,
                extra_cuda_cflags=["-O3"], extra_ldflags=_cublaslt_flags(), verbose=False)


def _cublaslt_flags() -> list[str]:
    """Link the libcublasLt torch itself loaded: NVIDIA's pip wheel when there is one, else the toolkit's."""

    try:
        import nvidia.cublas
    except ImportError:
        return ["-lcublasLt"]
    lib = Path(next(iter(nvidia.cublas.__path__))) / "lib"
    return [f"-L{lib}", "-l:libcublasLt.so.12", f"-Wl,-rpath,{lib}"]


def active() -> bool:
    """Whether this process's GPU is a Volta: 4-bit weights then take the ``Tiled`` layout and these kernels."""

    return volta()


def prep(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, K) bf16 -> (fp16 rows scaled by a power of two, the row scales), reusable across projections of ``x``."""

    if x.stride(1) != 1:
        x = x.contiguous()
    x16, rs = _ext().prep(x)
    return x16, rs


def prep884(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``prep`` for the decode kernel: the same values in fragment order [M/8][K/8][8 rows][8 inputs] (a few narrow
    rows stay row-major, see ``FRAG_MIN``)."""

    if x.stride(1) != 1:
        x = x.contiguous()
    x16, rs = _ext().prep884(x, 1 if x.shape[1] >= FRAG_ALWAYS_FROM_K else FRAG_MIN)
    return x16, rs


def split_k884(n: int, k: int, target: int = 640) -> int:
    """K slices for an (n, k) weight: a function of the shape only, enough blocks to fill 80 SMs."""

    tiles, groups, sk = -(-n // 128), k // 64, 1
    while sk < 16 and tiles * sk < target and groups % (sk * 2) == 0 and groups // (sk * 2) >= 4:
        sk *= 2
    return sk


def _row_tiles(m: int) -> int:
    """8-row tiles a block: one for a few rows, up to four so a weight load feeds more tensor-core work."""

    return 1 if m <= 8 else 2 if m <= 16 else 4


def _decode(x16: torch.Tensor, rs: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
            n: int, k: int, *, tiled: bool, f32: bool) -> torch.Tensor:
    m = rs.shape[0]
    sk = split_k884(n, k)
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x16.device)
    part = torch.empty((sk, m, n), dtype=torch.float32, device=x16.device) if sk > 1 else None
    _ext().qmm884(x16, rs, weight, scales, biases, out, part, sk, _row_tiles(m), f32, n, tiled)
    return out


def lane_matmul(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                xs: torch.Tensor | None = None, *, f32: bool = False) -> torch.Tensor:
    """x (M, K) bf16 times the stored 4-bit ``weight`` (N, K/8) transposed -> (M, N) bf16 (fp32 with ``f32``).

    ``xs`` is accepted for signature parity with the Triton lane matmul and ignored: biases are folded into the
    fp16 weights, so no input group sums are needed."""

    del xs
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        raise ValueError("lane_matmul: x must be a 2-D bf16 tensor")
    k, n = x.shape[1], weight.shape[0]
    if weight.shape[1] * 8 != k or k % 64:
        raise ValueError(f"lane_matmul: weight {tuple(weight.shape)} does not match K={k}")
    if scales.dtype != torch.bfloat16 or biases.dtype != torch.bfloat16:
        raise ValueError("lane_matmul: bf16 scales and biases only")
    x16, rs = prep884(x)
    return _decode(x16, rs, weight.view(torch.int32), scales.contiguous(), biases.contiguous(), n, k, tiled=False,
                   f32=f32)


class Tiled:
    """A 4-bit weight repacked for the decode kernel: [N/32][K/64][32 columns][8 words], scales and biases alike."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor):
        words = weight.view(torch.int32)
        n, k8 = words.shape
        kg = k8 // 8
        pad = -n % 32
        if pad:
            words = torch.cat([words, words.new_zeros((pad, k8))])
            scales = torch.cat([scales, scales.new_zeros((pad, kg))])
            biases = torch.cat([biases, biases.new_zeros((pad, kg))])
        np_ = n + pad
        self.weight = words.view(np_ // 32, 32, kg, 8).permute(0, 2, 1, 3).contiguous()
        self.scales = scales.view(np_ // 32, 32, kg).permute(0, 2, 1).contiguous()
        self.biases = biases.view(np_ // 32, 32, kg).permute(0, 2, 1).contiguous()
        self.n, self.k = n, k8 * 8

    @classmethod
    def wrap(cls, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, n: int) -> Tiled:
        """Already-tiled tensors (a ``QLinear`` with layout "volta") as a ``Tiled``, no copy."""

        t = object.__new__(cls)
        t.weight, t.scales, t.biases, t.n, t.k = weight, scales, biases, n, weight.shape[1] * 64
        return t

    def untile(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        np_, kg = self.weight.shape[0] * 32, self.k // 64
        w = self.weight.permute(0, 2, 1, 3).reshape(np_, kg * 8)[:self.n].contiguous()
        s = self.scales.permute(0, 2, 1).reshape(np_, kg)[:self.n].contiguous()
        b = self.biases.permute(0, 2, 1).reshape(np_, kg)[:self.n].contiguous()
        return w, s, b


def tiled_matmul(x: torch.Tensor, t: Tiled, *, f32: bool = False, xs=None) -> torch.Tensor:
    """``lane_matmul`` on a ``Tiled`` weight: the same bits as the stored layout, coalesced reads. ``xs``: ``prep884(x)``
    shared by the projections of one input."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != t.k:
        raise ValueError(f"tiled_matmul: x must be (M, {t.k}) bf16")
    x16, rs = xs if isinstance(xs, tuple) else prep884(x)
    return _decode(x16, rs, t.weight, t.scales, t.biases, t.n, t.k, tiled=True, f32=f32)


def dequant(t: Tiled) -> torch.Tensor:
    """The weight as dense fp16 (N, K): the values the decode kernel expands, the prompt GEMM's operand."""

    return _ext().dequant(t.weight, t.scales, t.biases, t.n)


def prompt_matmul(x: torch.Tensor, t: Tiled, *, f32: bool = False, xs=None) -> torch.Tensor:
    """Prompt rows: the weight expanded once to dense fp16, one cuBLASLt GEMM with fp32 sums under an algorithm pinned
    per shape (so no row's bits depend on the row count), the rows' power-of-two scales applied last. Not the decode
    kernel's bits."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != t.k:
        raise ValueError(f"prompt_matmul: x must be (M, {t.k}) bf16")
    x16, rs = xs if isinstance(xs, tuple) else prep(x)
    y = _ext().gemm(x16, dequant(t))
    if y.shape[1] % 4:
        y.mul_(rs[:, None])
        return y if f32 else y.to(torch.bfloat16)
    return _ext().unscale(y, rs, f32)
