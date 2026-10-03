"""NVFP4 checkpoints in their own math: rows quantized under the static input scales, FP4 x FP4 and FP8 x FP8."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import torch

A4, A8 = 0, 1
SWIGLU_FP32 = True          # the fused gate|up epilogue's SwiGLU in fp32 (False: through bf16, as the unfused path)
WS = 10                     # prompt tiles past WS run the warp-specialized GEMM on rows quantized into its tiles
TB = 128                    # its tile rows


def available() -> bool:
    """Whether this GPU has the block-scaled FP4 mma (compute capability 12.x)."""

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def bulk_tile(tile: int, capability: tuple[int, int]) -> bool:
    """Whether prompt ``tile`` runs the bulk-copy GEMM (past ``WS``, from sm_90); else gemm_ck's tile, same bits."""

    return tile > WS and tuple(int(v) for v in capability) >= (9, 0)


@lru_cache(maxsize=1)
def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability())


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda import precision
    from tensorfold.cuda.build import FP8, load

    here = Path(__file__).parent
    return load(name="tensorfold_nvfp4_ck_v6",
                sources=[str(here / "checkpoint.cpp"), str(here / "act.cu"), str(here / "lane4.cu"),
                         str(here / "gemm_ck.cu"), str(here / "gemm_ws.cu")],
                need=FP8, arch_specific=precision.own_math(_capability())["nvfp4"],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3"], verbose=False)


class Rows4(NamedTuple):
    """Rows in NVFP4: e2m1 codes [M, K/2] (input 2j in byte j's low nibble), e4m3 scales [K/64, mpad, 4]."""

    codes: torch.Tensor
    scales: torch.Tensor


def _rows(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or (x.shape[0] > 1 and x.stride(0) % 8) or x.data_ptr() % 16:
        x = x.to(torch.bfloat16).contiguous()
    return x


def _inv(act: float) -> float:
    return float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32))


def quant4(x: torch.Tensor, act: float, tb: int = 0) -> Rows4:
    """bf16 rows -> NVFP4 under input scale ``act``: per-16 e4m3 scales, e2m1 codes to nearest even (``tb``: tiled)."""

    x = _rows(x)
    m, k = x.shape
    mpad = -(-m // (tb or 64)) * (tb or 64)
    codes = torch.empty((mpad if tb else m, k // 2), dtype=torch.uint8, device=x.device)
    shape = (mpad // tb, k // 64, tb, 4) if tb else (k // 64, mpad, 4)
    scales = (torch.empty if tb else torch.zeros)(shape, dtype=torch.uint8, device=x.device)
    _ext().quant4(x, _inv(act), codes, scales, tb)
    return Rows4(codes, scales)


def quant8(x: torch.Tensor, act: float, tb: int = 0) -> torch.Tensor:
    """bf16 rows -> e4m3(x / act), saturating, in the FP8 weights' fragment order (``tb``: the GEMM's tiled rows)."""

    x = _rows(x)
    m, k = x.shape
    out = torch.empty((-(-m // tb) * tb if tb else m, k), dtype=torch.uint8, device=x.device)
    _ext().quant8(x, _inv(act), out, tb)
    return out


def pack4(weight: torch.Tensor, npad: int) -> torch.Tensor:
    """NVFP4 checkpoint bytes [N, K/2] -> the FP4 lane matmul's words [npad/64, K/64, 8, 32, 2] (int32)."""

    n, k = weight.shape[0], weight.shape[1] * 2
    words = torch.empty((npad // 64, k // 64, 8, 32, 2), dtype=torch.int32, device=weight.device)
    _ext().pack4(weight.contiguous().view(torch.uint8), words)
    return words


def alpha(act: float, scale: float) -> float:
    """The output factor: the input scale times the weight's (fp32, as the checkpoint's runtimes compute it)."""

    return float(torch.tensor(act, dtype=torch.float32) * torch.tensor(scale, dtype=torch.float32))


def _rowsq(mode: int, x: torch.Tensor, act: float, tb: int = 0):
    """Rows quantized for ``mode`` under the static input scale ``act`` (``tb``: the prompt GEMM's tiles)."""

    return quant4(x, act, tb) if mode == A4 else (quant8(x, act, tb), None)


def _out(m: int, lin, out: torch.Tensor | None, f32: bool) -> torch.Tensor:
    want = torch.float32 if f32 else torch.bfloat16
    return out if out is not None and out.is_contiguous() and out.dtype == want else \
        torch.empty((m, lin.n), dtype=want, device=lin.words.device if hasattr(lin, "words") else lin.w8.device)


FILL = 2                    # 128-wide lane blocks while at least SMs / FILL of them fill the GPU, else 64-wide
WIDE = 96                   # SMs from which wide lane blocks pay: a 48-SM GB10 streams faster on 64-wide ones


def lane_tile(m: int, n: int, sk: int, sms: int) -> int:
    """The lane matmul's block (BM * 1000 + BN past 32 rows) for m rows, n columns, sk K slices on sms SMs."""

    if m <= 32:
        return 16 if m <= 16 else 32                    # 16 or 32 rows by 64 columns; blocks never change bits
    rows = 64 if m <= 64 else 128
    if sms < WIDE:
        return 64                                       # 64 x 64 blocks side by side
    wide = -(-m // rows) * -(-n // 128) * sk * FILL >= sms
    return (64128 if wide else 64) if rows == 64 else (128128 if wide else 128064)


@lru_cache(maxsize=8)
def _sms(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def sm_count() -> int:
    """SMs on the current GPU (for block shapes only: never a bit of output)."""

    return _sms(torch.cuda.current_device())


def _lane(mode: int, rows, lin, y: torch.Tensor, f32: bool) -> None:
    from tensorfold.cuda.kernels import qmm

    m = y.shape[0]
    sk = qmm.split_k(lin.n, lin.k)
    part = torch.empty((sk, m, lin.n), dtype=torch.float32, device=y.device) if sk > 8 else None
    w, ws = (lin.words, lin.bs) if mode == A4 else (lin.w8, None)
    _ext().lane(mode, rows[0], rows[1], w, ws, alpha(lin.act, lin.scale), y, part, lin.n, lin.k, sk, lin.npad,
                lane_tile(m, lin.n, sk, sm_count()), f32)


def _tb(tile: int) -> int:
    """The quantizer's tile rows for a prompt ``tile``: the bulk-copy GEMM's, or 0 for row-major rows."""

    return TB if bulk_tile(tile, _capability()) else 0


def _gemm(mode: int, rows, lin, y: torch.Tensor, f32: bool, tile: int = 0) -> None:
    w, ws = (lin.words, lin.bs) if mode == A4 else (lin.w8, None)
    run = _ext().gemm_ws if bulk_tile(tile, _capability()) else _ext().gemm
    run(mode, rows[0], rows[1], w, ws, alpha(lin.act, lin.scale), y, lin.n, lin.k, lin.npad,
        tile % WS if tile > WS else tile, f32)


def matmul(mode: int, x: torch.Tensor, lin, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
    """bf16 rows (M, K) @ an NVFP4 (``A4``) or FP8 (``A8``) linear in checkpoint mode -> (M, n); K slices by shape."""

    y = _out(x.shape[0], lin, out, f32)
    _lane(mode, _rowsq(mode, x, lin.act), lin, y, f32)
    if out is not None and y is not out:
        out.copy_(y)
    return y


def prompt(mode: int, x: torch.Tensor, lin, out: torch.Tensor | None = None, f32: bool = False,
           tile: int = 0) -> torch.Tensor:
    """Prompt rows (M, K) bf16 @ an NVFP4 or FP8 linear in checkpoint math: one K chain a row (chunk-invariant bits)."""

    y = _out(x.shape[0], lin, out, f32)
    _gemm(mode, _rowsq(mode, x, lin.act, _tb(tile)), lin, y, f32, tile)
    if out is not None and y is not out:
        out.copy_(y)
    return y


def matmul_group(x: torch.Tensor, lins: list, prompt_rows: bool = False, outs: list[torch.Tensor] | None = None,
                 tile: int = 0) -> list[torch.Tensor] | None:
    """Projections of one input under one input scale: rows quantized once, each its own bits; None if they differ."""

    if len(lins) < 2 or any(getattr(lin, "act", None) is None for lin in lins):
        return None
    modes = {A4 if hasattr(lin, "words") else A8 for lin in lins}
    if len(modes) != 1 or len({float(lin.act) for lin in lins}) != 1:
        return None
    mode = modes.pop()
    rows = _rowsq(mode, x, lins[0].act, _tb(tile) if prompt_rows else 0)
    outs = [_out(x.shape[0], lin, None if outs is None else outs[j], False) for j, lin in enumerate(lins)]
    for lin, y in zip(lins, outs):
        _gemm(mode, rows, lin, y, False, tile) if prompt_rows else _lane(mode, rows, lin, y, False)
    return outs


def mlp_prompt(x: torch.Tensor, gate, up, down, out: torch.Tensor | None = None) -> torch.Tensor | None:
    """Prompt rows through gate|up -> SiLU(gate) * up -> down, the SwiGLU rows leaving as down's NVFP4 rows; or None."""

    if not all(hasattr(lin, "words") and getattr(lin, "act", None) is not None for lin in (gate, up, down)):
        return None
    if float(gate.act) != float(up.act) or gate.n != up.n or gate.npad != gate.n or down.k != gate.n:
        return None
    if gate.k % 128 or down.k % 128:
        return None
    rows = quant4(x, gate.act)
    m, mpad = x.shape[0], rows.scales.shape[1]
    codes = torch.empty((m, down.k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((down.k // 64, mpad, 4), dtype=torch.uint8, device=x.device)
    _ext().gemm_gu_ck(rows.codes, rows.scales, gate.words, gate.bs, up.words, up.bs, alpha(gate.act, gate.scale),
                      alpha(up.act, up.scale), gate.npad, gate.k, _inv(down.act), codes, scales, SWIGLU_FP32)
    y = _out(m, down, out, False)
    _gemm(A4, Rows4(codes, scales), down, y, False, 0)
    return y
