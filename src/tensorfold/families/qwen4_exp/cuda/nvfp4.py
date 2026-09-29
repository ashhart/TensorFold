"""NVFP4 (NVIDIA ModelOpt's FP4 format) as the Swift 1.5 Flash Next NVFP4 checkpoint stores them, and
the row-invariant kernels that read them: the definition the CUDA path is checked against.

NVFP4 quantizes a linear layer's weights (here: the 512 routed experts; everything else in the checkpoint is
BF16). With K inputs and N outputs a layer is stored as four tensors:

    weight          uint8  [N, K/2]     packed E2M1 nibbles: byte i holds q[2i] (low) and q[2i+1] (high)
    weight_scale    fp8e4m3 [N, K/16]   one scale per 16-value block
    weight_scale_2  fp32   []           a second, per-tensor scale
    input_scale     fp32   []           calibration-time activation scale (this engine serves BF16
                                        activations, so it is not read on the verify path)

The dequantized weight is

    W[n, k] = E2M1(nibble (n, k)) * (fp32(weight_scale[n, k // 16]) * weight_scale_2)

with E2M1 the 16 FP4 values {0, .5, 1, 1.5, 2, 3, 4, 6} and their negations (code = sign << 3 | magnitude),
the fp32 widening of the e4m3 scale exact and the fp32 product taking one rounding — ModelOpt's own
reference (``NVFP4QTensor.dequantize`` / ``fp4_dequantize``), not Marlin's kernel packing that multiplies
scales by ``2**7`` for a different on-device encoding. The quantization is experts-only: every other
linear (hyper-connections, DeltaNet, attention, the router, the shared expert, PLE, embeddings, lm_head,
the MTP head) is stored in BF16.

``FP4`` stores the two halves exactly: ``weight`` the E2M1 code values as bf16 bit patterns (uint16 —
an E2M1 value is a 3-bit bf16 pattern; the patterns are stored so the format's math never goes through
torch's fp8 casts, whose bf16 path yields zeros, and so the reference and the kernel widen the identical
grid), ``scale`` the per-row fp32 dequant weights. ``dequantize`` is the fp32 reference the kernels are
checked against.

The matmul (``matmul``) is the same contract as ``qmm.matmul``, with a 16-wide block and no bias (the
format is purely multiplicative, so no group-input sums travel with the activation): for each 16-input
block b of row m and output column n, with P the tensor-core dot of the block's bf16 inputs and the
block's stored code values,

    y[m, n] = sum over blocks in order, and within a K slice in block order, of  scale(b, n) * P[m, n, b]

K is split into slices fixed by the weight's shape (never the row count) and the slices are summed in slice
order by ``_reduce``. A row's bits depend only on its own input, its own K-slices and the block order:
drafted windows and serial decoding give a row the same bits
(``tests/cuda/test_flashnext_nvfp4.py`` checks it kernel-first, the way ``adding-a-cuda-family.md`` asks).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GS = 16                   # inputs per quantization block (NVFP4's block size)
BN = 64                   # output columns per stored tile

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)      # FP4 magnitudes by code & 7
BF16_BITS = (0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0)   # their bf16 patterns
E4M3_BF16_BITS = (0x0000, 0x3B00, 0x3B80, 0x3BC0, 0x3C00, 0x3C20, 0x3C40, 0x3C60)  # the fp8 subnormals m*2**-9 (m 0..7)
BF16_SCALE2 = 0x3F800000                                # 1.0 as an fp32 pattern: e4m3 -> bf16's exponent shift


def e2m1_table(device: str | torch.device = "cpu") -> torch.Tensor:
    """The 16 FP4 values by code: code = sign (bit 3) * 8 + magnitude (bits 0..2)."""

    mags = torch.tensor(_E2M1, dtype=torch.float32)
    return torch.cat([mags, -mags]).to(device)


@dataclass
class FP4:
    """A quantized matrix [n, k], in one of two stored forms that the same kernel decodes:

    * ``packed`` — the checkpoint's own bytes: ``weight`` is the 4-bit codes as they ship,
      ``[N/BN, K/64, 32, BN]`` uint8 (two codes a byte, the low nibble the even input), and ``scale`` is the
      block scale as it ships, ``[K/16, N]`` fp8e4m3 bytes. The kernel unpacks the nibbles and rebases the fp8
      exponent, so the device holds what the file holds (four times less than a widened grid).
    * otherwise — an exact BF16 operand (the shared expert): ``weight`` is one bf16 bit pattern a value,
      ``[N/BN, K/64, 64, BN]`` uint16, and ``scale`` is ``[K/16, N]`` fp32, identity for that case.

    ``weight[nb, kb, i, j]`` addresses ``W[nb*BN + j, kb*64 + i]`` — a program's K block is one contiguous
    [64, BN] block (the same read pattern ``qmm`` tiles for). ``scale2`` is the block scale's own per-tensor
    factor (the checkpoint's fp32 ``weight_scale_2``, one a tensor, one an expert), applied on the device: it
    is a kernel argument, never a widening of the scales in memory.

    A leading [E] axis stacks experts over both tensors (a tile slab and its scales each).
    """

    weight: torch.Tensor      # packed: [N/BN, K/64, 32, BN] uint8 | patterns: [N/BN, K/64, 64, BN] uint16
    scale: torch.Tensor       # packed: [K/16, N] uint8 (fp8e4m3)   | patterns: [K/16, N] fp32
    n: int
    k: int
    scale2: torch.Tensor | None = None    # packed: the fp32 per-tensor scale, read by the kernel
    packed: bool = False                  # True: the checkpoint's bytes (codes + fp8 scales), decoded in-kernel

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scale, self.scale2) if t is not None)



def _tensor_scale(weight_scale_2) -> float:
    """The checkpoint stores ``weight_scale_2`` as an fp32 scalar tensor; tests may pass a float."""

    return float(weight_scale_2.item()) if isinstance(weight_scale_2, torch.Tensor) else float(weight_scale_2)


def e2m1_bits(words: torch.Tensor) -> torch.Tensor:
    """(..., N, K/2) uint8 -> (..., N, K) uint16: the bf16 bit pattern of each E2M1 code (the kernel's
    grid). Stacked inputs (a leading expert axis) decode whole — the 512 experts' words in one pass.

    The patterns are gathered as int32 and widened once: torch's CUDA index kernel has no UInt16
    (``index_cuda`` is unimplemented for it), and the 16-bit patterns are exact either way."""

    w = words.to(torch.int32)
    code = torch.stack([w & 0xF, (w >> 4) & 0xF], dim=-1).reshape(*w.shape[:-1], w.shape[-1] * 2)
    table = torch.tensor(BF16_BITS, dtype=torch.int32, device=words.device)
    # the gather and the sign bit are int32 work (torch's CUDA kernels have neither index nor bitwise
    # ops for UInt16); the 16-bit patterns are exact in either width
    pat = table[(code & 0x7).to(torch.int64)] | ((code >> 3) * 0x8000)   # sign bit 15, the pattern's sign
    return pat.to(torch.uint16)


def quantized_values(words: torch.Tensor) -> torch.Tensor:
    """(N, K/2) uint8 -> [N, K] bf16: the E2M1 code grid (exact: the codes' bf16 patterns)."""

    return e2m1_bits(words).view(torch.bfloat16)


def _subnormal_bits(b: torch.Tensor) -> torch.Tensor:
    """The fp8 subnormals' bf16 patterns, value-wise: a subnormal is ``m * 2**-9`` (m 1..7) — exact
    bf16 values, the pattern table ``E4M3_BF16_BITS`` (m 0 is the signed zero)."""

    table = torch.tensor(E4M3_BF16_BITS, dtype=torch.int32, device=b.device)
    return table[b & 0x7]


def e4m3_bits(scale: torch.Tensor) -> torch.Tensor:
    """fp8e4m3 -> bf16 bit patterns (uint16), exact. Built by hand: torch's fp8 casts are unreliable
    (the bf16 cast of fp8 goes through an fp32 view of the storage and yields zeros).

    The fp8 byte is sign (bit 7), exponent (bits 3..6, bias 7), mantissa (bits 0..2). A normal fp8
    value widens by rebasing the exponent (fp8 bias 7 -> bf16 bias 127: ``+ 120``, the bf16 field
    ``((e + 120) << 7) | (m << 4)``); the fp8 subnormals (e == 0, value ``m * 2**-9``) are widened as
    values, not fields (the table — ``m * 2**-9`` is a bf16 power of two times a 3-bit significand,
    exact). The NaN codes (e == 15, m == 7) keep their payload as bf16 NaNs. Checked byte-for-byte
    against torch's fp32 widening over all 256 codes (NaNs compared as NaNs)."""

    b = scale.view(torch.uint8).to(torch.int32)
    e = (b >> 3) & 0xF
    m = b & 0x7
    sign = (b & 0x80) << 8
    normal = torch.where((e == 15) & (m == 7), 0x7FC0, ((e + 120) << 7) | (m << 4))   # NaN codes -> the NaN pattern
    pat = torch.where(e == 0, _subnormal_bits(b), normal) | sign
    return pat.to(torch.int64).to(torch.uint16)


def _bits_to_f32(bits16: torch.Tensor) -> torch.Tensor:
    """uint16 bf16 patterns -> fp32 (exact widening)."""

    return (bits16.to(torch.int32) << 16).view(torch.float32)


def unpack_codes(words: torch.Tensor) -> torch.Tensor:
    """(N, K/2) uint8 -> [N, K] fp32 E2M1 values (the low nibble is the even input)."""

    return _bits_to_f32(e2m1_bits(words))


def row_scales(weight_scale: torch.Tensor, weight_scale_2) -> torch.Tensor:
    """(N, K/16) fp8e4m3 + scalar -> [N, K/16] fp32: each quantization block's dequant weight,
    ``fp32(e4m3) * weight_scale_2`` (the widening exact, the fp32 product one rounding)."""

    return _bits_to_f32(e4m3_bits(weight_scale)) * _tensor_scale(weight_scale_2)


def dequantize(words: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2) -> torch.Tensor:
    """Reference: packed nibbles + scales -> (N, K) fp32 exact weight (the per-tensor scale included):
    code * (fp32(e4m3) * scale_2) per quantization block, the row-scale form ``FP4`` stores."""

    return unpack_codes(words) * row_scales(weight_scale, weight_scale_2).repeat_interleave(GS, dim=1)


def _untile_bits(bits: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The FP4 table's tile of bf16 patterns back to a pattern grid (a leading [E] axis stacked: the
    dataclass ``n`` counts rows *per expert*, so a stacked grid comes back [E*n, k])."""

    if bits.dim() == 5:                                                  # [E, N/BN, K/64, 64, BN]
        rows = n * bits.shape[0]
        bits = bits.permute(0, 1, 4, 2, 3).reshape(rows, k)
    else:                                                                # [N/BN, K/64, 64, BN]
        bits = bits.permute(0, 3, 1, 2).reshape(n, k)
    return bits


def _tile_words(words: torch.Tensor) -> torch.Tensor:
    """(E, N, K/2) (or (N, K/2)) stored uint8 words -> [.., N/BN, K/64, 32, BN]: the tile order
    ``_tile_bits`` produces, at half the width (a byte holds two codes, a block's 16 values its 8 bytes)."""

    *lead, n, k2 = words.shape
    if n % BN:
        raise ValueError(f"NVFP4 tiling needs N a multiple of {BN}, got {n}")
    if k2 % 32:
        raise ValueError(f"NVFP4 tiling needs K/2 a multiple of 32, got {k2}")
    e = words.reshape(*lead, n // BN, BN, k2 // 32, 32)
    return e.permute(*range(len(lead)), len(lead), len(lead) + 2, len(lead) + 3, len(lead) + 1).contiguous()


def _untile_words(tiles: torch.Tensor, n: int, k2: int) -> torch.Tensor:
    """The packed tile grid back to (E, N, K/2) (or (N, K/2)) stored words (the reference's inverse)."""

    if tiles.dim() == 5:                                                 # [E, N/BN, K/64, 32, BN]
        words = tiles.permute(0, 1, 4, 2, 3).reshape(tiles.shape[0] * n, k2)
    else:                                                                # [N/BN, K/64, 32, BN]
        words = tiles.permute(0, 3, 1, 2).reshape(n, k2)
    return words.contiguous()


def _scale2_rows(scale2, rows: int, per: int, device=None) -> torch.Tensor:
    """The per-tensor factors as one a row (the caller multiplies row blocks): one factor a matrix, or one an
    expert of a stacked table."""

    if scale2 is None:
        return torch.ones(rows, dtype=torch.float32, device=device)
    factors = scale2.to(torch.float32).reshape(-1)
    if factors.numel() == rows:                     # one a row already (a stacked table's per-expert factors)
        return factors
    if factors.numel() == 1:
        return factors.expand(rows)
    return factors.repeat_interleave(per)


def dequantize_fp4(fp: FP4) -> torch.Tensor:
    """The stored layout back to the exact fp32 weight [n, k] (the reference for kernel checks). The
    stacked-expert layout (a leading expert axis) comes back as E*N rows, the grouped kernels' row order."""

    if fp.packed:                                                        # the checkpoint's own bytes
        w = _bits_to_f32(e2m1_bits(_untile_words(fp.weight, fp.n, fp.k // 2)))
        s = _bits_to_f32(e4m3_bits(fp.scale))            # [K/16, N] (stacked: [E, K/16, N/E])
        s = s.permute(0, 2, 1).reshape(-1, fp.k // GS) if s.dim() == 3 else s.t()
        rows = s.shape[0]
        factor = _scale2_rows(fp.scale2, rows, fp.n, s.device)
        return w * (s * factor[:, None]).repeat_interleave(GS, dim=1)

    e = int(fp.weight.shape[0]) if fp.weight.dim() == 5 else 1   # the tile grid's leading axis, not the dataclass n
    w = _bits_to_f32(_untile_bits(fp.weight, fp.n, fp.k))
    s = fp.scale
    if s.dim() == 3:                                             # [E, K/16, N/E]
        s = s.permute(0, 2, 1).reshape(w.shape[0], fp.k // GS)
    else:
        s = s.t()                                                # [K/16, N] -> [N, K/16]
    return w * s.repeat_interleave(GS, dim=1)


def _tile_bits(bits: torch.Tensor) -> torch.Tensor:
    """(E, N, K) (or (N, K)) uint16 bf16-pattern grid -> [.., N/BN, K/64, 64, BN] (a program's K block
    contiguous, N tiles outer — the kernel's addressing)."""

    *lead, n, k = bits.shape
    if n % BN:
        raise ValueError(f"NVFP4 tiling needs N a multiple of {BN}, got {n}")
    if k % 64:
        raise ValueError(f"NVFP4 tiling needs K a multiple of 64, got {k}")
    e = bits.reshape(*lead, n // BN, BN, k // 64, 64)
    # [.., N/BN, K/64, 64, BN]: a program's K block is contiguous the way the kernel reads it (the 64 K
    # values a stride of BN apart, the BN columns of a row next to each other)
    return e.permute(*range(len(lead)), len(lead), len(lead) + 2, len(lead) + 3, len(lead) + 1).contiguous()


def _fp8_bytes(weight_scale: torch.Tensor) -> torch.Tensor:
    """The stored scale bytes as uint8 (a safetensors fp8e4m3 tensor arrives as fp8; the kernel reads bytes)."""

    return weight_scale.contiguous().view(torch.uint8)
