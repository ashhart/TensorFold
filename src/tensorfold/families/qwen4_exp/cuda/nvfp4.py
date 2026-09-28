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

    W[n, k] = E2M1(nibble (n, k)) * (fp32(weight_scale[n, k // 16]) * 2**-7 * weight_scale_2)

with E2M1 the 16 FP4 values {0, .5, 1, 1.5, 2, 3, 4, 6} and their negations (code = sign << 3 | magnitude),
the fp32 widening of the e4m3 scale exact and the fp32 product taking one rounding. The quantization is
experts-only: every other linear (hyper-connections, DeltaNet, attention, the router, the shared expert,
PLE, embeddings, lm_head, the MTP head) is stored in BF16.

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
    ``fp32(e4m3) * 2**-7 * weight_scale_2`` (the widening exact, the fp32 product one rounding)."""

    return _bits_to_f32(e4m3_bits(weight_scale)) * (_tensor_scale(weight_scale_2) * 2.0 ** -7)


def dequantize(words: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2) -> torch.Tensor:
    """Reference: packed nibbles + scales -> (N, K) fp32 exact weight (the per-tensor scale included):
    code * (fp32(e4m3) * 2**-7 * scale_2) per quantization block, the row-scale form ``FP4`` stores."""

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
        factor = _scale2_rows(fp.scale2, rows, fp.n, s.device) * (2.0 ** -7)
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


def make_fp4(words: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2) -> FP4:
    """One linear layer from the checkpoint's arrays: (N, K/2) uint8, (N, K/16) fp8e4m3, scalar -> FP4.

    The stored bytes are kept as they ship; the kernel decodes them, so nothing is widened here."""

    n, k2 = words.shape
    factor = torch.full((n,), _tensor_scale(weight_scale_2), dtype=torch.float32, device=words.device)
    return FP4(_tile_words(words), _fp8_bytes(weight_scale).t().contiguous(), n, k2 * 2,
               scale2=factor, packed=True)


def stacked_fp4(words: torch.Tensor, weight_scale: torch.Tensor, scale2) -> FP4:
    """Stacked per-expert arrays (words [E, N, K/2] uint8, scales [E, N, K/16] fp8e4m3, scale2 [E] fp32) -> one
    packed table with a leading expert axis (``n`` rows per expert; the kernels slice ``weight[e]``/``scale[e]``
    and read ``scale2[e]``)."""

    e, n, k2 = words.shape
    factors = torch.as_tensor(scale2, dtype=torch.float32, device=words.device)
    if factors.numel() == e:                        # one factor an expert: the same for its whole row block
        factors = factors.reshape(e, 1).expand(e, n)
    return FP4(_tile_words(words), _fp8_bytes(weight_scale).permute(0, 2, 1).contiguous(), n, k2 * 2,
               scale2=factors.reshape(e, n), packed=True)


def fp4_from_rows(weight_bits: torch.Tensor, scale: torch.Tensor) -> FP4:
    """An FP4 from a bf16-pattern grid (uint16 [N, K]) and [N, K/16] fp32 row scales: the gate/up stack
    (gate and up code grids joined, each projection's row scales its own) and any pre-decoded table. The
    per-tensor factor is materialised here, not lazily: a table both the eager step and a CUDA graph capture
    read must never allocate during capture."""

    n, k = weight_bits.shape
    return FP4(_tile_bits(weight_bits), scale.t().contiguous(), n, k,
               scale2=torch.ones(n, dtype=torch.float32, device=weight_bits.device))


def fp4_from_bf16(rows: torch.Tensor) -> FP4:
    """A BF16 matrix as an exact scale-1 FP4-style table (identity row scales): the shared expert riding
    in the FP4 kernels, its bf16 values as the stored operand grid."""

    n, k = rows.shape
    scale = torch.ones((n, k // GS), dtype=torch.float32, device=rows.device)
    return fp4_from_rows(rows.contiguous().view(torch.uint16), scale)


def split_k(n: int, k: int, target: int = 160) -> int:
    """K-slice count for an (n, k) weight: a function of the shape only (never the row count), a power of two,
    at least 4 quantization blocks (64 inputs) a slice."""

    tiles = -(-n // BN)
    blocks = k // GS
    sk = 1
    while sk < 32 and tiles * sk < target and blocks % (sk * 2) == 0 and blocks // (sk * 2) >= 4:
        sk *= 2
    return sk


def gpi_for(per: int, want: int) -> int:
    for g in (want, 8, 4, 2, 1):
        if g <= want and per % g == 0:
            return g
    return 1


def bucket(m: int) -> int:
    """Rows a program takes: 16 to 128, then tiles of 128 (a row's bits never depend on its tile)."""

    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    return 128


# -- the matmul kernel -------------------------------------------------------------------------------
HAS_TRITON = True
try:
    import triton                             # noqa: E402
    import triton.language as tl              # noqa: E402

    @triton.jit
    def _bf16_widen(bits):
        return (bits.to(tl.int32) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _e2m1_pattern(code):
        """An E2M1 code (sign in bit 3, magnitude bits 0..2) as its bf16 bit pattern: the magnitudes 0, .5, 1,
        1.5, 2, 3, 4, 6 — the same table ``e2m1_bits`` gathers, built from the code's fields."""

        m = code & 0x7
        pat = tl.where(m == 0, 0, ((126 + (m >> 1)) << 7) | (tl.where(m >= 2, m & 1, 0) << 6))
        return (pat | ((code & 0x8) << 12)).to(tl.uint16)

    @triton.jit
    def _e4m3_value(byte):
        """An fp8e4m3 byte as its exact fp32 value: the exponent rebases (bias 7 -> 127: +120), its subnormals
        are values (m * 2**-9), its NaN codes stay NaN — the rules ``e4m3_bits`` follows."""

        e = (byte >> 3) & 0xF
        m = byte & 0x7
        widened = ((((e + 120) << 7) | (m << 4)).to(tl.int32) << 16).to(tl.float32, bitcast=True)
        value = tl.where(e == 0, m.to(tl.float32) * (2.0 ** -9), widened)
        value = tl.where((e == 15) & (m == 7), float("nan"), value)
        return tl.where((byte & 0x80) != 0, -value, value)

    @triton.jit
    def _fp4mm(X, W, S, S2, OUT, PART, M, x_stride,
               N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
               SBN: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, F32: tl.constexpr,
               PACKED: tl.constexpr):
        """x [M, K] bf16 @ FP4.T -> [M, N] bf16 (split-K: fp32 partials, reduced in slice order). One program:
        a row tile x a column tile x one K slice; blocks in order, each block one tensor-core dot of the 16
        bf16 inputs against the block's stored codes, times the block's scale: one fp32 rounding per block
        product, the dequantize reference's order. ``SBN`` is the stored N tile (a constexpr: Triton reads
        no globals) and ``BLOCK_N`` the program's slice of it.

        ``PACKED`` reads the checkpoint's own bytes: the codes as stored nibbles (two a byte, the low nibble
        the even input) and the scales as stored fp8e4m3, both decoded here — the device keeps the file's
        bytes, a quarter of the widened grid. Otherwise the operand is a grid of bf16 patterns (an exact BF16
        matrix riding these kernels) with fp32 scales. ``S2`` is the fp32 per-tensor scale the checkpoint
        carries beside its block scales."""

        PER: tl.constexpr = (K // 16) // SK             # quantization blocks per slice
        SUB: tl.constexpr = SBN // BLOCK_N              # programs per stored N tile
        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        r16 = tl.arange(0, 16)
        m_ok = rm < M
        n_ok = rn < N
        # a quantization block b lives in stored K block kb = b // 4, rows (b % 4) * 16 .. + 15 of
        # [N/SBN, K/64, 64, SBN] — the packed form, 32 bytes a K block and 8 rows a quantization block
        tile = W + (pid_n // SUB) * ((K // 64) * (32 if PACKED else 64) * SBN)
        local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        s2 = tl.load(S2 + rn, mask=n_ok, other=1.0)     # the per-tensor factor, one a row (a stacked table)
        KT: tl.constexpr = K // 64
        for i in range(PER // GPI):
            for j in tl.static_range(GPI):
                b = pid_s * PER + i * GPI + j
                kb = b // 4
                row0 = (b % 4) * 16
                x = tl.load(X + rm[:, None] * x_stride + (b * 16 + r16)[None, :], mask=m_ok[:, None], other=0.0)
                if PACKED:
                    # a block's 16 values are its 8 stored bytes: read a byte a value (half the weight
                    # traffic of the widened grid) and take the low nibble for the even input
                    w8 = tl.load(tile + kb * (32 * SBN) + (row0 // 2 + r16 // 2)[:, None] * SBN + local[None, :])
                    code = tl.where((r16 % 2)[:, None] == 0, w8 & 0xF, w8 >> 4).to(tl.int32)
                    wv = _bf16_widen(_e2m1_pattern(code)).to(tl.bfloat16)
                else:
                    wbits = tl.load(tile + kb * (64 * SBN) + (row0 + r16)[:, None] * SBN + local[None, :])
                    wv = _bf16_widen(wbits).to(tl.bfloat16)
                p = tl.dot(x, wv)
                if PACKED:
                    s = _e4m3_value(tl.load(S + b * N + rn, mask=n_ok, other=0).to(tl.int32)) * (s2 * 0.0078125)
                else:
                    s = tl.load(S + b * N + rn, mask=n_ok, other=0.0)
                acc += p * s[None, :]
        out_mask = m_ok[:, None] & n_ok[None, :]
        if SK == 1:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc if F32 else acc.to(tl.bfloat16), mask=out_mask)
        else:
            tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)

    @triton.jit
    def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < total
        acc = tl.load(PART + offs, mask=ok, other=0.0)
        for s in tl.static_range(1, SK):
            acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)
except ModuleNotFoundError:                   # the CPU tests of the format import this module without Triton
    HAS_TRITON = False


# (blocks per unrolled step, warps, stages) by row bucket: every choice gives the same bits
CONFIG = {16: (4, 4, 3), 32: (2, 4, 3), 64: (2, 4, 2), 128: (1, 8, 2)}

# Per (N, K) at up to 16 rows: (K slices, blocks per step, warps, stages). The K slices set the sum order
# (a per-shape constant); the rest never changes bits. Expert shapes of the NVFP4 Flash Next (D = 2560,
# NI = 640): gate/up stacked (1280, 2560), gate and up (640, 2560), down (2560, 640).
SHAPES16 = {
    (640, 2560): (1, 4, 4, 3),
    (1280, 2560): (1, 4, 4, 3),
    (2560, 640): (8, 2, 4, 3),
}


def split_for(n: int, k: int) -> int:
    got = SHAPES16.get((n, k))
    return got[0] if got else split_k(n, k)


def matmul(x: torch.Tensor, fp: FP4, *, out: torch.Tensor | None = None, f32: bool = False,
           sk: int | None = None, part: torch.Tensor | None = None,
           gpi: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
           block_n: int | None = None) -> torch.Tensor:
    """x [M, K] bf16 (rows may be strided) @ fp.T -> [M, N] bf16 (or fp32 sums with ``f32``, as the MoE
    buffers keep). Split-K sums in slice order (``_reduce``), the same rule as ``qmm.matmul``."""

    if not HAS_TRITON:
        raise RuntimeError("the NVFP4 matmul needs Triton (the CUDA engine's environment)")

    m, k = x.shape
    if k != fp.k or x.stride(1) != 1:
        raise ValueError(f"fp4 matmul: x {tuple(x.shape)} does not match K={fp.k}")
    if fp.n % BN:
        raise ValueError(f"fp4 matmul: N {fp.n} is not a multiple of {BN}")
    bm = bucket(m)
    c_gpi, c_warps, c_stages = CONFIG[bm]
    tuned = SHAPES16.get((fp.n, fp.k)) if bm == 16 else None
    if tuned is not None:
        _, c_gpi, c_warps, c_stages = tuned
    sk = int(sk) if sk else split_for(fp.n, fp.k)
    per = (k // GS) // sk
    g = gpi_for(per, gpi or c_gpi)
    if out is None:
        out = torch.empty((m, fp.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, fp.n) or not out.is_contiguous():
        raise ValueError(f"fp4 matmul: out {tuple(out.shape)} must be a contiguous ({m}, {fp.n})")
    elif (out.dtype == torch.float32) != f32:
        raise ValueError(f"fp4 matmul: out dtype {out.dtype} does not match f32={f32}")
    if sk > 1 and part is None:
        part = torch.empty((sk, m, fp.n), dtype=torch.float32, device=x.device)
    bn = block_n or BN
    if fp.scale2 is None:                        # the pattern form: its fp32 scales already carry the factor
        fp.scale2 = torch.ones(fp.n, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), fp.n // bn, sk)
    _fp4mm[grid](x, fp.weight, fp.scale, fp.scale2, out, part if sk > 1 else out, m, x.stride(0),
                 N=fp.n, K=k, SK=sk, BM=bm, SBN=BN, BLOCK_N=bn, GPI=g, F32=f32, PACKED=fp.packed,
                 num_warps=num_warps or c_warps, num_stages=num_stages)
    if sk > 1:
        total = m * fp.n
        if f32:
            # the fp32 sums at the split: the caller's combine wants every slice's fp32 partial summed in
            # slice order, so the reduce's own rounding is the final bf16 one — for f32 outputs, sum the
            # slices here in slice order, one fp32 add per slice, no bf16 round
            out.copy_(part[0])
            for s in range(1, sk):
                out += part[s]
        else:
            _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, num_warps=4)
    return out
