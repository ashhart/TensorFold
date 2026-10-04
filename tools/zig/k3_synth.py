"""Kimi K3 test data: bf16 rounding, MXFP4 decode and hash-made weights the Metal fill kernel makes bit for bit."""
import numpy as np

M32 = 0xFFFFFFFF
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def bf16(x) -> np.ndarray:
    """Round to bfloat16 (nearest even) through fp32; the result is fp32 holding bf16 values."""
    u = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def bf16_bits(x) -> np.ndarray:
    """The bf16 words of values already on the bf16 grid."""
    return (np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32) >> 16).astype(np.uint16)


def from_bf16_bits(words) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)


def fnv1a(name: str) -> int:
    """32-bit FNV-1a of a tensor name: its synthetic seed."""
    h = 0x811C9DC5
    for b in name.encode():
        h = ((h ^ b) * 0x01000193) & M32
    return h


def mix(seed: int, index) -> np.ndarray:
    """A 32-bit hash of (seed, index) for each index; the Metal fill kernel runs the same steps."""
    x = np.asarray(index, dtype=np.uint32) ^ np.uint32((seed * 0x9E3779B9) & M32)
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def values(seed: int, count: int, kind: str, e0: int = 0, start: int = 0) -> np.ndarray:
    """Synthetic tensor values: 'w' = k 2^-(e0+j), 'one' = 1 + k 2^-7, 'u8' bytes, 'e8' = E8M0 127-e0-j; all exact."""
    x = mix(seed, np.arange(start, start + count, dtype=np.uint64).astype(np.uint32))
    if kind == "u8":
        return (x & 0xFF).astype(np.uint8)
    if kind == "e8":
        return (127 - e0 - ((x >> 8) & 3)).astype(np.uint8)
    if kind == "one":
        return (1.0 + ((x & 63).astype(np.float64) - 32.0) / 128.0).astype(np.float32)
    k = (x & 0xFF).astype(np.float64) - 128.0
    return (k * np.exp2(-(e0 + ((x >> 8) & 3)).astype(np.float64))).astype(np.float32)


def tensor(name: str, shape, kind: str, e0: int = 0) -> np.ndarray:
    """The synthetic tensor called `name`, its values set by the name's seed alone."""
    return values(fnv1a(name), int(np.prod(shape)), kind, e0).reshape(shape)


def fan_exp(fan_in: int) -> int:
    """The base exponent that gives a fan_in-wide product rows of unit scale (k has rms ~74)."""
    return int(round(np.log2(74.0 * np.sqrt(fan_in)))) - 1


def rule(name: str, shape) -> tuple:
    """The fill rule for a tensor name (zig/tests/kimi_k3/synth.zig's rule): kind and base exponent."""
    if name.endswith("_packed"):
        return "u8", 0
    if name.endswith("_scale"):
        return "e8", 7
    if name.endswith("A_log"):
        return "w", 7
    if name.endswith(("dt_bias", "conv1d.weight", "e_score_correction_bias")):
        return "w", 8
    if name.endswith("norm.weight"):
        return "one", 0
    if name.endswith("res_proj.weight"):
        return "w", 6
    if name.endswith("embed_tokens.weight"):
        return "w", 7
    return "w", fan_exp(shape[-1]) if len(shape) == 2 else 6


def synthetic(name: str, shape) -> np.ndarray:
    """The synthetic tensor the Zig checks make for `name` at `shape`."""
    kind, e0 = rule(name, shape)
    return tensor(name, shape, kind, e0)


def mxfp4_decode(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dense fp32 from compressed-tensors' MXFP4: low nibble first, E2M1 codes, E8M0 scale a group of 32."""
    lo, hi = packed & 0x0F, packed >> 4
    codes = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], -1)
    mag = E2M1[codes & 7]
    vals = np.where(codes & 8, -mag, mag)
    scale = np.exp2(scales.astype(np.float64) - 127.0).astype(np.float32)
    return vals * np.repeat(scale, 32, axis=1)
