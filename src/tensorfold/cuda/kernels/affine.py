"""Packed MLX affine projections without dense weight copies or row-dependent arithmetic."""

from __future__ import annotations

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)


def packed_shape(weight, scales, biases, bits: int, group: int) -> tuple[int, int]:
    if bits not in BITS or group not in GROUPS:
        raise ValueError("CUDA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    if len(weight) != 2 or len(scales) != 2 or tuple(scales) != tuple(biases):
        raise ValueError("affine projections need 2-D weights and matching scales and biases")
    n, groups = map(int, scales)
    k = groups * group
    if n <= 0 or groups <= 0 or tuple(weight) != (n, k * bits // 32):
        raise ValueError("packed affine words do not match their declared bit width and group metadata")
    return n, k


def input_slice(k: int, bits: int, group: int, rank: int, world: int = 2):
    """Split only whole quantization groups and packed words, without changing any values."""
    if world != 2 or rank not in (0, 1) or bits not in BITS or group not in GROUPS or k % (world * group):
        raise ValueError("tensor parallel affine inputs must split into two complete group-aligned halves")
    count = k // (world * group)
    a, b = rank * count, (rank + 1) * count
    return (a * group * bits // 32, b * group * bits // 32), (a, b)


def _validate(x, q):
    import torch

    if q.layout == "dense":
        if q.weight.ndim != 2 or q.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("dense projections need a 2-D floating-point checkpoint weight")
        n, k = q.weight.shape
    else:
        if q.layout != "mlx":
            raise ValueError("generic affine kernels read the original MLX word layout")
        n, k = packed_shape(q.weight.shape, q.scales.shape, q.biases.shape, q.bits, q.gs)
        if q.weight.dtype != torch.int32 or any(t.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                                               for t in (q.scales, q.biases)):
            raise ValueError("affine words must be int32 with floating-point scales and biases")
    if x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or x.shape[0] < 1:
        raise ValueError("affine inputs must be nonempty BF16 rows of the declared input width")
    if not x.is_cuda or any(t is not None and t.device != x.device for t in (q.weight, q.scales, q.biases)):
        raise ValueError("affine operands must share a CUDA device")
    if any(t is not None and not t.is_contiguous() for t in (q.weight, q.scales, q.biases)):
        raise ValueError("affine weights and group metadata must be contiguous")
    return n, k


def matmul(x, q, *, f32: bool = False):
    import torch
    import triton
    from . import affine_kernels as kernels

    n, k = _validate(x, q)
    x = x.contiguous()
    out = torch.empty((x.shape[0], n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    if q.layout == "dense":
        kernels.dense[(x.shape[0], triton.cdiv(n, 4))](x, q.weight, out, N=n, K=k, BLOCK=128,
                                                     num_warps=4, enable_fp_fusion=False)
    else:
        kernels.matmul[(triton.cdiv(x.shape[0], 16), triton.cdiv(n, 32))](
            x, q.weight, q.scales, q.biases, out, x.shape[0], N=n, K=k, BITS=q.bits, GS=q.gs,
            num_warps=4, enable_fp_fusion=False)
    return out


def embed(ids, q):
    import torch
    import triton
    from . import affine_kernels as kernels

    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or ids.device != q.weight.device:
        raise ValueError("embedding IDs must be a vector on the weights' CUDA device")
    if q.layout == "dense":
        return q.weight.index_select(0, ids.to(torch.int64)).to(torch.bfloat16)
    if q.layout != "mlx" or q.weight.dtype != torch.int32:
        raise ValueError("generic embeddings require original int32 MLX packed words")
    n, k = packed_shape(q.weight.shape, q.scales.shape, q.biases.shape, q.bits, q.gs)
    out = torch.empty((ids.shape[0], k), dtype=torch.bfloat16, device=ids.device)
    kernels.embed[(ids.shape[0], triton.cdiv(k, 256))](ids, q.weight, q.scales, q.biases, out,
                                                     N=n, K=k, BITS=q.bits, GS=q.gs, BLOCK=256,
                                                     num_warps=4, enable_fp_fusion=False)
    return out
