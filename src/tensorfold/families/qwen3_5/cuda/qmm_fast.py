"""Keep the optimized four-bit path and dispatch other affine formats without converting their weights."""

from __future__ import annotations

import torch

from tensorfold.cuda.kernels import qmm as shared

from .qmm import lane_matmul
from .weights import QLinear, Weights


def tile(q: QLinear) -> QLinear:
    if q.layout == "tiled" or not q.fast:
        return q
    p = shared.pack(q.weight, q.scales, q.biases, 64)
    return QLinear(p.weight, p.scales, p.biases, layout="tiled", rows=q.n)


def untile(q: QLinear) -> QLinear:
    """The stored MLX layout again (for the fp32 reference, TP sharding or slicing rows)."""

    if q.layout != "tiled":
        return q
    return QLinear(*shared.unpack(shared.Q4(q.weight, q.scales, q.biases, q.n, q.k, 64)))


def rows(q: QLinear, a: int, b: int) -> QLinear:
    """Rows [a, b) of a tiled weight: a view when they are whole 128-row blocks from a tile edge, else a small copy."""

    if a % 64 == 0 and (b - a) % 128 == 0:
        return QLinear(q.weight[a // 64:b // 64], q.scales[:, a:b], q.biases[:, a:b], layout="tiled", rows=b - a)
    t0, t1 = a // 64, -(-b // 64)
    part = shared.Q4(q.weight[t0:t1], q.scales[:, t0 * 64:t1 * 64].contiguous(),
                     q.biases[:, t0 * 64:t1 * 64].contiguous(), (t1 - t0) * 64, q.k, q.gs)
    w, s, bias = shared.unpack(part)
    lo, hi = a - t0 * 64, b - t0 * 64
    return tile(QLinear(w[lo:hi].contiguous(), s[lo:hi].contiguous(), bias[lo:hi].contiguous()))


def matmul_rows(x: torch.Tensor, parts: list[QLinear]) -> torch.Tensor:
    """``x`` against row blocks of one weight, with the bits of the stacked weight's matmul."""

    sk = shared.split_k(sum(p.n for p in parts), parts[0].k, parts[0].gs)
    xs = shared.group_sums(x, parts[0].gs)
    return torch.cat([shared.matmul(x, p, xs, sk=sk) for p in parts], dim=1)


def matmul(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """The lane matmul for either layout; both give the same bits."""

    if not q.fast:
        from tensorfold.cuda.kernels.affine import matmul as affine_matmul

        return affine_matmul(x, q)
    if q.layout == "tiled":
        return shared.matmul(x, q, xs)
    return lane_matmul(x, q.weight, q.scales, q.biases, xs=xs)


def matmul_partial(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 sums for a tiled weight, unrounded: a row-parallel rank's share of a projection."""

    if not q.fast:
        from tensorfold.cuda.kernels.affine import matmul as affine_matmul

        return affine_matmul(x, q, f32=True)
    if q.layout != "tiled":
        raise ValueError("matmul_partial takes tiled weights")
    return shared.matmul(x, q, xs, f32=True)


def stack(parts: list[QLinear]) -> QLinear:
    """Several projections of the same input as one: stored-layout rows concatenated in order."""

    if any(q.layout != "mlx" for q in parts):
        raise ValueError("stack the stored layout, then tile")
    if len({(q.bits, q.gs, q.k, q.scales.dtype, q.biases.dtype) for q in parts}) != 1:
        raise ValueError("stacked projections must share an affine format and input width")
    return QLinear(torch.cat([q.weight for q in parts]).contiguous(), torch.cat([q.scales for q in parts]).contiguous(),
                   torch.cat([q.biases for q in parts]).contiguous(), gs=parts[0].gs, bits=parts[0].bits)


def _stackable(parts: list[QLinear]) -> bool:
    return (all(q.layout == "mlx" for q in parts)
            and len({(q.bits, q.gs, q.k, q.scales.dtype, q.biases.dtype) for q in parts}) == 1)


def stack_small(layer) -> None:
    """[z | b | a] and [k | v] as one matmul each: the gates and k/v are too narrow to fill the GPU alone."""

    if layer.gdn is not None and layer.gdn.zba is None and _stackable([layer.gdn.z, layer.gdn.b, layer.gdn.a]):
        layer.gdn.zba = stack([layer.gdn.z, layer.gdn.b, layer.gdn.a])
    if layer.attn is not None and layer.attn.kv is None and _stackable([layer.attn.k, layer.attn.v]):
        layer.attn.kv = stack([layer.attn.k, layer.attn.v])


def prepare(w: Weights, *, fuse: bool = False) -> None:
    """Pack every projection and the head in place; ``fuse`` changes K splits and bits, so all rounds must share it."""

    for layer in w.layers:
        if fuse:
            stack_small(layer)
        for owner, names in ((layer, ("gate", "up", "down")), (layer.gdn, ("qkv", "z", "b", "a", "out")),
                             (layer.attn, ("q", "k", "v", "o"))):
            if owner is None:
                continue
            for name in names:
                setattr(owner, name, tile(getattr(owner, name)))
        if layer.gdn is not None and layer.gdn.zba is not None:
            layer.gdn.zba = tile(layer.gdn.zba)
        if layer.attn is not None and layer.attn.kv is not None:
            layer.attn.kv = tile(layer.attn.kv)
    w.head = tile(w.head)
    torch.cuda.empty_cache()
