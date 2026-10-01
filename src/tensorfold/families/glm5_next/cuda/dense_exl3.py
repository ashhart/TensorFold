"""GLM-5.3-Flash's dense projections from an EXL3 pack (``TF_GLM_DENSE_EXL3``), on ``tensorfold.cuda.exl3``'s Exl3Linear.

An EXL3 checkpoint quantizes the routed experts only; attention, the shared experts, the dense MLPs and the head stay
BF16, ~9.7 GiB a rank that every decode round reads. ``TF_GLM_DENSE_EXL3`` names one or more safetensors files
(``:``-separated) holding EXL3 groups (``<module>.trellis / suh / svh / mul1 | mcg``) for some of those matrices, under
the checkpoint's own module names, e.g. the non-expert groups of a full EXL3 quant of the model. Every matrix a pack
holds runs on Exl3Linear; the others keep the engine's BF16 path.

Each matrix is cut for its rank the way ``split.py`` cuts its BF16 twin. A "row" matrix (output rows split) keeps its
rank's output tiles and their svh; a "col" matrix (input split) keeps its rank's input tiles and suh; a "rep" matrix
stays whole; the head keeps its rank's vocabulary columns. Hadamard blocks are 128 wide and every cut lands on a
multiple of 128, so a "row" shard's outputs are the full layer's columns, and a "col" shard's output is
had(z_r) * svh, which the ranks' all-reduce sums into had(sum z) * svh: the full layer's output.

Projections that the engine fuses (KDA's q/k/v/f_a/g_a/b, DSA's q_a/kv_a, gate/up) become ``Dense3`` stacks: one part
a matrix, EXL3 or BF16, each written into its own columns of the output."""

from __future__ import annotations

import os

import torch

_PACK = None
_WORKSPACES: dict[str, object] = {}


def pack_paths(env=None) -> list[str]:
    value = (os.environ if env is None else env).get("TF_GLM_DENSE_EXL3", "").strip()
    return [p for p in value.split(":") if p]


class Pack:
    """The pack's tensors by name, read lazily from its safetensors files (a later file wins a duplicate name)."""

    def __init__(self, paths: list[str]) -> None:
        from safetensors import safe_open

        self.files = [safe_open(p, framework="pt", device="cpu") for p in paths]
        self.where = {k: f for f in self.files for k in f.keys()}

    def has(self, module: str) -> bool:
        return module + ".trellis" in self.where

    def get(self, key: str) -> torch.Tensor | None:
        f = self.where.get(key)
        return None if f is None else f.get_tensor(key)


def pack() -> Pack | None:
    global _PACK
    if _PACK is None and pack_paths():
        _PACK = Pack(pack_paths())
    return _PACK


class Dense3:
    """A projection as parts whose outputs are concatenated: Exl3Linear shards and/or the engine's BF16 matrices."""

    def __init__(self, parts: list, sizes: list[int], k: int) -> None:
        self.parts, self.sizes, self.k = parts, sizes, k
        self.n = sum(sizes)

    def tensors(self) -> list[torch.Tensor]:
        """Every tensor the parts hold (memory accounting)."""

        out = []
        for p in self.parts:
            if hasattr(p, "words"):
                out += [t for t in (p.words, p.suh, p.svh, p.bias) if t is not None]
            else:
                out += [t for t in vars(p).values() if isinstance(t, torch.Tensor)]
        return out


def cut(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, kind: str, rank: int, world: int,
        cols: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A rank's (trellis [K/16, N/16, 16 * bits], suh [K], svh [N]) for split rule ``kind``, or output ``cols``."""

    k, n = suh.numel(), svh.numel()
    if cols is not None:
        a, b = cols
        if a % 128 or b % 128:
            raise ValueError(f"EXL3 output columns {cols} must be multiples of 128")
        return trellis[:, a // 16:b // 16], suh, svh[a:b]
    if kind == "row":
        h = n // world
        if h % 128:
            raise ValueError(f"EXL3 row split of N={n} over {world} ranks is not a multiple of 128")
        return trellis[:, rank * h // 16:(rank + 1) * h // 16], suh, svh[rank * h:(rank + 1) * h]
    if kind == "col":
        h = k // world
        if h % 128:
            raise ValueError(f"EXL3 column split of K={k} over {world} ranks is not a multiple of 128")
        return trellis[rank * h // 16:(rank + 1) * h // 16], suh[rank * h:(rank + 1) * h], svh
    if kind == "rep":
        return trellis, suh, svh
    raise ValueError(f"no EXL3 split for rule {kind!r}")


def linear(module: str, kind: str, rank: int, world: int, device, cols: tuple[int, int] | None = None):
    """``module``'s shard on this rank as an Exl3Linear (``kind``: split.py's rule for its BF16 weight)."""

    from tensorfold.cuda.exl3.linear import Exl3Linear

    pk = pack()
    tr, suh, svh = cut(pk.get(module + ".trellis"), pk.get(module + ".suh"), pk.get(module + ".svh"), kind, rank,
                       world, cols)
    codebook = "mul1" if pk.get(module + ".mul1") is not None else "mcg" if pk.get(module + ".mcg") is not None \
        else "3inst"
    return Exl3Linear.from_tensors(tr.contiguous(), suh.contiguous(), svh.contiguous(), codebook, device=device)


def workspace(device):
    """One prompt-path workspace a device: the unpacked weight of the widest projection, reused by every layer."""

    from tensorfold.cuda.exl3.prefill import Workspace

    key = str(device)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = Workspace()
    return _WORKSPACES[key]


def run(layer, x: torch.Tensor, out: torch.Tensor) -> None:
    """out [M, n] = x [M, K] through Exl3Linear ``layer``: the decode kernel up to 128 rows, the prompt GEMM above
    (both row-invariant: a row's bits never depend on the row count)."""

    if x.shape[0] <= 128:
        layer(x, out=out)
    else:
        from tensorfold.cuda.exl3.prefill import matmul as prompt_matmul

        prompt_matmul(layer, x, out, workspace(x.device))


def matmul(x: torch.Tensor, q: Dense3, *, out: torch.Tensor | None, f32: bool, dense_matmul) -> torch.Tensor:
    """x [M, K] (bf16, rows may be strided) times ``q``: each part into its own columns of ``out`` (bf16, or fp32 with
    ``f32``); ``dense_matmul(x, part, f32)`` runs a BF16 part."""

    m = x.shape[0]
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    xc = x if x.is_contiguous() else x.contiguous()
    col = 0
    for part, n in zip(q.parts, q.sizes):
        dst = out[:, col:col + n]
        if hasattr(part, "words"):
            if dst.is_contiguous() or (m > 128 and dst.stride(1) == 1):
                run(part, xc, dst)
            else:                                 # a decode window's column slice: the kernel writes whole rows
                tmp = torch.empty((m, n), dtype=out.dtype, device=x.device)
                run(part, xc, tmp)
                dst.copy_(tmp)
        else:
            dst.copy_(dense_matmul(x, part, f32))
        col += n
    return out
