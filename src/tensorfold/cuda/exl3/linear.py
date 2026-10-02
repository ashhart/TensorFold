"""EXL3 linear layers on CUDA (``linear.cu``), any codebook and width: y = x @ W + bias for 1 to 128 rows, row-invariant, no cuBLAS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import torch

from . import format as fmt

CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}
# the kernel's walk and launch (TF_EXL3_LINEAR_LOADS overrides); none changes a bit of any output. 0: the original walk.
# Bit 0: the transposed walk (the decoded tile as the mma's A, 8 rows a mma: half the mma and accumulators up to 8 rows,
# x a step ahead); bit 1 (with bit 0): the weights as 16-byte loads a step ahead, staged in shared memory (3 to 6
# bits); bit 5: programmatic dependent launch of rot_in and the linear (their weight loads overlap the previous kernel)
LOADS = int(os.environ.get("TF_EXL3_LINEAR_LOADS", "35"))


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_linear_v5", sources=[str(here / "linear.cpp"), str(here / "linear.cu")],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False)


def k2_of(bits: float) -> int:
    return int(2 * fmt.check_bits(bits))


def plan(k: int, n: int, blocks: int = 192, min_tiles: int = 8) -> tuple[int, int]:
    """(K splits, warps a program) for a K x N layer: the shape's alone, so a layer keeps one reduction for every row count."""

    kt, nb = k // 16, n // 128
    sk, wk = (1, 8) if nb >= 64 else (1, 4)
    if nb < 64:                                   # a narrow layer needs K splits to fill the SMs at all
        while nb * sk < blocks and sk < 64:
            sk *= 2
    while (wk > 4 or sk > 1) and (kt % (sk * wk) or kt // (sk * wk) < min_tiles):
        if wk > 4:
            wk = 4
        elif sk > 1:
            sk //= 2
        else:
            break
    return sk, wk


def strips(words: torch.Tensor) -> torch.Tensor:
    """Trellis words [K/16, N/16, W] -> [N/128, K/16, 8, W]: each 128-column block's tiles in k order (a copy)."""

    kt, nt, w = words.shape
    return words.view(kt, nt // 8, 8, w).permute(1, 0, 2, 3).contiguous()


@dataclass
class Exl3Linear:
    """One EXL3 layer on the GPU: trellis words (``layout``), fp16 scales, optional fp16 bias."""

    words: torch.Tensor           # int32, [N/128, K/16, 8, 8 * bits] ("strips") or [K/16, N/16, 8 * bits] ("stored")
    suh: torch.Tensor             # fp16 [K]
    svh: torch.Tensor             # fp16 [N]
    bias: torch.Tensor | None     # fp16 [N]
    bits: float
    codebook: str
    k: int
    n: int
    layout: str = "strips"
    split: tuple[int, int] | None = None     # (K splits, warps a program); plan(k, n) when None, fixed thereafter
    loads: int | None = None                 # the kernel's load path (LOADS when None); never changes a bit

    @classmethod
    def from_tensors(cls, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str,
                     bias: torch.Tensor | None = None, device: str | torch.device = "cuda",
                     layout: str = "strips") -> "Exl3Linear":
        """From a group's tensors: trellis int16 [K/16, N/16, 16 * bits], suh/svh fp16 (or packed su/sv sign words), codebook name."""

        bits = fmt.bits_of(trellis.shape)
        if codebook not in CODEBOOK_IDS:
            raise ValueError(f"unknown EXL3 codebook {codebook!r}")
        if not float(bits).is_integer() and codebook != "mul1":
            raise ValueError(f"{bits}-bit EXL3 tiles need the mul1 codebook")
        k, n = 16 * trellis.shape[0], 16 * trellis.shape[1]
        if k % 128 or n % 128:
            raise ValueError(f"K={k} and N={n} must be multiples of 128")

        def scales(t: torch.Tensor, size: int) -> torch.Tensor:
            if t.dtype == torch.int16 and t.numel() * 16 == size:
                t = torch.from_numpy(fmt.unpack_signs(t))
            if t.dtype != torch.float16 or t.numel() != size:
                raise ValueError(f"scales must be fp16 [{size}] or int16 sign words [{size // 16}]")
            return t.reshape(size).to(device).contiguous()

        words = trellis.to(device).contiguous().view(torch.int32)
        if layout == "strips":
            words = strips(words)
        elif layout != "stored":
            raise ValueError(f"layout must be 'strips' or 'stored', got {layout!r}")
        b = None if bias is None else bias.to(device=device, dtype=torch.float16).contiguous()
        return cls(words, scales(suh, k), scales(svh, n), b, bits, codebook, k, n, layout)

    @classmethod
    def load(cls, model_dir: str | Path, prefix: str, device: str | torch.device = "cuda",
             layout: str = "strips") -> "Exl3Linear":
        """Layer ``prefix`` (e.g. "model.layers.0.self_attn.q_proj") of an EXL3 checkpoint folder."""

        from safetensors import safe_open

        root = Path(model_dir)
        index = root / "model.safetensors.index.json"
        if index.exists():
            import json

            weight_map = json.loads(index.read_text())["weight_map"]
            rel = weight_map.get(prefix + ".trellis")
            if rel is None:  # a group outside the index's coverage: find the shard that holds it
                for part in sorted(root.glob("*.safetensors")):
                    with safe_open(str(part), framework="pt") as g:
                        if f"{prefix}.trellis" in set(g.keys()):
                            rel = part.name
                            break
            if rel is None:
                raise ValueError(f"{prefix}.trellis is in no file of {root}")
            file = root / rel
        else:
            file = root / "model.safetensors"
        with safe_open(str(file), framework="pt") as f:
            names = set(f.keys())

            def get(part: str) -> torch.Tensor | None:
                return f.get_tensor(f"{prefix}.{part}") if f"{prefix}.{part}" in names else None

            parts = {p: get(p) for p in fmt.PARTS}
        codebook = "mul1" if parts["mul1"] is not None else "mcg" if parts["mcg"] is not None else "3inst"
        suh = parts["suh"] if parts["suh"] is not None else parts["su"]
        svh = parts["svh"] if parts["svh"] is not None else parts["sv"]
        return cls.from_tensors(parts["trellis"], suh, svh, codebook, parts["bias"], device, layout)

    @property
    def k2(self) -> int:
        return k2_of(self.bits)

    @property
    def strides(self) -> tuple[int, int]:
        """(words between k tiles, words between 128-column blocks) in ``words``."""

        tw = 4 * self.k2
        if self.layout == "strips":
            return 8 * tw, (self.k // 16) * 8 * tw
        return (self.n // 16) * tw, 8 * tw

    def nbytes(self) -> int:
        return self.words.numel() * 4

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = plan(self.k, self.n)

    @property
    def counters(self) -> torch.Tensor:
        c = getattr(self, "_counters", None)
        if c is None:
            c = torch.zeros((8 * (self.n // 128),), dtype=torch.int32, device=self.words.device)
            self._counters = c
        return c

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                 xh: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        """y [M, N] = x [M, K] @ W + bias for M = 1..128; scratch ``xh`` fp16 [M, K] and ``z`` fp32 (SK > 1) allocated when not given."""

        if x.dim() != 2 or x.shape[1] != self.k or not 1 <= x.shape[0] <= 128:
            raise ValueError(f"x must be [1..128, {self.k}], got {tuple(x.shape)}")
        x = x.contiguous()
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, self.n), dtype=out_dtype or x.dtype, device=x.device)
        sk, wk = self.split
        if xh is None:
            xh = torch.empty((m, self.k), dtype=torch.float16, device=x.device)
        if sk > 1 and z is None:
            z = torch.empty((sk * m * self.n,), dtype=torch.float32, device=x.device)
        loads = LOADS if self.loads is None else self.loads
        _launch([self], x, [xh], [out], [z], loads)
        return out

    def unpack(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """W_q [K, N] fp16 decoded on the GPU from either layout, into ``out`` when given."""

        w = out if out is not None else torch.empty((self.k, self.n), dtype=torch.float16, device=self.words.device)
        _ext().unpack(self.words, w, *self.strides, self.k2, CODEBOOK_IDS[self.codebook])
        return w


_NONE: dict = {}


def _none(device) -> torch.Tensor:
    """An empty tensor: no bias, no Z."""
    key = str(device)
    if key not in _NONE:
        _NONE[key] = torch.empty((0,), dtype=torch.float32, device=device)
    return _NONE[key]


def _launch(layers: list, x: torch.Tensor, xhs: list, outs: list, zs: list, loads: int) -> None:
    """rot_in and the linear of 1 to 3 layers of input x (one launch each)."""

    ext = _ext()
    ext.rot_in(x, [lin.suh for lin in layers], xhs, (loads >> 5) & 1)
    none = _none(x.device)
    ext.linear(xhs, [lin.words for lin in layers], [lin.strides[0] for lin in layers],
               [lin.strides[1] for lin in layers], [lin.svh for lin in layers],
               [none if lin.bias is None else lin.bias for lin in layers], outs,
               [z if lin.split[0] > 1 else none for lin, z in zip(layers, zs)], [lin.counters for lin in layers],
               layers[0].k2, CODEBOOK_IDS[layers[0].codebook], [lin.split[0] for lin in layers], layers[0].split[1],
               loads)


def groupable(layers: list) -> bool:
    """Whether layers of one input can run as one launch: 1 to 3 of them, the same K, width, codebook and warps a
    block (their K splits may differ). The launch gives every layer the bits of its own call."""

    a = layers[0]
    return 1 <= len(layers) <= 3 and all(
        lin.k == a.k and lin.k2 == a.k2 and lin.codebook == a.codebook and lin.split[1] == a.split[1]
        and (LOADS if lin.loads is None else lin.loads) == (LOADS if a.loads is None else a.loads) for lin in layers)


def group(layers: list, x: torch.Tensor, outs: list) -> list:
    """outs[j] = layers[j](x, out=outs[j]) in one rot_in and one linear launch when ``groupable`` (else one by one):
    the layers of one input (q_a and kv_a, gate and up, ...). Each output has the bits of the layer's own call."""

    if not groupable(layers) or x.dim() != 2 or not 1 <= x.shape[0] <= 128:
        return [lin(x, out=o) for lin, o in zip(layers, outs)]
    x = x.contiguous()
    m = x.shape[0]
    xhs = [torch.empty((m, lin.k), dtype=torch.float16, device=x.device) for lin in layers]
    zs = [torch.empty((lin.split[0] * m * lin.n,), dtype=torch.float32, device=x.device) if lin.split[0] > 1 else None
          for lin in layers]
    a = layers[0]
    _launch(layers, x, xhs, outs, zs, LOADS if a.loads is None else a.loads)
    return outs


def unpack_cuda(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """W_q [K, N] fp16 from trellis int16 [K/16, N/16, 16 * bits] on the GPU (``decode.cuh``)."""

    bits = fmt.bits_of(trellis.shape)
    words = trellis.cuda().contiguous().view(torch.int32)
    k, n = 16 * trellis.shape[0], 16 * trellis.shape[1]
    w = torch.empty((k, n), dtype=torch.float16, device=words.device)
    tw = 4 * k2_of(bits)
    _ext().unpack(words, w, (n // 16) * tw, 8 * tw, k2_of(bits), CODEBOOK_IDS[codebook])
    return w
