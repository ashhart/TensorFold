"""EXL3 linear layers on CUDA for every codebook and width (``linear.cu``): y = x @ W + bias for 1 to 128 rows,
row-invariant, no cuBLAS.

    layer = Exl3Linear.from_tensors(trellis, suh, svh, codebook="mul1")      # or .load(checkpoint, prefix)
    y = layer(x)                                                              # x [M, K] fp16/bf16/fp32 -> y [M, N]

Two kernels compute the layer. ``rot_in`` rotates the input once: xh = fp16((x * suh) @ H / sqrt(128)), rounded to
fp16 as ExLlamaV3 does. ``linear`` gives each program 128 output columns (one Hadamard block) and a fixed slice of
K; its warps read their k tiles with per-lane cached loads, decode them into tensor-core fragments
(``decode.cuh``), multiply in fp32 and add their sums in warp order, and the program (or, with K split over several
programs, the last of them to finish) rotates the outputs, scales by svh and adds the bias. The splits and warps
depend only on (K, N) (``plan``), so a row's bits never depend on the other rows.

At load the tiles are copied, not changed, into column strips: all k tiles of a 128-column block in k order
(``strips``), so each warp reads one contiguous run of words. ``layout="stored"`` reads the checkpoint's order
instead; both give the same bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from . import format as fmt

CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_linear_v2", sources=[str(here / "linear.cpp"), str(here / "linear.cu")],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False)


def k2_of(bits: float) -> int:
    return int(2 * fmt.check_bits(bits))


def plan(k: int, n: int, blocks: int = 192, min_tiles: int = 8) -> tuple[int, int]:
    """(K splits, warps a program) for a K x N layer, measured on a DGX Spark (docs/recipes/exl3.md): wide layers give
    every program one 128-column block and eight warps, narrow ones split K to reach ``blocks`` programs, and a warp
    always keeps at least ``min_tiles`` k tiles. A function of the shape only, never of the rows: a layer keeps its
    plan for every call, which is what makes its rows independent."""

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

    @classmethod
    def from_tensors(cls, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str,
                     bias: torch.Tensor | None = None, device: str | torch.device = "cuda",
                     layout: str = "strips") -> "Exl3Linear":
        """From a group's tensors: trellis int16 [K/16, N/16, 16 * bits]; suh [K] and svh [N] fp16 (or the packed
        int16 sign words su [K/16] and sv [N/16] of older checkpoints); codebook "3inst", "mcg" or "mul1"."""

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
        """y [M, N] = x [M, K] @ W + bias, M = 1..128; fp16/bf16/fp32 in and out (out_dtype defaults to x's).
        Scratch, allocated when not given: ``xh`` fp16 [M, K]; ``z`` fp32, at least SK * M * N (SK > 1 only)."""

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
        sk_stride, nb_stride = self.strides
        ext = _ext()
        ext.rot_in(x, self.suh, xh)
        ext.linear(xh, self.words, sk_stride, nb_stride, self.svh, self.bias, out, z if sk > 1 else None,
                   self.counters, self.k2, CODEBOOK_IDS[self.codebook], sk, wk)
        return out

    def unpack(self) -> torch.Tensor:
        """W_q [K, N] fp16 decoded on the GPU (the stored layout's words are rebuilt when needed)."""

        words = self.words
        if self.layout == "strips":
            nb, kt, _, tw = words.shape
            words = words.permute(1, 0, 2, 3).reshape(kt, nb * 8, tw).contiguous()
        w = torch.empty((self.k, self.n), dtype=torch.float16, device=words.device)
        _ext().unpack(words, w, self.k2, CODEBOOK_IDS[self.codebook])
        return w


def unpack_cuda(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """W_q [K, N] fp16 from trellis int16 [K/16, N/16, 16 * bits] on the GPU (``decode.cuh``)."""

    bits = fmt.bits_of(trellis.shape)
    words = trellis.cuda().contiguous().view(torch.int32)
    w = torch.empty((16 * trellis.shape[0], 16 * trellis.shape[1]), dtype=torch.float16, device=words.device)
    _ext().unpack(words, w, k2_of(bits), CODEBOOK_IDS[codebook])
    return w
