"""The fp32 reference on a checkpoint's own weights: EXL3 expert dequant, Engram rows, and layer/model/DSpark drivers.

Dense tensors come from ``dsv41_ref_pack`` (CPU, cached); routed EXL3 experts are decoded by the kernels' own lane
decode on the GPU and rotated there; Engram rows are read with the family's pread reader. Everything the reference
computes stays on the CPU in fp32.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from dsv41_ref_pack import POW2, RefPack
from dsv41_reference import FP32, Mode, Reference, State

from tensorfold.cuda.exl3 import experts as exl3
from tensorfold.families.deepseek_v41 import engram_hash
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.engram_hash import Hasher, TokenMap
from tensorfold.families.deepseek_v41.engram_table import Layout, Reader

HAD = 128
EXPERTS = 64            # dequantized experts kept, the least recently used dropped first
# |W - W_exact| <= EXL3_BOUND * |suh_k| |svh_n| * sum(|W_q| over the 128x128 block) / 128: two 7-add FWHTs and two
# scale products, each a rounding of at most 2^-24 relative (16 in all, one more for the float64 comparator)
EXL3_BOUND = 17 * 2.0**-24
_LAYER = re.compile(r"layers\.(\d+)\.")


def hadamard(x: torch.Tensor, dim: int) -> torch.Tensor:
    """The unnormalized 128x128 Sylvester Hadamard on every 128-block of ``dim``: 7 butterfly stages of fp32 adds."""

    y = x.movedim(dim, -1)
    shape = y.shape
    y = y.reshape(-1, HAD)
    h = 1
    while h < HAD:
        y = y.view(-1, HAD // (2 * h), 2, h)
        y = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), dim=2)
        h *= 2
    return y.reshape(shape).movedim(-1, dim)


def exl3_weight(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str) -> torch.Tensor:
    """W [K, N] fp32 = diag(suh) H_K W_q H_N diag(svh) / 128 on the trellis' device; W_q from ``exl3.dequant``."""

    w = hadamard(hadamard(exl3.dequant(trellis, codebook).float(), 0), 1)
    return w * (suh.float() / HAD)[:, None] * svh.float()[None, :]


class RefWeights:
    """A checkpoint's tensors by name for ``Reference``, the first ``layers`` backbone layers (all when None)."""

    def __init__(self, model_dir: str | Path, layers: int | None = None, *, experts: int = EXPERTS,
                 device: str = "cuda") -> None:
        self.dir = Path(model_dir)
        self.pack = RefPack(self.dir)
        self.cfg = Config.read(self.dir, layers)
        self.device = device
        self.experts = experts
        self._dense: dict[str, torch.Tensor] = {}
        self._lru: OrderedDict[tuple[str, int], tuple[torch.Tensor, ...]] = OrderedDict()

    def __call__(self, name: str) -> torch.Tensor:
        """Tensor ``name`` dequantized to fp32 [out, in] on the CPU, read once."""

        if name not in self._dense:
            found = _LAYER.match(name)
            if found and int(found[1]) >= self.cfg.num_hidden_layers:
                raise KeyError(f"{name}: past the {self.cfg.num_hidden_layers} layers this reference loads")
            if ".experts." in name or ".engram.embed." in name:
                raise KeyError(f"{name}: routed experts come from expert(), Engram rows from engram_rows()")
            self._dense[name] = self.pack.dense_fp32(name.removesuffix(".weight"))
        return self._dense[name]

    def exl3(self, name: str) -> torch.Tensor:
        """EXL3 projection ``name`` (its trellis, suh, svh and codebook marker) as W [K, N] fp32 on the device."""

        codebook = next((cb for cb in ("mcg", "mul1") if self.pack.has(f"{name}.{cb}")), "3inst")
        trellis, suh, svh = (self.pack.tensor(f"{name}.{part}").to(self.device) for part in ("trellis", "suh", "svh"))
        return exl3_weight(trellis, suh, svh, codebook)

    def expert(self, prefix: str, e: int) -> tuple[torch.Tensor, ...]:
        """Expert ``e`` of block ``prefix`` as fp32 (w1, w2, w3), each [out, in] on the CPU (EXL3 or MXFP4)."""

        key = (prefix, e)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        base = f"{prefix}.ffn.experts.{e}"
        if self.pack.has(f"{base}.w1.trellis"):
            mats = tuple(self.exl3(f"{base}.{w}").cpu().T for w in ("w1", "w2", "w3"))
        else:
            mats = tuple(self.pack.dense_fp32(f"{base}.{w}") for w in ("w1", "w2", "w3"))
        self._lru[key] = mats
        if len(self._lru) > self.experts:
            self._lru.popitem(last=False)
        return mats

    @cached_property
    def hasher(self) -> Hasher:
        """Engram's hasher over the checkpoint tokenizer's map (the real vocabulary's map checked by its sha256)."""

        c = self.cfg
        sha = engram_hash.TOKEN_MAP_SHA256 if c.engram_compressed_vocab_size == engram_hash.TOKEN_MAP_SIZE else None
        return Hasher(c, TokenMap.build(self.dir / "tokenizer.json", c.engram_compressed_vocab_size, sha))

    @cached_property
    def reader(self) -> Reader:
        return Reader(Layout.read(self.dir, self.cfg.engram_layer_ids, self.hasher.primes.tolist()))

    def engram_rows(self, layer: int, ids) -> torch.Tensor:
        """bf16 [*ids.shape, head_dim]: table rows ``ids`` (the layer's own) as bf16(e4m3 * 2^(e - 127)) (M:309-317)."""

        r = self.reader
        ids = torch.as_tensor(ids)
        flat = ids.reshape(-1).numpy() + r.layout.starts[self.cfg.engram_layer_ids.index(layer)]
        w = np.empty((flat.size, r.wrow), dtype=np.uint8)
        s = np.empty((flat.size, r.srow), dtype=np.uint8)
        r.gather(flat, w, s)
        v = torch.from_numpy(w).view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
        v = v * POW2[torch.from_numpy(s).long()][..., None]
        return v.flatten(-2).to(torch.bfloat16).view(*ids.shape, -1)

    def reference(self, mode: Mode = FP32, state: State | None = None) -> Reference:
        """A ``Reference`` on these weights, continuing ``state`` when given."""

        ref = Reference(self.cfg, self, expert=self.expert, engram_rows=self.engram_rows,
                        hasher=self.hasher if self.cfg.engram_layer_ids else None, mode=mode)
        if state is not None:
            ref.state = state
        return ref

    def layer(self, L: int, X: torch.Tensor, pre: torch.Tensor, state: State, mode: Mode = FP32,
              tokens=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Block ``L`` on the stream ``X`` [n, hc, D] entering it (before its Engram) and ``pre`` [n, hc].

        Rows sit at ``state.pos`` on; ``tokens`` (Engram layers only) are theirs, after ``state.tokens``. The state
        takes the block's caches; advancing ``pos`` and ``tokens`` after the forward's last layer is the caller's.
        """

        c, ref = self.cfg, self.reference(mode, state)
        X = X.float()
        if c.roles[L].engram:
            if tokens is None:
                raise ValueError(f"layer {L} adds Engram rows: pass the rows' tokens")
            ids = self.hasher.ids(state.tokens[-(c.engram_max_ngram_size - 1):], list(tokens))
            X = ref.engram(L, X, torch.as_tensor(ids[:, c.engram_layer_ids.index(L)]))
        return ref.block(L, X, pre.float(), state.pos)

    def model(self, ids, mode: Mode = FP32, state: State | None = None,
              all_logits: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, State]:
        """The reduced model (the kept layers, final norm, head) on ``ids`` at ``state.pos``: (logits, taps, state)."""

        ref = self.reference(mode, state)
        logits, taps = ref.forward([int(t) for t in ids], all_logits)
        return logits, taps, ref.state

    def dspark(self, main_hidden: torch.Tensor, ids, p: int, mode: Mode = FP32, state: State | None = None,
               sample=None) -> tuple[list[int], torch.Tensor, torch.Tensor, State]:
        """DSpark at position ``p``: absorb ``main_hidden`` (taps of rows p-n+1..p), then draft from ``ids`` (y).

        Returns (drafts for p+2.., block logits with the Markov bias, confidence logits, state).
        """

        state = state if state is not None else State(pos=p + 1)
        if state.pos != p + 1:
            raise ValueError(f"the state is committed to {state.pos}; DSpark at p={p} needs {p + 1}")
        ref = self.reference(mode, state)
        ref.absorb(main_hidden.float(), p + 1 - len(main_hidden))
        drafts, logits, conf = ref.propose(int(torch.as_tensor(ids).reshape(-1)[0]), sample)
        return drafts, logits, conf, state
