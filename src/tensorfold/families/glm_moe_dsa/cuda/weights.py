"""One rank's share of full GLM-5.3: EXL3 linears and mixed-width routed experts, read straight from the checkpoint.

Reads use safetensors slices, so a rank reads only its rows (row splits) or its whole tiles per row (column splits);
the split rules are ``split.rule``. EXL3 linears become the universal ``Exl3Linear`` (any codebook and width, 1-128
rows) and each MoE layer's routed experts one ``Exl3RoutedExperts`` (a width per expert).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3.linear import Exl3Linear

from ..config import Config
from . import split

EXL3_PARTS = ("trellis", "suh", "svh")


class RankReader:
    """Tensors of one rank's share, from the full checkpoint (safetensors slices)."""

    def __init__(self, model_dir: str | Path, rank: int, world: int) -> None:
        self.dir, self.rank, self.world = Path(model_dir), rank, world
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._open: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.index

    def _file(self, name: str):
        fn = self.index[name]
        if fn not in self._open:
            self._open[fn] = safe_open(str(self.dir / fn), framework="pt", device="cpu")
        return self._open[fn]

    def get(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        f = self._file(name)
        kind = split.rule(name)
        if kind == "rep":
            return f.get_tensor(name).to(device)
        sl = f.get_slice(name)
        shape = list(sl.get_shape())
        split._check_cut(name, kind, shape, self.world)
        axis = 0 if kind == "row" else 1
        n = shape[axis] // self.world
        a, b = self.rank * n, (self.rank + 1) * n
        part = sl[a:b] if axis == 0 else sl[:, a:b]
        return part.contiguous().to(device)

    def codebook(self, prefix: str) -> str:
        return "mul1" if self.has(prefix + ".mul1") else ("mcg" if self.has(prefix + ".mcg") else "3inst")


def load_linear(r: RankReader, prefix: str, device="cuda") -> Exl3Linear:
    """An EXL3 linear (this rank's share), as stored: kv_a keeps its 640 stored outputs (the engine keeps 576)."""
    t = {p: r.get(f"{prefix}.{p}") for p in EXL3_PARTS}
    return Exl3Linear.from_tensors(t["trellis"], t["suh"], t["svh"], r.codebook(prefix), device=device)


def load_experts(r: RankReader, cfg: Config, layer: int, device="cuda",
                 ids: list[int] | None = None) -> x3experts.Exl3RoutedExperts:
    """A MoE layer's routed experts on this rank: gate/up output tiles, down input tiles; a width per expert.
    ``ids`` (tests): only those experts, in that order."""
    base = f"model.layers.{layer}.mlp.experts"
    mats = {"gate": [], "up": [], "down": []}
    for e in (range(cfg.n_routed_experts) if ids is None else ids):
        for p in mats:
            pre = f"{base}.{e}.{p}_proj"
            mats[p].append(tuple(r.get(f"{pre}.{s}", device) if s == "trellis" else r.get(f"{pre}.{s}", device).half()
                                 for s in EXL3_PARTS))
    cb = r.codebook(f"{base}.0.gate_proj")
    return x3experts.prepare(mats["gate"], mats["up"], mats["down"], cb, device=device)


@dataclass
class Layer:
    index: int
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor
    q_a: Exl3Linear
    q_a_norm: torch.Tensor
    kv_a: Exl3Linear
    kv_a_norm: torch.Tensor
    q_b: Exl3Linear                     # 16 heads x (192 nope + 64 rope) on this rank
    kv_b: torch.Tensor                  # bf16 [16 heads x (192 + 256), 512]
    o_proj: Exl3Linear                  # this rank's 16 x 256 inputs -> 6144 (a partial sum)
    indexer: dict | None                # wq_b (EXL3), wk / weights_proj / k_norm (bf16) on full-indexer layers
    router: tuple[torch.Tensor, torch.Tensor] | None      # gate weight, score-correction bias (MoE layers)
    experts: x3experts.Exl3RoutedExperts | None
    shared: dict | None                 # shared expert (MoE) or the dense MLP (layers 0-2): gate, up, down
    extra: dict = field(default_factory=dict)


EXPERTS_IMPL = None      # "shared" (experts_cx: cuda-exl3 stacks shared with TensorFold's kernel) or "tf"; None: auto


def _experts(r: RankReader, cfg: Config, layer: int, device):
    global EXPERTS_IMPL
    if EXPERTS_IMPL is None:
        import os

        from . import experts_cx

        want = os.environ.get("TF_GLM53_EXPERTS", "auto")
        EXPERTS_IMPL = "shared" if want != "tf" and experts_cx.available() else "tf"
    if EXPERTS_IMPL == "shared":
        from .experts_cx import SharedExperts

        return SharedExperts(r, cfg, layer, device)
    return load_experts(r, cfg, layer, device)


def load_layer(r: RankReader, cfg: Config, layer: int, device="cuda", experts: bool = True) -> Layer:
    p = f"model.layers.{layer}"
    bf = lambda n: r.get(n, device)  # noqa: E731
    moe = layer >= cfg.first_k_dense_replace
    mlp = f"{p}.mlp.shared_experts" if moe else f"{p}.mlp"
    idx = None
    if r.has(f"{p}.self_attn.indexer.wq_b.trellis"):
        idx = {"wq_b": load_linear(r, f"{p}.self_attn.indexer.wq_b", device),
               "wk": bf(f"{p}.self_attn.indexer.wk.weight"),
               "weights_proj": bf(f"{p}.self_attn.indexer.weights_proj.weight"),
               "k_norm": (bf(f"{p}.self_attn.indexer.k_norm.weight"), bf(f"{p}.self_attn.indexer.k_norm.bias"))}
    return Layer(
        index=layer, input_norm=bf(f"{p}.input_layernorm.weight"), post_attn_norm=bf(f"{p}.post_attention_layernorm.weight"),
        q_a=load_linear(r, f"{p}.self_attn.q_a_proj", device), q_a_norm=bf(f"{p}.self_attn.q_a_layernorm.weight"),
        kv_a=load_linear(r, f"{p}.self_attn.kv_a_proj_with_mqa", device),
        kv_a_norm=bf(f"{p}.self_attn.kv_a_layernorm.weight"),
        q_b=load_linear(r, f"{p}.self_attn.q_b_proj", device), kv_b=bf(f"{p}.self_attn.kv_b_proj.weight"),
        o_proj=load_linear(r, f"{p}.self_attn.o_proj", device), indexer=idx,
        router=(bf(f"{p}.mlp.gate.weight"), bf(f"{p}.mlp.gate.e_score_correction_bias")) if moe else None,
        experts=_experts(r, cfg, layer, device) if (moe and experts) else None,
        shared={k: load_linear(r, f"{mlp}.{k}_proj", device) for k in ("gate", "up", "down")},
    )


@dataclass
class MtpHead:
    """The MTP layer (index num_hidden_layers): a full decoder layer plus its input projection and norms; the output
    head is the model's lm_head (shared_head.head is not stored)."""
    layer: Layer
    eh_proj: torch.Tensor               # bf16 [hidden, 2 * hidden]: [enorm(embedding) ; hnorm(hidden)] -> hidden
    enorm: torch.Tensor
    hnorm: torch.Tensor
    head_norm: torch.Tensor             # shared_head.norm


def load_mtp(r: RankReader, cfg: Config, device="cuda") -> MtpHead:
    i = cfg.num_hidden_layers
    p = f"model.layers.{i}"
    return MtpHead(layer=load_layer(r, cfg, i, device), eh_proj=r.get(f"{p}.eh_proj.weight", device),
                   enorm=r.get(f"{p}.enorm.weight", device), hnorm=r.get(f"{p}.hnorm.weight", device),
                   head_norm=r.get(f"{p}.shared_head.norm.weight", device))
