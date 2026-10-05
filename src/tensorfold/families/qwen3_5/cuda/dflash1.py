"""DFlash (v1) proposals on CUDA: z-lab's block drafter over target taps; rounding moves acceptance, never output."""
# Plain Qwen3 layers over fc + hidden_norm of the taps; sliding layers causal in the block, full ones not (z-lab's).

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.engine.exact_sampling import Sampling

from .dflash2 import DFlash2, _norm, _rope
from .draft_config import architecture, target_layers
from .draft_attention import append, block_attention
from .draft_tree import best_first
from .glue import embedding, swiglu
from .qmm import group_sums
from .qmm_fast import matmul, matmul_group, matmul_rows
from .weights import Weights

# a full-attention layer's context: z-lab keeps all of it; here its last FULL_CONTEXT rows (drafts only)
FULL_CONTEXT = int(os.environ.get("TF_DFLASH_FULL_CONTEXT", "32768"))
WIDE = 1 << 30                       # block_attention's window for a full layer: every context row is seen
CANDIDATES = 16                      # top ids a depth the tree policy weighs, as DFlash2's


def drafter_class(draft_dir: str | Path):
    """DFlash2 or DFlash1, by the drafter checkpoint's architecture."""

    cfg = json.loads((Path(draft_dir) / "config.json").read_text())
    return DFlash1 if architecture(cfg) == "DFlashDraftModel" else DFlash2


class DFlash1(DFlash2):
    """A DFlash (v1) drafter: context taps at the drafter's target layers, one window a layer, a 16-row block."""

    def __init__(self, draft_dir: str | Path, target: Weights, bits: int | None = None, block: int | None = None,
                 fast: bool = True, rank: int = 0, world: int = 1):
        bits = int(os.environ.get("TF_DFLASH_BITS", "4")) if bits is None else int(bits)   # 4, or 0: bf16
        if world != 1:
            raise ValueError("a DFlash (v1) drafter runs on one GPU")
        path = Path(draft_dir)
        cfg = json.loads((path / "config.json").read_text())
        dflash = cfg["dflash_config"]
        self.rank, self.world = 0, 1
        self.hidden = int(cfg["hidden_size"])
        if self.hidden != target.config.hidden:
            raise ValueError(f"the drafter's width {self.hidden} is not the target's {target.config.hidden}")
        self.head_dim = int(cfg["head_dim"])
        self.heads = int(cfg["num_attention_heads"])
        self.kv_heads = int(cfg["num_key_value_heads"])
        self.eps = float(cfg["rms_norm_eps"])
        rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        self.theta = float(cfg.get("rope_theta", rope.get("rope_theta", 10000.0)))
        self.mask_id = int(dflash["mask_token_id"])
        self.taps = target_layers(cfg)
        self.trained = int(dflash.get("block_size", cfg.get("block_size", 16)))
        self.block = int(block or self.trained)
        self.layers = int(cfg["num_hidden_layers"])
        kinds = cfg.get("layer_types") or ["full_attention"] * self.layers
        sliding = cfg.get("sliding_window")
        # z-lab: a sliding layer keeps window - 1 context rows and is causal in the block; a full one sees them all
        self.windows = [int(sliding) - 1 if k == "sliding_attention" else FULL_CONTEXT for k in kinds]
        self.causal = [k == "sliding_attention" if cfg.get("is_causal") is None else bool(cfg["is_causal"])
                       for k in kinds]
        self.window = max(self.windows)          # rows a prefill taps (prefill.prefill_state)
        self.is_causal = True
        self.group_size = 0
        self.target_embed = target.embed
        self.device = target.norm.device
        self.weights: dict[str, torch.Tensor] = {}
        f = SafeTensors(sorted(path.glob("*.safetensors")))
        for name in f.keys():
            self.weights[name] = f.get(name)           # on the host until packed
        del f
        self.inv_freq = (1.0 / self.theta ** (torch.arange(self.head_dim // 2, device=self.device,
                                                          dtype=torch.float32) * 2 / self.head_dim))
        self._head(target, 0, 1)
        from .weights import QLinear

        if isinstance(target.head, QLinear) and target.head.layout == "tiled" and self.sub_rows is None:
            from .qmm_fast import tile

            self.sub_head = tile(self.sub_head)
        self.heads_local, self.kv_local = self.heads, self.kv_heads
        self._weights_v1(bits)
        # an all-zero edge model: best_first's edge term vanishes and the tree follows the drafter's own scores
        self._zero = np.zeros((1, 1), dtype=np.float32)

    def _weights_v1(self, bits: int) -> None:
        """q, k and v fused (k and v alone for the context), every large projection packed to 4 bits."""

        from .affine_memory import packed_draft
        from .dflash2 import quantize4
        from .qmm_fast import tile

        self.fast = bits == 4
        for layer in range(self.layers):
            prefix = f"layers.{layer}.self_attn."
            q, k, v = (self.weights[prefix + n] for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
            self.weights[prefix + "qkv.weight"] = torch.cat((q, k, v)).contiguous()
            self.weights[prefix + "kv.weight"] = torch.cat((k, v)).contiguous()
            for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                del self.weights[prefix + n]
        self.q4 = {}
        if bits == 4:
            for name in list(self.weights):
                t = self.weights[name]
                if packed_draft(name, t.shape):
                    self.q4[name] = tile(quantize4(t.to(self.device, torch.bfloat16)))
                    del self.weights[name]
        for name, t in self.weights.items():
            self.weights[name] = t.to(self.device, torch.bfloat16) if t.is_floating_point() else t.to(self.device)
        torch.cuda.empty_cache()
        self.kc: list[torch.Tensor | None] = [None] * self.layers
        self.vc: list[torch.Tensor | None] = [None] * self.layers
        self.context_len = 0
        self.context_end = 0

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        q = self.q4.get(name)
        if q is None:
            return F.linear(x.to(torch.bfloat16), self.weights[name])
        return matmul(x.to(torch.bfloat16).contiguous(), q)

    def _row(self, x: torch.Tensor, name: str, xs: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q4.get(name)
        if q is None:
            return F.linear(x.to(torch.bfloat16), self.weights[name])
        return matmul(x.contiguous(), q, xs)

    def _context(self, taps: torch.Tensor) -> torch.Tensor:
        if taps.ndim != 2 or taps.shape[1] != len(self.taps) * self.hidden:
            raise ValueError(f"DFlash expects {len(self.taps)} target layer taps per committed row")
        return _norm(self._lin(taps, "fc.weight"), self.weights["hidden_norm.weight"], self.eps)

    @torch.no_grad()
    def add_taps(self, taps: torch.Tensor) -> None:
        projected = self._context(taps)
        n = projected.shape[0]
        cos, sin = self._rotary(self.context_end, n)
        for layer in range(self.layers):
            _, k, v = self._prep(self._lin(projected, f"layers.{layer}.self_attn.kv.weight"), layer, cos, sin, 0)
            keep = self.windows[layer]
            kc, vc = self.kc[layer], self.vc[layer]
            kc = k if kc is None else torch.cat((kc, k), dim=1)
            vc = v if vc is None else torch.cat((vc, v), dim=1)
            self.kc[layer] = kc[:, -keep:].contiguous()
            self.vc[layer] = vc[:, -keep:].contiguous()
        self.context_len = min(self.window, self.context_len + n)
        self.context_end += n

    @torch.no_grad()
    def add_taps_streams(self, snaps: list, taps: list[torch.Tensor]) -> list:
        """``add_taps`` for several streams, each projection run once over all their rows."""

        sizes = [t.shape[0] for t in taps]
        projected = self._context(torch.cat(taps) if len(taps) > 1 else taps[0])
        pos = torch.tensor([p for snap, n in zip(snaps, sizes) for p in range(snap[3], snap[3] + n)],
                           dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        phase = pos[:, None] * self.inv_freq[None, :]
        cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
        kcs, vcs = [list(snap[0]) for snap in snaps], [list(snap[1]) for snap in snaps]
        for layer in range(self.layers):
            _, k, v = self._prep(self._lin(projected, f"layers.{layer}.self_attn.kv.weight"), layer, cos, sin, 0)
            for cache, fresh in ((kcs, k), (vcs, v)):
                for c, out in zip(cache, append(fresh, [c[layer] for c in cache], sizes, self.windows[layer])):
                    c[layer] = out
        return [(kc, vc, min(self.window, snap[2] + n), snap[3] + n) for kc, vc, snap, n in zip(kcs, vcs, snaps, sizes)]

    def _layer_v1(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: list,
                  length: int) -> torch.Tensor:
        """One Qwen3 layer over several streams' blocks; each block attends its own context in this layer's window."""

        w = self.weights
        base = f"layers.{i}."
        normed = F.rms_norm(x, (self.hidden,), w[base + "input_layernorm.weight"], self.eps)
        q, k, v = self._prep(self._lin(normed, base + "self_attn.qkv.weight"), i, cos, sin, self.heads)
        out = block_attention(q, k, v, [snap[0][i] for snap in ctx], [snap[1][i] for snap in ctx], length,
                              WIDE if self.windows[i] >= FULL_CONTEXT else self.windows[i], self.head_dim ** -0.5,
                              self.causal[i])
        x = (x.float() + self._row(out, base + "self_attn.o_proj.weight").float()).to(torch.bfloat16)
        normed = F.rms_norm(x, (self.hidden,), w[base + "post_attention_layernorm.weight"], self.eps)
        if base + "mlp.gate_proj.weight" in self.q4:
            xs = group_sums(normed)
            act, act_xs = swiglu(*matmul_group(normed, [self.q4[base + "mlp.gate_proj.weight"],
                                                        self.q4[base + "mlp.up_proj.weight"]], xs))
        else:                                    # a bf16 drafter (TF_DFLASH_BITS=0): as trained
            gate, up = self._lin(normed, base + "mlp.gate_proj.weight"), self._lin(normed, base + "mlp.up_proj.weight")
            act, act_xs = (F.silu(gate.float()) * up.float()).to(torch.bfloat16), None
        return (x.float() + self._row(act, base + "mlp.down_proj.weight", act_xs).float()).to(torch.bfloat16)

    @torch.no_grad()
    def launch_blocks(self, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
        """GPU half of ``propose_tree`` for several streams: each block's top candidates and their log-probabilities."""

        out: list = [None] * len(snaps)
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live or max_nodes < 1:
            return out
        length = min(block or self.block, max_nodes + 1)
        tokens = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                              dtype=torch.int32, device=self.device)
        x = embedding(tokens, self.target_embed)
        ctx = [snaps[i] for i in live]
        rot = [self._rotary(snap[3], length) for snap in ctx]
        cos, sin = torch.cat([c for c, _ in rot]), torch.cat([s for _, s in rot])
        for layer in range(self.layers):
            x = self._layer_v1(layer, x, cos, sin, ctx, length)
        h = F.rms_norm(x.view(len(live), length, -1)[:, 1:].reshape(-1, self.hidden), (self.hidden,),
                       self.weights["norm.weight"], self.eps).contiguous()
        if self.sub_parts is not None:
            logits = torch.cat([part(h)[:, lo:hi] for part, lo, hi in self.sub_parts], dim=1)
        elif self.head_cols is not None:
            logits = self.sub_head(h).index_select(1, self.head_cols)
        elif self.sub_rows is not None:
            logits = matmul_rows(h, self.sub_rows)
        else:
            logits = matmul(h, self.sub_head)
        logp = torch.log_softmax(logits.float(), dim=-1)
        values, local_ids = torch.topk(logp, k=CANDIDATES, dim=-1, sorted=False)
        shared = [self.head_ids[local_ids], values, None]       # read back once, by the first finish
        for j, i in enumerate(live):
            out[i] = (shared, j * (length - 1), length - 1, int(pendings[i]))
        return out

    def finish_tree(self, launched, context_length: int, max_nodes: int,
                    sampling: Sampling | None = None) -> tuple[list[int], list[int], list[float]]:
        """The tree policy over the drafter's own scores (DFlash2's with its edge model zeroed)."""

        if launched is None:
            return [], [], []
        shared, start, rows, pending = launched
        if shared[2] is None:
            shared[2] = (shared[0].cpu().numpy().astype(np.int64), shared[1].cpu().numpy().astype(np.float64))
        ids, scores = shared[2][0][start:start + rows], shared[2][1][start:start + rows]
        self.last_candidates = ids
        zero = np.zeros((ids.shape[0], 1), dtype=np.float64)
        table = _ZeroTable()
        return best_first(ids, scores, zero, table, table, pending, min(127, max_nodes), sampling, context_length)


class _ZeroTable:
    """An edge codebook of zeros for any token: ``table[ids]`` -> zeros [*ids.shape, 1]."""

    def __getitem__(self, ids):
        return np.zeros((*np.shape(ids), 1), dtype=np.float64)
