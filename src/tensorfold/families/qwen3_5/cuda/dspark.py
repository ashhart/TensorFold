"""DSpark proposals on CUDA (speculators' ``dspark``, as vLLM runs it): DFlash v1's backbone, then a Markov head."""
# speculators index target layers as aux ids (the state entering layer i); taps are layer outputs, so i - 1

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.engine.exact_sampling import Sampling

from .dflash1 import DFlash1
from .glue import embedding
from .weights import Plain, Weights


def is_dspark(cfg: dict) -> bool:
    return (str(cfg.get("speculators_model_type", "")) == "dspark"
            or "Qwen3DSparkModel" in (cfg.get("architectures") or []))


class DSpark(DFlash1):
    """A speculators DSpark drafter: one window a layer, its own embedding and draft-vocabulary head, a Markov head."""

    def __init__(self, draft_dir: str | Path, target: Weights, bits: int | None = None, block: int | None = None,
                 fast: bool = True, rank: int = 0, world: int = 1):
        import os

        bits = int(os.environ.get("TF_DFLASH_BITS", "4")) if bits is None else int(bits)
        if world != 1:
            raise ValueError("a DSpark drafter runs on one GPU")
        path = Path(draft_dir)
        cfg = json.loads((path / "config.json").read_text())
        layer = cfg["transformer_layer_config"]
        if not cfg.get("sample_from_anchor", False):
            raise ValueError("this DSpark drafter predicts from masks only (sample_from_anchor false): not read yet")
        self.rank, self.world = 0, 1
        self.hidden = int(layer["hidden_size"])
        if self.hidden != target.config.hidden:
            raise ValueError(f"the drafter's width {self.hidden} is not the target's {target.config.hidden}")
        self.head_dim = int(layer["head_dim"])
        self.heads = int(layer["num_attention_heads"])
        self.kv_heads = int(layer["num_key_value_heads"])
        self.eps = float(layer["rms_norm_eps"])
        rope = layer.get("rope_parameters") or {}
        self.theta = float(layer.get("rope_theta", rope.get("rope_theta", 10000.0)))
        self.mask_id = int(cfg["mask_token_id"])
        self.taps = tuple(int(i) - 1 for i in cfg["aux_hidden_state_layer_ids"])
        self.trained = int(cfg.get("block_size", 8))
        self.block = min(int(block or self.trained), self.trained)
        self.layers = int(layer["num_hidden_layers"])
        kinds = layer.get("layer_types") or ["full_attention"] * self.layers
        from .dflash1 import FULL_CONTEXT

        sliding = layer.get("sliding_window")
        self.windows = [int(sliding) - 1 if k == "sliding_attention" else FULL_CONTEXT for k in kinds]
        # speculators: sliding layers causal in the block unless sliding_window_non_causal (vLLM's dflash causal)
        causal = not cfg.get("sliding_window_non_causal", True)
        self.causal = [causal if k == "sliding_attention" else False for k in kinds]
        self.window = max(self.windows)
        self.is_causal = True
        self.group_size = 0
        self.device = target.norm.device
        self.weights = {}
        f = SafeTensors(sorted(path.glob("*.safetensors")))
        for name in f.keys():
            self.weights[name] = f.get(name)
        del f
        self.inv_freq = (1.0 / self.theta ** (torch.arange(self.head_dim // 2, device=self.device,
                                                          dtype=torch.float32) * 2 / self.head_dim))
        # its own embedding (the masks' row included) and its own draft-vocabulary head
        self.target_embed = Plain(self.weights.pop("embed_tokens.weight").to(self.device, torch.bfloat16).contiguous())
        self.d2t = self.weights.pop("d2t").to(self.device, torch.int64)
        self.weights.pop("t2d", None)
        for name in [n for n in self.weights if n.startswith("confidence_head.")]:
            del self.weights[name]
        self.markov_w1 = self.weights.pop("markov_head.markov_w1.weight").to(self.device, torch.bfloat16).contiguous()
        self.markov_w2 = self.weights.pop("markov_head.markov_w2.weight").to(self.device, torch.bfloat16).contiguous()
        self.heads_local, self.kv_local = self.heads, self.kv_heads
        self.sub_parts = self.head_cols = self.sub_rows = None
        self.vocab_spans = ((0, int(target.config.vocab)),)
        self._weights_v1(bits)               # packs lm_head [draft vocab, hidden] too

    def in_vocab(self, token: int) -> bool:
        return True

    @torch.no_grad()
    def launch_blocks(self, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
        """The backbone over every live stream's block, then the Markov chain left to right, each stream's in turn."""

        out: list = [None] * len(snaps)
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live or max_nodes < 1:
            return out
        length = min(self.block, max_nodes)
        tokens = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                              dtype=torch.int32, device=self.device)
        x = embedding(tokens, self.target_embed)
        ctx = [snaps[i] for i in live]
        rot = [self._rotary(snap[3], length) for snap in ctx]
        cos, sin = torch.cat([c for c, _ in rot]), torch.cat([s for _, s in rot])
        for layer in range(self.layers):
            x = self._layer_v1(layer, x, cos, sin, ctx, length)
        h = F.rms_norm(x, (self.hidden,), self.weights["norm.weight"], self.eps).contiguous()
        base = self._lin(h, "lm_head.weight").float().view(len(live), length, -1)   # draft-vocabulary logits
        prev = torch.tensor([pendings[i] for i in live], dtype=torch.int64, device=self.device)
        ids = torch.empty((len(live), length), dtype=torch.int64, device=self.device)
        logp = torch.empty((len(live), length), dtype=torch.float32, device=self.device)
        for i in range(length):              # the sequential stage: each position biased by the token before it
            logits = base[:, i] + F.linear(self.markov_w1[prev], self.markov_w2).float()
            pick = logits.argmax(dim=-1)
            logp[:, i] = torch.log_softmax(logits, dim=-1).gather(1, pick[:, None])[:, 0]
            prev = pick + self.d2t[pick]
            ids[:, i] = prev
        shared = [ids, logp, None]
        for j, i in enumerate(live):
            out[i] = (shared, j, length, int(pendings[i]))
        return out

    def finish_tree(self, launched, context_length: int, max_nodes: int,
                    sampling: Sampling | None = None) -> tuple[list[int], list[int], list[float]]:
        """A chain: the Markov head's tokens in order, each the previous one's child; scores are path -log p."""

        if launched is None:
            return [], [], []
        shared, row, length, _ = launched
        if shared[2] is None:
            shared[2] = (shared[0].cpu().numpy(), shared[1].cpu().numpy().astype(np.float64))
        ids, logp = shared[2][0][row], shared[2][1][row]
        n = min(length, max_nodes)
        self.last_candidates = ids[:n, None]
        scores = list(np.cumsum(-logp[:n]))
        return [int(t) for t in ids[:n]], list(range(-1, n - 1)), [float(s) for s in scores]
