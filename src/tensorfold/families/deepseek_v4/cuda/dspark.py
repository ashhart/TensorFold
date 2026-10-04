"""Native DSpark GGUF execution using the target's TensorFold CUDA backbone pieces."""

from __future__ import annotations

import torch

from tensorfold.cuda.gguf import Weights
from tensorfold.cuda.gguf.linear import linear

from .attention import rope
from .model import Layer, norm

WINDOW = 128


class DSpark:
    def __init__(self, path, model):
        self.weights = Weights(path)
        try:
            self._load(model)
        finally:
            self.weights.close()

    def _load(self, model):
        meta = self.weights.metadata
        if meta.get("general.architecture") != "deepseek4-dspark":
            raise ValueError("drafter must be a deepseek4-dspark GGUF")
        self.size = int(meta["deepseek4.dspark.block_size"])
        self.noise = int(meta["deepseek4.dspark.noise_token_id"])
        self.taps = tuple(meta["deepseek4.dspark.target_layers"])
        model.tap_layers = self.taps
        count = int(meta["deepseek4.dspark.layer_count"])
        self.layers = [Layer(self.weights, f"dspark.{i}", model.meta, 0, model.limit) for i in range(count)]
        blocks = {str(i) for i in range(count)}
        self.w = {
            n[len("dspark.") :]: self.weights.tensor(n)
            for n in self.weights.inventory
            if n.startswith("dspark.") and n.split(".")[1] not in blocks
        }
        self.model = model
        self.ids = torch.full((self.size,), self.noise, device=model.embed.device, dtype=torch.long)
        self.graph = None
        self.graph_state = None
        for name in ("hc_head_fn.weight", "hc_head_base.weight", "hc_head_scale.weight"):
            self.w[name] = self.w[name].float()

    def state(self):
        return [layer.state() for layer in self.layers]

    def absorb(self, taps, state):
        x = norm(linear(taps, self.w["main_proj.weight"]), self.w["main_norm.weight"], self.layers[0].eps)
        for layer, cache in zip(self.layers, state):
            pos = layer.weights._positions[cache.offset : cache.offset + x.shape[0]]
            kv = rope(
                norm(linear(x, layer.w["attn_kv.weight"]), layer.w["attn_kv_a_norm.weight"], layer.eps), pos, layer.freq
            )
            # Ping-pong fixed windows: no concatenation or new allocation during decode.
            spare = getattr(cache, "_spare", None)
            if spare is None:
                spare = torch.empty((WINDOW, kv.shape[1]), device=kv.device, dtype=kv.dtype)
            count = min(WINDOW, kv.shape[0])
            keep = min(cache.keys.shape[0], WINDOW - count)
            if keep:
                spare[:keep].copy_(cache.keys[-keep:])
            spare[keep:keep + count].copy_(kv[-count:])
            old = cache.keys
            cache.keys = spare[:keep + count]
            cache._spare = old if old.shape[0] == WINDOW else None
            cache.offset += x.shape[0]

    def _logits(self, state):
        x = self.model.embed[self.ids].to(torch.bfloat16)[:, None].expand(-1, 4, -1).reshape(self.size, -1).contiguous()
        for layer, cache in zip(self.layers, state):
            x = layer(x, self.ids, cache, draft=True)
        return self.model.head(
            x,
            fn=self.w["hc_head_fn.weight"],
            base=self.w["hc_head_base.weight"],
            scale=self.w["hc_head_scale.weight"],
            weight=self.w["norm.weight"],
        )

    def logits(self, token, state):
        self.ids[:1].fill_(token)
        # Short prompts keep their true attention span. A full window has fixed
        # geometry; keys and absolute RoPE positions remain live graph inputs.
        if not all(cache.keys.shape[0] == WINDOW for cache in state):
            return self._logits(state)
        if self.graph_state is None:
            self.positions = torch.empty_like(self.ids)
            self.graph_state = [layer.state() for layer in self.layers]
            for live, cache in zip(self.graph_state, state):
                live.keys = torch.empty_like(cache.keys)
                live.offset = cache.offset
                live._positions = self.positions
        start = state[0].offset
        self.positions.copy_(self.weights._positions[start : start + self.size])
        for live, cache in zip(self.graph_state, state):
            live.keys.copy_(cache.keys)
        if self.graph is None:
            for _ in range(2):
                self._logits(self.graph_state)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.graph_logits = self._logits(self.graph_state)
        self.graph.replay()
        return self.graph_logits

    def markov(self, previous):
        # Int or 1-element CUDA index: proposal sampling can keep the chain on-device.
        if isinstance(previous, torch.Tensor):
            row = self.w["markov_w1.weight"].index_select(0, previous.to(dtype=torch.long).reshape(-1)[:1])
        else:
            row = self.w["markov_w1.weight"][int(previous) : int(previous) + 1]
        return linear(row.float(), self.w["markov_w2.weight"], dtype=torch.float32)[0]
