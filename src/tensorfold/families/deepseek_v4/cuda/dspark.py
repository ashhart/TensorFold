"""Native DSpark GGUF execution using the target's TensorFold CUDA backbone pieces."""

from __future__ import annotations

import torch

from tensorfold.cuda.gguf import Weights
from tensorfold.cuda.gguf.linear import linear

from .attention import rope
from .model import Layer, norm


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
        self.w = {
            n[len("dspark.") :]: self.weights.tensor(n)
            for n in self.weights.inventory
            if n.startswith("dspark.") and n.split(".")[1] not in [str(i) for i in range(count)]
        }
        self.model = model

    def state(self):
        return [layer.state() for layer in self.layers]

    def absorb(self, taps, state):
        x = norm(linear(taps, self.w["main_proj.weight"]), self.w["main_norm.weight"], self.layers[0].eps)
        for layer, cache in zip(self.layers, state):
            pos = torch.arange(cache.offset, cache.offset + x.shape[0], device=x.device)
            kv = rope(
                norm(linear(x, layer.w["attn_kv.weight"]), layer.w["attn_kv_a_norm.weight"], layer.eps), pos, layer.freq
            )
            raw = torch.cat((cache.keys, kv))
            cache.keys = raw[-128:].clone()
            cache.offset += x.shape[0]

    def logits(self, token, state):
        ids = torch.tensor([token] + [self.noise] * (self.size - 1), device=self.model.embed.device, dtype=torch.long)
        x = self.model.embed[ids].to(torch.bfloat16)[:, None].expand(-1, 4, -1).reshape(self.size, -1).contiguous()
        for layer, cache in zip(self.layers, state):
            x = layer(x, ids, cache, draft=True)
        return self.model.head(
            x,
            fn=self.w["hc_head_fn.weight"].float(),
            base=self.w["hc_head_base.weight"].float(),
            scale=self.w["hc_head_scale.weight"].float(),
            weight=self.w["norm.weight"],
        )

    def markov(self, previous):
        return linear(
            self.w["markov_w1.weight"][previous : previous + 1].float(), self.w["markov_w2.weight"], dtype=torch.float32
        )[0]
