"""Kolibri 1's fp32 reference forward, after Aleph Alpha's vLLM plugin: weights dequantized from FP8 per matmul."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


class Kolibri:
    def __init__(self, model_dir: str | Path, device: str = "cuda") -> None:
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.dev = device
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.w: dict[str, torch.Tensor] = {}
        for shard in sorted(set(index.values())):
            with safe_open(str(self.dir / shard), framework="pt", device=device) as f:
                for name in f.keys():
                    self.w[name] = f.get_tensor(name)
        c = self.cfg
        self.L, self.H = c["num_hidden_layers"], c["num_attention_heads"]
        self.HK, self.D = c["num_key_value_heads"], c["head_dim"]
        self.eps, self.window, self.top_k = c["rms_norm_eps"], c["sliding_window"], c["num_experts_per_tok"]
        self.full = [t == "full_attention" for t in c["layer_types"]]
        theta = float((c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta", 10000.0)))
        self.inv_freq = 1.0 / theta ** (torch.arange(0, self.D, 2, device=device, dtype=torch.float64) / self.D)
        self.caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.L

    # -- pieces ----------------------------------------------------------------------------------------------------
    def weight(self, name: str) -> torch.Tensor:
        """fp32 [N, K]: the FP8 bytes times their block's scale, or the stored bf16 weight."""

        w = self.w[name + ".weight"]
        s = self.w.get(name + ".weight_scale_inv")
        if s is None:
            return w.float()
        n, k = w.shape
        full = s.float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        return w.float() * full

    def linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x @ self.weight(name).T

    def norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w[name + ".weight"].float()

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Neox halves: x [T, heads, D] at positions ``pos`` [T]."""

        ang = (pos.double()[:, None] * self.inv_freq[None]).float()           # [T, D/2]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1)[:, None], torch.cat([ang.sin(), ang.sin()], -1)[:, None]
        half = self.D // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * cos + rot * sin

    def attention(self, i: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{i}.self_attn."
        t = x.shape[0]
        q = self.linear(x, p + "q_proj").view(t, self.H, self.D)
        k = self.linear(x, p + "k_proj").view(t, self.HK, self.D)
        v = self.linear(x, p + "v_proj").view(t, self.HK, self.D)
        q, k = self.norm(q, p + "q_norm"), self.norm(k, p + "k_norm")
        if not self.full[i]:
            q, k = self.rope(q, pos), self.rope(k, pos)
        old = self.caches[i]
        if old is not None:
            k, v = torch.cat([old[0], k]), torch.cat([old[1], v])
        self.caches[i] = (k, v)
        kpos = torch.arange(k.shape[0], device=x.device)
        allowed = kpos[None] <= pos[:, None]
        if not self.full[i]:
            allowed &= pos[:, None] - kpos[None] < self.window
        g = self.H // self.HK
        kk, vv = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)            # [S, H, D]
        scores = torch.einsum("thd,shd->hts", q, kk) * self.D ** -0.5
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        out = torch.einsum("hts,shd->thd", scores.softmax(-1), vv).reshape(t, self.H * self.D)
        return self.linear(out, p + "o_proj")

    def mlp(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        act = torch.nn.functional.silu(self.linear(x, prefix + "gate_proj")) * self.linear(x, prefix + "up_proj")
        return self.linear(act, prefix + "down_proj")

    def moe(self, i: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = f"model.layers.{i}.mlp."
        logits = x @ self.w[p + "gate.weight"].float().T                          # [T, E] fp32
        bias = self.w[f"model.layers.{i}.moe.router.expert_bias"].float()
        ids = torch.topk(logits + bias, self.top_k, dim=-1).indices
        weights = torch.sigmoid(logits.gather(1, ids))
        out = self.mlp(x, p + "shared_experts.")
        for e in ids.unique().tolist():
            rows, slot = (ids == e).nonzero(as_tuple=True)
            out[rows] += weights[rows, slot, None] * self.mlp(x[rows], f"{p}experts.{e}.")
        return out, ids

    # -- model -----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, tokens: list[int], start: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """fp32 logits [T, vocab] for ``tokens`` at positions start.., extending the caches; each layer's routing."""

        pos = torch.arange(start, start + len(tokens), device=self.dev)
        res = self.w["model.embed_tokens.weight"][torch.tensor(tokens, device=self.dev)].float()
        routes = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            a = self.norm(self.attention(i, self.norm(res, p + "input_layernorm"), pos), p + "post_attn_norm")
            res = res + a
            m, ids = self.moe(i, self.norm(res, p + "post_attention_layernorm"))
            res = res + self.norm(m, p + "post_ffn_norm")
            routes.append(ids)
        return self.norm(res, "model.norm") @ self.w["lm_head.weight"].float().T, routes

    def reset(self) -> None:
        self.caches = [None] * self.L

    @torch.no_grad()
    def greedy(self, prompt: list[int], new: int, eos: int | None = None) -> tuple[list[int], list[torch.Tensor]]:
        """Greedy reply (ties to the lower id) and each step's logits row."""

        self.reset()
        logits, _ = self.forward(prompt, 0)
        rows, out = [logits[-1]], []
        for step in range(new):
            tok = int(rows[-1].argmax())
            out.append(tok)
            if tok == eos or step == new - 1:
                break
            logits, _ = self.forward([tok], len(prompt) + step)
            rows.append(logits[-1])
        return out, rows
