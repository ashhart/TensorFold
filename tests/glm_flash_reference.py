"""GLM-5.3-Flash in float64 torch, written from the Mac engine's definition (families/glm5_next: model.py's
hyper-connections, kda.py with mlx-lm's gated delta rule, mla.py's sparse attention and caches.py's pooled keys,
mlp.py, mtp.py), sharing no code with the CUDA path but the EXL3 reference decoder (cuda/exl3.py)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

PREFIX = "model.language_model."
TIE = 0.1          # a row whose last chosen pool and the next score within this share of its largest score: a tie
ROUTE_TIE = 0.01   # a row whose last chosen expert and the next within this (sigmoid plus bias): a tie


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def swiglu(g: torch.Tensor, u: torch.Tensor, limit: float) -> torch.Tensor:
    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    return g * torch.sigmoid(g) * u


class Reference:
    """The whole model over a token sequence: every position's logits, and the MTP head's."""

    def __init__(self, folder) -> None:
        folder = Path(folder)
        self.t = {}
        for f in sorted(folder.glob("*.safetensors")):
            self.t.update(load_file(str(f)))
        c = json.loads((folder / "config.json").read_text())["text_config"]
        self.c = c
        self.eps, self.S = c["rms_norm_eps"], c["hc_mult"]
        self.kinds = c["layer_types"]
        self.mlps = c["mlp_layer_types"]
        self.experts: dict[str, torch.Tensor] = {}
        self.tied: set[int] = set()              # rows whose pools or experts bf16 rounding may change

    def w(self, name: str) -> torch.Tensor:
        key = name if name in self.t else PREFIX + name
        return self.t[key].double()

    def lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x @ self.w(name + ".weight").T

    # -- hyper-connections (model.py's HC.split, hc_expand and final_norm) ------------------------------------------
    def hc_split(self, x: torch.Tensor, i: int, site: str):
        c, S = self.c, self.S
        T = x.shape[0]
        fn, base = self.w(f"layers.{i}.hc_{site}_fn"), self.w(f"layers.{i}.hc_{site}_base")
        scale, eps = self.w(f"layers.{i}.hc_{site}_scale"), c["hc_eps"]
        mixes = rms(x.reshape(T, -1), None, self.eps) @ fn.T
        pre = torch.sigmoid(mixes[:, :S] * scale[0] + base[:S]) + eps
        post = 2 * torch.sigmoid(mixes[:, S:2 * S] * scale[1] + base[S:2 * S])
        comb = mixes[:, 2 * S:].reshape(T, S, S) * scale[2] + base[2 * S:].reshape(S, S)
        comb = torch.softmax(comb, -1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(c["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return (pre[..., None] * x).sum(1), post, comb

    @staticmethod
    def hc_expand(branch, x, post, comb):
        return post[..., None] * branch[:, None, :] + comb.transpose(-1, -2) @ x

    # -- KDA (kda.py's _step, mlx-lm's gated_delta_ops) ---------------------------------------------------------------
    def kda(self, x: torch.Tensor, i: int) -> torch.Tensor:
        lc = self.c["linear_attn_config"]
        H, d = lc["num_heads"], lc["head_dim"]
        p = f"layers.{i}.self_attn."
        T = x.shape[0]
        mixed = torch.cat([self.lin(x, p + f"{n}_proj") for n in "qkv"], -1)
        cw = torch.cat([self.w(p + f"{n}_conv1d.weight").reshape(H * d, -1) for n in "qkv"])     # [3 H d, taps]
        taps = cw.shape[1]
        ci = torch.cat([torch.zeros(taps - 1, mixed.shape[1], dtype=x.dtype), mixed])
        acc = sum(ci[t:t + T] * cw[:, t] for t in range(taps))
        co = (acc * torch.sigmoid(acc)).view(T, 3, H, d)
        q, k, v = co[:, 0], co[:, 1], co[:, 2]
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * d ** -0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        a = self.lin(self.lin(x, p + "f_a_proj"), p + "f_b_proj").view(T, H, d)
        A = torch.exp(self.w(p + "A_log"))[:, None]
        g = torch.exp(lc["gate_lower_bound"] * torch.sigmoid(A * (a + self.w(p + "dt_bias").view(H, d))))
        beta = torch.sigmoid(self.lin(x, p + "b_proj"))
        state = torch.zeros(H, d, d, dtype=x.dtype)                          # [H, value, key]
        ys = []
        for t in range(T):
            state = state * g[t][:, None, :]
            kv = (state * k[t][:, None, :]).sum(-1)
            state = state + ((v[t] - kv) * beta[t][:, None])[:, :, None] * k[t][:, None, :]
            ys.append((state * q[t][:, None, :]).sum(-1))
        gate = self.lin(self.lin(x, p + "g_a_proj"), p + "g_b_proj").view(T, H, d)
        o = rms(torch.stack(ys), self.w(p + "o_norm.weight"), self.eps) * torch.sigmoid(gate)
        return self.lin(o.reshape(T, H * d), p + "o_proj")

    # -- sparse MLA (mla.py, caches.py's pool_blocks) -----------------------------------------------------------------
    def mla(self, x: torch.Tensor, i: int) -> torch.Tensor:
        c = self.c
        p = f"layers.{i}.self_attn."
        T, Hh, nope, vd = x.shape[0], c["num_attention_heads"], c["qk_nope_head_dim"], c["v_head_dim"]
        Hi, Di, kp, topk = c["index_n_heads"], c["index_head_dim"], c["index_kpool"], c["index_topk"]
        qr = rms(self.lin(x, p + "q_a_proj"), self.w(p + "q_a_layernorm.weight"), self.eps)
        q = self.lin(qr, p + "q_b_proj").view(T, Hh, nope)
        lat = rms(self.lin(x, p + "kv_a_proj_with_mqa"), self.w(p + "kv_a_layernorm.weight"), self.eps)
        kvb = self.w(p + "kv_b_proj.weight").view(Hh, nope + vd, -1)
        keys = torch.einsum("tl,hdl->thd", lat, kvb[:, :nope])
        values = torch.einsum("tl,hdl->thd", lat, kvb[:, nope:])
        ik = torch.nn.functional.layer_norm(self.lin(x, p + "indexer.wk"), (Di,), self.w(p + "indexer.k_norm.weight"),
                                            self.w(p + "indexer.k_norm.bias"), 1e-6)
        ig = x @ self.w(p + "indexer.index_kpool_compress_gate").T
        iw = self.lin(x, p + "indexer.weights_proj") * (Hi ** -0.5) * (Di ** -0.5)
        iq = self.lin(qr, p + "indexer.wq_b").view(T, Hi, Di)
        blocks = T // kp
        logit = ig[:blocks * kp].view(blocks, kp, Di) + self.w(p + "indexer.index_kpool_compress_ape")[None]
        pools = (torch.softmax(logit, 1) * ik[:blocks * kp].view(blocks, kp, Di)).sum(1)
        out = []
        for t in range(T):
            n = t + 1
            ids = torch.arange(n)
            if n > topk:
                nb = n // kp
                scores = (iw[t][:, None] * torch.relu(iq[t] @ pools[:nb].T)).sum(0)
                order = torch.sort(-scores, stable=True).indices                          # ties to the lower pool
                best = order[:min(topk // kp, nb)]
                if nb > topk // kp:
                    gap = scores[order[topk // kp - 1]] - scores[order[topk // kp]]
                    if gap < TIE * scores.abs().max():
                        self.tied.add(t)
                tail = torch.arange(nb * kp, n)
                ids = torch.cat([(best[:, None] * kp + torch.arange(kp)).reshape(-1), tail])
            s = torch.einsum("hd,nhd->hn", q[t], keys[ids]) * nope ** -0.5
            out.append(torch.einsum("hn,nhd->hd", torch.softmax(s, -1), values[ids]).reshape(-1))
        return self.lin(torch.stack(out), p + "o_proj")

    # -- feed-forward (mlp.py) ----------------------------------------------------------------------------------------
    def mlp(self, x: torch.Tensor, p: str) -> torch.Tensor:
        lim = self.c["swiglu_limit"]
        return self.lin(swiglu(self.lin(x, p + "gate_proj"), self.lin(x, p + "up_proj"), lim), p + "down_proj")

    def moe(self, x: torch.Tensor, i: int) -> torch.Tensor:
        from tensorfold.families.glm5_next.cuda import exl3

        c = self.c
        p = f"layers.{i}.mlp."
        scores = torch.sigmoid(x @ self.w(p + "gate.weight").T)
        choice = scores + self.w(p + "gate.e_score_correction_bias")
        k = c["num_experts_per_tok"]
        idx = torch.topk(choice, k).indices
        top = choice.sort(-1, descending=True).values
        self.tied.update(int(t) for t in torch.nonzero(top[:, k - 1] - top[:, k] < ROUTE_TIE).flatten())
        wts = torch.gather(scores, 1, idx)
        wts = wts / wts.sum(-1, keepdim=True) * c["routed_scaling_factor"]

        def mat(e: int, proj: str) -> torch.Tensor:
            n = PREFIX + p + f"experts.{e}.{proj}."
            if n not in self.experts:
                self.experts[n] = exl3.dequantize(self.t[n + "trellis"], self.t[n + "suh"], self.t[n + "svh"])
            return self.experts[n]

        out = self.mlp(x, p + "shared_experts.")
        for t in range(x.shape[0]):
            for e, wt in zip(idx[t].tolist(), wts[t]):
                act = swiglu(x[t] @ mat(e, "gate_proj"), x[t] @ mat(e, "up_proj"), c["swiglu_limit"])
                out[t] = out[t] + wt * (act @ mat(e, "down_proj"))
        return out

    # -- the model ----------------------------------------------------------------------------------------------------
    def embed(self, tokens: list[int]) -> torch.Tensor:
        return self.w("embed_tokens.weight")[torch.tensor(tokens)]

    def head(self, normed: torch.Tensor) -> torch.Tensor:
        return normed @ self.t["lm_head.weight"].double().T

    def forward(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits [T, V], final-normed rows [T, D]) of every position; ``tied`` names the rows near a tie."""

        self.tied = set()
        x = self.embed(tokens)[:, None, :].expand(-1, self.S, -1)
        for i, kind in enumerate(self.kinds):
            xc, post, comb = self.hc_split(x, i, "attn")
            normed = rms(xc, self.w(f"layers.{i}.input_layernorm.weight"), self.eps)
            g = self.kda(normed, i) if kind == "linear_attention" else self.mla(normed, i)
            x = self.hc_expand(g, x, post, comb)
            xc, post, comb = self.hc_split(x, i, "ffn")
            normed = rms(xc, self.w(f"layers.{i}.post_attention_layernorm.weight"), self.eps)
            g = self.mlp(normed, f"layers.{i}.mlp.") if self.mlps[i] == "dense" else self.moe(normed, i)
            x = self.hc_expand(g, x, post, comb)
        normed = rms(x.mean(1), self.w("norm.weight"), self.eps)
        return self.head(normed), normed

    def mtp(self, normed: torch.Tensor, next_tokens: list[int]) -> torch.Tensor:
        """The MTP head's logits for rows 0.. (row j: main row j and token j + 1); row 0's embedding zeroed, as the
        CUDA head (and vLLM's DeepSeek MTP) take position 0. ``tied`` keeps the rows ``forward`` found near a tie."""

        i = self.c["num_hidden_layers"]
        e = self.embed(next_tokens)
        e[0] = 0
        x = torch.cat([rms(e, self.w(f"layers.{i}.enorm.weight"), self.eps),
                       rms(normed, self.w(f"layers.{i}.hnorm.weight"), self.eps)], -1)
        x = self.lin(x, f"layers.{i}.eh_proj")
        x = x + self.mla(rms(x, self.w(f"layers.{i}.input_layernorm.weight"), self.eps), i)
        x = x + self.moe(rms(x, self.w(f"layers.{i}.post_attention_layernorm.weight"), self.eps), i)
        return self.head(rms(x, self.w(f"layers.{i}.shared_head.norm.weight"), self.eps))
