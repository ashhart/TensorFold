"""Native DeepSeek-V4 GGUF backbone; TensorFold's hyper-connections and packed CUDA linears."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda.experts import Plan, route
from tensorfold.cuda.gguf import Packed, Weights
from tensorfold.cuda.gguf.linear import linear
from tensorfold.families.glm5_next.cuda import glue

from .attention import attend, frequencies, norm_rope, rope, select
from .moe import combine, swiglu


def norm(x, weight, eps):
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.bfloat16)
    glue.rmsnorm(x, weight, eps, out)
    return out.reshape(shape)


@dataclass
class State:
    keys: torch.Tensor
    values: torch.Tensor | None = None
    gates: torch.Tensor | None = None
    ivals: torch.Tensor | None = None
    igates: torch.Tensor | None = None
    pool: torch.Tensor | None = None
    ipool: torch.Tensor | None = None
    offset: int = 0

    def clone(self, *, pool_rows=None):
        values = {}
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            if pool_rows is not None and k in ("pool", "ipool") and v is not None:
                v = v[:pool_rows]
            values[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return State(**values)


class Layer:
    def __init__(self, weights, prefix, meta, ratio, limit, hash_layer=False):
        self.weights, self.prefix, self.meta = weights, prefix, meta
        self.ratio, self.limit, self.hash_layer = ratio, limit, hash_layer
        self.dims = int(meta["deepseek4.embedding_length"])
        self.heads = int(meta["deepseek4.attention.head_count"])
        self.hdim = int(meta["deepseek4.attention.key_length"])
        self.groups = int(meta["deepseek4.attention.output_group_count"])
        self.rank = int(meta["deepseek4.attention.output_lora_rank"])
        self.slots = int(meta["deepseek4.expert_used_count"])
        self.experts = int(meta["deepseek4.expert_count"])
        self.eps = float(meta.get("deepseek4.attention.layer_norm_rms_epsilon", 1e-6))
        self.hc_eps = float(meta.get("deepseek4.hyper_connection.epsilon", 1e-6))
        self.iters = int(meta.get("deepseek4.hyper_connection.sinkhorn_iterations", 20))
        self.freq = frequencies(meta, ratio, weights.device)
        if not hasattr(weights, "_rope_tables"):
            weights._rope_tables = {}
            weights._positions = torch.arange(limit + 8, device=weights.device)
        kind = bool(ratio)
        if kind not in weights._rope_tables:
            angles = weights._positions.float()[:, None] * self.freq[None]
            weights._rope_tables[kind] = (angles.cos(), angles.sin())
        self.tables = weights._rope_tables[kind]
        self._buffers = {}
        names = [n for n in weights.inventory if n.startswith(prefix + ".")]
        self.w = {n[len(prefix) + 1 :]: weights.tensor(n) for n in names}
        # Hyper-connection dots keep full precision, matching TensorFold's shared kernel.
        for k, v in self.w.items():
            if k.startswith("hc_") and isinstance(v, torch.Tensor):
                self.w[k] = v.float()
        if hash_layer:
            table = self.w["ffn_gate_tid2eid.weight"]
            if int(table.min()) < 0 or int(table.max()) >= self.experts:
                raise ValueError(f"{prefix}: hash routing contains invalid expert ids")
        self.wo = []
        w = self.w["attn_output_a.weight"]
        for group in range(self.groups):
            if isinstance(w, Packed):
                row_bytes = w.data.numel() // w.shape[1]
                lo, hi = group * self.rank, (group + 1) * self.rank
                self.wo.append(Packed(w.data[lo * row_bytes : hi * row_bytes], (w.shape[0], self.rank), w.format))
            else:
                self.wo.append(w[group * self.rank : (group + 1) * self.rank])

    def state(self):
        dev = self.weights.device
        pool = (
            torch.empty((self.limit // self.ratio + 1, self.hdim), device=dev, dtype=torch.bfloat16)
            if self.ratio
            else None
        )
        ipool = torch.empty((self.limit // 4 + 1, 128), device=dev, dtype=torch.bfloat16) if self.ratio == 4 else None
        return State(torch.empty((0, self.hdim), device=dev, dtype=torch.bfloat16), pool=pool, ipool=ipool)

    def trim(self, state, end):
        if not state.offset >= end >= 0:
            raise ValueError("cache trim outside committed range")
        raw, base = state._frame["raw"]
        lo = max(0, end - int(self.meta["deepseek4.attention.sliding_window"]))
        state.keys = raw[lo - base : end - base].clone()
        if self.ratio:
            keep = max(0, (end // self.ratio - int(self.ratio == 4)) * self.ratio)
            values, gates, base = state._frame["compress"]
            state.values, state.gates = (
                values[keep - base : end - base].clone(),
                gates[keep - base : end - base].clone(),
            )
            if self.ratio == 4:
                values, gates, base = state._frame["index"]
                state.ivals, state.igates = (
                    values[keep - base : end - base].clone(),
                    gates[keep - base : end - base].clone(),
                )
        state.offset = end

    def buffers(self, rows):
        if rows not in self._buffers:
            if rows > 16:
                # Keep decode widths, but never accumulate every prompt-tail size.
                self._buffers = {k: v for k, v in self._buffers.items() if k <= 16}
            d, dev = self.dims, self.weights.device
            self._buffers[rows] = (
                torch.empty((rows, d), device=dev, dtype=torch.bfloat16),
                torch.empty((rows, d // 64), device=dev, dtype=torch.float32),
                torch.empty((rows, 4), device=dev, dtype=torch.float32),
                torch.empty((rows, 4, 4), device=dev, dtype=torch.float32),
                torch.empty((rows, glue.HC_BLOCKS, 32), device=dev, dtype=torch.float32),
            )
        return self._buffers[rows]

    def split(self, x, kind):
        out, xs, post, comb, part = self.buffers(x.shape[0])
        w = self.w
        glue.hc_pre(
            x,
            w[f"hc_{kind}_fn.weight"],
            w[f"hc_{kind}_base.weight"],
            w[f"hc_{kind}_scale.weight"],
            w["attn_norm.weight" if kind == "attn" else "ffn_norm.weight"],
            out,
            xs,
            post,
            comb,
            part,
            self.eps,
            self.hc_eps,
            self.iters,
        )
        return out, post, comb

    def expand(self, x, y, post, comb):
        out = torch.empty_like(x)
        glue.hc_post(x, out, y.float()[None].contiguous(), post, comb)
        return out

    def compress(self, x, state, positions, index=False):
        r = self.ratio
        name = "indexer_compressor" if index else "attn_compressor"
        values = linear(x, self.w[f"{name}_kv.weight"], dtype=torch.float32)
        gates = linear(x, self.w[f"{name}_gate.weight"], dtype=torch.float32)
        ape = self.w[f"{name}_ape.weight"].float()
        gates = gates + ape[positions % r]
        old_values, old_gates = (state.ivals, state.igates) if index else (state.values, state.gates)
        oldrows = 0 if old_values is None else old_values.shape[0]
        lo = state.offset - oldrows
        if oldrows:
            values = torch.cat((old_values, values))
            gates = torch.cat((old_gates, gates))
        if not hasattr(state, "_frame"):
            state._frame = {}
        state._frame["index" if index else "compress"] = (values, gates, lo)
        first, last = state.offset // r, (state.offset + x.shape[0]) // r
        if last > first:
            starts = torch.arange(first, last, device=x.device) * r
            dim = self.w[f"{name}_norm.weight"].numel()
            slots = torch.arange(r, device=x.device)
            if r == 4:
                prev = starts[:, None] - r + slots[None]
                cur = starts[:, None] + slots[None]
                pv = values[(prev - lo).clamp(min=0), :dim]
                ps = gates[(prev - lo).clamp(min=0), :dim].masked_fill((prev < 0)[..., None], -torch.inf)
                cv = values[cur - lo, dim:]
                cs = gates[cur - lo, dim:]
                vs = torch.cat((pv, cv), 1)
                gs = torch.cat((ps, cs), 1)
            else:
                idx = starts[:, None] + slots[None] - lo
                vs, gs = values[idx], gates[idx]
            pooled = (gs.softmax(1) * vs).sum(1).to(torch.bfloat16)
            pooled = rope(norm(pooled, self.w[f"{name}_norm.weight"], self.eps), starts, self.freq)
            target = state.ipool if index else state.pool
            target[first:last].copy_(pooled)
        # Keep the current partial block and (for ratio 4) its predecessor.
        end = state.offset + x.shape[0]
        keep = max(0, (end // r - int(r == 4)) * r)
        values, gates = values[keep - lo :].clone(), gates[keep - lo :].clone()
        if index:
            state.ivals, state.igates = values, gates
        else:
            state.values, state.gates = values, gates

    def attention(self, x, state, *, draft=False):
        w = self.w
        rows = x.shape[0]
        start = state.offset
        pos = self.weights._positions[start : start + rows]
        qr = norm(linear(x, w["attn_q_a.weight"]), w["attn_q_a_norm.weight"], self.eps)
        kv = norm_rope(linear(x, w["attn_kv.weight"]), pos, self.tables, w["attn_kv_a_norm.weight"], self.eps)
        q = linear(qr, w["attn_q_b.weight"]).reshape(rows, self.heads, self.hdim)
        q = norm_rope(q, pos, self.tables, eps=self.eps)
        raw = torch.cat((state.keys, kv))
        base = start - state.keys.shape[0]
        state._frame = {"raw": (raw, base)}
        picks = None
        pool = None
        if self.ratio:
            self.compress(x, state, pos)
            count = (start + rows) // self.ratio
            pool = state.pool[:count]
            if self.ratio == 4:
                self.compress(x, state, pos, index=True)
                if count > 512:
                    iq = linear(qr, w["indexer.attn_q_b.weight"]).reshape(rows, 64, 128)
                    iq = norm_rope(iq, pos, self.tables, normalize=False)
                    iw = linear(x, w["indexer.proj.weight"]).float() * (64**-0.5)
                    picks = select(iq, iw, state.ipool[:count], start)
        out = attend(
            q,
            raw,
            pool,
            picks,
            w["attn_sinks.weight"],
            start=start,
            rawbase=base,
            ratio=self.ratio,
            window=int(self.meta["deepseek4.attention.sliding_window"]),
            draft=draft,
        )
        out = norm_rope(out, pos, self.tables, normalize=False, inverse=True).reshape(rows, self.groups, -1)
        parts = [linear(out[:, g].contiguous(), self.wo[g]) for g in range(self.groups)]
        result = linear(torch.cat(parts, 1), w["attn_output_b.weight"])
        if not draft:
            state.keys = raw[-int(self.meta["deepseek4.attention.sliding_window"]) :].clone()
            state.offset += rows
        return result

    def moe(self, x, ids):
        w = self.w
        scores = torch.nn.functional.softplus(linear(x, w["ffn_gate_inp.weight"], dtype=torch.float32)).sqrt()
        if self.hash_layer:
            picks = w["ffn_gate_tid2eid.weight"][ids].to(torch.int32)
        else:
            bias = w.get("exp_probs_b.bias", 0)
            picks = (scores + bias).topk(self.slots, dim=1).indices.to(torch.int32)
        picks = picks.sort(1).values.contiguous()
        weights = scores.gather(1, picks.long())
        weights = (
            weights
            / (weights.sum(1, keepdim=True) + 1e-20)
            * float(self.meta.get("deepseek4.expert_weights_scale", 1.5))
        )
        plan = None
        if x.shape[0] > 16:
            plan = Plan(x.shape[0], self.slots, self.experts, x.device, prefill=True)
            route(picks, plan, tile=64)
        gate = w["ffn_gate_exps.weight"].linear(x, picks, plan=plan, validated=True)
        up = w["ffn_up_exps.weight"].linear(x, picks, plan=plan, validated=True)
        inner = swiglu(gate, up)
        down = w["ffn_down_exps.weight"].linear(inner, picks, plan=plan, validated=True)
        shared = swiglu(linear(x, w["ffn_gate_shexp.weight"]), linear(x, w["ffn_up_shexp.weight"]))
        shared = linear(shared, w["ffn_down_shexp.weight"])
        return combine(down, weights, shared)

    def __call__(self, x, ids, state, *, draft=False):
        y, post, comb = self.split(x, "attn")
        x = self.expand(x, self.attention(y, state, draft=draft), post, comb)
        y, post, comb = self.split(x, "ffn")
        return self.expand(x, self.moe(y, ids), post, comb)


class Model:
    def __init__(self, path, limit=163840):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.weights = Weights(path)
        try:
            self._load(limit)
        finally:
            self.weights.close()

    def _load(self, limit):
        self.meta = self.weights.metadata
        if self.meta.get("general.architecture") != "deepseek4":
            raise ValueError("native DeepSeek engine requires a deepseek4 GGUF")
        from types import SimpleNamespace

        from tensorfold.families.deepseek_v4.gguf import validate_cuda_shapes

        validate_cuda_shapes(SimpleNamespace(tensors=tuple(self.weights.inventory.values())), self.meta)
        self.limit = limit
        count = int(self.meta["deepseek4.block_count"])
        ratios = self.meta["deepseek4.attention.compress_ratios"]
        hashes = int(self.meta.get("deepseek4.hash_layer_count", 3))
        self.layers = []
        import sys

        for i in range(count):
            print(f"[tensorfold] loading DeepSeek layer {i + 1}/{count}", file=sys.stderr, flush=True)
            self.layers.append(Layer(self.weights, f"blk.{i}", self.meta, int(ratios[i]), limit, i < hashes))
        self.embed = self.weights.tensor("token_embd.weight")
        self.output = self.weights.tensor("output.weight")
        self.head_fn = self.weights.tensor("output_hc_fn.weight").float()
        self.head_base = self.weights.tensor("output_hc_base.weight").float()
        self.head_scale = self.weights.tensor("output_hc_scale.weight").float()
        self.head_norm = self.weights.tensor("output_norm.weight")
        self.tap_layers = tuple(range(max(0, count - 3), count))
        self.last_taps = None

    def state(self):
        return [layer.state() for layer in self.layers]

    def head(self, x, fn=None, base=None, scale=None, weight=None):
        d = self.embed.shape[1]
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(1, keepdim=True) + self.layers[0].eps)
        fn = self.head_fn if fn is None else fn
        base = self.head_base if base is None else base
        scale = self.head_scale if scale is None else scale
        pre = torch.sigmoid(linear(xf, fn, dtype=torch.float32) * inv * scale[0] + base) + self.layers[0].hc_eps
        xs = xf.reshape(-1, 4, d)
        collapsed = pre[:, 0, None] * xs[:, 0]
        for j in range(1, 4):
            collapsed += pre[:, j, None] * xs[:, j]
        h = norm(collapsed.to(torch.bfloat16), self.head_norm if weight is None else weight, self.layers[0].eps)
        return linear(h, self.output, dtype=torch.float32)

    def forward(self, ids, state, *, head_all=True):
        ids = torch.as_tensor(ids, device=self.embed.device, dtype=torch.long)
        if not ids.numel() or ids.numel() + state[0].offset > self.limit:
            raise ValueError("empty input or context capacity exceeded")
        x = self.embed[ids].to(torch.bfloat16)[:, None].expand(-1, 4, -1).reshape(ids.numel(), -1).contiguous()
        taps = []
        for i, (layer, cache) in enumerate(zip(self.layers, state)):
            x = layer(x, ids, cache)
            if i in self.tap_layers:
                means = torch.empty((ids.numel(), self.embed.shape[1]), device=x.device, dtype=torch.bfloat16)
                glue.stream_mean(x, means)
                taps.append(means)
        self.last_taps = torch.cat(taps, 1) if taps else None
        return self.head(x if head_all else x[-1:])
