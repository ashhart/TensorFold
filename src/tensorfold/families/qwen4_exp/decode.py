"""Flash Next decode steps (1-16 consecutive rows) through ``kernels``.

A layer is: hyper-connection (2 kernels, the previous block's write-back
folded into the first) -> Gated DeltaNet (projection, ``gdn_step``, out
projection) or sparse attention (projection, ``attn_prep``, cache write,
attention, ``attn_gate``, out projection) -> hyper-connection -> MoE (router,
route, expert gate/up/down, shared expert). The MoE's combine is folded into
the next hyper-connection's write-back. A one-row projection is MLX's quantized
matmul; a multi-row one is ``kernels.qmv_rows``, which gives every row MLX's
one-row bits (MLX's own matmul does not on every GPU). Everything between them
is a row-invariant kernel, so each row of a multi-row step gets a one-row
step's bits.

Past 2,048 keys a row's attention reads only its selected blocks: ``kernels``
pools each completed block's indexer key once, scores every block for all rows
in one kernel and selects each row's best 512 in another; one attention kernel
then reads each row's selected keys and unfinished tail (or, before 2,048 keys,
all its keys) for all rows at once.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tensorfold.families.qwen4_exp import kernels as K


def _stacked(linears: list[Any]) -> tuple[nn.QuantizedLinear, list[int]]:
    """One quantized linear for projections of the same input; each original keeps a row view of it."""

    first = linears[0]
    rows = sum(int(l.weight.shape[0]) for l in linears)
    stacked = nn.QuantizedLinear(int(first.weight.shape[1]) * 32 // first.bits, rows, bias=False,
                                 group_size=first.group_size, bits=first.bits)
    stacked.weight = mx.concatenate([l.weight for l in linears])
    stacked.scales = mx.concatenate([l.scales for l in linears])
    stacked.biases = mx.concatenate([l.biases for l in linears])
    mx.eval(stacked.weight, stacked.scales, stacked.biases)
    cuts, at = [], 0
    for l in linears:
        n = int(l.weight.shape[0])
        l.weight, l.scales, l.biases = (stacked.weight[at:at + n], stacked.scales[at:at + n],
                                        stacked.biases[at:at + n])
        # evaluated here (views of the stacked buffer): a lazy op made on this thread would need its stream
        # in the server's scheduler thread
        mx.eval(l.weight, l.scales, l.biases)
        at += n
        cuts.append(at)
    return stacked, cuts[:-1]


def project(x: mx.array, linear: Any) -> mx.array:
    """x [..., R, K] through a 4-bit linear: MLX's quantized matmul for one row, ``kernels.qmv_rows`` (MLX's
    one-row bits for every row, a simdgroup a row, weight reads shared) for more; MLX's own matmul sums a row
    differently in a window of 2-4 rows on some GPUs (M3 Ultra, MLX 0.32.0, 2026-09-25)."""

    rows = x.size // x.shape[-1]
    return linear(x) if rows == 1 else K.qmv_rows(x, linear)


class _HC:
    """A hyper-connection's weights for hc_down / hc_up."""

    def __init__(self, hc: Any, *, inject: bool) -> None:
        parts = [hc.input_mix_weight_down] + ([hc.block_inject_weight] if inject else [])
        self.down = K.QWeights.of(*parts)
        mx.eval(self.down.weight, self.down.scales, self.down.biases)
        if len(parts) > 1:
            # keep the modules' own weights as views of the stacked matrix
            at = 0
            for lin in parts:
                n = int(lin.weight.shape[0])
                lin.weight, lin.scales, lin.biases = (self.down.weight[at:at + n], self.down.scales[at:at + n],
                                                      self.down.biases[at:at + n])
                mx.eval(lin.weight, lin.scales, lin.biases)
                at += n
        self.up = K.QWeights.of(hc.input_mix_weight_up)
        self.scale = (1.0 + hc.hc_norm.weight.astype(mx.float32))
        self.low = int(hc.input_mix_weight_down.weight.shape[0])
        mx.eval(self.scale)


class FusedDecode:
    """Decode rows of Flash Next through the fused kernels, with the reference model's caches."""

    def __init__(self, model: Any) -> None:
        cfg = model.args
        self.model = model
        self.cfg = cfg
        self.streams = cfg.hc_count
        self.eps = mx.array([cfg.rms_norm_eps], dtype=mx.float32)
        self.layers: list[dict[str, Any]] = []
        for layer in model.layers:
            entry: dict[str, Any] = {
                "attn_hc": _HC(layer.attn_hyper_connection, inject=True),
                "mlp_hc": _HC(layer.mlp_hyper_connection, inject=True),
            }
            if layer.is_linear:
                g = layer.linear_attn
                proj, _ = _stacked([g.in_proj_qkv, g.in_proj_z, g.in_proj_b, g.in_proj_a])
                conv_w = mx.contiguous(g.conv1d.weight[:, :, 0])
                mx.eval(conv_w)
                entry["gdn"] = (proj, conv_w, g)
            else:
                a = layer.self_attn
                proj, _ = _stacked([a.q_proj, a.k_proj, a.v_proj, a.indexer.index_qk_proj])
                scales = [1.0 + n.weight.astype(mx.float32) for n in
                          (a.q_norm, a.k_norm, a.indexer.q_layernorm, a.indexer.k_layernorm)]
                mx.eval(*scales)
                entry["attn"] = (proj, *scales, a)
            moe = layer.mlp
            # the router's bf16 rows and the shared expert's gate row (dequantized): one matvec
            sgate = moe.shared_expert_gate
            sgate_row = mx.dequantize(sgate.weight, sgate.scales, sgate.biases, group_size=sgate.group_size,
                                      bits=sgate.bits).astype(moe.gate.weight.dtype)
            router_rows = mx.concatenate([moe.gate.weight, sgate_row])
            mx.eval(router_rows)
            entry["moe"] = (moe, router_rows)
            self.layers.append(entry)
        self.mixer = _HC(model.model.hyper_connection_mixer, inject=False)
        # the last call's recurrent states after each of its rows, by layer (for keeping a prefix of a window)
        self.row_states: dict[int, tuple[mx.array, mx.array]] = {}
        self.eval_every = 4

    # -- blocks ------------------------------------------------------------------
    def _gdn(self, index: int, x: mx.array, cache: Any) -> mx.array:
        proj, conv_w, g = self.layers[index]["gdn"]
        cfg = self.cfg
        rows = x.shape[0]
        conv_state = cache.conv[0] if cache.conv is not None else mx.zeros(
            (cfg.linear_conv_kernel_dim - 1, conv_w.shape[0]), dtype=x.dtype)
        ssm_state = cache.ssm[0] if cache.ssm is not None else None
        out, conv_rows, ssm_rows = K.gdn_step(project(x, proj), conv_state, ssm_state, conv_w, g.A_log, g.dt_bias,
                                              g.norm.weight, self.eps, nk=cfg.linear_num_key_heads,
                                              nv=cfg.linear_num_value_heads, dk=cfg.linear_key_head_dim,
                                              dv=cfg.linear_value_head_dim)
        cache.conv, cache.ssm = conv_rows[rows - 1:rows], ssm_rows[rows - 1:rows]
        cache.offset += rows
        self.row_states[index] = (conv_rows, ssm_rows)
        return project(out, g.out_proj)

    def _attention(self, index: int, x: mx.array, cache: Any) -> mx.array:
        proj, q_scale, k_scale, iq_scale, pool_scale, a = self.layers[index]["attn"]
        cfg = self.cfg
        rows = x.shape[0]
        heads, kv_heads, dims = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        index_heads, index_dims = cfg.indexer_n_heads, cfg.indexer_head_dim
        past = cache.offset
        p = project(x, proj)
        positions = mx.arange(past, past + rows, dtype=mx.int32)
        q, k, iq = K.attn_prep(p, positions, q_scale, k_scale, iq_scale, self.eps, q_heads=heads,
                               kv_heads=kv_heads, head_dim=dims, index_heads=index_heads, index_dim=index_dims,
                               rotary_dim=cfg.rotary_dim, base=cfg.rope_theta)
        at = heads * 2 * dims
        v = p[:, at + kv_heads * dims: at + 2 * kv_heads * dims].reshape(rows, kv_heads, dims)
        raw_key = p[:, at + 2 * kv_heads * dims + index_heads * index_dims:]
        _, _, raw = cache.update(k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None], raw_key[None])
        ratio, top = cfg.indexer_compress_ratio, a.indexer.top_blocks
        ends = [past + r + 1 for r in range(rows)]
        complete = [e // ratio for e in ends]
        ids = self._select(iq, raw, cache, complete, ends, pool_scale, top) if complete[-1] > top else None
        # every row in one kernel: a row past ``top`` complete blocks reads its selected blocks' keys and its
        # tail, a shorter row all its keys
        sparse = [c > top for c in complete]
        counts = [ratio * top + e - ratio * c if sp else e for e, c, sp in zip(ends, complete, sparse)]
        out = K.attention_rows(q, cache.keys, cache.values, counts, ids, sparse, a.scale)
        gated = K.attn_gate(out, p, q_heads=heads, head_dim=dims)
        return project(gated, a.o_proj)

    def _select(self, iq: mx.array, raw: mx.array, cache: Any, complete: list[int], ends: list[int],
                pool_scale: mx.array, top: int) -> mx.array:
        """The rows' key ids (``kernels.index_select``), after pooling the blocks the last row completes."""

        cfg = self.cfg
        done = 0 if cache.pooled is None else int(cache.pooled.shape[1])
        if complete[-1] > done:
            fresh = K.index_pool(raw[0], done, complete[-1], pool_scale, self.eps, rotary_dim=cfg.rotary_dim,
                                 base=cfg.rope_theta)[None]
            cache.pooled = fresh if cache.pooled is None else mx.concatenate([cache.pooled, fresh], axis=1)
        return K.index_select(iq, cache.pooled[0], complete, ends, top=top)

    # windows of at least this many rows read each distinct expert once (expert_group + grouped_*): 8 consecutive
    # tokens pick ~40 distinct experts a layer of 80 (2026-09-25); fewer rows keep the per-slot kernels, which have
    # no grouping kernel to wait on. Both give a row the same bits.
    group_rows = 3

    def _moe(self, index: int, x: mx.array) -> tuple[str, tuple[mx.array, ...]]:
        """Routed experts + the shared expert: the write-back ("plain": the branch [R, D]; "grouped": the slots'
        outputs, weights and router logits, combined by the next hc_norm)."""

        moe, router_rows = self.layers[index]["moe"]
        cfg = self.cfg
        logits = K.router(x, router_rows)                                      # [R, E + 1] fp32
        sw, se = moe.switch_mlp, moe.shared_expert
        k, experts = cfg.num_experts_per_tok, cfg.num_experts
        if x.shape[0] < self.group_rows:
            act = K.expert_gateup(x, logits, k, experts, sw.gate_proj, sw.up_proj, shared=(se.gate_proj, se.up_proj))
            return "plain", (K.expert_down(act, logits, k, experts, sw.down_proj, shared=se.down_proj),)
        group = K.expert_group(logits, k, experts)
        act = K.grouped_gateup(x, group, sw.gate_proj, sw.up_proj, (se.gate_proj, se.up_proj))
        return "grouped", (K.grouped_down(act, group, sw.down_proj, se.down_proj), group[1], logits)

    # -- a step --------------------------------------------------------------------
    def __call__(self, tokens: np.ndarray, cache: list[Any]) -> mx.array:
        """Mixed hidden states [1, R, D] after the last layer for R consecutive tokens (batch 1, host ids)."""

        h = self.model.model.embed_tokens(mx.array(tokens.reshape(-1).astype(np.int32)))     # [R, D]
        return self.run(mx.tile(h, (1, self.streams)), tokens, cache)

    def run(self, h: mx.array, tokens: np.ndarray | None, cache: list[Any]) -> mx.array:
        """The layers and the final mixer from residual streams h [R, S*D]: mixed hidden states [1, R, D]."""

        model = self.model
        pending: tuple[str, tuple[mx.array, ...], mx.array | None] = ("none", (), None)
        for i, layer in enumerate(model.layers):
            c = cache[i]
            if "ple" in layer:
                h = self._write_back(h, pending)
                pending = ("none", (), None)
                h = h + self._ple(layer.ple, h, tokens.reshape(1, -1), c)
            entry = self.layers[i]
            ahc = entry["attn_hc"]
            kind, branch, inject = pending
            hn, ssp = K.hc_norm(h, streams=self.streams, write_back=kind, branch=branch, inject=inject)
            mixed, inj = K.hc_project(hn, ssp, ahc.down, ahc.up, ahc.scale, eps=self.eps, streams=self.streams, low=ahc.low)
            out = self._gdn(i, mixed, c) if layer.is_linear else self._attention(i, mixed, c)
            mhc = entry["mlp_hc"]
            hm, ssp = K.hc_norm(hn, streams=self.streams, write_back="plain", branch=(out,), inject=inj)
            mixed, inj2 = K.hc_project(hm, ssp, mhc.down, mhc.up, mhc.scale, eps=self.eps, streams=self.streams, low=mhc.low)
            h = hm
            kind, branch = self._moe(i, mixed)
            pending = (kind, branch, inj2)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(h, *pending[1])
        kind, branch, inject = pending
        mix = self.mixer
        hn, ssp = K.hc_norm(h, streams=self.streams, write_back=kind, branch=branch, inject=inject)
        self.last_streams = hn                                    # [R, S*D] before the final mixer (the MTP reads it)
        return K.hc_project(hn, ssp, mix.down, mix.up, mix.scale, eps=self.eps, streams=self.streams, low=mix.low)[0][None]

    def _ple(self, ple: Any, h: mx.array, tokens: np.ndarray, cache: Any) -> mx.array:
        """model.PLELayer on rows h [R, S*D], its projections through ``project`` (row-invariant)."""

        rows = h.shape[0]
        emb_mod = ple.ple_embedding
        history = cache.history
        if history is None:
            history = np.full((1, emb_mod.context), emb_mod.eos, dtype=np.int64)
        ids = emb_mod.ids(history, tokens)
        cache.history = np.concatenate([history, tokens.astype(np.int64)], axis=1)[:, -emb_mod.context:]
        emb = emb_mod(ids)[0]                                                  # [R, E]
        shape = (rows, ple.streams, ple.dims)
        keys = ple.norm_key(project(emb, ple.key_proj)).reshape(shape)
        values = project(emb, ple.value_proj)
        queries = ple.norm_query(h).reshape(shape)
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / float(np.sqrt(ple.dims))
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * values[:, None, :]).reshape(h.shape)
        normed = ple.norm_conv(gated)[None]
        tail = cache.ple_conv if cache.ple_conv is not None else mx.zeros((1, ple.tail, h.shape[-1]), h.dtype)
        conv_in = mx.concatenate([tail, normed], axis=1)
        cache.ple_conv = conv_in[:, -ple.tail:]
        cache.ple_rollback = (history, tokens.astype(np.int64), conv_in)
        return gated + nn.silu(ple.conv1d(conv_in))[0]

    def _write_back(self, h: mx.array, pending: tuple[str, tuple[mx.array, ...], mx.array | None]) -> mx.array:
        kind, branch, inject = pending
        if kind == "none":
            return h
        # the PLE layer reads the streams themselves: apply the pending write-back first
        return K.hc_norm(h, streams=self.streams, write_back=kind, branch=branch, inject=inject)[0]

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call on ``rows`` rows, make ``cache`` hold only its first ``keep`` rows."""

        if keep == rows:
            return
        drop = rows - keep
        ratio = self.cfg.indexer_compress_ratio
        for i, layer in enumerate(self.model.layers):
            c = cache[i]
            if layer.is_linear:
                conv_rows, ssm_rows = self.row_states[i]
                c.conv, c.ssm = conv_rows[keep - 1:keep], ssm_rows[keep - 1:keep]
                c.offset -= drop
                if "ple" in layer:
                    history, tokens, conv_in = c.ple_rollback
                    ple = layer.ple
                    c.history = np.concatenate([history, tokens[:, :keep]], axis=1)[:, -ple.ple_embedding.context:]
                    c.ple_conv = conv_in[:, keep:keep + ple.tail]
            else:
                c.trim(drop, ratio)
