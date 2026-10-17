"""Gemma 4 dense kernel fixtures from gilby's RowDecode: each decode kernel's inputs and outputs as npy, by rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,5", help="layers to dump (a sliding one and a global one)")
    parser.add_argument("--rows", default="1,2,16", help="window widths, each after the same prompt")
    parser.add_argument("--prompt", type=int, default=1100, help="tokens decoded before each window (past the window)")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.gemma4_text import geglu
    from tensorfold.families.gemma4.model import load
    from tensorfold.kernels.gemma.v1.attention import Rows, attend
    from tensorfold.kernels.gemma.v1.glue import attn_tail, dense_tail, qkv_prep

    family, _ = load(args.model, backend="lane", check=False)
    d = family.decode
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(7)

    def save(name: str, a: mx.array) -> None:
        a = a.astype(mx.float32) if a.dtype == mx.bfloat16 else a
        np.save(out / f"{name}.npy", np.asarray(a))

    for i in (int(x) for x in args.layers.split(",")):
        save(f"L{i}.inv_freq", d.inv_freq[i])
    meta = {"layers": args.layers, "prompt": args.prompt, "eps": d.eps_value, "windows": {}}
    for rows in (int(x) for x in args.rows.split(",")):
        cache = family.make_cache()
        layers = family._layers(cache)
        prompt = rng.integers(1000, 200000, args.prompt).tolist()
        for t in prompt:
            mx.eval(family.hidden(mx.array([t]), cache))
        tokens = mx.array(rng.integers(1000, 200000, rows).tolist())
        p0 = int(layers[0].offset)
        positions = list(range(p0, p0 + rows))
        at = mx.array(positions, dtype=mx.int32)
        tag = f"r{rows}"
        h = d.backbone.embed_tokens(tokens) * d.backbone.embed_scale
        normed = mx.fast.rms_norm(h, d.layers[0].input_layernorm.weight, d.eps_value)
        save(f"{tag}.tokens", tokens.astype(mx.int32))
        save(f"{tag}.embed", h)
        save(f"{tag}.normed0", normed)
        wanted = {int(x) for x in args.layers.split(",")}
        for i in range(max(wanted) + 1):
            layer, attn = d.layers[i], d.layers[i].self_attn
            qkv = d.qkv[i](normed)
            q, k, v = qkv_prep(qkv, attn.q_norm.weight, attn.k_norm.weight, d.inv_freq[i], at, d.eps,
                               heads=attn.n_heads, kv_heads=attn.n_kv_heads, head_dim=attn.head_dim,
                               values_are_keys=attn.use_k_eq_v)
            sliding = d.sliding(i)
            keys, values = layers[i].write(k[None], v[None])
            kinds = Rows(positions, d.window if sliding else 0, layers[i].ring, attn.head_dim)
            o_in = attend(q, keys, values, kinds, k, v, float(attn.scale))
            o = d.o[i](o_in.reshape(rows, -1))
            pre = layer.pre_feedforward_layernorm.weight
            hn, n_mlp, _, _ = attn_tail(h, o, layer.post_attention_layernorm.weight, pre, pre, pre, d.eps)
            gu = d.gate_up[i](n_mlp)
            gate, up = d.gate_up[i].split(gu)
            act = geglu(gate, up)
            y = d.down[i](act)
            nxt = d.layers[i + 1].input_layernorm.weight if i + 1 < len(d.layers) else d.backbone.norm.weight
            h2, normed2 = dense_tail(hn, y, layer.post_feedforward_layernorm.weight, layer.layer_scalar, nxt, d.eps)
            if i in wanted:
                p = f"{tag}.L{i}"
                for name, a in (("x", normed), ("qkv", qkv), ("q", q), ("k", k), ("v", v), ("o_in", o_in), ("o", o),
                                ("h", h), ("hn", hn), ("n_mlp", n_mlp), ("gate_up", gu), ("act", act), ("y", y),
                                ("h2", h2), ("normed2", normed2)):
                    save(f"{p}.{name}", a)
                # the cache rows the window's attention read: every earlier key and value in position order
                lo = max(0, p0 - d.window + 1) if sliding else 0
                idx = mx.array([(x % layers[i].ring) if sliding else x for x in range(lo, p0)], dtype=mx.int32)
                save(f"{p}.keys_before", keys[0][:, idx])
                save(f"{p}.values_before", values[0][:, idx])
                meta["windows"][f"{p}"] = {"positions": positions, "first_key": lo, "scale": float(attn.scale)}
            h, normed = h2, normed2
        final = mx.random.normal((rows, d.layers[0].input_layernorm.weight.shape[0])).astype(mx.bfloat16)
        logits = d.logits(final)
        save(f"{tag}.head_in", final)
        save(f"{tag}.head_raw", d.head(final))
        save(f"{tag}.logits", logits)
        mx.eval(logits)
        print(f"rows {rows}: dumped layers {sorted(wanted)} at positions {p0}..{p0 + rows - 1}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
