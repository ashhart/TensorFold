"""Python numerical oracle for native Nemotron/Flash Next; never launches native code."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", help="Explicit prompt IDs, including for generation")
    p.add_argument("--dump-logits", type=Path)
    p.add_argument("--generate", type=int, default=0)
    p.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--metal-sampling", action="store_true")
    p.add_argument("--simd", action="store_true")
    p.add_argument("--synthetic-glm", action="store_true")
    p.add_argument("--synthetic-glm-layout", action="store_true")
    p.add_argument("--synthetic-glm-mixed", action="store_true")
    p.add_argument("--serial-rows", action="store_true")
    p.add_argument("--trace-layers", action="store_true")
    p.add_argument("--state-directory", type=Path)
    args = p.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    if args.synthetic_glm or args.synthetic_glm_mixed:
        from tests.glm5_fakes import write_checkpoint
        formats = {
            "model.language_model.layers.0.self_attn.q_proj": (2, 32),
            "model.language_model.layers.0.self_attn.k_proj": (3, 64),
            "model.language_model.layers.0.self_attn.v_proj": (6, 128),
            "model.language_model.layers.0.self_attn.f_b_proj": (2, 32),
            "model.language_model.layers.0.self_attn.g_b_proj": (3, 32),
            "model.language_model.layers.0.mlp.gate_proj": (2, 128),
            "model.language_model.layers.3.mlp.shared_experts.gate_proj": (6, 64),
            "model.language_model.layers.3.mlp.shared_experts.up_proj": (3, 32),
            "model.language_model.embed_tokens": (3, 32),
            "lm_head": (2, 128),
        } if args.synthetic_glm_mixed else {}
        write_checkpoint(args.model, overrides={key: dict(bits=bits, group_size=group) for key, (bits, group) in formats.items()})
        args.synthetic_glm = True
    if args.synthetic_glm_layout:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from test_glm5_layouts import write_mlxlm_checkpoint
        if not (args.model / "mlxlm/config.json").exists():
            write_mlxlm_checkpoint(args.model)
        args.model = args.model / "mlxlm"
    from tensorfold.engine.exact_sampling import Sampling, sample_rows
    kind = json.loads((args.model / "config.json").read_text())["model_type"]
    if kind == "glm5_next":
        from tensorfold.families.glm5_next.weights import load_backbone
        model = load_backbone(args.model)
        cache = model.make_cache()
        tokenizer = None
        if args.trace_layers:
            from tensorfold.families.glm5_next.model import Layer, hc_expand
            original_layer = Layer.__call__
            layer_ids = {id(layer): i for i, layer in enumerate(model.layers)}
            positions = [0] * len(model.layers)
            args.state_directory.mkdir(parents=True, exist_ok=True)
            def traced_layer(self, x, *a, **kw):
                if id(self) in layer_ids:
                    i = layer_ids[id(self)]
                    caches, lengths, decode = a
                    def save_stage(label, value):
                        for row in range(value.shape[0]):
                            np.save(args.state_directory / f"trace-{positions[i] + row}-{i}-{label}.npy", np.asarray(value[row:row + 1].astype(mx.float32)))
                    xc, post, comb = self.attn_hc.split(x, decode)
                    normed = mx.fast.rms_norm(xc, self.in_norm, self.eps)
                    save_stage("attn-input", normed)
                    branch = self.attn(normed, caches, lengths, decode)
                    save_stage("attn-output", branch)
                    x = hc_expand(branch, x, post, comb, decode)
                    save_stage("attn-expanded", x)
                    xc, post, comb = self.ffn_hc.split(x, decode)
                    normed = mx.fast.rms_norm(xc, self.post_norm, self.eps)
                    save_stage("ffn-input", normed)
                    branch = self.mlp(normed, decode)
                    save_stage("ffn-output", branch)
                    result = hc_expand(branch, x, post, comb, decode)
                else:
                    result = original_layer(self, x, *a, **kw)
                if id(self) in layer_ids:
                    i = layer_ids[id(self)]
                    for row in range(result.shape[0]):
                        np.save(args.state_directory / f"trace-{positions[i] + row}-{i}.npy", np.asarray(result[row:row + 1].astype(mx.float32)))
                    positions[i] += result.shape[0]
                return result
            Layer.__call__ = traced_layer
        if args.synthetic_glm or args.synthetic_glm_layout:
            from tensorfold.families.glm5_next.mtp import load as load_mtp
            mtp = load_mtp(model)
            mtp_cache = mtp.make_cache()
        forward = lambda ids: model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))
    elif kind in ("gemma4", "gemma4_text"):
        from tensorfold.families.gemma4 import load
        model, tokenizer = load(args.model, lane_kernels="off")
        cache = model.make_cache()
        forward = lambda ids: model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))
    elif kind == "nemotron_h":
        from mlx_lm import load
        from tensorfold.kernels.nemotron.lightning.v1 import kernels
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm
        from tools.native_legacy import nemotron_rows, nemotron as legacy_nemotron
        model, tokenizer = load(str(args.model))
        # Native retains the original combined conv/scan and per-slot experts.
        kernels.mamba_step = legacy_nemotron.mamba_step
        fused = kernels.FusedDecode(model)
        # Build the same explicit operations as native, without mx.compile
        # combining neighboring elementwise operations.
        fused._block = lambda index, kind, nxt: (fused._mamba_block(index, nxt) if kind == "M"
                                                else fused._moe_block(index, nxt))
        if args.simd:
            fused.lane_attention = False
        def experts(index, mixer, x):
            logits = kernels.router_logits(x, mixer.gate.weight)
            ids, weights = kernels.route(logits, fused.gate_bias[index], fused.top_k, fused.scaling)
            return nemotron_rows.experts(mixer.switch_mlp, x, ids), weights, mixer.shared_experts(x)
        fused._moe = experts
        holder = nn.Module()
        holder.model = model
        holder.stacked = [x for x, _ in fused.qkv.values()]
        if args.simd:
            for _, module in holder.named_modules():
                if isinstance(module, nn.QuantizedLinear):
                    module.__class__ = nemotron_rows.RowLinear
        else:
            lane_qmm.install(holder, rows=128, tile=True, wide=True)
        cache = model.make_cache()
        forward = lambda ids: model.lm_head(fused(mx.array([ids], dtype=mx.uint32), cache))
    elif kind == "qwen4_exp":
        from tensorfold.families.qwen4_exp.model import load
        from tensorfold.families.qwen4_exp.decode import FusedDecode
        from tensorfold.families.qwen4_exp.runtime import FlashNext
        from types import SimpleNamespace
        from tensorfold.kernels.qwen.flash_next.v1 import embed as flash_kernels
        from tensorfold.families.qwen4_exp import decode
        decode.DENSE = "rows"
        # Keep the 32 GB PLE tables sharded. The reference embedding performs the
        # same lookup/dequantization without materializing a second concatenated copy.
        flash_kernels.PleTables = lambda embedding: embedding
        flash_kernels.ple_lookup = lambda ids, tables: tables(ids)
        model, tokenizer = load(args.model, lazy=True)
        model.__dict__["fused"] = FusedDecode(model)
        # The lookup adapter holds the original sharded embedding. Remove its
        # fused alias so calling it cannot recurse into this same adapter.
        for layer in model.layers:
            if "ple" in layer:
                layer.ple.ple_embedding.__dict__.pop("fused_tables", None)
        cache = model.make_cache()
        # The serving runtime uses a row-invariant vocabulary projection; the raw
        # model's __call__ uses MLX's batch-dependent quantized matmul instead.
        runtime = SimpleNamespace(model=model)
        forward = lambda ids: FlashNext.head(runtime, model.hidden(mx.array([ids], dtype=mx.int32), cache))
    else:
        raise ValueError(kind)
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else list(range(1, 41)) if args.synthetic_glm or args.synthetic_glm_layout else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    # Native and oracle prefill on the same fixed 16-token grid.
    for start in range(0, len(tokens), 16):
        chunk = tokens[start:start + 16]
        if kind == "glm5_next" and args.serial_rows:
            hidden_rows = []
            logit_rows = []
            for token in chunk:
                logit_rows.append(forward([token]))
                hidden_rows.append(model.last_normed)
            logits = mx.concatenate(logit_rows, axis=1)
            model.last_normed = mx.concatenate(hidden_rows)
        else:
            logits = forward(chunk)
        mx.eval(logits)
        if kind == "glm5_next" and (args.synthetic_glm or args.synthetic_glm_layout):
            next_tokens = [t + 1 for t in tokens[start:start + 16]]
            if args.serial_rows:
                mtp_hidden = mx.concatenate([mtp(model, model.last_normed[j:j + 1], mx.array([token]), [mtp_cache], (1,), True) for j, token in enumerate(next_tokens)])
            else:
                mtp_hidden = mtp(model, model.last_normed, mx.array(next_tokens), [mtp_cache], (len(next_tokens),), True)
            mtp_logits = mtp.logits(model, mtp_hidden)
            from tensorfold.families.glm5_next.linear import project
            mtp_input = mx.concatenate([mx.fast.rms_norm(model.embed_tokens(mx.array(next_tokens)), mtp.enorm, mtp.eps), mx.fast.rms_norm(model.last_normed, mtp.hnorm, mtp.eps)], axis=-1)
            mtp_projection = project(mtp_input, mtp.eh_proj, rows_exact=True)
            if args.state_directory:
                args.state_directory.mkdir(parents=True, exist_ok=True)
                np.save(args.state_directory / f"mtp-projection-{start // 16}.npy", np.asarray(mtp_projection.astype(mx.float32)))
                np.save(args.state_directory / f"mtp-input-{start // 16}.npy", np.asarray(mtp_input.astype(mx.float32)))
                np.save(args.state_directory / f"hidden-{start // 16}.npy", np.asarray(model.last_normed.astype(mx.float32)))
                np.save(args.state_directory / f"logits-{start // 16}.npy", np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
            mx.eval(mtp_hidden, mtp_logits)
        if start % 512 == 0:
            print(f"Prefill {start + min(16, len(tokens) - start)}/{len(tokens)}", flush=True)
    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_logits, np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state_directory:
        args.state_directory.mkdir(parents=True, exist_ok=True)
        if kind == "glm5_next" and (args.synthetic_glm or args.synthetic_glm_layout):
            np.save(args.state_directory / "mtp-hidden.npy", np.asarray(mtp_hidden.astype(mx.float32)))
            np.save(args.state_directory / "mtp-logits.npy", np.asarray(mtp_logits.astype(mx.float32)))
            np.save(args.state_directory / "mtp-projection.npy", np.asarray(mtp_projection.astype(mx.float32)))
            for key in ("keys", "ik", "ig", "pool"):
                array = getattr(mtp_cache, key)
                length = mtp_cache.offset // model.args.index_kpool if key == "pool" else mtp_cache.offset
                np.save(args.state_directory / f"mtp-{key}.npy", np.asarray(array[:length].astype(mx.float32)))
        for i, layer in enumerate(cache):
            for key, source in (("conv", "conv"), ("state", "ssm"), ("keys", "keys"), ("ik", "ik"), ("ig", "ig"), ("pool", "pool")):
                array = getattr(layer, source, None)
                if array is None:
                    continue
                if key in ("keys", "ik", "ig"):
                    array = array[:layer.offset]
                elif key == "pool":
                    array = array[:layer.offset // model.args.index_kpool]
                np.save(args.state_directory / f"layer{i}-{key}.npy", np.asarray(array.astype(mx.float32)))
    if not args.generate:
        np.save(args.output, np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
        print(f"Saved {args.output}: {logits.shape}")
        return
    settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    pos = len(tokens)
    result = []
    eos = model.args.eos_token_id if kind == "glm5_next" else (1, 106, 50) if kind in ("gemma4", "gemma4_text") else (2, 11) if kind == "nemotron_h" else (248044, 248046)
    while len(result) < args.generate:
        if args.metal_sampling:
            from tensorfold.engine.gpu_sampling import sample
            token = int(sample(logits.reshape(-1, logits.shape[-1])[-1:], settings if args.temperature else None, [pos]).item())
        else:
            token = sample_rows(logits.reshape(-1, logits.shape[-1])[-1:], [pos], settings)[0] if args.temperature else int(mx.argmax(logits.reshape(-1, logits.shape[-1])[-1]).item())
        result.append(token)
        if token in eos:
            break
        logits = forward([token])
        mx.eval(logits)
        pos += 1
    digest = hashlib.sha256(np.asarray(result, dtype="<u4").tobytes()).hexdigest()
    args.output.write_text(json.dumps(dict(prompt_tokens=tokens, tokens=result, token_sha256=digest,
                                         peak_mlx_bytes=mx.get_peak_memory(), active_mlx_bytes=mx.get_active_memory())))
    print(f"Saved {args.output}: {len(result)} tokens, {digest}")


if __name__ == "__main__":
    main()
