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
    args = p.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from tensorfold.engine.exact_sampling import Sampling, sample_rows
    kind = json.loads((args.model / "config.json").read_text())["model_type"]
    if kind in ("gemma4", "gemma4_text"):
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
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    # Native and oracle prefill on the same fixed 16-token grid.
    for start in range(0, len(tokens), 16):
        logits = forward(tokens[start:start + 16])
        mx.eval(logits)
        if start % 512 == 0:
            print(f"Prefill {start + min(16, len(tokens) - start)}/{len(tokens)}", flush=True)
    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_logits, np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.generate:
        np.save(args.output, np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
        print(f"Saved {args.output}: {logits.shape}")
        return
    settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    pos = len(tokens)
    result = []
    eos = (1, 106, 50) if kind in ("gemma4", "gemma4_text") else (2, 11) if kind == "nemotron_h" else (248044, 248046)
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
