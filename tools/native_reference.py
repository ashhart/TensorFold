"""Numerical oracle for the native Qwen port (no native process launching).

Generate reference logits with this checkout's actual lane forward, or compare
already generated .npy files. Artifacts live under build/native-checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/models/Qwen3.8-27B-MLX-4bit")
    parser.add_argument("--tokens", help="Exact prompt IDs, also honored during generation")
    parser.add_argument("--dump-logits", type=Path, help="Save the final prefill block before generation")
    parser.add_argument("--metal-sampling", action="store_true")
    parser.add_argument("--simd", action="store_true", help="Use original SIMD projections and row attention, without stacked projections")
    parser.add_argument("--production-kernels", action="store_true", help="Diagnostic: use production loading/fusion with lane-tree prefill (not the serving engine's prefill)")
    parser.add_argument("--disable-fusion", action="store_true", help="Diagnostic: turn off stacked consumers after the production load")
    parser.add_argument("--output", default="build/native-checks/reference.npy")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--compare-reports", nargs="+")
    parser.add_argument("--require-rounds", action="store_true", help="Reject trivial EOS-before-decode parity runs")
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.compare_reports:
        reports = [json.loads(Path(p).read_text()) for p in args.compare_reports]
        if args.require_rounds:
            for report in reports:
                assert report["rounds"] > 0 and len(report["tokens"]) > 1, "No decode rounds exercised"
        for report in reports[1:]:
            assert report["prompt_tokens"] == reports[0]["prompt_tokens"], "Tokenizer mismatch"
            assert report["tokens"] == reports[0]["tokens"], "Output token mismatch"
        print(f"PASS: {len(reports)} reports have identical prompts and {len(reports[0]['tokens'])} output token IDs")
        return
    if args.compare:
        a, b = [np.load(p) for p in args.compare]
        if b.dtype.kind == "V" and b.dtype.itemsize == 2:
            b = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        print(f"shapes: {a.shape} / {b.shape}")
        print(f"equal elements: {np.count_nonzero(a == b)}/{a.size}")
        print(f"max absolute difference: {np.max(np.abs(a - b))}")
        print(f"argmax reference: {a.argmax(axis=-1).tolist()}")
        print(f"argmax native: {b.argmax(axis=-1).tolist()}")
        if not np.array_equal(a, b):
            raise SystemExit(1)
        return
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from tensorfold.families.qwen3_5 import load_lane_model
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_tree, row_attention, simd_qmm

    if args.production_kernels:
        if args.simd:
            parser.error("--production-kernels cannot be combined with --simd")
        from tensorfold.families.qwen3_5 import load
        family, tokenizer = load(Path(args.model))
        model = family.inner
        if args.disable_fusion:
            from tensorfold.kernels.qwen.dense.v1 import lane_fuse
            lane_fuse.enabled = False
    else:
        if args.disable_fusion:
            parser.error("--disable-fusion requires --production-kernels")
        model, tokenizer = load_lane_model(Path(args.model))
    if args.simd:
        # Native keeps the individual checkpoint projections. row_forward.install
        # stacks GDN projections, changing simd_qmm's shape-dependent split sums.
        # Compose the original unstacked lane host with the original SIMD kernels.
        simd_qmm.install(model)
        # tree_forward now delegates to lane_multi. Replace its attention entry
        # point, rather than setting the removed lane_attention.lane_tree_sdpa.
        from types import SimpleNamespace
        from tensorfold.kernels.qwen.dense.v1 import stream_attention
        def plan(parents, starts, heads, kv_heads):
            if len(parents) != 1 or len(starts) != 1:
                raise ValueError("The native SIMD oracle supports one stream")
            return SimpleNamespace(streams=1, rows=len(parents[0]), parents=parents[0], start=starts[0])
        def attention(q, kv, scale, layout):
            k, v = kv[0]
            return row_attention.row_sdpa(q, k, v, scale, layout.start, layout.parents)
        stream_attention.Plan = plan
        stream_attention.tree_sdpa = attention
    elif not args.production_kernels:
        lane_qmm.install(model, rows=128, tile=True, wide=True)
    core = model.language_model.model
    head = model.language_model.lm_head
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    if not tokens:
        raise ValueError("Empty prompt")
    cache = make_prompt_cache(model)
    # Same 128-row grid as the native CLI; commit the whole prompt before decoding.
    for start in range(0, len(tokens), 128):
        block = tokens[start:start + 128]
        logits, record = lane_tree.tree_forward(core, head, block, list(range(-1, len(block)-1)), cache, start)
        mx.eval(logits)
        lane_tree.commit_tree(cache, record, list(range(len(block))), len(block), start)
        if start % 1024 == 0:
            print(f"Prefill {start + len(block)}/{len(tokens)}", flush=True)
    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_logits, np.array(logits.astype(mx.float32)))
    if args.generate:
        from tensorfold.engine.exact_sampling import Sampling, sample_rows
        settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        def select(logits, position):
            last = logits[0, -1:]
            if args.metal_sampling:
                from tensorfold.engine.gpu_sampling import sample
                return int(sample(last, settings if args.temperature else None, [position]).item())
            return sample_rows(last, [position], settings)[0] if args.temperature else int(mx.argmax(last).item())
        position = len(tokens)
        pending = select(logits, position)
        generated = [pending]
        while len(generated) < args.generate and pending not in (248044, 248046):
            logits, record = lane_tree.tree_forward(core, head, [pending], [-1], cache, position)
            lane_tree.commit_tree(cache, record, [0], 1, position)
            position += 1
            pending = select(logits, position)
            generated.append(pending)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(np.array(generated, dtype="<u4").tobytes()).hexdigest()
        out.write_text(json.dumps(dict(prompt_tokens=tokens, tokens=generated, token_sha256=sha,
                                      peak_mlx_bytes=mx.get_peak_memory(), active_mlx_bytes=mx.get_active_memory())))
        print(f"Saved {out}: {len(generated)} tokens, SHA-256 {sha}")
        return
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(logits.astype(mx.float32)))
    print(f"Saved {out}: shape={logits.shape}, argmax={mx.argmax(logits, axis=-1).tolist()}")


if __name__ == "__main__":
    main()
