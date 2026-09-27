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
    parser.add_argument("--tokens", default="1,2,3,4")
    parser.add_argument("--output", default="build/native-checks/reference.npy")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--compare-reports", nargs="+")
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.compare_reports:
        reports = [json.loads(Path(p).read_text()) for p in args.compare_reports]
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
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_tree

    model, tokenizer = load_lane_model(Path(args.model))
    lane_qmm.install(model, rows=128, tile=True, wide=True)
    core = model.language_model.model
    head = model.language_model.lm_head
    tokens = [int(x) for x in args.tokens.split(",")]
    cache = make_prompt_cache(model)
    if args.generate:
        from tensorfold.engine.exact_sampling import Sampling, sample_rows
        settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
        logits, record = lane_tree.tree_forward(core, head, tokens, list(range(-1, len(tokens)-1)), cache, 0)
        lane_tree.commit_tree(cache, record, list(range(len(tokens))), len(tokens), 0)
        position = len(tokens)
        pending = sample_rows(logits[0, -1:], [position], settings)[0] if args.temperature else int(mx.argmax(logits[0, -1]).item())
        generated = [pending]
        while len(generated) < args.generate and pending not in (248044, 248046):
            logits, record = lane_tree.tree_forward(core, head, [pending], [-1], cache, position)
            lane_tree.commit_tree(cache, record, [0], 1, position)
            position += 1
            pending = sample_rows(logits[0], [position], settings)[0] if args.temperature else int(mx.argmax(logits[0, -1]).item())
            generated.append(pending)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(np.array(generated, dtype="<u4").tobytes()).hexdigest()
        out.write_text(json.dumps(dict(prompt_tokens=tokens, tokens=generated, token_sha256=sha)))
        print(f"Saved {out}: {len(generated)} tokens, SHA-256 {sha}")
        return
    logits, _ = lane_tree.tree_forward(core, head, tokens, list(range(-1, len(tokens)-1)), cache, 0)
    mx.eval(logits)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(logits.astype(mx.float32)))
    print(f"Saved {out}: shape={logits.shape}, argmax={mx.argmax(logits, axis=-1).tolist()}")


if __name__ == "__main__":
    main()
