"""Compare production Qwen family forwards with an independent serial cache.

Uses the current LaneEngine family API, including its actual prefill. This checks
engine cache handling, not native/Python kernel parity or inference performance.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.family_common import cache_contents
from tensorfold.engine.lane_engine import LaneEngine, LaneStream
from tensorfold.families import qwen3_5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/models/Qwen3.8-27B-MLX-4bit"))
    parser.add_argument("--output", type=Path, default=Path("build/native-bench/qwen-engine-trace"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = qwen3_5.load(args.model)
    reference = model.make_cache()
    sampling = Sampling(5678, 0.7, 12, 0.8)
    prompt = tokenizer.encode("Write a short Python function that computes the Fibonacci sequence.", add_special_tokens=False)
    engine = LaneEngine(model, **qwen3_5.engine_settings(model))
    stream = LaneStream("trace", prompt, 32, sampling=sampling, drafts=False)
    counts = {"prefill": 0, "hidden": 0, "hidden_rows": 0}
    expected = []
    original_prefill, original_hidden, original_rows = model.prefill, model.hidden, model.hidden_rows

    def equal(label, left, right):
        a, b = np.array(left.astype(mx.float32)), np.array(right.astype(mx.float32))
        if not np.array_equal(a, b):
            np.save(args.output / f"{label}-reference.npy", a)
            np.save(args.output / f"{label}-engine.npy", b)
            detail = f"shapes {a.shape}/{b.shape}"
            if a.shape == b.shape:
                detail += f", different {np.count_nonzero(a != b)}, max error {np.max(np.abs(a - b))}"
            raise AssertionError(f"{label}: {detail}")

    def caches_equal(label, cache):
        assert len(reference) == len(cache)
        for index, (a, b) in enumerate(zip(reference, cache)):
            if hasattr(a, "offset"):
                assert a.offset == b.offset, (label, index, a.offset, b.offset)
            if hasattr(a, "keys") and a.keys is None:
                assert b.keys is None and a.values is None and b.values is None
                continue
            left, right = cache_contents(a), cache_contents(b)
            assert len(left) == len(right)
            for part, (x, y) in enumerate(zip(left, right)):
                equal(f"{label}-layer{index}-part{part}", x, y)

    def compare(kind, cache, reference_call, engine_call):
        index = sum(counts.values())
        caches_equal(f"{index}-before", cache)
        ref = reference_call()
        ref_logits = model.head(ref)
        mx.eval(ref_logits)
        actual = engine_call()  # Leave the family's commit metadata on the real cache.
        equal(f"{index}-hidden", ref, actual)
        equal(f"{index}-logits", ref_logits, model.head(actual))
        caches_equal(f"{index}-after", cache)
        position = model._position(reference)
        draw = model.sample(ref_logits[:, -1:], sampling, [position])
        expected.append(int(draw[0]))
        counts[kind] += 1
        print(f"PASS {kind} {counts[kind]}: position={position}, all {len(cache)} caches exact", flush=True)
        return actual

    def prefill(inputs, cache):
        return compare("prefill", cache, lambda: original_prefill(inputs, reference),
                       lambda: original_prefill(inputs, cache))

    def hidden(inputs, cache, parents=None):
        assert parents is None, "This trace checks serial decoding only"
        return compare("hidden", cache, lambda: original_hidden(inputs, reference),
                       lambda: original_hidden(inputs, cache))

    def hidden_rows(windows, caches, parents=None):
        assert len(windows) == len(caches) == 1 and parents is None
        return compare("hidden_rows", caches[0], lambda: original_hidden(windows[0], reference),
                       lambda: original_rows(windows, caches))

    model.prefill, model.hidden, model.hidden_rows = prefill, hidden, hidden_rows
    engine.add_stream(stream)
    engine.run()
    assert counts["prefill"] > 0 and counts["hidden"] + counts["hidden_rows"] > 0, counts
    assert len(stream.emitted) == 32, stream.emitted
    # A queued terminal step may be evaluated without emitting its draw.
    assert stream.emitted == expected[:len(stream.emitted)], (stream.emitted, expected)
    report = {"checks": counts, "tokens": stream.emitted, "cache_layers": len(reference)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {sum(counts.values())} forwards and 32 emitted tokens match independent serial execution", flush=True)


if __name__ == "__main__":
    main()
