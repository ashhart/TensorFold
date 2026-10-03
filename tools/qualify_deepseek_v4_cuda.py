#!/usr/bin/env python3
"""Real-GGUF qualification: native 160Ki context, retained 128Ki prefix, seeded serial/draft agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizers import Tokenizer

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda.app import DeepSeekTemplate
from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    engine = DeepSeekEngine(args.model_dir)
    if engine.limit != 163840 or engine.retained_prefix != 131072 or engine.drafter is None:
        raise ValueError("qualification requires 160Ki context, retained 128Ki prefix and a DSpark GGUF")
    engine.request.stop_eos = False
    sampling = Sampling(43, 0.6, 20, 0.95)
    results = []

    def run(case, prompt, count, draft):
        reply = []

        def feed(ids):
            reply.extend(ids)
            return False  # Shared CUDA serving: True stops, False continues.

        stats = engine.generate(prompt, count, sampling, feed, draft=draft)
        result = {
            "case": case,
            "prompt_tokens": len(prompt),
            "draft": draft,
            "stats": stats,
            "ids": reply,
            "text": tok.decode(reply, skip_special_tokens=False),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"passed": False, "results": results}, indent=2) + "\n")
        return reply, stats

    try:
        short = tok.encode(
            DeepSeekTemplate().render(
                [
                    {
                        "role": "user",
                        "content": "Write a Python function that checks whether an integer is prime and explain how it works.",
                    }
                ],
                tools=None,
                enable_thinking=False,
            )
        ).ids
        plain, _ = run("short", short, 32, False)
        drafted, _ = run("short", short, 32, True)
        assert plain == drafted, "short seeded replies differ"
        repeated = tok.encode("You are a helpful assistant. Explain arithmetic clearly. ").ids
        long = ([0] + repeated * 15000)[:131093]
        plain, _ = run("128Ki-cold", long, 128, False)
        drafted, warm = run("128Ki-warm", long, 128, True)
        assert plain == drafted and warm["cached"] == 131072, "retained-prefix draft mismatch"
        again, warm = run("128Ki-warm", long, 128, False)
        assert again == plain and warm["cached"] == 131072, "retained-prefix serial mismatch"
        grown = long + plain + tok.encode(" Now continue.").ids
        plain, _ = run("128Ki-growing", grown, 64, False)
        drafted, warm = run("128Ki-growing", grown, 64, True)
        assert plain == drafted and warm["cached"] == 131072, "growing-prefix seeded replies differ"
        # Changed-prefix invalidation needs a short fixture, not a second 128Ki cold pass.
        engine.retained_prefix = 2048
        small = ([0] + repeated * 300)[:2053]
        run("changed-prefix", small, 8, False)
        small[100] = 3 if small[100] != 3 else 4
        _, cold = run("changed-prefix", small, 8, True)
        assert cold["cached"] == 0, "changed prefix reused stale state"
        if args.output:
            args.output.write_text(
                json.dumps(
                    {"passed": True, "context": engine.limit, "retained_prefix": 131072, "results": results}, indent=2
                )
                + "\n"
            )
        print("QUALIFICATION PASSED", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
