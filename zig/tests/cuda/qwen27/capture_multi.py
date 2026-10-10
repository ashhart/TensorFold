#!/usr/bin/env python3
"""The 27B's drafted shared rounds as the Zig engine's oracle: MultiDecoder's windows each round, and every reply."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TOPICS = ["lighthouses", "volcanoes", "bread", "rivers", "chess", "glaciers", "owls", "trains"]
CODE = '''def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
'''


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()
    from tokenizers import Tokenizer

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
    from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder

    model = Path(a.model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    users = [f"Rewrite this function with a docstring that explains each step, keeping the code the same:\n\n{CODE}"]
    users += [f"Write four sentences about {t}." for t in TOPICS]
    prompts = [tok.encode(chat(u), add_special_tokens=False).ids for u in users[:a.streams]]
    eng = Qwen27Engine(model, Path(a.draft), streams=1, context=8192, context_explicit=True)
    if rec is not None:
        from tensorfold.cuda.kernels import gdn, prefill_attention, qmm

        rec.wrap(qmm._ext(), ("qmm", "qmm_group", "qmm_prefill"), "qmm")
        rec.wrap(gdn._ext(), ("tree", "replay", "prefill"), "gdn")
        rec.wrap(prefill_attention._ext(), ("prefill_attention",), "prefill_attention")
    # GB10's planning: no measured costs (every tree node verified), the drafter's block 16 rows a stream
    md = MultiDecoder(eng.w, eng.draft, allow_copy=True, context=8192, keep=1)
    md.depth = False
    rounds: list[list[dict]] = []
    windows = md._windows

    def logged(plan, copied, blocks):
        wins = windows(plan, copied, blocks)
        rounds.append([{"stream": sid, "mode": ["copy", "tree", "one"][mode], "tokens": t, "parents": p}
                       for (sid, mode, _, _), (t, p) in zip(plan, wins)])
        return wins

    md._windows = logged
    scores: list[list[float]] = []                  # each tree's path scores, in finish order
    finish = md.draft.finish_tree

    def scored(*args, **kw):
        t = finish(*args, **kw)
        scores.append([float(x) for x in t[2]])
        return t

    md.draft.finish_tree = scored
    streams = [Stream(list(p), a.max_tokens, None, draft=True, stop_eos=True) for p in prompts]
    for s in streams:
        md.admit(s)
    while any(not s.done for s in streams):
        md.finish(md.round())
    results = [{"tokens": s.out, "sha": sha12(s.out), "rounds": s.rounds, "drafted": s.drafted,
                "accepted": s.accepted} for s in streams]
    for k, r in enumerate(results):
        print(f"stream {k}: {len(r['tokens'])} tokens sha {r['sha']} rounds {r['rounds']} accepted {r['accepted']}",
              flush=True)
    (out / "multi.json").write_text(json.dumps({"prompts": prompts, "max_tokens": a.max_tokens, "results": results,
                                                "rounds": rounds, "scores": scores}) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
    print(f"{len(rounds)} rounds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
