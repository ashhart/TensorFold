#!/usr/bin/env python3
"""The 27B's DFlash2 drafting as the Zig engine's oracle: drafted replies (equal to serial), rounds, Triton launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompts", required=True, help="capture.py's prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="")
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()
    import torch

    from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    eng = Qwen27Engine(Path(a.model), Path(a.draft), streams=1, context=16384, context_explicit=True)
    if rec is not None:
        from tensorfold.cuda.kernels import gdn, prefill_attention, qmm

        rec.wrap(qmm._ext(), ("qmm", "qmm_group", "qmm_prefill"), "qmm")
        rec.wrap(gdn._ext(), ("tree", "replay", "prefill"), "gdn")
        rec.wrap(prefill_attention._ext(), ("prefill_attention",), "prefill_attention")
    prompts = json.loads(Path(a.prompts).read_text())
    results = {}
    for name, ids in prompts.items():
        for _ in range(2):                         # the second run is the timed one
            drafter = eng.draft
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            st, pending = prefill(eng.w, ids, None, drafter, limit=16384)
            trace: list = []
            r = draft_decode(eng.w, st, ids, pending, a.max_tokens, None, drafter, max_rows=eng.max_rows,
                             tree_rows=eng.tree_rows, allow_copy=True, stop_eos=True, trace=trace, inplace=True)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        results[name] = {"tokens": r.tokens, "sha": sha12(r.tokens), "rounds": r.rounds, "drafted": r.drafted_rows,
                         "accepted": r.accepted_drafts, "decode_s": round(r.seconds, 4), "wall_s": round(wall, 4),
                         "widths": r.widths, "paths": [len(t.get("path", [])) for t in trace]}
        print(f"{name}: {len(r.tokens)} tokens sha {sha12(r.tokens)} rounds {r.rounds} accepted {r.accepted_drafts} "
              f"{len(r.tokens) / r.seconds:.1f} tok/s", flush=True)
    (out / "draft.json").write_text(json.dumps({"max_rows": eng.max_rows, "results": results}) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
