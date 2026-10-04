"""The Python engine's greedy tokens with prompts prefilled in chunks of at most --chunk tokens (decode kernels)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("ids", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--drafts", action="store_true", help="also time drafted decoding (MTP head)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0: greedy; else keyed sampling (--seed ...)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    args = ap.parse_args()

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.nemotron_h import load

    fam, tok = load(args.model)
    eos: set[int] = set()
    for attr in ("eos_token_ids", "eos_token_id"):
        v = getattr(tok, attr, None)
        eos |= set(v) if isinstance(v, (list, set, tuple)) else ({int(v)} if v is not None else set())
    import time

    sampling = None if args.temperature == 0 else Sampling(seed=args.seed, temperature=args.temperature,
                                                          top_k=args.top_k, top_p=args.top_p, min_p=args.min_p)

    def once(ids: list[int], tokens: int, drafts: bool) -> tuple[Any, float]:
        engine = LaneEngine(fam, prefill_plan=PrefillPlan(args.chunk))
        stream = LaneStream(stream_id=f"{tokens}-{time.perf_counter()}", prompt_ids=ids, max_new_tokens=tokens,
                            eos_ids=frozenset(eos), sampling=sampling, drafts=drafts)
        engine.add_stream(stream)
        started = time.perf_counter()
        engine.run()
        return stream, time.perf_counter() - started

    results = {}
    for name, ids in json.loads(args.ids.read_text()).items():
        entry: dict[str, Any] = {}
        for drafts in (False, True) if args.drafts else (False,):
            _, first = once(ids, 1, drafts)          # the harness's rule: decode time is the run past its first token
            stream, total = once(ids, args.tokens, drafts)
            out = [int(t) for t in stream.emitted]
            rate = (len(out) - 1) / max(total - first, 1e-9)
            kind = "drafted" if drafts else "serial"
            entry[kind] = {"tok_s": rate, "rounds": int(getattr(stream, "rounds", 0)),
                           "accepted": int(getattr(stream, "accepted", 0)), "sha": sha(out)}
            if not drafts:
                entry.update({"tokens": out, "sha": sha(out)})
            print(f"RESULT {name} chunk {args.chunk} {kind} n {len(out)} sha {sha(out)} decode {rate:.1f} tok/s "
                  f"rounds {entry[kind]['rounds']} accepted {entry[kind]['accepted']}", flush=True)
        results[name] = entry
    args.out.write_text(json.dumps(results) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
