"""The Zig engine's tokens and speed against a Python oracle's JSON, per prompt (no MLX needed)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def sha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("ids", type=Path)
    ap.add_argument("oracle", type=Path)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--drafts", action="store_true")
    ap.add_argument("--sample", default="", help="sampling flags for both sides, e.g. '--temperature 0.7 --seed 1234'")
    args = ap.parse_args()
    oracle = json.loads(args.oracle.read_text())
    bad = 0
    for name, ids in json.loads(args.ids.read_text()).items():
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "r.json"
            cmd = [args.binary, "run", args.model, "--tokens", ",".join(map(str, ids)), "--max-tokens",
                   str(args.tokens), "--warmup", "--prefill-chunk", str(args.chunk),
                   "--report", str(report)] + ([] if args.drafts else ["--no-drafts"]) + args.sample.split()
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = json.loads(report.read_text())
        got, want = r["tokens"], oracle[name]["tokens"]
        diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        if diff is None and len(got) != len(want):
            diff = min(len(got), len(want))
        same = diff is None
        bad += not same
        rate = (len(got) - 1) / max(r["decode_seconds"], 1e-9)
        print(f"{name}: first {'==' if got[:1] == want[:1] else '!='} sequence "
              f"{'== (all ' + str(len(got)) + ')' if same else '!= at ' + str(diff)} sha {sha(got)} "
              f"(python {oracle[name]['sha']}); decode {rate:.1f} tok/s, {r['token_ms_wall']:.3f} ms a token, "
              f"{r['rounds']} rounds, {r['accepted_drafts']} accepted, GPU ms step {r['step_ms_gpu']:.3f} "
              f"verify {r['verify_ms_gpu']:.3f} draft {r['draft_ms_gpu']:.3f}; prefill "
              f"{r['prefill_seconds'] * 1e3:.1f} ms", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
