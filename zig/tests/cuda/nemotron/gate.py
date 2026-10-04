#!/usr/bin/env python3
"""Runs the Zig engine on each captured prompt and compares its tokens and speed with the Python engine's."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--prompts", required=True, help="the capture's prompts.json")
    ap.add_argument("--python", required=True, help="the Python run's results.json or bench.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("serial", "drafted"), default="serial")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--extra", default="", help="more tensorfold run options, space separated")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text())
    python = json.loads(Path(a.python).read_text())["results"]
    bad = 0
    rows = []
    for name, ids in prompts.items():
        report = out / f"{a.mode}-{name}.json"
        cmd = [a.bin, "run", a.model, "--tokens", ",".join(map(str, ids)), "--max-tokens", str(a.max_tokens),
               "--report", str(report), "--kernels", a.kernels, *a.extra.split()]
        if a.mode == "serial":
            cmd.append("--no-drafts")
        rc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / f"{a.mode}-{name}.log").write_text(rc.stdout)
        if rc.returncode:
            print(f"FAIL {name}: tensorfold exited {rc.returncode}\n{rc.stdout[-2000:]}")
            bad += 1
            continue
        z = json.loads(report.read_text())
        p = python[f"{a.mode}/{name}"]
        serial = python[f"serial/{name}"]
        same = z["tokens"] == serial["tokens"]
        bad += not same
        rows.append((name, sha12(z["tokens"]), serial["sha"], same, z["ms_per_token"], p["ms_per_token"],
                     z["prefill_seconds"], p.get("prefill_s"), z.get("rounds"), p.get("rounds"),
                     z.get("accepted_drafts"), p.get("accepted")))
    print(f"| prompt | Zig sha | Python serial sha | equal | Zig ms/token | Python {a.mode} ms/token | Zig prefill s | "
          f"Python prefill s | Zig rounds | Python rounds | Zig accepted | Python accepted |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")
    print(f"{'PASS' if bad == 0 else 'FAIL'} {a.mode}: {len(rows) - bad} of {len(prompts)} prompts equal to Python serial")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
