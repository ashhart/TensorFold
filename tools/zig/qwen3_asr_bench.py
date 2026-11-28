"""One-stream throughput of tf-qwen3-asr over a clip set: realtime factor, tower rows/s, LM prompt and decode tok/s."""
# usage: python3 -I tools/zig/qwen3_asr_bench.py run BINARY MODEL CLIPS REF OUT [--rounds 3]; then summary OUT CLIPS

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

TIMES = re.compile(r"^TIMES mel ([\d.]+) ms, tower ([\d.]+) ms \((\d+) rows\), prompt pass ([\d.]+) ms \((\d+) rows\), "
                   r"decode (\d+) tokens ([\d.]+) tok/s")


def run(a):
    os.makedirs(a.out, exist_ok=True)
    with open(a.binary, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    with open(os.path.join(a.out, "md5.txt"), "w") as f:
        f.write(f"{md5}  {a.binary}\n")
    with open(os.path.join(a.clips, "manifest.json"), encoding="utf-8") as f:
        clips = json.load(f)
    for r in range(1, a.rounds + 1):
        for c in clips:
            wav = os.path.join(a.clips, c["id"] + ".wav")
            case = os.path.join(a.ref, c["id"] + ".win")
            cmd = [a.binary, a.model, wav, "--case", case, "--ids-only", "--tokens", "448"]
            with open(os.path.join(a.out, f"{c['id']}.r{r}.txt"), "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
            print(f"{c['id']} r{r} rc={rc}", flush=True)


def summary(a):
    with open(os.path.join(a.clips, "manifest.json"), encoding="utf-8") as f:
        clips = json.load(f)
    rounds = {}  # round -> totals
    for name in sorted(os.listdir(a.out)):
        m = re.match(r"(.+)\.r(\d+)\.txt$", name)
        if not m:
            continue
        with open(os.path.join(a.out, name), encoding="utf-8") as f:
            t = next((TIMES.match(line) for line in f if line.startswith("TIMES")), None)
        if t is None:
            sys.exit(f"{name}: no TIMES line")
        mel, tower, rows, prompt, ids, toks, tps = (float(x) for x in t.groups())
        secs = next(c["seconds"] for c in clips if c["id"] == m.group(1))
        s = rounds.setdefault(int(m.group(2)), {k: 0.0 for k in ("audio", "mel", "tower", "rows", "prompt", "ids", "toks", "dec")})
        for k, v in (("audio", secs), ("mel", mel / 1e3), ("tower", tower / 1e3), ("rows", rows), ("prompt", prompt / 1e3),
                     ("ids", ids), ("toks", toks), ("dec", toks / tps if tps else 0.0)):
            s[k] += v
    per = []
    for r, s in sorted(rounds.items()):
        busy = s["mel"] + s["tower"] + s["prompt"] + s["dec"]
        per.append({"round": r, "rtf": busy / s["audio"], "speed": s["audio"] / busy, "mel ms/s audio": 1e3 * s["mel"] / s["audio"],
                    "tower rows/s": s["rows"] / s["tower"], "prompt tok/s": s["ids"] / s["prompt"],
                    "decode tok/s": s["toks"] / s["dec"], "audio s": s["audio"]})
    keys = [k for k in per[0] if k != "round"]
    print(f"{'':<16}" + "".join(f"{'round ' + str(p['round']):>12}" for p in per) + f"{'median':>12}")
    for k in keys:
        vals = [p[k] for p in per]
        print(f"{k:<16}" + "".join(f"{v:>12.4g}" for v in vals) + f"{statistics.median(vals):>12.4g}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    for x in ("binary", "model", "clips", "ref", "out"):
        r.add_argument(x)
    r.add_argument("--rounds", type=int, default=3)
    s = sub.add_parser("summary")
    s.add_argument("out")
    s.add_argument("clips")
    a = p.parse_args()
    run(a) if a.mode == "run" else summary(a)


if __name__ == "__main__":
    main()
