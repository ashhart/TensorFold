#!/usr/bin/env python3
"""R536: the prompt pass's fixed term, shape by shape -- the price of `prefill_many`.

One serve, cache off, TF_DSV41_STAGE_CB=1 (the GPU's own per-stage seconds; the wall is the instrument's).
A prompt of N rows is one forward; the same text at 280/560/1,120/2,240/4,480 rows gives each stage's
fixed + marginal cost. The merge question: does one 2,240-row pass cost less than eight 280-row passes?
The stage sums answer it directly, and the same table prices the per-chunk intercept the cached turn pays
(R-534's 0.7-2.8 s suffix ttfts).

Usage (on studio): python3 r536-prefill-sweep.py OUT.json
"""
import bisect
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.expanduser("~/rehearse"))
import rehearse as R  # noqa: E402

STAGE = re.compile(r"stage (\w+) layer (\d+): gpu\s+([\d.]+) ms, dispatches (\d+), threadgroups (\d+)")
DONE = re.compile(r"done \S+ prompt=(\d+) cached=(\d+) .*?tokens=(\d+) sha=(\w+) .*?ttft=([\d.]+) decode=([\d.]+)")

LENGTHS = [280, 560, 1120, 2240, 4480]
CHARS_PER_TOKEN = 4.006  # 88,000 chars -> 21,966 tokens in the standing corpus prompt


def request(messages, max_tokens):
    body = {"model": R.MODEL_NAME, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "top_p": 0.95, "top_k": 20, "seed": 1234,
            "draft": False, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(R.BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    usage = None
    out = ""
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.time()
                if first is None:
                    first = now
                last = now
                out += ch[0]["delta"]["content"]
    return {"prompt": usage["prompt_tokens"] if usage else None,
            "gen": usage["completion_tokens"] if usage else None,
            "wall": round(time.time() - t0, 3),
            "ttft": round((first - t0), 3) if first else None,
            "decode_s": round((last - first), 3) if first and last else None,
            "sha16": __import__("hashlib").sha256(out.encode()).hexdigest()[:16]}


def cycles(log):
    out, cur = [], None
    for m in STAGE.finditer(log):
        label, layer = m.group(1), int(m.group(2))
        if label == "engram" and layer == 0:
            if cur:
                out.append(cur)
            cur = []
        if cur is not None:
            cur.append({"label": label, "layer": layer, "ms": float(m.group(3)),
                        "dispatches": int(m.group(4)), "threadgroups": int(m.group(5)),
                        "at": m.start()})
    if cur:
        out.append(cur)
    return out


def summary(rows):
    tot = {}
    for r in rows:
        a = tot.setdefault(r["label"], {"ms": 0.0, "dispatches": 0})
        a["ms"] += r["ms"]
        a["dispatches"] += r["dispatches"]
    return {k: {"ms": round(v["ms"], 3), "dispatches": v["dispatches"]} for k, v in tot.items()}


def total(cyc):
    return sum(r["ms"] for r in cyc)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/rehearse/r536-prefill-sweep.json")
    text = R.corpus()
    srv = R.Serve("r536-prefill", 0, {"TF_DSV41_STAGE_CB": "1"})
    boots = [srv.start()]
    results = []
    try:
        for n in LENGTHS:
            chars = int(n * CHARS_PER_TOKEN)
            prompt = text[:chars] + R.S
            # two arms: the first touch (Engram pages cold for this slice) and the warm replay
            for rep in (0, 1):
                a = request(R.msgs(prompt), 2)
                a["tag"] = "rows~%d-rep%d" % (n, rep)
                a["target"] = n
                results.append(a)
    finally:
        boots.append(srv.stop())
        log = open(srv.log_path, "rb").read().decode("utf-8", "replace")

    done_at = [m.start() for m in DONE.finditer(log)]
    allc = cycles(log)
    report = {"tree": R.tree_state(), "boots": boots, "arms": [], "lengths": LENGTHS}
    for a in results:
        report["arms"].append({"tag": a["tag"], "arm": a, "cycles": []})
    for cyc in allc:
        i = bisect.bisect_left(done_at, cyc[0]["at"])
        i = min(i, len(report["arms"]) - 1)
        report["arms"][i]["cycles"].append(cyc)
    for e in report["arms"]:
        cycs = e["cycles"]
        biggest = max((total(c) for c in cycs), default=0.0)
        prompt = [c for c in cycs if total(c) > 0.25 * biggest]
        e["prompt_cycles"] = [round(total(c), 2) for c in prompt]
        e["prompt"] = [summary(c) for c in prompt]
        e["stage_ms"] = {}
        for s in e["prompt"]:
            for k, v in s.items():
                e["stage_ms"][k] = round(e["stage_ms"].get(k, 0.0) + v["ms"], 3)
        e["gpu_ms"] = round(sum(total(c) for c in prompt), 3)
    report["arms"] = [{k: v for k, v in e.items() if k != "cycles"} for e in report["arms"]]

    # the decision table: per stage, the fixed + marginal fit, and the merge comparison
    rows = []
    for e in report["arms"]:
        if e["arm"]["tag"].endswith("rep1"):
            rows.append((e["arm"]["prompt"], e["stage_ms"], e["gpu_ms"]))
    print("%-8s %-10s %-8s %s" % ("prompt", "gpu_ms", "ttft", "stages"))
    for e in report["arms"]:
        print("%-14s %-8s %-8s %s" % (e["tag"], e["gpu_ms"], e["arm"]["ttft"],
                                      " ".join("%s=%.1f" % (k, v) for k, v in sorted(e["stage_ms"].items()))))
    print("\nmerge price (warm arms):")
    for e in report["arms"]:
        if e["arm"]["tag"].endswith("rep1"):
            n = e["arm"]["prompt"]
            print("  %d rows: gpu %.1f ms = %.3f ms/row; routed %.1f" % (
                n, e["gpu_ms"], e["gpu_ms"] / max(n, 1), e["stage_ms"].get("swiglu", 0) + e["stage_ms"].get("down", 0)))
    json.dump(report, open(out_path, "w"), indent=1)
    print("json", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
