"""The receipt's tables from run.sh's output folders (out/LABEL): startup and memory for every run, then each suite's
checks and timings. Markdown to stdout; a failed or missing check prints FAIL. Standard library only.

    python3 summarize.py OUT_DIR LABEL...
"""
import json
import os
import re
import statistics
import sys


def path(d, r, run, published):
    """A rank's file as run.sh leaves it (mem-rankN/NAME) or as the receipt publishes it."""
    p = os.path.join(d, f"mem-rank{r}", run)
    return p if os.path.exists(p) else os.path.join(d, published.format(r))


def log(d, r):
    return open(path(d, r, "process.log", "server-rank{}.log"), errors="replace").read()


def startup(d):
    text = log(d, 0)
    def find(pat):
        m = re.search(pat, text)
        return m.group(1) if m else "-"
    gib = 2 ** 30
    mem = []
    for r in (0, 1):
        rec = json.load(open(path(d, r, "receipt.json", "memory-rank{}.json")))
        rows = {x["phase"]: x["usage"] for x in rec["native"]["rows"]}
        mem.append(f"rank {r}: {rows['load_end']['device'] / gib:.2f} at load end, {rows['cleanup_begin']['peak'] / gib:.2f} peak, "
                   f"{rows['cleanup_end']['device'] / gib:.2f} after cleanup, exit {rec['returncode']}")
    return {"loaded": find(r"loaded in ([0-9.]+)s"), "digest": find(r"checkpoint digest: ([^\n]+)"),
            "plan": find(r"\(NVIDIA GB10[^)]*\): ([^\n]+)"), "windows": find(r"glm: windows (\((?:timed|kept)\): [^\n]+)"),
            "drafts": find(r"drafts: ([^;]+);"), "memory": "; ".join(mem)}


def ex(d):
    out = []
    g = json.load(open(os.path.join(d, "golden.json")))
    gl = open(os.path.join(d, "check-golden.log")).read()
    for kind in ("count", "prose", "code"):
        ok = f"PASS {kind}: drafts and plain equal" in gl
        rnd = re.search(rf"{kind} drafts: \d+ tokens, \w+, ([0-9.]+) s, ([0-9.]+) a round", gl)
        out.append(f"| golden {kind} | `{g[kind]['token_sha256'][:12]}` | {'PASS' if ok else 'FAIL'} drafted = plain, "
                   f"{float(rnd.group(2)):.2f} tokens a round, {rnd.group(1)} s for 512 |")
    c = json.load(open(os.path.join(d, "conc.json")))
    bad = 0
    for cell in c["cells"]:
        a, s = cell["alone"], cell["serial"]
        bad += a["unequal"] + a["failed"] + s["unequal"] + s["failed"] + cell["failed"]
    n = len(c["cells"])
    out.append(f"| concurrent | {n} cells | {'PASS' if bad == 0 else 'FAIL'}: every reply = its alone run = `\"draft\": false` "
               f"({sum(x['alone']['equal'] for x in c['cells'])} and {sum(x['serial']['equal'] for x in c['cells'])} equal, {bad} not) |")
    r = json.load(open(os.path.join(d, "resume.json")))["rows"]
    first = r[0]["token_sha256"]
    same = all(x["token_sha256"] == first for x in r)
    cached = sorted({x["cached_tokens"] for x in r[1:]})
    out.append(f"| resume | `{first[:12]}` | {'PASS' if same and r[0]['cached_tokens'] == 0 and min(cached) > 0 else 'FAIL'}: "
               f"{len(r) - 1} resumed replies ({', '.join(x['phase'] for x in r[1:])}) from {cached} cached of "
               f"{r[0]['prompt_tokens']} = cold |")
    rows = ["| prompt | t | streams | aggregate tok/s | per stream | first token max (s) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for cell in c["cells"]:
        rows.append(f"| {cell['prompt']} | {cell['temperature']:g} | {cell['streams']} | {cell['aggregate_tps']} | "
                    f"{cell['per_stream_tps']} | {cell['ttft_s_max']} |")
    return out, rows


def sp(d):
    rows = ["| decode | prompt | t | median tok/s | all | tokens a reply equal |", "| --- | --- | ---: | ---: | --- | --- |"]
    for kind in ("nodraft", "drafted"):
        for r in json.load(open(os.path.join(d, f"{kind}.json"))):
            shas = {e["token_sha"] for e in r["evidence"]}
            same = "yes" if r["temperature"] == 0 and len(shas) == 1 else ("n/a (sampled)" if r["temperature"] else "NO")
            rows.append(f"| {'`\"draft\": false`' if kind == 'nodraft' else 'drafted'} | {r['prompt']} | {r['temperature']:g} | "
                        f"{r['decode_tps_median']:.1f} | {', '.join(f'{x:g}' for x in r['decode_tps_all'])} | {same} |")
    pre = ["| cold prompt (tokens) | first token, s (first, two repeats) | prompt tok/s (median) | two at once, s | cached | ids equal |",
           "| ---: | --- | ---: | --- | --- | --- |"]
    for n in (2048, 8192, 9600, 32768, 38400, 65536):
        p = os.path.join(d, f"prefill{n}.json")
        if not os.path.exists(p):
            pre.append(f"| {n} | FAIL: no result | | | | |")
            continue
        r = json.load(open(p))["rows"]
        t = [x["ttft_s"] for x in r if x["phase"] in ("first", "repeated")]
        two = sorted(x["ttft_s"] for x in r if x["phase"] == "concurrent")
        pre.append(f"| {r[0]['prompt_tokens']:,} | {', '.join(f'{x:.2f}' for x in t)} | {r[0]['prompt_tokens'] / statistics.median(t):,.0f} | "
                   f"{', '.join(f'{x:.2f}' for x in two)} | "
                   f"{sorted({x['cached_tokens'] for x in r})} | {'yes' if len({x['token_sha256'] for x in r}) == 1 else 'NO'} |")
    return rows, pre


def main(base, labels):
    for label in labels:
        d = os.path.join(base, label)
        s = startup(d)
        print(f"### {label}\n\n- loaded in {s['loaded']} s; digest {s['digest']}; drafts {s['drafts']}")
        print(f"- plan: {s['plan']}\n- windows: {s['windows']}\n- memory (GiB, process-owned CUDA buffers): {s['memory']}\n")
        if os.path.exists(os.path.join(d, "golden.json")):
            checks, rows = ex(d)
            print("| check | ids | result |\n| --- | --- | --- |")
            print("\n".join(checks) + "\n")
            print("\n".join(rows) + "\n")
        if os.path.exists(os.path.join(d, "nodraft.json")):
            rows, pre = sp(d)
            print("\n".join(rows) + "\n")
            print("\n".join(pre) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
