"""Concurrent two-request bench with two different real-text prompts built from a checkout: PORT OUT.json CHARS."""
import concurrent.futures, glob, json, os, statistics, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3_len_abba as base


def corpus(patterns, chars):
    out, total = [], 0
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.expanduser(pattern), recursive=True)):
            try:
                text = open(path, encoding="utf-8").read()
            except Exception:
                continue
            out.append("// file: %s\n%s\n" % (os.path.basename(path), text))
            total += len(out[-1])
            if total >= chars:
                return "".join(out)[:chars]
    return "".join(out)[:chars]


def post(port, prompt, tokens):
    return base.one(port, prompt, tokens, False)


if __name__ == "__main__":
    port, out_path, chars, tokens = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(os.environ.get("TOK", "128"))
    root = "~/tensorfold-twochunk-dbg"
    a = corpus([root + "/zig/src/families/flashnext/*.zig", root + "/zig/src/core/**/*.zig"], chars)
    b = corpus([root + "/docs/**/*.md", root + "/*.md", root + "/zig/tests/**/*.zig"], chars)
    prompts = [a, b]
    report = {"chars": [len(a), len(b)], "batches": []}
    for phase in ("warmup-1", "measure-1", "measure-2"):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(lambda p: post(port, p, tokens), prompts))
        batch = {
            "phase": phase,
            "requests": rows,
            "mean_prefill_s": statistics.mean(r["prefill_s"] for r in rows),
            "batch_wall_s": time.perf_counter() - start,
        }
        report["batches"].append(batch)
        print(json.dumps(batch), flush=True)
    json.dump(report, open(out_path, "w"), indent=2)
