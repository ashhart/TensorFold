"""Compare simultaneous KV-page requests with their solo token hashes through the native HTTP server."""
import argparse
import json
import re
import sys
import threading
import time


def fit(log: str, lengths: list, tokens: int) -> int:
    """The longest of `lengths` whose request takes the pool's pages once but not twice (0 when none)."""
    m = re.search(r"taking (\d+) pages of 512 rows", open(log, errors="replace").read())
    if not m:
        print(f"fit: no page count in {log}", flush=True)
        return 0
    pages = int(m.group(1))
    need = lambda n: -(-(n + 64 + tokens + 16) // 512)  # noqa: E731 - the prompt with its template, the reply, the slack
    ok = [n for n in lengths if need(n) <= pages < 2 * need(n)]
    print(f"fit: {pages} pages; " + (f"{ok[-1]}-token prompts take {need(ok[-1])} each" if ok else "no length fits once and not twice"), flush=True)
    return ok[-1] if ok else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--length", type=int, default=0)
    ap.add_argument("--fit", default="")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--long", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--tools", default="tools")
    o = ap.parse_args()
    sys.path.insert(0, o.tools)
    import bench_concurrent as bench  # noqa: E402 - the tree's own client

    items = json.load(open(o.prompts))
    if o.fit:
        o.length = fit(o.fit, sorted({it["length"] for it in items}), o.tokens)
    if not o.length:
        return 2
    pick = lambda length: [it for it in items if it["length"] == length]  # noqa: E731
    short = [{"name": f"p{it['length']}-{it['rep']}", "kind": "chat", "prompt": it["messages"][0]["content"]}
             for it in pick(o.length)]
    if len(short) < o.n:
        short = [dict(s, name=f"{s['name']}-{k}") for k in range(o.n) for s in short][: o.n]
    short = short[: o.n]
    long = None
    if o.long:
        it = pick(o.long)[0]
        long = {"name": f"long{o.long}", "kind": "chat", "prompt": it["messages"][0]["content"]}
    alone = []
    for i, s in enumerate(short):
        r = bench.stream(o.base, o.model, s, o.tokens, 1.0, 1000 + i)
        alone.append(r)
        print(f"alone {s['name']}: prompt {r.get('prompt_tokens')} ttft {r.get('ttft_s')} sha {r.get('token_sha256')}", flush=True)
    runs: list = [None] * (len(short) + (1 if long else 0))
    threads = []

    def go(k, item, seed):
        runs[k] = bench.stream(o.base, o.model, item, o.tokens, 1.0, seed)

    t0 = time.perf_counter()
    if long:
        threads.append(threading.Thread(target=go, args=(len(short), long, 999)))
        threads[-1].start()
        time.sleep(2.0)  # its stream is admitted before the others arrive
    for i, s in enumerate(short):
        threads.append(threading.Thread(target=go, args=(i, s, 1000 + i)))
        threads[-1].start()
    for t in threads:
        t.join()
    rows, equal = [], 0
    for k, r in enumerate(runs):
        name = long["name"] if long and k == len(short) else short[k]["name"]
        same = None if k == len(short) else (r.get("token_sha256") is not None and r.get("token_sha256") == alone[k].get("token_sha256"))
        equal += 1 if same else 0
        rows.append({"name": name, "prompt": r.get("prompt_tokens"), "sent": r["sent"] - t0 if r.get("sent") else None,
                     "first": r["first"] - t0 if r.get("first") else None, "last": r["last"] - t0 if r.get("last") else None,
                     "error": r.get("error"), "equal_alone": same})
    ends = sorted((x["last"], x["name"]) for x in rows if x["last"] is not None)
    for x in rows:
        before = [n for e, n in ends if x["first"] is not None and e < x["first"] and n != x["name"]]
        x["started_after"] = before
        print(f"{x['name']:>12}: prompt {x['prompt']} first {x['first']:.2f} s last {x['last']:.2f} s"
              f" equal_alone {x['equal_alone']} started after {before or '-'}" if x["first"] is not None else f"{x['name']}: {x['error']}", flush=True)
    waited = sum(1 for x in rows if x["started_after"])
    print(f"together == alone: {equal}/{len(short)}; {waited} request(s) started only after another ended", flush=True)
    json.dump({"alone": alone, "together": runs, "rows": rows}, open(o.out, "w"))
    return 0 if equal == len(short) else 1


if __name__ == "__main__":
    sys.exit(main())
