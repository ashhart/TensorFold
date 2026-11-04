"""A long prompt filling beside a running reply: what each side waits, every reply checked against its run alone."""
import argparse
import json
import math
import sys
import threading
import time
import urllib.request

from bench_auth import authorize, read_api_key
from bench_evidence import token_equal
import bench_openai as bench

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/bench_fill.py ...")

REPLY = "Explain how a hash map handles collisions, with a small worked example in Python, then compare the approaches."
REPLY_LEAD_S = 0.5  # a reply is decoding this long after it is sent


def largest_gaps(pieces: list) -> tuple[float | None, float | None]:
    """The 95th percentile and the largest ms between a reply's text pieces."""
    gaps = sorted((b[0] - a[0]) * 1000 for a, b in zip(pieces, pieces[1:]))
    if not gaps:
        return None, None
    return round(gaps[max(0, math.ceil(len(gaps) * 0.95) - 1)], 1), round(gaps[-1], 1)


def row(reply: dict, long_: dict) -> dict:
    """One order's waits; both_done_s counts from whichever request went first."""
    p95, largest = largest_gaps(reply["pieces"])
    done = max(reply["pieces"][-1][0], long_["pieces"][-1][0])
    return {"reply_s": round(reply["pieces"][-1][0] - reply["sent"], 2), "long_ttft_s": round(long_["ttft_s"], 2),
            "both_done_s": round(done - min(reply["sent"], long_["sent"]), 2),
            "reply_gap_p95_ms": p95, "reply_gap_max_ms": largest}


def summary(alone: dict, reply_first: dict, long_first: dict) -> dict:
    """Each order beside the two runs alone; a reply is exact only when its hash is known and equals its run alone."""
    return {"alone": {"reply_s": round(alone["reply"]["pieces"][-1][0] - alone["reply"]["sent"], 2),
                      "long_ttft_s": round(alone["long"]["ttft_s"], 2)},
            "reply_first": row(reply_first["reply"], reply_first["long"]),
            "long_first": row(long_first["reply"], long_first["long"]),
            "exact": all(token_equal(alone[side], order[side]) is True
                         for order in (reply_first, long_first) for side in ("reply", "long")),
            "cold": all(r["long"]["cached_tokens"] == 0 for r in (alone, reply_first, long_first))}


def request(args, key, item, tokens) -> dict:
    """One greedy streamed request, stamped with when it was sent."""
    sent = time.perf_counter()
    return {**bench.stream(args.base, args.model, item, tokens, 0, 1234, api_key=key), "sent": sent}


def together(args, key, first: tuple, second: tuple, lead_s: float) -> tuple[dict, dict]:
    """`first` under way for `lead_s`, then `second` beside it; the two receipts in that order."""
    results = [None, None]

    def run(at, spec):
        results[at] = request(args, key, *spec)

    threads = [threading.Thread(target=run, args=(at, spec)) for at, spec in enumerate((first, second))]
    threads[0].start()
    time.sleep(lead_s)
    threads[1].start()
    for t in threads:
        t.join()
    return results[0], results[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--corpus", required=True, help="a text file of at least --long tokens")
    p.add_argument("--long", type=int, default=65536, help="the long prompt's tokens")
    p.add_argument("--reply", type=int, default=1024, help="the running reply's tokens")
    p.add_argument("--lead", type=float, default=5.0, help="seconds the long prompt fills before the reply is sent")
    p.add_argument("--output")
    p.add_argument("--api-key-env", metavar="NAME", help="read the bearer API key from this environment variable")
    args = p.parse_args()
    key = read_api_key(p, args.api_key_env)
    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        body = json.dumps({"prompt": f.read(), "add_special_tokens": False}).encode()
    req = urllib.request.Request(args.base + "/tokenize", body, {"Content-Type": "application/json"})
    authorize(req, key)
    with urllib.request.urlopen(req, timeout=600) as r:
        ids = json.load(r)["tokens"]
    if len(ids) < args.long:
        p.exit(1, f"the corpus has {len(ids)} tokens; --long needs {args.long}\n")
    long_ask = ({"name": "long", "kind": "completion", "prompt": ids[:args.long]}, 64)
    reply_ask = ({"name": "reply", "kind": "chat", "prompt": REPLY}, args.reply)
    request(args, key, reply_ask[0], 16)  # a warm start, so no order pays for a cold one
    alone = {"long": request(args, key, *long_ask), "reply": request(args, key, *reply_ask)}
    reply, long_ = together(args, key, reply_ask, long_ask, REPLY_LEAD_S)
    long2, reply2 = together(args, key, long_ask, reply_ask, args.lead)
    report = summary(alone, {"reply": reply, "long": long_}, {"reply": reply2, "long": long2})
    print(json.dumps(report), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    if not report["cold"]:
        p.exit(1, "the long prompt reused cached tokens: serve with --prompt-cache-gib 0\n")
    if not report["exact"]:
        p.exit(1, "a reply's tokens differ beside the other request\n")


if __name__ == "__main__":
    main()
