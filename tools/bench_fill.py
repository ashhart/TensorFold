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
REPLY_LEAD_S = 0.5  # offered-load cue only; the native trace decides whether decode mixed with prefill


def largest_gaps(pieces: list) -> tuple[float | None, float | None]:
    """The 95th percentile and the largest ms between a reply's text pieces."""
    gaps = sorted((b[0] - a[0]) * 1000 for a, b in zip(pieces, pieces[1:]))
    if not gaps:
        return None, None
    return round(gaps[max(0, math.ceil(len(gaps) * 0.95) - 1)], 1), round(gaps[-1], 1)


def seconds(value):
    """A finite nonnegative native duration; missing or malformed evidence stays unknown."""
    return round(float(value), 6) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def widths(value):
    """Observed positive native prompt widths; an empty or malformed list is not a zero-width prefill."""
    return value if isinstance(value, list) and value and all(type(v) is int and v > 0 for v in value) else None


def row(reply: dict, long_: dict, mix: str = "unknown") -> dict:
    """One order's waits; native prefill wall, server TTFT and full client wall stay distinct."""
    p95, largest = largest_gaps(reply["pieces"])
    done = max(reply["pieces"][-1][0], long_["pieces"][-1][0])
    completed = long_.get("completed", long_["pieces"][-1][0])
    return {"reply_s": round(reply["pieces"][-1][0] - reply["sent"], 2), "long_ttft_s": round(long_["ttft_s"], 2),
            "long_native_prefill_wall_s": seconds(long_.get("prefill_seconds")),
            "long_server_ttft_s": seconds(long_.get("time_to_first_token")),
            "long_wall_s": round(completed - long_["sent"], 2), "long_prefill_widths": widths(long_.get("prefill_widths")),
            "native_mix": mix, "both_done_s": round(done - min(reply["sent"], long_["sent"]), 2),
            "reply_gap_p95_ms": p95, "reply_gap_max_ms": largest}


def valid_event(event):
    """One complete native trace event with internally valid rows and clock bounds."""
    if not isinstance(event, dict) or event.get("kind") not in ("submit", "fill", "decode", "round", "complete", "cancel"):
        return False
    if type(event.get("ok")) is not bool:
        return False
    numbers = [event.get(k) for k in ("t0_ns", "t1_ns", "id", "stream", "rows_from", "rows_to")]
    if any(type(v) is not int for v in numbers):
        return False
    return event["t1_ns"] >= event["t0_ns"] and event["id"] >= 0 and event["stream"] >= 0 and \
        0 <= event["rows_from"] <= event["rows_to"]


def phase_mix(events, long_id, reply_id, long_rows):
    """Owned reply output between two of the long request's fill lines, never a timing inference."""
    fills = [(i, e) for i, e in enumerate(events) if e["kind"] == "fill" and e["id"] == long_id]
    if not fills:
        return "unknown"
    at = 0
    for _, fill in fills:
        if not fill["ok"] or fill["rows_from"] != at or fill["rows_to"] <= at:
            return "invalid"
        at = fill["rows_to"]
    if at != long_rows:
        return "invalid"
    decodes = [(i, e) for i, e in enumerate(events) if e["kind"] == "decode" and e["id"] == reply_id]
    if not decodes:
        return "unknown"
    at = 0
    for _, decode in decodes:
        if not decode["ok"] or decode["rows_from"] != at or decode["rows_to"] <= at:
            return "invalid"
        at = decode["rows_to"]
    for (left_at, _), (right_at, _) in zip(fills, fills[1:]):
        if any(left_at < decode_at < right_at for decode_at, _ in decodes):
            return "verified_mixed"
    return "verified_serialized"


def native_mix(trace, long_rows):
    """Match the tool's seven submits, then judge both orders from complete owned trace evidence."""
    unknown = {"reply_first": "unknown", "long_first": "unknown"}
    if trace is None or long_rows is None:
        return unknown
    if long_rows <= 512:
        return dict.fromkeys(unknown, "not_long")
    if not all(valid_event(e) for e in trace):
        return dict.fromkeys(unknown, "invalid")
    submits = [e for e in trace if e["kind"] == "submit"]
    expected = [None, long_rows, None, None, long_rows, long_rows, None]
    if len(submits) != len(expected) or len({e["id"] for e in submits}) != len(submits):
        return dict.fromkeys(unknown, "invalid")
    if any(e["rows_to"] != rows for e, rows in zip(submits, expected) if rows is not None):
        return dict.fromkeys(unknown, "invalid")
    return {"reply_first": phase_mix(trace, submits[4]["id"], submits[3]["id"], long_rows),
            "long_first": phase_mix(trace, submits[5]["id"], submits[6]["id"], long_rows)}


def combined_mix(orders):
    """A mixed-load verdict requires owned decode during long prefill in both arrival orders."""
    if all(v == "verified_mixed" for v in orders.values()):
        return "verified_mixed"
    for state in ("invalid", "unknown", "not_long"):
        if state in orders.values():
            return state
    return "verified_serialized"


def summary(alone: dict, reply_first: dict, long_first: dict, trace=None, long_rows=None) -> dict:
    """Legacy waits plus native evidence; exact and cold retain their affirmative-only meanings."""
    mixes = native_mix(trace, long_rows)
    return {"alone": {"reply_s": round(alone["reply"]["pieces"][-1][0] - alone["reply"]["sent"], 2),
                      "long_ttft_s": round(alone["long"]["ttft_s"], 2)},
            "reply_first": row(reply_first["reply"], reply_first["long"], mixes["reply_first"]),
            "long_first": row(long_first["reply"], long_first["long"], mixes["long_first"]),
            "mixed_load": combined_mix(mixes),
            "exact": all(token_equal(alone[side], order[side]) is True
                         for order in (reply_first, long_first) for side in ("reply", "long")),
            "cold": all(r["long"]["cached_tokens"] == 0 for r in (alone, reply_first, long_first))}


def request(args, key, item, tokens) -> dict:
    """One greedy streamed request, stamped with send and completed-read times."""
    sent = time.perf_counter()
    result = bench.stream(args.base, args.model, item, tokens, 0, 1234, api_key=key)
    return {**result, "sent": sent, "completed": time.perf_counter()}


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


def read_trace(path, offset):
    """Only complete JSON lines appended after this benchmark started."""
    with open(path, encoding="utf-8") as f:
        f.seek(offset)
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--corpus", required=True, help="a text file of at least --long tokens")
    p.add_argument("--long", type=int, default=65536, help="the long prompt's tokens")
    p.add_argument("--reply", type=int, default=1024, help="the running reply's tokens")
    p.add_argument("--lead", type=float, default=5.0, help="seconds the long prompt fills before the reply is sent")
    p.add_argument("--output")
    p.add_argument("--trace", help="TF_LANE_TRACE JSONL from this server; required for a verified mixed-load verdict")
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
    if args.trace:
        with open(args.trace, "rb") as f:
            f.seek(0, 2)
            trace_start = f.tell()
    else:
        trace_start = None
    long_ask = ({"name": "long", "kind": "completion", "prompt": ids[:args.long]}, 64)
    reply_ask = ({"name": "reply", "kind": "chat", "prompt": REPLY}, args.reply)
    request(args, key, reply_ask[0], 16)  # a warm start, so no order pays for a cold one
    alone = {"long": request(args, key, *long_ask), "reply": request(args, key, *reply_ask)}
    reply, long_ = together(args, key, reply_ask, long_ask, REPLY_LEAD_S)
    long2, reply2 = together(args, key, long_ask, reply_ask, args.lead)
    trace = read_trace(args.trace, trace_start) if args.trace else None
    report = summary(alone, {"reply": reply, "long": long_}, {"reply": reply2, "long": long2}, trace, args.long)
    print(json.dumps(report), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    if not report["cold"]:
        p.exit(1, "the long prompt reused cached tokens: serve with --prompt-cache-gib 0\n")
    if not report["exact"]:
        p.exit(1, "a reply's tokens differ beside the other request\n")
    if args.trace and report["mixed_load"] != "verified_mixed":
        p.exit(1, f"native trace did not verify mixed load: {report['mixed_load']}\n")

if __name__ == "__main__":
    main()
