"""A request cancelled in its prompt pass releases its lane within a layer; usage below (--help).

serve_cancel.py measures only the next short request. This one splits the case into what can be told apart:

  alone    a short request on the idle server, best of three (the baseline)
  release  the client of a long request leaves; seconds from its socket closing until /metrics says no request runs
  lifetime the server's own `ended ... after=` for that request, arrival to end (with --server-log)
  after    a short request sent 0.05 s after another long request's client left, as serve_cancel.py sends it

Every trial sends a fresh prompt (a nonce first), so no kept prompt state shortens it. A trial passes when its
release is within --release-max and its short request within --slack of alone, with the same text as alone.
The defaults hold the 27B to the 9B's and 35B's footing: a lane released at the next layer edge (about 0.08 s
of the 27B's 5 s chunk) plus the server's 50 ms disconnect poll, against the 4.6 s the 1.0.12 receipt waited.

--baseline-27b prints the receipt's 27B numbers next to these (docs/receipts/gfx1151-qwen-833bbfcb): the next
short request 6.49 to 6.51 s, alone 1.90 s, the cancelled request ending 5.04 to 5.07 s after it arrived.
Exit 0 only when every trial passes; the summary line is one JSON object for the receipt.
"""

import argparse
import http.client
import json
import re
import socket
import sys
import time
import urllib.parse

RECEIPT_27B = {"after_short_s": [6.49, 6.51], "alone_s": 1.90, "lifetime_s": [5.04, 5.07]}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("--words", type=int, default=2200, help="long prompt words: 2,200 are 14,994 tokens, inside a 16,384 window")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--drop-after", type=float, default=0.4, help="seconds from sending the long request to closing its socket")
ap.add_argument("--slack", type=float, default=0.5, help="seconds a short request after a cancel may take past alone")
ap.add_argument("--release-max", type=float, default=0.5, help="seconds from the disconnect to no running request")
ap.add_argument("--server-log", help="the server's stderr log, for each cancelled request's ended line")
ap.add_argument("--record", help="save each short request and full response as JSONL")
ap.add_argument("--baseline-27b", action="store_true", help="print the 1.0.12 receipt's 27B numbers beside these")
args = ap.parse_args()
HOST = urllib.parse.urlparse(args.url)


def body(text, max_tokens):
    return json.dumps({"model": args.model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}})


def conn():
    return http.client.HTTPConnection(HOST.hostname, HOST.port, timeout=600)


def short():
    c = conn()
    began = time.perf_counter()
    request = body("Count from one to ten in words.", 32)
    c.request("POST", "/v1/chat/completions", request, {"Content-Type": "application/json"})
    response = c.getresponse()
    status = response.status
    reply = json.load(response)
    c.close()
    seconds = time.perf_counter() - began
    if args.record:
        with open(args.record, "a") as f:
            f.write(json.dumps({"request": json.loads(request), "status": status, "response": reply, "seconds": seconds}) + "\n")
    if status != 200:
        sys.exit(f"short request returned HTTP {status}: {reply}")
    digest = reply.get("tensorfold", {}).get("token_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        sys.exit("short reply has no complete native token_sha256")
    return reply["choices"][0]["message"]["content"], digest, seconds


def running():
    c = conn()
    c.request("GET", "/metrics")
    page = c.getresponse().read().decode()
    c.close()
    m = re.search(r"^tensorfold:requests_running (\d+)$", page, re.M)
    if not m:
        sys.exit("no tensorfold:requests_running on /metrics (an API key gates it: start the server with --metrics-open)")
    return int(m.group(1))


def wait_running(want_at_least, limit):
    began = time.perf_counter()
    while time.perf_counter() - began < limit:
        if running() >= want_at_least:
            return True
        time.sleep(0.01)
    return False


def long_then_drop(nonce, watch):
    """The long request, its socket closed --drop-after later: the perf_counter time of the close, and whether
    /metrics saw it running before then (`watch`; the after trial does not ask, as serve_cancel.py does not)."""
    text = f"{nonce} " + " ".join(f"item{i % 97}x{(i * 7919) % 1013}" for i in range(args.words))
    sock = socket.create_connection((HOST.hostname, HOST.port))
    payload = body(text, 8).encode()
    sent = time.perf_counter()
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
                 + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    seen = watch and wait_running(1, args.drop_after)
    time.sleep(max(0.0, args.drop_after - (time.perf_counter() - sent)))
    sock.close()
    closed = time.perf_counter()
    if args.record:
        with open(args.record, "a") as f:
            f.write(json.dumps({"kind": "cancelled_request", "request": json.loads(payload),
                                "socket_closed_monotonic_s": closed, "running_at_drop": seen}) + "\n")
    return closed, seen


def lifetimes():
    if not args.server_log:
        return []
    with open(args.server_log, errors="replace") as f:
        return [float(m.group(1)) for m in re.finditer(r"ended \S+ reason=client-left prompt=\d+ tokens=\d+ after=([0-9.]+)s", f.read())]


def idle(limit=120, interval=0.05):
    began = time.perf_counter()
    while running() != 0:
        if time.perf_counter() - began > limit:
            sys.exit("the server never went idle")
        time.sleep(interval)


idle()
before, before_hash, _ = short()
baselines = [short() for _ in range(3)]
if any(text != before or digest != before_hash for text, digest, _ in baselines):
    sys.exit("idle baseline replies do not have identical text and native token hashes")
alone = min(seconds for _, _, seconds in baselines)
ended_before = len(lifetimes())
trials, failed = [], 0
for i in range(args.runs):
    # release: nothing else in flight, so the running count is the long request alone
    idle()
    closed, seen = long_then_drop(f"release{i}-{time.time_ns()}", True)
    idle(interval=0.005)
    release = time.perf_counter() - closed
    # after: the receipt's case, a short request right behind the disconnect
    idle()
    long_then_drop(f"after{i}-{time.time_ns()}", False)
    time.sleep(0.05)
    text, digest, after = short()
    same = text == before and digest == before_hash
    ok = seen and same and release <= args.release_max and after <= alone + args.slack
    failed += not ok
    trials.append({"release_s": round(release, 3), "after_short_s": round(after, 3), "running_at_drop": seen, "same_text": text == before, "same_native_tokens": digest == before_hash, "ok": ok})
    print(("ok   " if ok else "FAIL ") + f"cancel {i}: released {release:.2f} s after the disconnect (max {args.release_max:.2f}); "
          f"the next short request took {after:.2f} s (alone {alone:.2f} s, slack {args.slack:.2f})"
          + ("" if seen else "; /metrics never saw the long request run before the disconnect")
          + ("" if same else "; its text or native tokens differ from alone"), flush=True)
time.sleep(0.2)
ended = lifetimes()[ended_before:]
if args.server_log and len(ended) != 2 * args.runs:
    failed += 1
    print(f"FAIL expected {2 * args.runs} client-left completions in the server log, got {len(ended)}")
summary = {"model": args.model, "words": args.words, "drop_after_s": args.drop_after, "alone_s": round(alone, 3),
           "slack_s": args.slack, "release_max_s": args.release_max, "trials": trials,
           "baseline_native_token_sha256": before_hash, "lifetime_s": ended, "failed": failed}
if ended:
    print(f"server: {len(ended)} cancelled requests ended {min(ended):.2f} to {max(ended):.2f} s after they arrived (arrival to end, not disconnect to release)")
if args.baseline_27b:
    summary["receipt_27b"] = RECEIPT_27B
    r = RECEIPT_27B
    print(f"1.0.12 receipt 27B: next short {r['after_short_s'][0]:.2f} to {r['after_short_s'][1]:.2f} s, alone {r['alone_s']:.2f} s, "
          f"cancelled requests ended {r['lifetime_s'][0]:.2f} to {r['lifetime_s'][1]:.2f} s after arrival")
    worst = max(t["after_short_s"] for t in trials)
    print(f"this run: next short at most {worst:.2f} s, alone {alone:.2f} s: {r['after_short_s'][0] - worst:+.2f} s against the receipt's best")
print("summary " + json.dumps(summary))
print("failed", failed)
sys.exit(1 if failed else 0)
