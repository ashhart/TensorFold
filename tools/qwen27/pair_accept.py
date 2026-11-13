#!/usr/bin/env python3
"""Accept a paired TensorFold serve against the checked-in single-host receipts.

run   send every oracle case twice; both replies must be identical, and their token_sha must equal the oracle's
cold  one chat turn, then that conversation's next turn: the pair must report no reused prompt tokens, because a
      pair runs no prompt cache (its one slot's states are two Macs', so the serve enables none)

The oracle (qwen38_27b_oracle.json, next to this file) is the single-Mac, undrafted, temperature-0 record, made on
the GPU family it names: run refuses a serve whose Mac is in any other family, because before the M5 the bits come
from simd_qmm and on the M5 from the lane kernels with their own K splits, so one family's token_sha cannot accept
the other. The family is read from each rank's own serve (/health's device), not the Mac this script runs on: run
from rank 0 or from a third machine, the pair itself is what the gate checks, so --peer-url must reach rank 1.
A pair passes when the same token ids come out, so neither Mac's rank count nor the seam can show up in the text.
One chat case carries two turns: turn 1 is a recorded reply, fed back as the assistant message, so the decoded span
an earlier reply produced is part of the prompt the pair must reproduce.
Every case is compared to its oracle token_sha and must match: no case is exempt, because accepting a known
different value would hide exactly the pair-only divergence this run exists to catch.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen3.8-27b"


def post(base, path, body, timeout):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def reply_text(reply, endpoint):
    choice = reply["choices"][0]
    if endpoint.endswith("/chat/completions"):
        return choice.get("message", {}).get("content") or ""
    return choice.get("text", "")


def token_sha(reply):
    return reply.get("tensorfold", {}).get("token_sha")


def request_body(case, model):
    body = {
        "model": model,
        "max_tokens": case["max_tokens"],
        "temperature": 0.0,
        "seed": 1234,
    }
    if case["endpoint"].endswith("/chat/completions"):
        body["messages"] = case["messages"] if "messages" in case else [{"role": "user", "content": case["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": bool(case.get("thinking", False))}
    else:
        body["prompt"] = case["prompt"]
    return body


def load_oracle(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def server_device(base, timeout):
    """The GPU family a rank's own serve reports (/health's device), or None when it names none.

    Mirrors the engine: zig/src/native/metal.zig chip() names the family "apple-m<d>" from the Metal device
    name, which is the same string tensorfold-native capabilities --json prints and /health reports as device.
    Reading it from each rank's serve makes the gate check the pair itself, not the Mac this script runs on.
    """
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=timeout) as reply:
            if reply.status != 200:
                return None
            return json.load(reply).get("device")
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


def rank_refusal(recorded, rank, base, timeout):
    """Why `rank`'s serve may not carry this oracle, or None when its Mac is the family the oracle names."""
    if not isinstance(recorded, str) or not recorded:
        return "the oracle records no gpu_family, so it cannot accept a pair on any Mac; re-record it on rank 0's Mac"
    family = server_device(base, timeout)
    if family is None:
        return (
            "{}'s serve ({}) did not report its GPU family on /health, so the gate cannot check the pair itself;"
            " start both ranks first (pair_supervise.sh up) or point --peer-url at rank 1"
        ).format(rank, base)
    if family == recorded:
        return None
    return (
        "the oracle was recorded on {} but {}'s serve ({}) reports {}: before the M5 the bits come from simd_qmm"
        " and on the M5 from the lane kernels with their own K splits, so {}'s token_sha cannot accept a pair on {}."
        " Record the oracle on rank 0's Mac (record_oracle.py) or run acceptance against a pair in the recorded family"
    ).format(recorded, rank, base, family, recorded, family)


def pair_refusal(oracle, base, peer, timeout):
    """Why this pair may not accept this record, checking each rank's own serve."""
    refusal = rank_refusal(oracle.get("gpu_family"), "rank 0", base, timeout)
    if refusal is not None:
        return refusal
    return rank_refusal(oracle.get("gpu_family"), "rank 1", peer, timeout)


def run(base, oracle, model, repeat, timeout, only, peer):
    refusal = pair_refusal(oracle, base, peer, timeout)
    if refusal is not None:
        print("pair_accept: " + refusal, file=sys.stderr)
        return 2
    cases = [c for c in oracle["cases"] if only is None or c["name"] in only]
    if not cases:
        print("pair_accept: no case matches --only", file=sys.stderr)
        return 2
    if only is not None:
        print("pair_accept: running {} of {} cases; --only is a subset, not a full acceptance".format(len(cases), len(oracle["cases"])))
    failures = 0
    for case in cases:
        body = request_body(case, model)
        shas = []
        texts = []
        for _ in range(repeat):
            reply = post(base, case["endpoint"], body, timeout)
            shas.append(token_sha(reply))
            texts.append(reply_text(reply, case["endpoint"]))
        same = len(set(shas)) == 1 and len(set(texts)) == 1
        if same and shas[0] == case["token_sha"]:
            print("PASS {} token_sha={} ({} identical runs)".format(case["name"], shas[0], repeat))
        else:
            failures += 1
            if not same:
                why = "the {} runs differ".format(repeat)
            else:
                why = "token_sha {} != oracle {}".format(shas[0], case["token_sha"])
            print("FAIL {} {} shas={}".format(case["name"], why, ",".join(str(s) for s in shas)))
    total = len(cases)
    print("pair_accept: {}/{} cases match the oracle on {}; {} did not".format(total - failures, total, base, failures))
    return 0 if failures == 0 else 1


def cached_tokens(reply):
    usage = reply.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return value if value is not None else usage.get("cached_tokens")


def cold(base, oracle, model, timeout, peer):
    """Two turns of one conversation: the pair must report no prompt state reused between them.

    A pair runs no prompt cache at all (`qwen27_served.attach`: a pair's slots are one and its prompt states are
    two Macs', so `enableCache` is skipped), so the second turn has to prefill the whole conversation again. A cached
    count here would mean a pair kept one Mac's states and served them as the other's: a wrong answer, not a
    faster one. This pin fails if the count ever becomes positive without the shard handling the cache needs.
    """
    refusal = pair_refusal(oracle, base, peer, timeout)
    if refusal is not None:
        print("pair_accept: " + refusal, file=sys.stderr)
        return 2
    case = next((c for c in oracle["cases"] if c["name"].startswith("chat")), None)
    if case is None:
        print("pair_accept: the oracle has no chat case to build a conversation from", file=sys.stderr)
        return 2
    first = post(base, case["endpoint"], request_body(case, model), timeout)
    used = cached_tokens(first)
    text = reply_text(first, case["endpoint"])
    if not text:
        print("FAIL cold: the first turn returned no text", file=sys.stderr)
        return 1
    second_body = {
        "model": model,
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 1234,
        "messages": [
            {"role": "user", "content": case["prompt"]},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Now give the same example for a 3 by 3 matrix."},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    second = post(base, case["endpoint"], second_body, timeout)
    cached = cached_tokens(second)
    print("cold: turn 1 prefill marked {}; turn 2 cached={}".format(used, cached))
    if cached is None:
        print("FAIL cold: the next turn reported no cached-token count")
        return 1
    if cached > 0:
        print("FAIL cold: the pair reused {} prompt tokens; a pair runs no prompt cache".format(cached))
        return 1
    print("PASS cold: the pair kept no prompt cache, as its one slot over two Macs' states implies")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Accept a paired TensorFold serve against single-host receipts")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "cold"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:4100")
        p.add_argument("--peer-url", default="http://127.0.0.1:4101")
        p.add_argument("--oracle", default="tools/qwen27/qwen38_27b_oracle.json")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--timeout", type=float, default=600.0)
        if name == "run":
            p.add_argument("--repeat", type=int, default=2)
            p.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    oracle = load_oracle(args.oracle)
    try:
        if args.command == "run":
            return run(args.base_url, oracle, args.model, args.repeat, args.timeout, args.only, args.peer_url)
        return cold(args.base_url, oracle, args.model, args.timeout, args.peer_url)
    except urllib.error.URLError as err:
        print("pair_accept: {} did not answer: {}".format(args.base_url, err), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
