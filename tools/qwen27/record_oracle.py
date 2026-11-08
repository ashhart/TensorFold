#!/usr/bin/env python3
"""Record the two-turn chat case of qwen38_27b_oracle.json from a live, single-Mac serve.

Start the serve the oracle describes on this Mac (same checkpoint, no drafter, temperature 0, seed 1234), then run
this recorder against it. It reads the serve's /health first: a serve that reports the two-Mac pair policy is refused,
because rank 0 of a pair would pass the chat-128 check below just as well, and an oracle always has to come from one
Mac; the version /health reports becomes the case's runtime. Turn 1 is the oracle's chat prompt; its reply becomes
the assistant turn, turn 2 asks the follow-up, and the reply's token_sha is checked against its own decimal token ids
before it is stored. The first turn must also reproduce chat-128's token_sha, so the new receipt cannot come from a
serve that is not the record's single-Mac run. With --write the case replaces (or appends) itself in the oracle.
"""
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request

from pair_accept import DEFAULT_MODEL, family_refusal, load_oracle, post, reply_text, request_body, token_sha

CASE_NAME = "chat-two-turn-128"
OPENING_CASE = "chat-128"
SECOND_TURN = "Now give the same example for a 3 by 3 matrix."


def refuse(message):
    """Print a plain refusal on stderr and exit 2, the status the family refusal also uses."""
    print("record_oracle: " + message, file=sys.stderr)
    raise SystemExit(2)


def served_runtime(base, timeout):
    """The serve's runtime, "tensorfold-native <version>", read from /health; a pair serve is refused.

    /health is unauthenticated and cheap, and it is the only place a serve says what it is: a pair serve names the
    two-Mac speed-up mode in its policy line, so it is refused before any completion request is sent.
    """
    url = base.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=min(timeout, 30.0)) as response:
            health = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        refuse("{} did not answer with a /health document: {}; start the single-Mac serve first".format(url, error))
    if not isinstance(health, dict):
        refuse("{} is not a /health document".format(url))
    version = health.get("version")
    if not isinstance(version, str) or not version.strip():
        refuse("{} has no version string, so the build that recorded the receipt is unknown".format(url))
    policy = health.get("policy")
    if isinstance(policy, str) and "pair" in policy.lower():
        refuse(
            "{} reports the two-Mac pair policy ({}); an oracle must come from a single-Mac serve".format(url, policy)
        )
    return "tensorfold-native " + version.strip()


def two_turn_messages(oracle, base, model, timeout):
    opening = next((c for c in oracle["cases"] if c["name"] == OPENING_CASE), None)
    if opening is None:
        raise SystemExit("record_oracle: the oracle has no {} case to open the conversation".format(OPENING_CASE))
    reply = post(base, opening["endpoint"], request_body(opening, model), timeout)
    text = reply_text(reply, opening["endpoint"])
    if not text:
        raise SystemExit("record_oracle: turn 1 returned no text")
    if token_sha(reply) != opening["token_sha"]:
        raise SystemExit(
            "record_oracle: turn 1 token_sha {} != {}'s oracle {}; this serve is not the oracle's single-Mac run".format(
                token_sha(reply), OPENING_CASE, opening["token_sha"]
            )
        )
    print("record_oracle: turn 1 decoded {} characters, token_sha={}".format(len(text), token_sha(reply)))
    return [
        {"role": "user", "content": opening["prompt"]},
        {"role": "assistant", "content": text},
        {"role": "user", "content": SECOND_TURN},
    ]


def main():
    parser = argparse.ArgumentParser(description="Record the two-turn chat receipt from a live single-Mac serve")
    parser.add_argument("--base-url", default="http://127.0.0.1:4100")
    parser.add_argument("--oracle", default="tools/qwen27/qwen38_27b_oracle.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--write", action="store_true", help="store the case in the oracle instead of printing it")
    args = parser.parse_args()

    oracle = load_oracle(args.oracle)
    refusal = family_refusal(oracle)
    if refusal is not None:
        print("record_oracle: " + refusal, file=sys.stderr)
        return 2
    runtime = served_runtime(args.base_url, args.timeout)
    print("record_oracle: serve reports {} at {}".format(runtime, args.base_url.rstrip("/")))
    messages = two_turn_messages(oracle, args.base_url, args.model, args.timeout)
    case = {
        "name": CASE_NAME,
        "endpoint": "/v1/chat/completions",
        "thinking": False,
        "messages": messages,
        "max_tokens": 128,
        "runtime": runtime,
        "token_sha": "",
    }
    body = request_body(case, args.model)
    body["return_token_ids"] = True
    reply = post(args.base_url, case["endpoint"], body, args.timeout)
    ids = (reply.get("tensorfold") or {}).get("token_ids")
    if not ids:
        raise SystemExit("record_oracle: the reply carried no token_ids; is the serve built from this commit?")
    computed = hashlib.sha256(",".join(str(token) for token in ids).encode("utf-8")).hexdigest()[:12]
    reported = token_sha(reply)
    if computed != reported:
        raise SystemExit(
            "record_oracle: the decimal token ids hash to {} but the reply reports {}; refusing to record".format(
                computed, reported
            )
        )
    case["token_sha"] = reported
    print("record_oracle: {} token_sha={} ({} tokens, sha256 of decimal ids joined by commas)".format(
        CASE_NAME, reported, len(ids)))

    oracle["cases"] = [c for c in oracle["cases"] if c["name"] != CASE_NAME] + [case]
    if args.write:
        with open(args.oracle, "w", encoding="utf-8") as handle:
            json.dump(oracle, handle, indent=2)
            handle.write("\n")
        print("record_oracle: wrote {} to {}".format(CASE_NAME, args.oracle))
    else:
        print(json.dumps(case, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
