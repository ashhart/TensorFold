"""Measure chat turns that add a short message to a long retained conversation; never reset server state."""
import argparse
import hashlib
import json
import statistics
import sys

import bench_concurrent as bench
from bench_auth import read_api_key, redact
from bench_cache import receipt
from bench_evidence import token_equal


WORDS = ("alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "river", "stone", "amber", "cedar")


def words(count, state):
    """`count` fixture words from a 64-bit LCG, and the generator's next state."""
    out = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(WORDS[(state >> 33) % len(WORDS)])
    return " ".join(out), state


def conversation(messages, user_words, seed):
    """A reproducible chat of `messages` user turns of `user_words` words, each with a short assistant reply."""
    out, state = [], seed
    for _ in range(messages):
        text, state = words(user_words, state)
        reply, state = words(24, state)
        out += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    return out, state


def qualify(rows, reference=None):
    """Every turn completed, each later turn reused a prefix, and token hashes equal the reference report's."""
    if not rows or any(r.get("error") or not r.get("complete") or not r.get("tokens") for r in rows):
        return False
    if any(r["cache_state"] != "reused" for r in rows[1:]):
        return False
    if reference is None:
        return True
    return len(reference) == len(rows) and all(token_equal(a, b) is True for a, b in zip(reference, rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--messages", type=int, default=64, help="user turns of the retained conversation")
    p.add_argument("--user-words", type=int, default=400, help="words a retained user turn holds")
    p.add_argument("--tail-words", type=int, default=5, help="words of the user message each later turn adds")
    p.add_argument("--turns", type=int, default=8, help="turns after the first")
    p.add_argument("--tokens", type=int, default=8, help="reply tokens a turn")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--compare-file", help="an earlier report on the same fixture: hashes must be equal turn by turn")
    p.add_argument("--api-key-env", metavar="NAME", help="read the bearer API key from this environment variable")
    p.add_argument("--output")
    args = p.parse_args(argv)
    api_key = read_api_key(p, args.api_key_env)
    if min(args.messages, args.user_words, args.tail_words, args.turns, args.tokens) < 1:
        p.error("messages, user-words, tail-words, turns and tokens must be positive")
    messages, state = conversation(args.messages, args.user_words, args.seed)
    sent, rows = [], []
    for turn in range(args.turns + 1):
        text, state = words(args.tail_words, state)
        messages = messages + [{"role": "user", "content": text}]
        sent.append(text)
        item = {"name": f"turn{turn}", "kind": "chat", "messages": messages, "thinking_switch": True}
        run = bench.stream(args.base.rstrip("/"), args.model, item, args.tokens, 0, args.seed, api_key=api_key)
        rows.append(receipt(run, "first" if turn == 0 else "tail"))
        messages = messages + [{"role": "assistant", "content": "ok"}]
    fixture_sha = hashlib.sha256(json.dumps([args.messages, args.user_words, args.seed, sent]).encode()).hexdigest()
    reference = None
    if args.compare_file:
        with open(args.compare_file, encoding="utf-8") as f:
            other = json.load(f)
        if other.get("fixture_sha256") != fixture_sha:
            p.error("the compared report used another fixture")
        reference = other["rows"]
    tails = [r["ttft_s"] for r in rows[1:] if r.get("ttft_s") is not None]
    report = {"schema_version": 1, "fixture_sha256": fixture_sha, "messages": args.messages,
              "user_words": args.user_words, "tail_words": args.tail_words, "requested_reply_tokens": args.tokens,
              "temperature": 0, "seed": args.seed, "rows": rows,
              "tail_ttft_median_s": statistics.median(tails) if tails else None,
              "equal_reference": None if reference is None else qualify(rows, reference),
              "qualified": qualify(rows, reference)}
    encoded = json.dumps(redact(report, api_key), indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(encoded)
    print(encoded, end="", flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
