"""Verify native token hashes across cached restoration, speculative decode and concurrent GLM requests."""
import argparse
import concurrent.futures
import json
import threading
import urllib.request
import uuid
from pathlib import Path


def signature(reply):
    token_hash = reply.get("tensorfold", {}).get("token_sha")
    if not token_hash:
        raise ValueError("missing native token_sha; text parity cannot qualify this gate")
    return token_hash, reply["usage"]["completion_tokens"], reply["choices"][0]["finish_reason"]


def cached(reply):
    return reply["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def check(request, bodies):
    records = []
    for body in bodies:
        plain = request(dict(body, draft=False))
        if cached(plain) != 0:
            raise ValueError("baseline was not cold; use a new gate nonce")
        restored = request(dict(body, draft=False))
        if cached(restored) <= 0:
            raise ValueError("snapshot restoration was not exercised: no cached tokens")
        drafted = request(dict(body, draft=True))
        if signature(plain) != signature(restored) or signature(plain) != signature(drafted):
            raise ValueError("fresh/restored or plain/drafted token mismatch")
        records.append({"request": body, "plain": plain, "restored": restored, "drafted": drafted})
    for order in [list(range(len(bodies))), list(reversed(range(len(bodies))))]:
        start = threading.Barrier(len(order))

        def concurrent_request(body):
            start.wait(timeout=30)
            return request(body)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(order)) as pool:
            replies = list(pool.map(concurrent_request, [dict(bodies[i], draft=True) for i in order]))
        for i, reply in zip(order, replies):
            if signature(reply) != signature(records[i]["plain"]):
                raise ValueError(f"concurrent/solo token mismatch for request {i}")
            records[i].setdefault("concurrent", []).append(reply)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-repeats", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    if args.prefix_repeats <= 0 or args.tokens <= 0:
        parser.error("prefix repeats and tokens must be positive")
    nonce = str(uuid.uuid4())
    configs = [
        {"temperature": 0, "seed": 11},
        {"temperature": 1, "seed": 12, "top_k": 0, "top_p": 0.75, "min_p": 0},
        {"temperature": 0.7, "seed": 13, "top_k": 3000, "top_p": 0.5, "min_p": 0},
        {"temperature": 1, "seed": 14, "top_k": 0, "top_p": 0.5, "min_p": 0.8},
    ]
    bodies = []
    for i, cfg in enumerate(configs):
        prefix = f"Synthetic gate {nonce}, case {i}.\n"
        prefix += "A matrix maps an input vector to an output vector.\n" * args.prefix_repeats
        bodies.append(dict(cfg, model=args.model, max_tokens=args.tokens, stream=False, ignore_eos=True,
                           messages=[{"role": "user", "content": prefix + "Explain this in three sentences."}],
                           chat_template_kwargs={"enable_thinking": False}))

    def request(body):
        req = urllib.request.Request(args.base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)

    records = check(request, bodies)
    Path(args.output).write_text(json.dumps({"nonce": nonce, "checks": records}, indent=2) + "\n")
    print(f"PASS: {len(records)} requests, restored/plain/drafted, mixed concurrency in both orders")


if __name__ == "__main__":
    main()
