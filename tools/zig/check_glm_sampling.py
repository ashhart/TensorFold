"""Verify native token hashes across cached restoration, speculative decode and concurrent GLM requests."""
import argparse
import concurrent.futures
import json
import threading
import urllib.request
import uuid
from pathlib import Path


def drafted_count(reply):
    return reply.get("speculative", {}).get("drafted", 0)


def signature(reply):
    token_hash = reply.get("tensorfold", {}).get("token_sha")
    if not token_hash:
        raise ValueError("missing native token_sha; text parity cannot qualify this gate")
    return token_hash, reply["usage"]["completion_tokens"], reply["choices"][0]["finish_reason"]


def cached(reply):
    return reply["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def require_drafted(reply, label):
    """A drafted run must prove drafting happened; zero drafted tokens is unqualified, never a pass."""
    if drafted_count(reply) <= 0:
        raise ValueError(f"{label} ran without drafting (speculative.drafted = 0): the engine did not "
                         "draft, so this gate cannot qualify sampled decoding on it")
    return reply


def check(request, bodies, draft_available=True):
    records = []
    for body in bodies:
        plain = request(dict(body, draft=False))
        if cached(plain) != 0:
            raise ValueError("baseline was not cold; use a new gate nonce")
        restored = request(dict(body, draft=False))
        if cached(restored) <= 0:
            raise ValueError("snapshot restoration was not exercised: no cached tokens")
        drafted = require_drafted(request(dict(body, draft=True)), "the drafted run") if draft_available \
            else request(dict(body, draft=True))
        if signature(plain) != signature(restored) or signature(plain) != signature(drafted):
            raise ValueError("fresh/restored or plain/drafted token mismatch")
        records.append({"request": body, "plain": plain, "restored": restored, "drafted": drafted})
    for order in [list(range(len(bodies))), list(reversed(range(len(bodies))))]:
        start = threading.Barrier(len(order))

        def concurrent_request(pair):
            index, mode = pair
            start.wait(timeout=30)
            return index, request(dict(bodies[index], draft=mode))

        # Mixed concurrency: draft:true and draft:false together, both orders, against the solo baselines.
        pairs = [(i, i % 2 == 0) for i in order]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(order)) as pool:
            results = list(pool.map(concurrent_request, pairs))
        for index, reply in results:
            if signature(reply) != signature(records[index]["plain"]):
                raise ValueError(f"concurrent/solo token mismatch for request {index}")
            records[index].setdefault("concurrent", []).append(reply)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-repeats", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--skip-drafted-qualification", action="store_true",
                        help="the engine under test has no MTP head: drafting cannot run, so the drafted "
                             "runs are checked for token parity only (an honest unqualified-drafting run)")
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

    records = check(request, bodies, draft_available=not args.skip_drafted_qualification)
    Path(args.output).write_text(json.dumps({"nonce": nonce, "checks": records}, indent=2) + "\n")
    qualification = "drafting proven (speculative.drafted > 0)" if not args.skip_drafted_qualification \
        else "drafting unavailable (no MTP head): parity only, unqualified for drafting"
    print(f"PASS: {len(records)} requests, restored/plain/drafted, mixed draft modes in concurrent runs, "
          f"{qualification}")


if __name__ == "__main__":
    main()
