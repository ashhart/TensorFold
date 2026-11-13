"""Compare served token IDs and hashes for bias, drafting and concurrency. Run against two loaded servers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import Request, urlopen


def draw(url, model, prompt, bias=None, draft=False, sampled=False, drafted=None):
    body = {"model": model, "prompt": prompt, "max_tokens": 64, "seed": 1234,
            "temperature": 0.7 if sampled else 0, "top_k": 20, "top_p": 0.95,
            "draft": draft, "return_token_ids": True}
    if bias is not None:
        body["logit_bias"] = bias
    req = Request(url.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                  {"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as response:
        value = json.load(response)
    runtime = value["tensorfold"]
    if drafted is not None:  # a plain-only engine's equality is no draft evidence
        rounds = value.get("speculative", {}).get("drafted", 0)
        assert runtime.get("drafts") is drafted and (rounds > 0) is drafted, ("drafts", drafted, rounds)
    return runtime["token_ids"], runtime["token_sha"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    prompts = ["Continue: 1, 2, 3,", "Explain matrix multiplication in three sentences."]
    checks = 0
    for sampled in [False, True]:
        for prompt in prompts:
            base = draw(args.base, args.model, prompt, sampled=sampled)
            plain = draw(args.candidate, args.model, prompt, sampled=sampled)
            assert plain == base, ("default token IDs/hash changed", sampled, prompt)
            assert draw(args.candidate, args.model, prompt, draft=True, sampled=sampled) == \
                draw(args.base, args.model, prompt, draft=True, sampled=sampled), ("default drafted tokens changed", sampled)
            assert plain[0], "test requires at least one generated token"
            # Ban the first ordinary pick, ensuring the test changes a draw in this checkpoint.
            banned = plain[0][0]
            bias = {str(banned): -100}
            solo = draw(args.candidate, args.model, prompt, bias, sampled=sampled, drafted=False)
            assert banned not in solo[0], ("banned token drawn", banned)
            drafted = draw(args.candidate, args.model, prompt, bias, draft=True, sampled=sampled, drafted=True)
            assert drafted == solo, ("drafted differs from serial", sampled, prompt)
            with ThreadPoolExecutor(max_workers=2) as pool:
                biased_future = pool.submit(draw, args.candidate, args.model, prompt, bias, False, sampled)
                plain_future = pool.submit(draw, args.candidate, args.model, prompts[1], None, False, sampled)
                together = biased_future.result()
                other = plain_future.result()
            assert together == solo, "concurrent biased request differs from solo"
            assert other == draw(args.candidate, args.model, prompts[1], sampled=sampled), "bias leaked across streams"
            assert draw(args.candidate, args.model, prompt, sampled=sampled) == plain, "bias changed cached prompt logits"
            checks += 1
    print(f"PASS {checks} cases: default IDs/hash, bans, draft equality, concurrent equality and cache reuse")


if __name__ == "__main__":
    main()
