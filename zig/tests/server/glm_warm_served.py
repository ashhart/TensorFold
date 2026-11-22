"""Five-turn token and cache receipts for GLM CUDA TP servers, one server a run, then compared."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from urllib.request import Request, urlopen


def post(url, path, body):
    req = Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as response:
        return json.load(response)


def chat(url, model, messages, seed, drafted):
    return post(url, "/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": 128, "seed": seed,
        "temperature": 0 if seed == 0 else 0.7, "top_k": 20, "top_p": 0.95,
        "draft": drafted, "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    })


def record(args):
    turns = []
    for seed in [0, 1234]:
        talks = [[] for _ in range(args.streams)]
        for turn in range(5):
            for stream, messages in enumerate(talks):
                messages.append({"role": "user", "content": f"Conversation {stream}, turn {turn}: give one useful fact about matrix multiplication."})
            # the streams' drafted replies together, as served; then each one alone with draft off
            with ThreadPoolExecutor(max_workers=args.streams) as pool:
                drafted = list(pool.map(lambda m: chat(args.url, args.model, m, seed, True), talks))
            for stream, (messages, got) in enumerate(zip(talks, drafted)):
                plain = chat(args.url, args.model, messages, seed, False)
                previous = 0
                if turn:
                    previous = post(args.url, "/tokenize", {"model": args.model, "messages": messages[:-1],
                        "add_generation_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})["count"]
                turns.append({"seed": seed, "turn": turn + 1, "stream": stream, "previous_tokens": previous,
                    "cached_tokens": got["usage"]["prompt_tokens_details"]["cached_tokens"],
                    "ttft": got["tensorfold"]["time_to_first_token"],
                    "token_ids": got["tensorfold"]["token_ids"], "token_sha": got["tensorfold"]["token_sha"],
                    "plain_token_ids": plain["tensorfold"]["token_ids"], "plain_token_sha": plain["tensorfold"]["token_sha"]})
                messages.append(got["choices"][0]["message"])
            time.sleep(args.wait_seconds)
    with open(args.out, "w") as f:
        json.dump({"label": args.label, "url": args.url, "streams": args.streams, "turns": turns}, f, indent=2)


def compare(args):
    runs = {}
    for name in ["warm", "cache_off", "main"]:
        with open(getattr(args, name)) as f:
            runs[name] = json.load(f)
    if len({r["streams"] for r in runs.values()}) != 1:
        sys.exit("the three receipts ran different --streams")
    bad = 0
    for at, warm in enumerate(runs["warm"]["turns"]):
        key = (warm["seed"], warm["turn"], warm["stream"])
        shas = {warm["token_sha"], warm["plain_token_sha"]}
        for name in ["cache_off", "main"]:
            other = runs[name]["turns"][at]
            if (other["seed"], other["turn"], other["stream"]) != key:
                sys.exit(f"{name}: turn order differs at {key}")
            shas |= {other["token_sha"], other["plain_token_sha"]}
        if len(shas) != 1:
            bad += 1
            print("token mismatch", key, sorted(shas))
        base = runs["main"]["turns"][at]
        if warm["cached_tokens"] < base["cached_tokens"]:
            bad += 1
            print("warm resumed less than main", key, warm["cached_tokens"], base["cached_tokens"])
        if warm["cached_tokens"] > base["cached_tokens"] and warm["ttft"] >= base["ttft"]:
            bad += 1
            print("warm TTFT not below main's after a longer resume", key, warm["ttft"], base["ttft"])
    print(json.dumps({"turns": len(runs["warm"]["turns"]), "failures": bad}))
    sys.exit(1 if bad else 0)


def main():
    parser = argparse.ArgumentParser()
    steps = parser.add_subparsers(dest="step", required=True)
    one = steps.add_parser("record", help="one server's replies, cache and TTFT to JSON")
    one.add_argument("--url", required=True)
    one.add_argument("--model", required=True)
    one.add_argument("--label", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--streams", type=int, choices=[1, 4], required=True)
    one.add_argument("--wait-seconds", type=float, default=5)
    both = steps.add_parser("compare", help="warm, cache-off and main receipts: equal tokens, cache and TTFT")
    both.add_argument("--warm", required=True)
    both.add_argument("--cache-off", required=True)
    both.add_argument("--main", required=True)
    args = parser.parse_args()
    if args.step == "record":
        record(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
