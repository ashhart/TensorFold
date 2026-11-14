"""Five-turn token and cache receipts for separately loaded GLM CUDA TP servers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
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


def signature(reply):
    runtime = reply["tensorfold"]
    return runtime["token_ids"], runtime["token_sha"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--warm", required=True)
    parser.add_argument("--cache-off", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--streams", type=int, choices=[1, 4], required=True)
    parser.add_argument("--wait-seconds", type=float, default=5)
    args = parser.parse_args()
    receipts = []
    for seed in [0, 1234]:
        talks = [[] for _ in range(args.streams)]
        for turn in range(5):
            for stream, messages in enumerate(talks):
                messages.append({"role": "user", "content": f"Conversation {stream}, turn {turn}: give one useful fact about matrix multiplication."})
            with ThreadPoolExecutor(max_workers=args.streams) as pool:
                warm = list(pool.map(lambda m: chat(args.warm, args.model, m, seed, True), talks))
            for stream, (messages, got) in enumerate(zip(talks, warm)):
                serial = chat(args.warm, args.model, messages, seed, False)
                fresh = chat(args.cache_off, args.model, messages, seed, False)
                base = chat(args.base, args.model, messages, seed, True)
                assert signature(got) == signature(serial) == signature(fresh) == signature(base), (seed, turn, stream, "token mismatch")
                if turn:
                    previous = post(args.warm, "/tokenize", {"model": args.model, "messages": messages[:-1],
                        "add_generation_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})["count"]
                    cached = got["usage"]["prompt_tokens_details"]["cached_tokens"]
                    assert cached >= previous, (seed, turn, stream, "cache did not reach the reply end", cached, previous)
                else:
                    cached = got["usage"]["prompt_tokens_details"]["cached_tokens"]
                receipts.append({"seed": seed, "turn": turn + 1, "stream": stream, "cached_tokens": cached,
                    "ttft": got["tensorfold"]["time_to_first_token"], "base_ttft": base["tensorfold"]["time_to_first_token"],
                    "token_sha": got["tensorfold"]["token_sha"]})
                messages.append(got["choices"][0]["message"])
            time.sleep(args.wait_seconds)
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
