"""One chat session of N turns against an OpenAI-compatible server, resending one image each turn.

Every request keeps the first user message's image part and adds a short new question; the assistant's replies stay in
the history. Each turn records the reply text's sha256, completion tokens, cached prompt tokens and first-token time;
the last line summarizes the session. Failed or incomplete turns stop the session and exit nonzero, preserving the
failed turn and summary in the output. Standard library only.

  python3 tools/bench_image_session.py http://127.0.0.1:8080 --image picture.png --model MODEL --out session.jsonl
"""

import argparse
import base64
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request

from bench_auth import authorize, read_api_key
from bench_evidence import count

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/bench_image_session.py ...")

QUESTIONS = [
    "What is in the image?",
    "What colors stand out?",
    "How many objects are visible?",
    "Is there any text in the image?",
    "What is in the background?",
    "Is the scene indoors or outdoors?",
    "What is the largest object?",
    "Are there any people?",
    "What time of day does it look like?",
    "Summarize the image in one sentence.",
]


def data_url(path: str) -> str:
    """A PNG file as the data: URL a chat request carries, refusing anything without the PNG signature."""
    with open(path, "rb") as f:
        png = f.read()
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("--image must be a PNG file")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def cached_tokens(usage: dict) -> int | None:
    """usage.prompt_tokens_details.cached_tokens when it is a nonnegative integer, else null."""
    details = usage.get("prompt_tokens_details")
    return count(details.get("cached_tokens")) if isinstance(details, dict) else None


def turn(base: str, model: str, messages: list[dict], tokens: int, temperature: float, seed: int | None,
         api_key: str | None) -> dict:
    """One streamed chat turn: its reply text, sha256, usage counters and first-content time."""
    body = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    authorize(req, api_key)
    start = time.perf_counter()
    first = None
    text = []
    usage: dict = {}
    complete = False
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                complete = True
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError("server_stream_error")
            got_usage = chunk.get("usage")
            if isinstance(got_usage, dict):
                usage = got_usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
    if not complete:
        raise RuntimeError("incomplete_stream")
    reply = "".join(text)
    return {"reply": reply, "sha256": hashlib.sha256(reply.encode()).hexdigest(),
            "completion_tokens": count(usage.get("completion_tokens")),
            "cached_tokens": cached_tokens(usage),
            "ttft_s": round(first - start, 3) if first is not None else None}


def summary(turns: list[dict]) -> dict:
    """The session's totals and per-turn values; a missing counter stays null."""
    got = [t["completion_tokens"] for t in turns if t.get("completion_tokens") is not None]
    cached = [t["cached_tokens"] for t in turns if t.get("cached_tokens") is not None]
    return {"summary": True, "turns": len(turns), "completion_tokens": sum(got) if got else None,
            "cached_tokens": sum(cached) if cached else None, "errors": sum(1 for t in turns if t.get("error")),
            "ttft_s": [t.get("ttft_s") for t in turns], "sha256": [t.get("sha256") for t in turns]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--image", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--api-key-env", metavar="NAME", help="read the bearer API key from this environment variable")
    args = p.parse_args()
    if args.turns < 1:
        p.error("--turns must be greater than 0")
    if args.max_tokens < 1:
        p.error("--max-tokens must be greater than 0")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        p.error("--temperature must be finite and nonnegative")
    split = urllib.parse.urlsplit(args.base)
    if split.scheme not in ("http", "https") or not split.netloc:
        p.error("base must be an http or https URL")
    if split.query or split.fragment:
        p.error("base must not contain a query or fragment")
    base = args.base.rstrip("/")
    api_key = read_api_key(p, args.api_key_env)
    try:
        url = data_url(args.image)
    except OSError as exc:
        p.error(str(exc))
    except ValueError as exc:
        p.error(str(exc))
    messages = [{"role": "user", "content": [{"type": "text", "text": QUESTIONS[0]},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
    try:
        out = open(args.out, "w")
    except OSError as exc:
        p.error(f"--out: {exc}")
    records = []
    with out:
        for n in range(args.turns):
            try:
                got = turn(base, args.model, messages, args.max_tokens, args.temperature, args.seed, api_key)
            except Exception as exc:  # noqa: BLE001 - a failed turn is recorded, not fatal
                got = {"sha256": None, "completion_tokens": None, "cached_tokens": None, "ttft_s": None,
                       "error": type(exc).__name__}
            record = {"turn": n + 1, **{k: got[k] for k in ("sha256", "completion_tokens", "cached_tokens", "ttft_s")}}
            if got.get("error"):
                record["error"] = got["error"]
            records.append(record)
            line = json.dumps(record)
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
            if got.get("error"):
                break
            messages.append({"role": "assistant", "content": got["reply"]})
            if n + 1 < args.turns:
                messages.append({"role": "user", "content": QUESTIONS[(n + 1) % len(QUESTIONS)]})
        end = summary(records)
        print(json.dumps(end), flush=True)
        out.write(json.dumps(end) + "\n")
    return 1 if end["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
