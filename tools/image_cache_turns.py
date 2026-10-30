"""Multi-turn image chats against a running native server: TTFT, cached and computed prompt tokens, replies vs fresh.

Each scenario is one conversation whose client resends the whole history every turn, as chat clients do:

  text     the same questions with no image (the control)
  image    image A in turn 1, then text-only turns
  repeat   the image chat, then a turn that sends image A again
  changed  the image chat's history with image B in A's place (an edit): must not resume past the image

Per request: prompt and cached tokens (computed = prompt - cached), time to first streamed piece, total time, the
server's prefill seconds and token hash. With --fresh-url (the same model served with --prompt-cache-gib 0), every
request is sent there too and its token hash must equal the resumed one. GET /v1/memory is recorded before and after.
Standard library only; Python 3.11+.

  python3 tools/image_cache_turns.py --url http://127.0.0.1:8080 --image a.png --image-b b.png \
      --fresh-url http://127.0.0.1:8081 --label main-qwen27 --output turns.json
"""

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_auth import authorize, read_api_key  # noqa: E402
from bench_evidence import Evidence, token_equal  # noqa: E402

if sys.version_info < (3, 11):
    sys.exit("Python 3.11+ is required")

QUESTIONS = ["Describe this picture in two sentences.", "What colours stand out?", "Is there any text in it?",
             "Summarise everything you said in one sentence.", "Name one thing you are unsure about."]
REPLY = "Noted."


def data_url(path: Path) -> str:
    """A local picture as a data URL (PNG, JPEG or WebP)."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def user(text: str, image: str | None = None) -> dict:
    """A user message: the picture first when there is one, then the text."""
    if image is None:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": text}]}


def ask(base: str, model: str, messages: list, tokens: int, key: str | None) -> dict:
    """One streamed greedy request; usage, runtime hashes, TTFT and total seconds."""
    body = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    authorize(req, key)
    sent, first, usage, runtime, complete = time.perf_counter(), None, {}, {}, False
    evidence = Evidence()
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    complete = True
                    break
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    return {"error": "server_stream_error", "complete": False}
                usage = chunk.get("usage") or usage
                runtime = chunk.get("tensorfold") or runtime
                for choice in chunk.get("choices", []):
                    evidence.add(choice)
                    delta = choice.get("delta") or {}
                    if first is None and (delta.get("content") or delta.get("reasoning_content")):
                        first = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - a failed request is reported, not fatal
        return {"error": type(exc).__name__, "complete": False}
    done = time.perf_counter()
    out = evidence.finish(usage, runtime, complete)
    if out["prompt_tokens"] is not None and out["cached_tokens"] is not None:
        out["computed_tokens"] = out["prompt_tokens"] - out["cached_tokens"]
    out.update(ttft_s=None if first is None else round(first - sent, 4), elapsed_s=round(done - sent, 4),
               prefill_s=runtime.get("prefill_seconds"))
    return out


def memory(base: str, key: str | None) -> dict | None:
    """The server's memory report, unchanged (None when the route is absent)."""
    req = urllib.request.Request(base + "/v1/memory")
    authorize(req, key)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 - optional evidence
        return None


def scenarios(a: str, b: str | None, turns: int) -> dict:
    """Each scenario's requests in order, each the whole conversation so far."""
    def chat(first_image: str | None) -> list[list]:
        msgs, out = [], []
        for i in range(turns):
            msgs = msgs + ([{"role": "assistant", "content": REPLY}] if i else []) + [user(QUESTIONS[i % len(QUESTIONS)], first_image if i == 0 else None)]
            out.append(msgs)
        return out
    image = chat(a)
    out = {"text": chat(None), "image": image,
           "repeat": [image[-1] + [{"role": "assistant", "content": REPLY}, user("Here it is again: is it the same?", a)]]}
    if b:
        out["changed"] = [[user(QUESTIONS[0], b) if i == 0 else m for i, m in enumerate(image[-1])]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--fresh-url", help="the same model and flags with --prompt-cache-gib 0: the replies' reference")
    ap.add_argument("--image", required=True, type=Path, help="picture A")
    ap.add_argument("--image-b", type=Path, help="picture B, for the changed-image scenario")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--only", help="comma-separated scenarios (text,image,repeat,changed)")
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--api-key-env", help="environment variable holding a bearer key")
    args = ap.parse_args()
    key = read_api_key(ap, args.api_key_env)
    req = urllib.request.Request(args.url + "/v1/models")
    authorize(req, key)
    with urllib.request.urlopen(req, timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]
    plan = scenarios(data_url(args.image), data_url(args.image_b) if args.image_b else None, args.turns)
    if args.only:
        plan = {k: v for k, v in plan.items() if k in args.only.split(",")}
    report = {"label": args.label, "model": model, "turns": args.turns, "tokens": args.tokens,
              "memory_before": memory(args.url, key), "scenarios": {}}
    failures = 0
    for name, requests in plan.items():
        rows = []
        for i, messages in enumerate(requests):
            row = {"turn": i + 1, **ask(args.url, model, messages, args.tokens, key)}
            if args.fresh_url:
                ref = ask(args.fresh_url, model, messages, args.tokens, key)
                row["fresh_cached_tokens"] = ref.get("cached_tokens")
                row["fresh_ttft_s"] = ref.get("ttft_s")
                row["equal_fresh"] = token_equal(ref, row)
                failures += row["equal_fresh"] is not True
            rows.append(row)
            print(f"{name:8} turn {i + 1}: prompt {row.get('prompt_tokens')} cached {row.get('cached_tokens')} "
                  f"computed {row.get('computed_tokens')} ttft {row.get('ttft_s')} s"
                  + (f" equal_fresh {row['equal_fresh']}" if 'equal_fresh' in row else "")
                  + (f" ERROR {row['error']}" if row.get("error") else ""))
            failures += bool(row.get("error"))
        report["scenarios"][name] = rows
    report["memory_after"] = memory(args.url, key)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
