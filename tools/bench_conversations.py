"""Repeat-turn TTFT over several long conversations served round-robin, against any OpenAI server: N distinct
chat prompts of about the same token length (stdlib source text, a unique first line each), turn 1 of each cold, then
turns 2..T round-robin, each extending the previous prompt with the reply and a short question, so a server that kept
(or spilled to host RAM) the stored prompt state resumes it instead of prefilling again. Compare a server with and
without a host prefix tier when N is more than the conversations it holds on the GPU.

build (where tensorfold or transformers is installed): python3 tools/bench_conversations.py build MODEL_DIR CONVS.json
run (any client): python3 tools/bench_conversations.py run URL MODEL CONVS.json OUT.json --turns 4 --label tier-on"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ASK = "\nSay in one sentence what the code above does."
FOLLOW = "Continue: one more sentence. (turn {turn})"


def counter(model_dir: str):
    """Tokens of a rendered chat prompt: the server's own template where tensorfold is installed, else transformers."""
    try:
        from pathlib import Path

        from tokenizers import Tokenizer

        from tensorfold.cuda.server import ChatTemplate

        tok, template = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")), ChatTemplate(Path(model_dir))
        return lambda m: len(tok.encode(template.render(m, tools=None, enable_thinking=False),
                                        add_special_tokens=False).ids)
    except ImportError:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        return lambda m: len(tok.encode(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                                enable_thinking=False), add_special_tokens=False))


def build(model_dir: str, out: str, conversations: int, tokens: int) -> None:
    from prefill_cold import corpus                       # the same stdlib source text as the cold-prefill bench

    count, text, items = counter(model_dir), corpus(), []

    def messages(i: int, start: int, chars: int):
        return [{"role": "user", "content": f"Conversation {i}.\n" + text[start:start + chars] + ASK}]

    for i in range(conversations):
        start = (i * 1_000_003 + tokens * 7) % (len(text) - 20 * tokens)
        lo, hi = 0, 8 * tokens
        while lo < hi:                                    # the most characters that stay within the length
            mid = (lo + hi + 1) // 2
            if count(messages(i, start, mid)) <= tokens:
                lo = mid
            else:
                hi = mid - 1
        m = messages(i, start, lo)
        items.append({"conversation": i, "tokens": count(m), "messages": m})
        print(json.dumps({"conversation": i, "tokens": items[-1]["tokens"]}), flush=True)
    json.dump({"items": items}, open(out, "w"))


def one(url: str, model: str, m, max_tokens: int, seed: int) -> dict:
    body = {"model": model, "messages": m, "max_tokens": max_tokens, "temperature": 0, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sent, first, last, usage, text = time.perf_counter(), None, None, {}, []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("error"):
                raise RuntimeError(f"stream error: {chunk['error']}")
            for c in chunk.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    first = first or time.perf_counter()
                    last = time.perf_counter()
                text.append(d.get("content") or "")
            usage = chunk.get("usage") or usage
    if first is None:
        raise RuntimeError("the stream carried no tokens")
    n = usage.get("completion_tokens")
    return {"prompt_tokens": usage.get("prompt_tokens"), "ttft_s": round(first - sent, 4),
            "total_s": round(time.perf_counter() - sent, 4), "text": "".join(text),
            "decode_tps": round((n - 1) / (last - first), 2) if n and last > first else None}


def percentile(xs, q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def stats(rows) -> dict:
    t = [r["ttft_s"] for r in rows]
    return {"n": len(t), "ttft_median_s": round(percentile(t, 0.5), 4), "ttft_p90_s": round(percentile(t, 0.9), 4)}


def summarize(rows, label: str = "") -> dict:
    turns = sorted({r["turn"] for r in rows})
    return {"label": label, "cold": stats([r for r in rows if r["turn"] == 1]),
            "repeat": stats([r for r in rows if r["turn"] > 1]) if len(turns) > 1 else None,
            "per_turn": {str(t): stats([r for r in rows if r["turn"] == t]) for t in turns}}


def run(url: str, model: str, convs: str, out: str, turns: int, reply_tokens: int, label: str) -> None:
    items = json.load(open(convs))["items"]
    history = {it["conversation"]: list(it["messages"]) for it in items}
    rows = []
    for turn in range(1, turns + 1):                      # turn 1 of every conversation, then each later turn in turn
        for it in items:
            c = it["conversation"]
            try:
                r = one(url, model, history[c], reply_tokens, 1234 + c)
            except urllib.error.HTTPError as e:
                sys.exit(f"conversation {c} turn {turn}: the server refused: HTTP {e.code}: "
                         f"{e.read().decode(errors='replace')[:400]}")
            except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
                sys.exit(f"conversation {c} turn {turn}: {e}")
            history[c] += [{"role": "assistant", "content": r.pop("text")},
                           {"role": "user", "content": FOLLOW.format(turn=turn + 1)}]
            rows.append({"label": label, "conversation": c, "turn": turn, **r})
            print(json.dumps(rows[-1]), flush=True)
    summary = summarize(rows, label)
    print(json.dumps(summary), flush=True)
    json.dump({"label": label, "model": model, "rows": rows, "summary": summary}, open(out, "w"), indent=1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("model_dir")
    b.add_argument("out")
    b.add_argument("--conversations", type=int, default=8)
    b.add_argument("--tokens", type=int, default=24000)
    r = sub.add_parser("run")
    for name in ("url", "model", "convs", "out"):
        r.add_argument(name)
    r.add_argument("--turns", type=int, default=4)
    r.add_argument("--reply-tokens", type=int, default=32)
    r.add_argument("--label", default="")
    a = p.parse_args(argv)
    if a.cmd == "build":
        build(a.model_dir, a.out, a.conversations, a.tokens)
    else:
        run(a.url.rstrip("/"), a.model, a.convs, a.out, a.turns, a.reply_tokens, a.label)


if __name__ == "__main__":
    main()
