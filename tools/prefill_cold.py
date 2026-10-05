"""Cold prefill against any OpenAI server: chat prompts of fixed token lengths from the Python standard library,
a unique first line each so no cached prefix resumes; TTFT and prompt tok/s per length.

build (where tensorfold is installed): python3 tools/prefill_cold.py build MODEL_DIR PROMPTS.json
run (any client):                      python3 tools/prefill_cold.py run URL MODEL PROMPTS.json OUT.json"""

import json
import statistics
import sys
import time
import urllib.request

LENGTHS = (2048, 8192, 16384, 32768, 65536)
REPS = 3
ASK = "\nSay in one sentence what the code above does."


def corpus() -> str:
    import sysconfig
    from pathlib import Path

    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib",
                                                                                     "site-packages", "__pycache__")))
    return "".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files)


def build(model_dir: str, out: str) -> None:
    from pathlib import Path

    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    template = ChatTemplate(Path(model_dir))
    text = corpus()

    def messages(nonce: str, start: int, chars: int):
        return [{"role": "user", "content": f"Request {nonce}.\n" + text[start:start + chars] + ASK}]

    def count(m) -> int:
        return len(tok.encode(template.render(m, tools=None, enable_thinking=False), add_special_tokens=False).ids)

    items = []
    for length in (1024,) + LENGTHS:
        for rep in range(1 if length == 1024 else REPS):
            nonce = f"{length}-{rep}" if length != 1024 else "warm"
            start = (rep * 1_000_003 + length * 7) % (len(text) - 20 * length)
            lo, hi = 0, 8 * length
            while lo < hi:                               # the most characters that stay within the length
                mid = (lo + hi + 1) // 2
                if count(messages(nonce, start, mid)) <= length:
                    lo = mid
                else:
                    hi = mid - 1
            m = messages(nonce, start, lo)
            items.append({"length": length, "rep": rep, "tokens": count(m), "messages": m})
            print(json.dumps({"length": length, "rep": rep, "tokens": items[-1]["tokens"]}), flush=True)
    json.dump({"items": items}, open(out, "w"))


def one(url: str, model: str, m) -> dict:
    body = {"model": model, "messages": m, "max_tokens": 2, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sent, first, usage = time.perf_counter(), None, {}
    done = False
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done = True
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError("cold prefill stream reported a server error")
            for c in chunk.get("choices") or []:
                d = c.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter()
            usage = chunk.get("usage") or usage
    if not done:
        raise RuntimeError("cold prefill stream ended before [DONE]")
    if first is None:
        raise RuntimeError("cold prefill stream completed without an output token")
    prompt_tokens = usage.get("prompt_tokens")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise RuntimeError("cold prefill stream did not report positive integer prompt_tokens")
    return {"ttft_s": round(first - sent, 4), "prompt_tokens": prompt_tokens}


def run(url: str, model: str, prompts: str, out: str) -> None:
    items = json.load(open(prompts))["items"]
    rows = []
    for it in items:
        r = one(url, model, it["messages"])
        r.update(length=it["length"], rep=it["rep"])
        rows.append(r)
        print(json.dumps(r), flush=True)
    summary = []
    for length in LENGTHS:
        rs = [r for r in rows if r["length"] == length]
        t = statistics.median(r["ttft_s"] for r in rs)
        n = statistics.median(r["prompt_tokens"] or 0 for r in rs)
        summary.append({"length": length, "prompt_tokens": n, "ttft_s": round(t, 3), "tok_s": round(n / t, 1),
                        "ttft_all": [r["ttft_s"] for r in rs]})
        print(json.dumps(summary[-1]), flush=True)
    json.dump({"model": model, "rows": rows, "summary": summary}, open(out, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        run(*sys.argv[2:6])
