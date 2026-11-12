"""Cold and resumed Qwen CUDA replies as token hashes, for an unread-prefill-chunk A/B."""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import bench_concurrent as bench

LENGTHS = (2048, 8192, 32768)


def tokenize(base, model):
    text = "".join(open(os.path.join(os.path.dirname(os.__file__), n), encoding="utf-8", errors="replace").read()
                   for n in sorted(os.listdir(os.path.dirname(os.__file__))) if n.endswith(".py"))
    body = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(base + "/v1/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["tokens"]


def request(base, model, name, prompt, tokens=16):
    item = {"name": name, "kind": "completion", "prompt": prompt}
    r = bench.stream(base, model, item, tokens, 0, None, draft=False)
    return {k: r.get(k) for k in ("prompt_tokens", "cached_tokens", "token_sha256", "output_sha256", "ttft_s", "error")}


def chat(base, model, name, prompt, tokens=16):
    item = {"name": name, "kind": "chat", "prompt": prompt}
    r = bench.stream(base, model, item, tokens, 0, None, draft=False)
    return {k: r.get(k) for k in ("prompt_tokens", "cached_tokens", "token_sha256", "output_sha256", "ttft_s", "error")}


def main():
    mode, base, model, out = sys.argv[1:5]
    ids = tokenize(base, model)
    if mode == "cold":
        rows = []
        for n in LENGTHS:
            row = request(base, model, f"cold-{n}", ids[:n])
            row["length"] = n
            rows.append(row)
            print(json.dumps(row), flush=True)
        json.dump(rows, open(out, "w"), indent=2)
    elif mode == "chat":
        text = "".join(open(os.path.join(os.path.dirname(os.__file__), n), encoding="utf-8", errors="replace").read()
                       for n in sorted(os.listdir(os.path.dirname(os.__file__))) if n.endswith(".py"))[:30000]
        first = chat(base, model, "prefix", "Explain this Python code in one sentence.\n\n" + text, 2)
        second = chat(base, model, "resumed", "Explain this Python code in one sentence.\n\n" + text + "\n\nAnswer in one word.", 16)
        print(json.dumps({"prefix": first, "resumed": second}), flush=True)
        json.dump({"prefix": first, "resumed": second}, open(out, "w"), indent=2)
    else:
        raise SystemExit(f"mode must be cold or resume, not {mode}")


if __name__ == "__main__":
    main()
