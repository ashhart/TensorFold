#!/usr/bin/env python3
"""Two turns over a long document (turn 2 resumes turn 1's prompt): token_sha and cached tokens of each, as JSON."""
# usage: xpu_resume_receipt.py BASE_URL MODEL DOC_FILE [--chars N] [--tokens N]; compare with --prompt-cache-gib 0
import argparse
import json
import urllib.request

SAMPLINGS = {"greedy": {"temperature": 0}, "seeded": {"temperature": 0.8, "top_p": 0.95, "seed": 11}}


def ask(base, model, messages, tokens, extra):
    body = {"model": model, "messages": messages, "max_tokens": tokens, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    details = out["usage"].get("prompt_tokens_details") or {}
    return (out["choices"][0]["message"]["content"], out["tensorfold"]["token_sha"], out["usage"]["prompt_tokens"],
            details.get("cached_tokens", 0), out["tensorfold"].get("prefill_seconds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("doc")
    ap.add_argument("--chars", type=int, default=26000)
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    doc = open(a.doc).read()[: a.chars]
    for name, s in SAMPLINGS.items():
        m1 = [{"role": "user", "content": f"Here is a document:\n\n{doc}\n\nIn two sentences, what is it about?"}]
        r1 = ask(a.base, a.model, m1, a.tokens, s)
        m2 = m1 + [{"role": "assistant", "content": r1[0]}, {"role": "user", "content": "List three numbers the document reports, with what each measures."}]
        r2 = ask(a.base, a.model, m2, a.tokens, s)
        print(json.dumps({"sampling": name, "turn1": {"sha": r1[1], "prompt": r1[2], "cached": r1[3], "prefill_s": r1[4]},
                          "turn2": {"sha": r2[1], "prompt": r2[2], "cached": r2[3], "prefill_s": r2[4]}}), flush=True)


if __name__ == "__main__":
    main()
