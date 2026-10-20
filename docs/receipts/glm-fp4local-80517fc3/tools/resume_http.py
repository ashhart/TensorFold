"""Against `tensorfold serve` of GLM-5.3-Flash on CUDA: a resumed prompt equals a fresh one. The server keeps prompt
states for chat messages only (a completion's prompt has no history: server/prompt.zig) and GLM's at chunk starts, so
the request is a conversation of three messages (a user message of about 5,000 tokens with a nonce no other request
has, a fixed assistant reply, a follow-up question), thinking off, greedy, ignore_eos, 64 tokens:
  - first: cold (cached_tokens 0)
  - repeated: resumed from the state the first kept (cached_tokens above 0), the same ids
  - "draft": false: the same ids
  - two at once: each the same ids
Every reply's counters and hashes to OUT; one line a request, then "exit 0|1". Standard library only.

    python3 resume_http.py PORT OUT
"""
import json
import sys
import threading
import time
import urllib.request

LINE = "Line {i}: the river carries {n} stones past the mill, and the miller counts them twice before the bell. "


def body(nonce, draft=True):
    text = f"Reference notes {nonce}.\n" + "".join(LINE.format(i=i, n=(i * 37) % 101) for i in range(260))
    b = {"model": "local-model", "max_tokens": 64, "temperature": 0, "ignore_eos": True,
         "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": text + "\nHow many stones does line 7 mention?"},
                      {"role": "assistant", "content": "Line 7 mentions 57 stones."},
                      {"role": "user", "content": "Which line mentions the most stones? Answer in one sentence."}]}
    if not draft:
        b["draft"] = False
    return b


def post(port, b):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    usage = d.get("usage") or {}
    return {"seconds": round(time.perf_counter() - t0, 3), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "tokens": usage.get("completion_tokens"),
            "token_sha256": (d.get("tensorfold") or {}).get("token_sha256"),
            "content": d["choices"][0]["message"].get("content")}


def main(port, out):
    nonce = f"{time.time_ns():x}"
    rows, ok = [], True

    def keep(phase, row, first=None):
        nonlocal ok
        row = {"phase": phase, **row}
        if first is not None:
            row["same_as_first"] = (row["token_sha256"] or row["content"]) == (first["token_sha256"] or first["content"])
            ok &= row["same_as_first"]
        rows.append(row)
        print(f"{phase}: prompt {row['prompt_tokens']} cached {row['cached_tokens']} tokens {row['tokens']} "
              f"{row['seconds']} s sha256 {str(row['token_sha256'])[:12]}"
              + ("" if first is None else f" same_as_first {row['same_as_first']}"), flush=True)
        return row

    first = keep("first", post(port, body(nonce)))
    ok &= first["cached_tokens"] == 0 and first["tokens"] == 64
    again = keep("repeated", post(port, body(nonce)), first)
    ok &= (again["cached_tokens"] or 0) > 0
    keep("draft_false", post(port, body(nonce, draft=False)), first)
    both = [None, None]
    threads = [threading.Thread(target=lambda i=i: both.__setitem__(i, post(port, body(nonce)))) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for row in both:
        keep("concurrent", row, first)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"nonce": nonce, "requested_reply_tokens": 64, "temperature": 0, "rows": rows}, f, indent=1)
    print(f"exit {0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), sys.argv[2]))
