"""Against `tensorfold serve --vision` of GLM-5.3-Flash on CUDA: a prompt resumed from a state after an image equals a
fresh one. media_http.py's request is one message, so nothing before it is kept and its repeat is cold; here the image
sits in the first of three messages (resume_http.py's form: the image as a data URL first, as media_http.py sends it,
then a nonce and about 5,000 tokens of notes, a fixed assistant reply, a follow-up question), thinking off, greedy,
ignore_eos, 64 tokens:
  - first: cold (cached_tokens 0)
  - repeated: resumed from the state the first kept (cached_tokens above 0, past the image), the same ids
  - "draft": false: the same ids
  - two at once: each the same ids
Every reply's counters and hashes to OUT; one line a request, then "exit 0|1". Standard library only.

    python3 media_resume_http.py PORT IMAGE OUT
"""
import base64
import json
import sys
import threading
import time
import urllib.request

LINE = "Line {i}: the river carries {n} stones past the mill, and the miller counts them twice before the bell. "


def body(nonce, url, draft=True):
    notes = "".join(LINE.format(i=i, n=(i * 37) % 101) for i in range(220))
    b = {"model": "local-model", "max_tokens": 64, "temperature": 0, "ignore_eos": True,
         "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": url}},
                          {"type": "text", "text": f"Reference notes {nonce}, after the picture.\n" + notes
                           + "\nWhat does the picture show, in one sentence?"}]},
                      {"role": "assistant", "content": "The picture shows a logo on a plain background."},
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


def main(port, image, out):
    with open(image, "rb") as f:
        url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
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

    first = keep("first", post(port, body(nonce, url)))
    ok &= first["cached_tokens"] == 0 and first["tokens"] == 64
    again = keep("repeated", post(port, body(nonce, url)), first)
    ok &= (again["cached_tokens"] or 0) > 0
    keep("draft_false", post(port, body(nonce, url, draft=False)), first)
    both = [None, None]
    threads = [threading.Thread(target=lambda i=i: both.__setitem__(i, post(port, body(nonce, url)))) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for row in both:
        keep("concurrent", row, first)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"nonce": nonce, "image_bytes": len(url), "requested_reply_tokens": 64, "temperature": 0,
                   "rows": rows}, f, indent=1)
    print(f"exit {0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), sys.argv[2], sys.argv[3]))
