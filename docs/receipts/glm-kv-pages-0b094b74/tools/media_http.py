"""Compare image-request token hashes across fresh, repeated, plain and simultaneous native HTTP requests."""
import base64
import json
import sys

import bench_concurrent as bench
from bench_evidence import token_equal

FIELDS = ("prompt_tokens", "cached_tokens", "tokens", "cache_state", "complete", "token_sha256", "error")


def main(base, model, image, out):
    with open(image, "rb") as f:
        url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    item = {"name": "image", "kind": "chat", "prompt": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Describe this image in two sentences."}]}
    rows, ok = [], True

    def keep(phase, run, ref=None):
        nonlocal ok
        row = {"phase": phase, **{k: run.get(k) for k in FIELDS}}
        if ref is not None:
            row["equal_first"] = token_equal(ref, run)
            ok &= row["equal_first"] is True
        rows.append(row)
        print(f"{phase}: prompt {row['prompt_tokens']} cached {row['cached_tokens']} ({row['cache_state']}) tokens "
              f"{row['tokens']} sha256 {str(row['token_sha256'])[:12]}"
              + ("" if ref is None else f" equal_first {row['equal_first']}"), flush=True)

    first = bench.stream(base, model, item, 128, 0, 1234)
    ok &= first.get("complete") is True and first.get("token_sha256") is not None
    keep("first", first)
    keep("repeated", bench.stream(base, model, item, 128, 0, 1234), first)
    keep("draft_false", bench.stream(base, model, item, 128, 0, 1234, draft=False), first)
    for run in bench.together(base, model, [(item, 1234)] * 3, 128, 0):
        keep("concurrent", run, first)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"image_bytes": len(url), "requested_reply_tokens": 128, "temperature": 0, "seed": 1234,
                   "rows": rows}, f, indent=1)
    print(f"exit {0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
