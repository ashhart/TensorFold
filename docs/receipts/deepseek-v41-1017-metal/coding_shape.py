"""Run a synthetic coding brief at the posted prompt/output lengths."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


MODEL = "deepseek-v4.1-flash-1017"
TARGET = 17639
BLOCK = """
Review the component below as part of a calendar synchronization service. The
caller can submit a duplicate operation after any network interruption. Preserve
the supplied public interface. Discuss what should happen if a calendar moves
between time zones on the same date that a client edits a recurring event.
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Change:
    calendar_id: str
    revision: int
    start: datetime
    payload: bytes

def accept(changes: list[Change], floor: int) -> list[Change]:
    pending = sorted(changes, key=lambda item: (item.revision, item.calendar_id))
    return [item for item in pending if item.revision > floor]
```
Add cases for an empty list, two equal revisions, daylight-saving overlap and a
server that acknowledges a write before publishing the corresponding update.
Document which ordering properties belong to the cursor and which belong to the
storage transaction. Consider how to test restoration from a cached prefix of
the event journal, and where the test fixture should inject a clock reset.
"""


def request(base, path, data=None, timeout=180):
    payload = json.dumps(data).encode() if data else None
    req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.load(reply)


def fixture(base, index):
    titles = ("Quartz", "Bluebird", "Mandolin", "Zinnia", "Aster", "Jupiter")
    content = "\n\n".join(f"Repository component {i+1}\n{BLOCK}" for i in range(240))
    lead = f"{titles[index]} coding review. Treat these fragments as separate module fixtures.\n"
    suffix = "\nWrite a long, rigorous implementation and test plan for the calendar service."
    kwargs = {"enable_thinking": False}
    def candidate(length):
        text = lead + content[:length] + suffix
        sample = {"messages": [{"role": "user", "content": text}], "chat_template_kwargs": kwargs}
        return sample, request(base, "/v1/tokenize", sample)["count"]
    lo, hi = 0, len(content)
    assert candidate(hi)[1] > TARGET
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        sample, count = candidate(mid)
        if count <= TARGET and (best is None or best[1] < count):
            best = sample, count
        if count == TARGET:
            break
        if count < TARGET:
            lo = mid + 1
        else:
            hi = mid - 1
    assert best and TARGET - best[1] <= 2
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--save-prompts", type=Path)
    parser.add_argument("--prompts", type=Path)
    a = parser.parse_args()
    results, prompts = [], []
    if a.prompts:
        prompts = json.loads(a.prompts.read_text())
    else:
        prompts = [fixture(a.base, a.seed_index + i) for i in range(a.runs)]
    if a.save_prompts:
        a.save_prompts.write_text(json.dumps(prompts) + "\n")
    for sample, count in prompts[:a.runs]:
        health = request(a.base, "/health")
        live = health["live"]
        assert live["connections"] == live["waiting"] == 0, live
        started = time.monotonic()
        data = request(a.base, "/v1/chat/completions", dict(
            sample, model=MODEL, max_tokens=a.new_tokens, seed=1234, temperature=0, ignore_eos=True))
        wall = time.monotonic() - started
        usage = data["usage"]
        stats = data["tensorfold"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        assert usage["prompt_tokens"] == count and usage["completion_tokens"] == a.new_tokens
        assert cached <= 2, usage
        text = data["choices"][0]["message"]["content"] or ""
        record = {
            "prompt_tokens": count, "cached_tokens": cached, "completion_tokens": a.new_tokens,
            "prefill_seconds": stats["prefill_seconds"], "ttft_seconds": stats["time_to_first_token"],
            "prefill_tps": (count-cached)/stats["prefill_seconds"],
            "pp_including_ttft_tps": (count-cached)/stats["time_to_first_token"],
            "prefill_widths": stats["prefill_widths"], "prefill_raised": stats["prefill_raised"],
            "decode_tps": stats["tokens_per_second"], "wall_seconds": wall, "output_ids_sha256": stats["token_sha256"],
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
        a.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
