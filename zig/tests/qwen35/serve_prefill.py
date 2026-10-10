"""Cold prefill of a running server; usage: serve_prefill.py URL MODEL IDS.npy OUT_DIR [LENGTHS] [REPS]."""

import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_ids(path: Path) -> list[int]:
    """The token ids of a 1-D little-endian int32 or int64 .npy file."""
    raw = path.read_bytes()
    header_len = struct.unpack("<H", raw[8:10])[0] if raw[6] == 1 else struct.unpack("<I", raw[8:12])[0]
    start = (10 if raw[6] == 1 else 12) + header_len
    header = raw[start - header_len:start].decode("latin1")
    kind = "q" if "<i8" in header else "i"
    count = (len(raw) - start) // struct.calcsize(kind)
    return list(struct.unpack(f"<{count}{kind}", raw[start:]))


def stream(url: str, body: dict) -> tuple[float, dict]:
    """Seconds to the first streamed token and the final usage block; a refusal or a short prompt raises."""
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    began, first, usage = time.perf_counter(), None, {}
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    raise SystemExit(f"refused: {event['error']}")
                if first is None and event.get("choices") and event["choices"][0].get("text") is not None:
                    first = time.perf_counter() - began
                usage = event.get("usage") or usage
    except urllib.error.HTTPError as e:
        raise SystemExit(f"refused: HTTP {e.code} {e.read().decode(errors='replace')[:300]}")
    if first is None or usage.get("prompt_tokens") != len(body["prompt"]):
        raise SystemExit(f"no prefill of {len(body['prompt'])} tokens: first token {first}, usage {usage}")
    return first, usage


def main() -> int:
    url, model, ids_path, out = sys.argv[1].rstrip("/"), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    lengths = [int(n) for n in (sys.argv[5] if len(sys.argv) > 5 else "2048,8192,32768,65536").split(",")]
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    ids, rows = read_ids(ids_path), []
    out.mkdir(parents=True, exist_ok=True)
    for n in lengths:
        if n > len(ids):
            print(f"skip {n}: the ids hold {len(ids)} tokens")
            continue
        body = {"model": model, "prompt": ids[:n], "max_tokens": 1, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        (out / f"request_{n}.json").write_text(json.dumps(body))
        times, cached = [], []
        for _ in range(reps):
            seconds, usage = stream(url, body)
            times.append(seconds)
            cached.append((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        best = sorted(times)[len(times) // 2]
        rows.append({"tokens": n, "seconds": times, "median_tok_s": n / best, "cached_tokens": cached})
        print(f"prefill {n}: median {best:.3f} s, {n / best:.0f} tok/s, cached {cached}")
    (out / "prefill.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
