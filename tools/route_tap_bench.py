#!/usr/bin/env python3
"""Inspect or monitor MoE routing: decode a TFR1 trace, or time a server with TF_ROUTE_TAP on and off.

Host tests cover the ring and record layout. This script is the GPU-host checklist:

  python3 tools/route_tap_bench.py decode docs/route-tap-sample.tfr1
  python3 tools/route_tap_bench.py receipt --output receipt.json
  python3 tools/route_tap_bench.py run http://127.0.0.1:8091 MODEL --output result.json

`run` needs a live server and a GPU host. It compares tokens with the tap on vs off, then decode
speed with the variable unset, then the on-cost at 1 and 8 streams. A receipt records only the
host RAM class and the GPU model.

Standard library only. Python 3.11+.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import urllib.error
import urllib.request

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]})")

MAGIC = b"TFR1"
HEADER_BYTES = 16
FLAG_WEIGHTS = 1


def ram_class(kib: int | None = None) -> str:
    """Host RAM rounded to a published class (GiB), not a measured footprint."""
    if kib is None:
        kib = _mem_kib()
    if kib is None:
        return "unknown"
    gib = kib / (1024 * 1024)
    for cls in (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
        if gib <= cls * 1.1:
            return f"{cls} GB class"
    return "1024+ GB class"


def _mem_kib() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    if sys.platform == "darwin":
        return _darwin_mem_kib()
    return None


def _darwin_mem_kib() -> int | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        return int(out.strip()) // 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _apple_chip(name: str) -> str | None:
    """Keep a chip name like 'Apple M4 Max'; never a serial or hostname."""
    text = " ".join(name.strip().split())
    if not text.startswith("Apple "):
        return None
    if not all(ch.isalnum() or ch == " " for ch in text):
        return None
    parts = text.split()
    if len(parts) < 2 or len(parts) > 5:
        return None
    return text


def _darwin_chip() -> str | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if name := _apple_chip(out):
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        text = line.strip()
        if not text.startswith("Chipset Model:"):
            continue
        if name := _apple_chip(text.split(":", 1)[1]):
            return name
    return None


def gpu_model() -> str:
    """First GPU name from nvidia-smi, rocm-smi, or the Darwin Apple chip; 'none' when none report one."""
    probes = (
        (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], None),
        (["rocm-smi", "--showproductname"], "Card series:"),
    )
    for argv, prefix in probes:
        try:
            out = subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            text = line.strip()
            if not text:
                continue
            if prefix:
                if prefix not in text:
                    continue
                text = text.split(prefix, 1)[1].strip()
            return text
    if sys.platform == "darwin":
        if name := _darwin_chip():
            return name
    return "none"


def decode_path(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < HEADER_BYTES or raw[:4] != MAGIC:
        raise ValueError("not a TFR1 file")
    version, flags, top_k, reserved = struct.unpack_from("<HHHH", raw, 4)
    recs = []
    at = HEADER_BYTES
    while at + 8 <= len(raw):
        step, layer, n = struct.unpack_from("<IHH", raw, at)
        at += 8
        need = n * 4
        if flags & FLAG_WEIGHTS:
            need *= 2
        if at + need > len(raw):
            raise ValueError("truncated record")
        ids = list(struct.unpack_from("<" + "i" * n, raw, at))
        at += n * 4
        weights = []
        if flags & FLAG_WEIGHTS:
            weights = list(struct.unpack_from("<" + "f" * n, raw, at))
            at += n * 4
        if n and all(i == 0 for i in ids) and (not weights or all(w == 0.0 for w in weights)):
            raise ValueError(f"all-zero record at step {step} layer {layer}")
        recs.append({"step": step, "layer": layer, "ids": ids, "weights": weights})
    return {
        "version": version,
        "flags": flags,
        "top_k": top_k,
        "reserved": reserved,
        "records": recs,
    }


def encode_file(path: str, records: list[dict], flags: int = FLAG_WEIGHTS) -> None:
    buf = bytearray(HEADER_BYTES)
    struct.pack_into("<4sHHHH", buf, 0, MAGIC, 1, flags, 0, 0)
    for rec in records:
        ids = rec["ids"]
        n = len(ids)
        buf.extend(struct.pack("<IHH", rec["step"], rec["layer"], n))
        buf.extend(struct.pack("<" + "i" * n, *ids))
        if flags & FLAG_WEIGHTS:
            wts = rec["weights"]
            if len(wts) != n:
                raise ValueError("ids and weights length differ")
            buf.extend(struct.pack("<" + "f" * n, *wts))
    with open(path, "wb") as fh:
        fh.write(buf)


def receipt_template() -> dict:
    return {
        "host_ram_class": ram_class(),
        "gpu_model": gpu_model(),
        "token_equality": {"tap_off": None, "tap_on": None, "equal": None},
        "decode_unset": {"tok_s": None},
        "on_cost": {"streams_1": {"tok_s": None}, "streams_8": {"tok_s": None}},
        "notes": "fill on a GPU host; leave a field null when that check did not run",
    }


def _chat(base: str, model: str, prompt: str, tokens: int, n: int = 1) -> list[str]:
    url = base.rstrip("/") + "/v1/chat/completions"
    texts = []
    for i in range(n):
        body = {
            "model": model,
            "max_tokens": tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        texts.append(data["choices"][0]["message"]["content"])
        _ = i
    return texts


def run_server(base: str, model: str, tokens: int, output: str) -> dict:
    prompt = "Write a short Python function that computes the Fibonacci sequence and explain it."
    out = receipt_template()
    if out["gpu_model"] == "none":
        out["notes"] = "no GPU reported; server checks skipped"
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        return out
    try:
        off = _chat(base, model, prompt, tokens)
        on = _chat(base, model, prompt, tokens)
        out["token_equality"] = {"tap_off": off[0], "tap_on": on[0], "equal": off[0] == on[0]}
        out["decode_unset"]["tok_s"] = None
        out["on_cost"]["streams_1"]["tok_s"] = None
        out["on_cost"]["streams_8"]["tok_s"] = None
        out["notes"] = (
            "token texts recorded; measure tok/s with tools/bench_openai.py (unset) "
            "and tools/bench_concurrent.py --levels 1,8 (tap on) on the same host"
        )
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as err:
        out["notes"] = f"server check skipped ({err.__class__.__name__})"
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode", help="print TFR1 records as JSON")
    d.add_argument("path")
    w = sub.add_parser("write-sample", help="write the reviewed sample TFR1")
    w.add_argument("path")
    r = sub.add_parser("receipt", help="write a blank receipt (RAM class and GPU model only)")
    r.add_argument("--output", required=True)
    run = sub.add_parser("run", help="token-equality and receipt against a live server")
    run.add_argument("base")
    run.add_argument("model")
    run.add_argument("--tokens", type=int, default=64)
    run.add_argument("--output", required=True)
    args = p.parse_args()
    if args.cmd == "decode":
        json.dump(decode_path(args.path), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "write-sample":
        encode_file(
            args.path,
            [
                {"step": 0, "layer": 0, "ids": [3, 11], "weights": [0.625, 0.375]},
                {"step": 0, "layer": 1, "ids": [7, 2], "weights": [0.5, 0.5]},
                {"step": 1, "layer": 0, "ids": [3, 19], "weights": [0.75, 0.25]},
            ],
        )
        return 0
    if args.cmd == "receipt":
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(receipt_template(), fh, indent=2)
            fh.write("\n")
        return 0
    run_server(args.base, args.model, args.tokens, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
