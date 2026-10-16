"""LFM2-MoE: teacher-forced NLL and top-token agreement against stock mlx-lm, one-row and chunked."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VOCAB = 128000
sys.path.insert(0, str(ROOT / "tools"))
from bench_openai import PROMPTS


def forward(model, ids, width):
    cache = make_prompt_cache(model)
    logits = np.empty((len(ids), VOCAB), np.float32)
    for at in range(0, len(ids), width):
        x = mx.array(ids[at:at+width].tolist())[None]
        y = model(x, cache=cache).astype(mx.float32)
        mx.eval(y, [c.state for c in cache])
        logits[at:at+width] = np.array(y)[0]
    return logits


def nll(logits, ids):
    total = 0.0
    for at in range(0, len(ids) - 1, 32):
        z = logits[at:min(at+32, len(ids)-1)].astype(np.float64)
        top = z.max(-1)
        logz = np.log(np.exp(z-top[:, None]).sum(-1)) + top
        targets = ids[at+1:at+1+len(z)]
        total += float(np.sum(logz-z[np.arange(len(z)), targets]))
    return total / (len(ids)-1)


def compare(actual, reference):
    squared, max_abs = 0.0, 0.0
    for at in range(0, len(actual), 32):
        delta = actual[at:at+32] - reference[at:at+32]
        max_abs = max(max_abs, float(np.abs(delta).max()))
        squared += float(np.sum(delta * delta, dtype=np.float64))
    a_top, r_top = actual.argmax(-1), reference.argmax(-1)
    differences = []
    for row in np.nonzero(a_top != r_top)[0]:
        a2, r2 = np.sort(actual[row])[-2:], np.sort(reference[row])[-2:]
        differences.append(dict(row=int(row), native=int(a_top[row]), reference=int(r_top[row]),
                                native_margin=float(a2[1] - a2[0]), reference_margin=float(r2[1] - r2[0])))
    return dict(top1_agreement=float(np.mean(a_top == r_top)), max_abs=max_abs, rms=float(np.sqrt(squared / actual.size)),
                finite=bool(np.isfinite(actual).all()), top1_differences=differences)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--native", type=Path, default=ROOT / "zig-out/bin/tf-lfm2-forward")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, tok = load(str(args.model))
    cases = []
    for fixture in PROMPTS:
        text = fixture["prompt"]
        ids = (tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                 add_generation_prompt=True) if fixture["kind"] == "chat" else tok.encode(text))
        cases.append((fixture["name"], np.array(ids, np.uint32)))
    text = (ROOT / "CONTRIBUTING.md").read_text()
    cases.append(("public-contribution-guide-512", np.array(tok.encode(text)[:512], np.uint32)))
    text = (ROOT / "CHANGELOG.md").read_text()
    cases.append(("public-changelog-2048", np.array(tok.encode(text)[:2048], np.uint32)))
    report = dict(mlx=importlib.metadata.version("mlx"), mlx_lm=importlib.metadata.version("mlx-lm"), fixtures=[])
    for name, ids in cases:
        tokens = args.output / (name + ".npy")
        native_file = args.output / (name + ".bin")
        np.save(tokens, ids)
        subprocess.run([args.native, args.model, tokens, native_file], check=True)
        words = np.fromfile(native_file, np.uint16)
        native = (words.astype(np.uint32) << 16).view(np.float32).reshape(len(ids), VOCAB)
        serial = forward(model, ids, 1)
        chunked = forward(model, ids, 32)
        full = forward(model, ids, len(ids))
        item = dict(fixture=name, tokens=len(ids), token_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            native_nll=nll(native, ids), mlx_serial_nll=nll(serial, ids), mlx_chunk32_nll=nll(chunked, ids), mlx_full_nll=nll(full, ids),
            native_vs_serial=compare(native, serial), native_vs_chunk32=compare(native, chunked),
            mlx_chunk32_vs_serial=compare(chunked, serial), mlx_full_vs_serial=compare(full, serial))
        report["fixtures"].append(item)
        print(json.dumps(item), flush=True)
        (args.output / "fidelity.json").write_text(json.dumps(report, indent=2) + "\n")
        if not item["native_vs_serial"]["finite"]:
            raise AssertionError("nonfinite native logits")


if __name__ == "__main__":
    main()
