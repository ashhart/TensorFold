"""Native Nemotron teacher-forced NLL and top-1 vs stock mlx-lm; mlx-lm's own row/chunk spread is the floor."""
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
from mlx_lm.utils import load_tokenizer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bench_openai import PROMPTS


def forward(model, ids, width, vocab):
    cache = make_prompt_cache(model)
    logits = np.empty((len(ids), vocab), np.float32)
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


def log_softmax(z):
    z = z.astype(np.float64)
    top = z.max(-1, keepdims=True)
    return z - (np.log(np.exp(z - top).sum(-1, keepdims=True)) + top)


def compare(actual, reference):
    squared, max_abs, kl = 0.0, 0.0, []
    for at in range(0, len(actual), 32):
        delta = actual[at:at+32] - reference[at:at+32]
        max_abs = max(max_abs, float(np.abs(delta).max()))
        squared += float(np.sum(delta * delta, dtype=np.float64))
        ref, act = log_softmax(reference[at:at+32]), log_softmax(actual[at:at+32])
        kl.extend((np.exp(ref) * (ref - act)).sum(-1).tolist())  # KL(reference || actual) a position, in nats
    kl = np.maximum(np.array(kl), 0.0)
    return dict(top1_agreement=float(np.mean(actual.argmax(-1) == reference.argmax(-1))),
                max_abs=max_abs, rms=float(np.sqrt(squared / actual.size)), finite=bool(np.isfinite(actual).all()),
                kl_mean=float(kl.mean()), kl_p99=float(np.quantile(kl, 0.99)), kl_max=float(kl.max()))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def pins(model, weights):
    """The checkpoint and engine the numbers belong to: config, index and (with --hash-weights) every shard."""

    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    shards = sorted(model.glob("*.safetensors"))
    return dict(model=str(model), config_sha256=sha256(model / "config.json"),
                index_sha256=sha256(model / "model.safetensors.index.json"),
                shards={p.name: (sha256(p) if weights else p.stat().st_size) for p in shards},
                engine_commit=git("rev-parse", "HEAD"), engine_dirty=bool(git("status", "--porcelain", "--", "zig")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--native", type=Path, default=ROOT / "zig-out/bin/tf-nemotron-forward")
    p.add_argument("--long", type=int, default=2048, help="tokens of the long code fixture (0: skip it)")
    p.add_argument("--hash-weights", action="store_true", help="pin every shard by sha256 (else by size)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((args.model / "config.json").read_text()).get("vocab_size", 131072)
    tok = load_tokenizer(args.model)
    cases = []
    for fixture in PROMPTS:
        text = fixture["prompt"]
        ids = (tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                 add_generation_prompt=True, enable_thinking=False) if fixture["kind"] == "chat" else tok.encode(text))
        cases.append((fixture["name"], np.array(ids, np.uint32)))
    text = (ROOT / "CONTRIBUTING.md").read_text()
    cases.append(("public-contribution-guide-512", np.array(tok.encode(text)[:512], np.uint32)))
    if args.long:  # public, deterministic code: the running Python's own json.decoder and argparse sources
        import inspect
        from json import decoder as json_decoder
        text = inspect.getsource(json_decoder) + inspect.getsource(argparse)
        cases.append((f"python-stdlib-{args.long}", np.array(tok.encode(text[:16 * args.long])[:args.long], np.uint32)))

    for name, ids in cases:  # every native pass first: the native process holds the checkpoint, then exits
        np.save(args.output / (name + ".npy"), ids)
        subprocess.run([args.native, args.model, args.output / (name + ".npy"), args.output / (name + ".bin")], check=True)

    model, _ = load(str(args.model))
    report = dict(mlx=importlib.metadata.version("mlx"), mlx_lm=importlib.metadata.version("mlx-lm"),
                  pins=pins(args.model, args.hash_weights), fixtures=[])
    for name, ids in cases:
        words = np.fromfile(args.output / (name + ".bin"), np.uint16)
        native = (words.astype(np.uint32) << 16).view(np.float32).reshape(len(ids), vocab)
        serial = forward(model, ids, 1, vocab)
        chunked = forward(model, ids, 32, vocab)
        full = forward(model, ids, len(ids), vocab)
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
