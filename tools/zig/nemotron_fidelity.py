"""Native Nemotron teacher-forced NLL and top-1 vs stock mlx-lm; mlx-lm's own row/chunk spread is the floor.

The native runs each fixture twice: prompt rows (fused_rows a forward) and decode rows (one a forward). Every position
whose native top-1 differs from mlx-lm serial is listed with mlx-lm's margin (its top-1's lead over the native's token,
in logits and in bf16 ulps at that logit). mlx-lm's own chunk32 and full runs against its serial run are the floor:
they flip near-ties too. The run fails on a non-finite logit; on a differing token whose margin is above the largest at
which mlx-lm's own runs flip on that fixture (or --tie-ulps, if larger); on more differing tokens than twice mlx-lm's
own count plus 2; or on a native NLL further from mlx-lm serial than --nll-spread times mlx-lm's own spread (floored
at 0.002 nats)."""
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


def differing(native, reference, ids):
    """Every position whose native top-1 differs from the reference's, with the reference's margin there."""
    out = []
    a, r = native.argmax(-1), reference.argmax(-1)
    for t in np.nonzero(a != r)[0].tolist():
        top, ours = float(reference[t, r[t]]), float(reference[t, a[t]])
        ulp = 2.0 ** (np.floor(np.log2(max(abs(top), 2.0 ** -126))) - 7)
        out.append(dict(position=t, reference_token=int(r[t]), native_token=int(a[t]),
                        next_token=int(ids[t + 1]) if t + 1 < len(ids) else None,
                        margin=top - ours, margin_ulps=(top - ours) / ulp,
                        native_margin=float(native[t, a[t]] - native[t, r[t]])))
    return out


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
    p.add_argument("--tie-ulps", type=float, default=2.0, help="the least margin a differing top-1 may have, in bf16 ulps, where mlx-lm's own runs flip only closer ties")
    p.add_argument("--skip-native", action="store_true", help="reuse the native logits already in OUTPUT")
    p.add_argument("--nll-spread", type=float, default=2.0, help="the native NLL may differ from mlx-lm serial by this many times mlx-lm's own spread")
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
        if args.skip_native:
            continue
        subprocess.run([args.native, args.model, args.output / (name + ".npy"), args.output / (name + ".bin")], check=True)
        subprocess.run([args.native, args.model, args.output / (name + ".npy"), args.output / (name + ".decode.bin"), "--decode-from", "0"], check=True)

    model, _ = load(str(args.model))
    report = dict(mlx=importlib.metadata.version("mlx"), mlx_lm=importlib.metadata.version("mlx-lm"),
                  pins=pins(args.model, args.hash_weights), fixtures=[])
    failures = []
    for name, ids in cases:
        rows = {}
        for phase, suffix in (("prompt", ".bin"), ("decode", ".decode.bin")):
            words = np.fromfile(args.output / (name + suffix), np.uint16)
            rows[phase] = (words.astype(np.uint32) << 16).view(np.float32).reshape(len(ids), vocab)
        serial = forward(model, ids, 1, vocab)
        chunked = forward(model, ids, 32, vocab)
        full = forward(model, ids, len(ids), vocab)
        serial_nll, chunk_nll = nll(serial, ids), nll(chunked, ids)
        spread = max(abs(chunk_nll - serial_nll), abs(nll(full, ids) - serial_nll), 0.002)
        item = dict(fixture=name, tokens=len(ids), token_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            mlx_serial_nll=serial_nll, mlx_chunk32_nll=chunk_nll, mlx_full_nll=nll(full, ids),
            mlx_chunk32_vs_serial=compare(chunked, serial), mlx_full_vs_serial=compare(full, serial),
            mlx_chunk32_vs_serial_differing=differing(chunked, serial, ids), mlx_full_vs_serial_differing=differing(full, serial, ids))
        own = item["mlx_chunk32_vs_serial_differing"] + item["mlx_full_vs_serial_differing"]
        tie = max([args.tie_ulps] + [d["margin_ulps"] for d in own])
        most = 2 * max(len(item["mlx_chunk32_vs_serial_differing"]), len(item["mlx_full_vs_serial_differing"])) + 2
        item["allowed"] = dict(margin_ulps=tie, differing=most)
        for phase, native in rows.items():
            diff = differing(native, serial, ids)
            item[phase] = dict(native_nll=nll(native, ids), vs_serial=compare(native, serial), vs_chunk32=compare(native, chunked),
                               differing=diff)
            if not item[phase]["vs_serial"]["finite"]:
                failures.append(f"{name} {phase}: nonfinite native logits")
            for d in diff:
                if d["margin_ulps"] > tie:
                    failures.append(f"{name} {phase} position {d['position']}: mlx-lm's top-1 leads the native's token by {d['margin_ulps']:.2f} bf16 ulps (mlx-lm's own runs flip up to {tie:.2f})")
            if len(diff) > most:
                failures.append(f"{name} {phase}: {len(diff)} differing tokens (mlx-lm's own runs allow {most})")
            if abs(item[phase]["native_nll"] - serial_nll) > args.nll_spread * spread:
                failures.append(f"{name} {phase}: native NLL {item[phase]['native_nll']:.4f} vs mlx-lm serial {serial_nll:.4f} (allowed {args.nll_spread * spread:.4f})")
        report["fixtures"].append(item)
        print(json.dumps(item), flush=True)
        report["failures"] = failures
        (args.output / "fidelity.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise AssertionError("fidelity failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
