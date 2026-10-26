#!/usr/bin/env python3
"""Rebuild a Clef runtime adapter (`--clef DIR`) from a merged Clef release and its stock base checkpoint.

Cloudflare publishes Clef as merged weights. Every tensor that differs between the merge and the base must be a 2-D
projection weight; each difference dW is factored exactly by an fp64 eigendecomposition of its Gram matrix and kept
at rank min(--rank, out, in) as bf16 factors with A's rows orthonormal and the scale folded into B, so that
y = x W^T + (x A^T) B^T. The joint head files and LICENSE are copied from the merge; `lm_head_rows.safetensors` is
the base's output embedding (its untied `lm_head.weight`, else the tied embedding), which must equal the merge's.

    python tools/clef_adapter.py --merged Cloudflare-clef/ --base Qwen3.8-27B/ --out clef-adapter/ \\
        [--merged-revision SHA] [--base-revision SHA] [--rank 256] [--threads 32] [--check OTHER_ADAPTER_DIR]

Writes adapter.safetensors (keys `layers.{i}.{module}.lora_A` / `.lora_B`), manifest.json (both revisions, rank,
convention, every module's shape and rank), extract_report.json (per projection: the factored residual against the
bf16 rounding floor of the merge), the head files and SHA256SUMS.json. --check compares the reconstructed deltas
B A with another adapter directory (factors may differ in sign and order; the products may not).

Needs torch and safetensors; CPU only, one projection in memory at a time. Its unit tests
(tools/tests/test_clef_adapter.py) need them too and are skipped without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import shutil
import struct
import sys
import time
from pathlib import Path

HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py", "LICENSE")
CONVENTION = "y = x @ W.T + (x @ A.T) @ B.T ; A [rank,in], B [out,rank]; scale folded into B (A rows orthonormal)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tensors(path: Path) -> dict[str, tuple[str, list[int], Path, int, int]]:
    """Every tensor of a checkpoint directory: name -> (dtype, shape, file, start, end)."""
    index = path / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out = {}
    for shard in files:
        f = path / shard
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, m in header.items():
            if name != "__metadata__":
                a, b = m["data_offsets"]
                out[name] = (m["dtype"], m["shape"], f, 8 + n + a, 8 + n + b)
    return out


def raw(entry) -> bytes:
    _, _, f, a, b = entry
    with open(f, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm[a:b]


def differing(merged: dict, base: dict) -> list[str]:
    """Names whose bytes differ; tensors only the base holds (an MTP head the merge dropped) are not used."""
    if set(merged) - set(base):
        raise SystemExit(f"tensors only in the merge: {sorted(set(merged) - set(base))[:6]}")
    if set(base) - set(merged):
        log(f"{len(set(base) - set(merged))} base tensors are not in the merge (unused), e.g. {sorted(set(base) - set(merged))[0]}")
    out = []
    for name in sorted(merged):
        if merged[name][:2] != base[name][:2]:
            raise SystemExit(f"{name}: dtype/shape {merged[name][:2]} vs {base[name][:2]}")
        if hashlib.sha256(raw(merged[name])).digest() != hashlib.sha256(raw(base[name])).digest():
            out.append(name)
    return out


def load(entry):
    import torch

    dtype, shape, _, _, _ = entry
    if dtype != "BF16":
        raise SystemExit(f"expected bf16 weights, found {dtype}")
    return torch.frombuffer(bytearray(raw(entry)), dtype=torch.bfloat16).reshape(shape)


def ulp_bf16(w):
    """The bf16 unit in the last place of each value (zero takes the smallest normal one)."""
    import torch

    a = w.abs().double()
    _, e = torch.frexp(a)
    u = torch.ldexp(torch.ones_like(a), e - 8)
    u[a == 0] = 2.0 ** -133
    return u


def factor(wb, wm, rank: int):
    """dW = wm - wb at rank r: bf16 A [r,in] with orthonormal rows and B [out,r]; the residual against the floor."""
    import torch

    d = wm.double() - wb.double()
    out_dim, in_dim = d.shape
    r = min(rank, out_dim, in_dim)
    if in_dim <= out_dim:
        ev, v = torch.linalg.eigh(d.T @ d)
        v = v.flip(1)[:, :r]
        a, b = v.T, d @ v
        s = ev.flip(0).clamp_min(0).sqrt()
    else:
        ev, u = torch.linalg.eigh(d @ d.T)
        ev, u = ev.flip(0).clamp_min(0), u.flip(1)
        s = ev.sqrt()
        # a rank-deficient wide delta: only the nonzero singular values divide; the factor keeps the delta's rank
        r = min(r, int((s > s[0] * 1e-7).sum().item()) if s.numel() and s[0] > 0 else 0)  # above fp64 Gram eig noise
        if r == 0:
            raise SystemExit(f"a {out_dim}x{in_dim} delta is zero: nothing to factor")
        u = u[:, :r]
        b = u * s[:r]
        a = (u.T @ d) / s[:r, None]
    ab, bb = a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous()
    if not (torch.isfinite(ab).all() and torch.isfinite(bb).all()):
        raise SystemExit(f"a {out_dim}x{in_dim} delta factored to non-finite values at rank {r}")
    norm = d.norm().item()
    resid = (d - bb.double() @ ab.double()).norm().item() / norm if norm else 0.0
    floor = (ulp_bf16(wm).square().sum() / 12).sqrt().item() / norm * max(0.0, 1 - r / min(out_dim, in_dim)) ** 0.5
    rec = {"shape": [out_dim, in_dim], "rank": r, "resid": resid, "floor": floor,
           "resid_over_floor": resid / floor if floor else 0.0,
           "sigma_next_over_sigma1": (s[r].item() / s[0].item()) if s.numel() > r and s[0] > 0 else 0.0}
    return ab, bb, rec


def head_rows(merged: dict, base: dict):
    """The base's output embedding, which the head scores against; the merge must hold the same bytes."""
    for name in ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
        if name in base:
            if name not in merged:
                raise SystemExit(f"{name} is in the base but not in the merge: the head needs the output embedding")
            if raw(base[name]) != raw(merged[name]):
                raise SystemExit(f"{name} differs between the merge and the base")
            return name, load(base[name])
    raise SystemExit("no output embedding found in the base")


def sha256s(out: Path) -> dict:
    sums = {}
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name != "SHA256SUMS.json":
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 24), b""):
                    h.update(block)
            sums[f.name] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
    return sums


def check(out: Path, other: Path) -> dict:
    """Relative difference of each reconstructed delta B A between this adapter and another."""
    from safetensors import safe_open

    mine, theirs = safe_open(str(out / "adapter.safetensors"), "pt"), safe_open(str(other / "adapter.safetensors"), "pt")
    if set(mine.keys()) != set(theirs.keys()):
        raise SystemExit(f"key sets differ: {sorted(set(mine.keys()) ^ set(theirs.keys()))[:6]}")
    worst = {"key": None, "rel": 0.0}
    keys = sorted(k[: -len(".lora_A")] for k in mine.keys() if k.endswith(".lora_A"))
    for k in keys:
        dm = mine.get_tensor(k + ".lora_B").double() @ mine.get_tensor(k + ".lora_A").double()
        dt = theirs.get_tensor(k + ".lora_B").double() @ theirs.get_tensor(k + ".lora_A").double()
        rel = ((dm - dt).norm() / dt.norm()).item()
        if rel > worst["rel"]:
            worst = {"key": k, "rel": rel}
    return {"modules": len(keys), "worst_reconstructed_delta_rel_diff": worst}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merged", type=Path, required=True, help="the merged Clef release (e.g. Cloudflare/clef)")
    ap.add_argument("--base", type=Path, required=True, help="its stock base checkpoint (bf16)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merged-revision", default=None)
    ap.add_argument("--base-revision", default=None)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check", type=Path, default=None, help="another adapter directory to compare against")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    if args.threads:
        torch.set_num_threads(args.threads)
    merged, base = tensors(args.merged), tensors(args.base)
    names = differing(merged, base)
    bad = [n for n in names if len(merged[n][1]) != 2 or not n.endswith(".weight")]
    if bad:
        raise SystemExit(f"not projection weights, cannot be an adapter: {bad[:6]}")
    prefix = names[0][: names[0].index("layers.")] if names and "layers." in names[0] else ""
    log(f"{len(names)} projections differ; key prefix {prefix!r}")
    args.out.mkdir(parents=True, exist_ok=True)
    weights, modules, report = {}, [], []
    t0 = time.time()
    for i, n in enumerate(names):
        key = n[len(prefix):].removesuffix(".weight")
        a, b, rec = factor(load(base[n]), load(merged[n]), args.rank)
        weights[key + ".lora_A"], weights[key + ".lora_B"] = a, b
        layer = int(key.split(".")[1])
        modules.append({"key": key, "layer": layer, "proj": key.split(".", 2)[2], "out": rec["shape"][0],
                        "in": rec["shape"][1], "rank": rec["rank"]})
        report.append({"key": key, **rec})
        log(f"[{i + 1}/{len(names)} {time.time() - t0:5.0f}s] {key} {rec['shape']} r={rec['rank']} "
            f"resid={rec['resid']:.4f} floor={rec['floor']:.4f} ratio={rec['resid_over_floor']:.3f}")
    save_file(weights, str(args.out / "adapter.safetensors"), metadata={"format": "lora", "convention": CONVENTION})
    for f in HEAD_FILES:
        if (args.merged / f).exists():
            shutil.copy2(args.merged / f, args.out / f)
    rows_name, rows = head_rows(merged, base)
    save_file({"lm_head.weight": rows.contiguous()}, str(args.out / "lm_head_rows.safetensors"),
              metadata={"source": f"base {rows_name} (identical in the merge)"})
    manifest = {"base_model": str(args.base), "base_revision": args.base_revision, "merged_model": str(args.merged),
                "merged_revision": args.merged_revision, "rank": args.rank, "dtype": "bfloat16", "scale": 1.0,
                "convention": CONVENTION, "hf_key_prefix": prefix, "layers": sorted({m["layer"] for m in modules}),
                "modules": modules, "n_modules": len(modules), "n_params": int(sum(t.numel() for t in weights.values()))}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    worst = max(report, key=lambda r: r["resid_over_floor"]) if report else {}
    summary = {"modules": len(report), "worst_resid_over_floor": worst, "seconds": round(time.time() - t0, 1)}
    if args.check:
        summary["check"] = check(args.out, args.check)
    (args.out / "extract_report.json").write_text(json.dumps({"summary": summary, "projections": report}, indent=1))
    (args.out / "SHA256SUMS.json").write_text(json.dumps(sha256s(args.out), indent=1))
    log("summary", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
