#!/usr/bin/env python3
"""Step-5-Preview FP8 (stepfun-ai/Step-5-Preview) -> an MLX affine checkpoint one 256 GB Mac holds.

The routed experts (`moe.{gate,up,down}_proj`, stacked [352, out, in], FP8 E4M3 with fp32 128x128 block scales, or
bf16 in the layers that ship unquantised) become affine 2-bit in groups of 128: 2.25 bits a weight, about 153 GiB for
the 88 MoE layers. Each group's range is searched rather than taken from its min and max: 25 shrinks of the low and
high ends toward the group mean are tried and the one with the least squared error is kept, which at 2 bits removes
a large part of the rounding error the plain min/max range leaves; then 8 rounds of least squares refit each
group's scale and bias to its codes. On heavy-tailed synthetic experts that is 40% less squared error than
`mx.quantize` alone (relative MSE 0.210 to 0.126). The format is MLX's own (`mx.quantize` of the
clipped weights), so any affine 2-bit reader dequantises it.

Every other linear weight (attention, gate, sparse indexer projections, shared experts, the three dense MLPs, the
embedding and the head) becomes affine 8-bit in groups of 64. The router (`moe.gate.weight`), `router_bias`,
`sparse_indexer_w`, `ssmax_s` and every norm keep their stored dtype. Names are the checkpoint's own, plus
`.scales` / `.biases` beside each quantised `.weight`; norms stay zero-centred (the model applies 1 + w).

The MTP layers (92-94) and the vision tower are left out unless --mtp / --vision are given. --layers N writes only
layers [0, N) with a config that says so, for bring-up against a reference on a small slice; --unquantized
writes bf16 instead (experts dequantised from FP8), to check a reference before any rounding.

    ~/vllm-mlx/.venv/bin/python tools/step5_fp8_to_mlx.py --src ~/models/step5-preview-fp8 \\
        --out ~/models/Step-5-Preview-MLX-2bit [--layers 5] [--mtp] [--vision]

Streams one tensor at a time (experts in chunks of 32), so it needs a few GiB of RAM plus the GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

EXPERT = re.compile(r"\.moe\.(gate|up|down)_proj\.weight$")
KEEP = re.compile(r"(norm\.(weight|bias)$|\.moe\.gate\.weight$|router_bias$|sparse_indexer_w\.weight$|ssmax_s$)")
LAYER = re.compile(r"^model\.layers\.(\d+)\.")
EXPERT_BITS, EXPERT_GROUP = 2, 128
DENSE_BITS, DENSE_GROUP = 8, 64
SHRINKS = (1.0, 0.9, 0.8, 0.7, 0.6)
CHUNK = 32
REFINE = 8
SHARD_BYTES = 5 * 2**30


def fp8_dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """real = w_fp8 * s[block] over 128x128 blocks of the last two axes; bf16 out."""
    out_, in_ = w.shape[-2:]
    s = s.repeat_interleave(128, dim=-2)[..., :out_, :].repeat_interleave(128, dim=-1)[..., :in_]
    return (w.to(torch.float32) * s).to(torch.bfloat16)


def to_mx(t: torch.Tensor) -> mx.array:
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())


def quantize_searched(w: mx.array, group: int, bits: int):
    """MLX affine quantisation of bf16 `w` [..., K], each group's range picked from SHRINKS^2 by squared error."""
    shape = w.shape
    g = w.reshape(-1, group).astype(mx.float32)
    lo, hi = g.min(axis=1, keepdims=True), g.max(axis=1, keepdims=True)
    mid = g.mean(axis=1, keepdims=True)
    best = None
    for a in SHRINKS:
        for b in SHRINKS:
            l, h = mid + a * (lo - mid), mid + b * (hi - mid)
            c = mx.clip(g, l, h).astype(mx.bfloat16).reshape(shape)
            q, s, z = mx.quantize(c, group_size=group, bits=bits)
            err = ((mx.dequantize(q, s, z, group_size=group, bits=bits).reshape(-1, group).astype(mx.float32) - g) ** 2).sum(axis=1)
            q, s, z = q.reshape(err.shape[0], -1), s.reshape(-1), z.reshape(-1)
            if best is None:
                best = [err, q, s, z]
            else:
                take = err < best[0]
                best = [mx.where(take, err, best[0]), mx.where(take[:, None], q, best[1]),
                        mx.where(take, s, best[2]), mx.where(take, z, best[3])]
            mx.eval(best)
    _, q, s, z = best
    q, s, z = refine(g, q, s, z, bits)
    lead = shape[:-1]
    return q.reshape(*lead, -1), s.reshape(*lead, -1), z.reshape(*lead, -1)


def refine(g: mx.array, q: mx.array, s: mx.array, z: mx.array, bits: int, rounds: int = REFINE):
    """Alternate least squares on each group (fp32 rows `g`): fit scale and bias to the codes, round the codes to the
    bf16 scale and bias, keep the result only where the group's squared error falls. Returns packed words again."""
    n, sh = (1 << bits) - 1, mx.arange(0, 32, bits, dtype=mx.uint32)
    c = ((q[..., None] >> sh) & n).reshape(g.shape).astype(mx.float32)
    s, z = s.astype(mx.float32)[:, None], z.astype(mx.float32)[:, None]
    gm = g.mean(axis=1, keepdims=True)
    for _ in range(rounds):
        cm = c.mean(axis=1, keepdims=True)
        var = ((c - cm) ** 2).sum(axis=1, keepdims=True)
        s2 = mx.where(var > 0, ((c - cm) * (g - gm)).sum(axis=1, keepdims=True) / mx.maximum(var, 1e-30), s)
        s2 = s2.astype(mx.bfloat16).astype(mx.float32)
        z2 = (gm - s2 * cm).astype(mx.bfloat16).astype(mx.float32)
        c2 = mx.clip(mx.round((g - z2) / mx.where(s2 == 0, 1, s2)), 0, n)
        take = ((c2 * s2 + z2 - g) ** 2).sum(axis=1, keepdims=True) < ((c * s + z - g) ** 2).sum(axis=1, keepdims=True)
        c, s, z = mx.where(take, c2, c), mx.where(take, s2, s), mx.where(take, z2, z)
        mx.eval(c, s, z)
    words = (c.astype(mx.uint32).reshape(g.shape[0], -1, 32 // bits) << sh).sum(axis=-1).astype(mx.uint32)
    return words, s[:, 0].astype(mx.bfloat16), z[:, 0].astype(mx.bfloat16)


class Writer:
    def __init__(self, out: Path, stem: str = "model"):
        self.out, self.stem, self.buf, self.size, self.n, self.map, self.total = out, stem, {}, 0, 0, {}, 0

    def add(self, name: str, a: mx.array):
        mx.eval(a)
        self.buf[name] = a
        self.size += a.nbytes
        self.total += a.nbytes
        if self.size >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.n += 1
        fname = f"{self.stem}-{self.n:05d}.safetensors"
        mx.save_safetensors(str(self.out / fname), self.buf, metadata={"format": "mlx"})
        for k in self.buf:
            self.map[k] = fname
        self.buf, self.size = {}, 0

    def finish(self, merge: bool = False):
        self.flush()
        path = self.out / "model.safetensors.index.json"
        if merge:  # beside an existing checkpoint: add to its index
            old = json.loads(path.read_text())
            self.map = {**old["weight_map"], **self.map}
            self.total += old["metadata"]["total_size"]
        idx = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.map.items()))}
        path.write_text(json.dumps(idx, indent=1) + "\n")


def wanted(name: str, layers: int, args) -> bool:
    if name.startswith(("vision_model.", "vit_large_projector")):
        return args.vision
    m = LAYER.match(name)
    if m:
        i = int(m.group(1))
        if args.only_mtp:
            return i >= layers
        return i < layers or (args.mtp and args.layers is None and i >= layers)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, help="write only layers [0, N) (bring-up slice)")
    ap.add_argument("--mtp", action="store_true", help="keep the MTP layers (full model only)")
    ap.add_argument("--only-mtp", action="store_true", help="add the MTP layers to the checkpoint already at --out")
    ap.add_argument("--vision", action="store_true", help="keep the vision tower and projector")
    ap.add_argument("--only-vision", action="store_true", help="add the vision tower and projector to the checkpoint at --out")
    ap.add_argument("--unquantized", action="store_true", help="write bf16 (experts dequantised), for reference checks")
    args = ap.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    tc = cfg["text_config"]
    layers = args.layers or tc["num_hidden_layers"]
    wmap = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    names = [n for n in wmap if not n.endswith("weight_scale_inv") and wanted(n, layers, args)]
    if args.only_mtp:
        names = [n for n in names if LAYER.match(n)]
    if args.only_vision:
        args.vision = True
        names = [n for n in wmap if n.startswith(("vision_model.", "vit_large_projector"))]
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(wmap[n], []).append(n)

    args.out.mkdir(parents=True, exist_ok=True)
    w = Writer(args.out, "model-mtp" if args.only_mtp else "model-vision" if args.only_vision else "model")
    quant: dict[str, dict] = {}
    t0, done = time.time(), 0
    for fname in sorted(by_file):
        with safe_open(str(args.src / fname), framework="pt") as f:
            for name in sorted(by_file[fname]):
                base = name[: -len(".weight")] if name.endswith(".weight") else name
                t = f.get_tensor(name)
                if EXPERT.search(name):
                    sname = name + "_scale_inv"
                    if wmap.get(sname) == fname:
                        s = f.get_tensor(sname)
                    elif sname in wmap:
                        with safe_open(str(args.src / wmap[sname]), framework="pt") as g:
                            s = g.get_tensor(sname)
                    else:
                        s = None
                    qs, ss, zs = [], [], []
                    for e in range(0, t.shape[0], CHUNK):
                        we = fp8_dequant(t[e : e + CHUNK], s[e : e + CHUNK]) if s is not None else t[e : e + CHUNK].to(torch.bfloat16)
                        if args.unquantized:
                            qs.append(to_mx(we))
                            continue
                        q, sc, z = quantize_searched(to_mx(we), EXPERT_GROUP, EXPERT_BITS)
                        qs.append(q), ss.append(sc), zs.append(z)
                    w.add(name, mx.concatenate(qs))
                    if not args.unquantized:
                        w.add(base + ".scales", mx.concatenate(ss)), w.add(base + ".biases", mx.concatenate(zs))
                        quant[base] = {"group_size": EXPERT_GROUP, "bits": EXPERT_BITS}
                elif KEEP.search(name) or t.ndim != 2 or args.unquantized or name.startswith(("vision_model.", "vit_large_projector")):
                    w.add(name, to_mx(t))
                else:
                    q, sc, z = mx.quantize(to_mx(t.to(torch.bfloat16)), group_size=DENSE_GROUP, bits=DENSE_BITS)
                    w.add(name, q), w.add(base + ".scales", sc), w.add(base + ".biases", z)
                    quant[base] = {"group_size": DENSE_GROUP, "bits": DENSE_BITS}
                done += 1
                print(f"[{time.time() - t0:7.0f}s] {done}/{len(names)} {name} {tuple(t.shape)} {t.dtype}", flush=True)
                del t
    if args.only_vision:
        added = w.total
        w.finish(merge=True)
        old = json.loads((args.out / "config.json").read_text())
        for k, v in cfg.items():  # the multimodal keys the text-only conversion left out
            if k not in ("text_config", "quantization", "quantization_config"):
                old.setdefault(k, v)
        (args.out / "config.json").write_text(json.dumps(old, indent=1) + "\n")
        print(f"added {added / 2**30:.1f} GiB of vision tower and projector to {args.out} ({time.time() - t0:.0f}s)")
        return 0
    if args.only_mtp:
        added = w.total
        w.finish(merge=True)
        old = json.loads((args.out / "config.json").read_text())
        old_tc = old.get("text_config", old)
        old_tc["num_nextn_predict_layers"] = tc["num_nextn_predict_layers"]
        old["quantization"] = {**old.get("quantization", {}), **quant}
        (args.out / "config.json").write_text(json.dumps(old, indent=1) + "\n")
        print(f"added {added / 2**30:.1f} GiB of MTP layers to {args.out} ({time.time() - t0:.0f}s)")
        return 0
    w.finish()

    if args.layers is not None:
        tc["num_hidden_layers"] = layers
        tc["num_nextn_predict_layers"] = 0
        for k in ("layer_types", "rope_theta", "partial_rotary_factors", "swiglu_limits", "swiglu_limits_shared"):
            if isinstance(tc.get(k), list):
                tc[k] = tc[k][:layers]
        tc["moe_layer_list"] = [i for i in tc["moe_layer_list"] if i < layers]
        tc["moe_layers_enum"] = ",".join(map(str, tc["moe_layer_list"]))
    elif not args.mtp:
        tc["num_nextn_predict_layers"] = 0
    cfg.pop("quantization_config", None)
    tc.pop("quantization_config", None)
    if not args.unquantized:
        cfg["quantization"] = {"group_size": DENSE_GROUP, "bits": DENSE_BITS, "mode": "affine", **quant}
    if not args.vision:
        cfg.pop("vision_config", None)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                  "generation_config.json", "LICENSE.txt", "LICENSE-CN.txt"):
        if (args.src / extra).exists():
            shutil.copy2(args.src / extra, args.out / extra)
    print(f"wrote {w.total / 2**30:.1f} GiB in {w.n} shards to {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
