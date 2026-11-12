#!/usr/bin/env python3
"""Clef-Flash's backbone rows: the gate-1 records, transformers' BF16 (GPU) and FP32 (CPU) rows, and the comparison."""
# usage:
#   clef_backbone_rows.py records <model dir> <out dir>                 the 40 records' ids (.npy) and index.json
#   clef_backbone_rows.py reference <model dir> <records> <out> --dtype bf16|fp32 [--device cuda|cpu] [--only N...]
#   clef_backbone_rows.py native <tf-qwen35-test> <model dir> <records> <out> [--cut N|whole] [--only N...]
#       [--backend cuda]: <tf-qwen35-cuda> instead, its `rows` mode (pieces of --cut, whole = one 4,096-row chunk)
#   clef_backbone_rows.py compare <model dir> <records> <native> <bf16> <fp32> [--report FILE] [--device cuda|cpu]
# The records are public synthetic text from a seeded generator; rows go to <out>/rec_NN.npy (bf16 as uint16 bits).
import argparse, json, math, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np

# the pinned reference: the ROCm torch and transformers this oracle ran with, and the checkpoint revision
SOURCES = {
    "torch": "2.15.0a0+rocm10.1.0a20260822",
    "transformers": "5.22.1",
    "checkpoint": "Cloudflare/clef-flash@fb4417d31c1d0f9e7e77f1141b40aaede7491e00",
}

# the gate's four length buckets in tokens, ten records each
BUCKETS = [(245, 1024), (1025, 4096), (4097, 10240), (10241, 16384)]
PER_BUCKET = 10
SEED = 1952

WORDS = ("account order status refund device policy region customer ticket delivery invoice warranty item "
         "service window carrier battery screen payment address shipment courier label return review note "
         "agent queue priority channel summary reason request update delay damage partial complete pending "
         "approved rejected escalated morning evening weekly monthly first second third final quick slow").split()
SYSTEM = ("You read a record and decide each schema field from the record alone. Answer every field with one of "
          "its allowed options.")
PREFIX = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"


def lengths():
    # every bucket's ten lengths, its two edges included: 245 and 16,384 are the gate's shortest and longest
    rng = random.Random(SEED)
    out = []
    for lo, hi in BUCKETS:
        picks = sorted(rng.randint(lo, hi) for _ in range(PER_BUCKET - 2))
        out.append([lo] + picks + [hi])
    return out


def state_text(rng, words):
    # a JSON-like customer record: nested events with seeded words, numbers and dates
    parts = []
    for i in range(words // 12 + 1):
        ev = {"id": f"E{rng.randint(1000, 99999)}", "kind": rng.choice(WORDS), "at": f"2026-{rng.randint(1, 12):02d}-"
              f"{rng.randint(1, 28):02d}", "amount": round(rng.uniform(1, 900), 2),
              "note": " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 14)))}
        parts.append(json.dumps(ev, separators=(",", ":"), sort_keys=True))
    return '{"events":[' + ",".join(parts) + "]}"


def schema_text(rng):
    lines = ["\n\nSCHEMA FIELDS:\n"]
    for i in range(1, rng.randint(2, 5) + 1):
        kind = rng.choice(["noul", "choice", "score"])
        lines.append(f"\nFIELD {i}\nID: {rng.choice(WORDS)}_{i}\nTYPE: {kind}\nINSTRUCTION: "
                     f"{' '.join(rng.choice(WORDS) for _ in range(8))}\nALLOWED OPTIONS:\n")
        for j in range(1, rng.randint(2, 4) + 1):
            lines.append(f'OPTION {j}: {{"description":"{rng.choice(WORDS)} {rng.choice(WORDS)}","option_id":"o{j}"}}\n')
        lines.append("END FIELD\n")
    return "".join(lines)


def tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def cmd_records(a):
    tok = tokenizer(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for b, sizes in enumerate(lengths()):
        for j, n in enumerate(sizes):
            rng = random.Random(SEED * 1000 + b * 100 + j)
            head = tok(PREFIX, add_special_tokens=False)["input_ids"]
            tail = tok(schema_text(rng) + SUFFIX, add_special_tokens=False)["input_ids"]
            body = tok(state_text(rng, n), add_special_tokens=False)["input_ids"]
            room = n - len(head) - len(tail)
            if room <= 0 or len(body) < room:
                sys.exit(f"record {b}.{j}: no room for {n} tokens")
            ids = np.array(head + body[:room] + tail, dtype=np.int32)
            name = f"rec_{len(index):02d}"
            np.save(out / f"{name}.npy", ids)
            index.append({"name": name, "bucket": b, "tokens": int(ids.size)})
    (out / "index.json").write_text(json.dumps({"sources": SOURCES, "records": index}, indent=1))
    print(f"{len(index)} records into {out}")


def chosen(records, only):
    index = json.loads((Path(records) / "index.json").read_text())["records"]
    return [r for r in index if not only or r["name"] in only or str(index.index(r)) in only]


def cmd_reference(a):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    device = a.device or ("cuda" if a.dtype == "bf16" else "cpu")
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa")
    text = model.model.language_model.to(device).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"sources": SOURCES, "torch": torch.__version__, "device": str(device), "dtype": a.dtype,
            "gdn": text.layers[0].linear_attn.__class__.__module__, "records": {}}
    for r in chosen(a.records, a.only):
        ids = torch.from_numpy(np.load(Path(a.records) / f"{r['name']}.npy").astype(np.int64))[None].to(device)
        t0 = time.time()
        with torch.inference_mode():
            rows = text(input_ids=ids, use_cache=False).last_hidden_state[0]
        rows = rows.cpu()
        if a.dtype == "bf16":
            np.save(out / f"{r['name']}.npy", rows.view(torch.int16).numpy().view(np.uint16))
        else:
            np.save(out / f"{r['name']}.npy", rows.float().numpy())
        meta["records"][r["name"]] = {"tokens": r["tokens"], "seconds": round(time.time() - t0, 1)}
        print(f"{r['name']}: {r['tokens']} rows in {time.time() - t0:.1f} s", flush=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))


def cmd_native(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.backend == "cuda":
        return native_cuda(a, out)
    for r in chosen(a.records, a.only):
        cmd = [a.binary, "hidden", a.model, str(Path(a.records) / f"{r['name']}.npy"), str(out / f"{r['name']}.npy")]
        if a.cut:
            cmd += ["--cut", a.cut]
        subprocess.run(cmd, check=True)


def native_cuda(a, out):
    # tf-qwen35-cuda reads ids as text and writes raw bf16 rows; saved as the HIP runner's uint16 .npy
    hidden = json.loads((Path(a.model) / "config.json").read_text())["text_config"]["hidden_size"]
    piece = "4096" if a.cut in (None, "whole") else a.cut
    for r in chosen(a.records, a.only):
        ids = np.load(Path(a.records) / f"{r['name']}.npy")
        text, raw = out / f"{r['name']}.ids", out / f"{r['name']}.bf16"
        text.write_text(",".join(str(int(i)) for i in ids))
        subprocess.run([a.binary, a.model, "rows", str(text), str(raw), piece], check=True)
        rows = np.fromfile(raw, dtype=np.uint16).reshape(-1, hidden)
        if rows.shape[0] != len(ids):
            sys.exit(f"{r['name']}: {rows.shape[0]} rows for {len(ids)} ids")
        np.save(out / f"{r['name']}.npy", rows)
        text.unlink()
        raw.unlink()


def widen(path):
    x = np.load(path)
    if x.dtype == np.uint16:
        return (x.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float32)


class Errors:
    """One contender's mean absolute error against FP32 over a bucket's pooled rows."""

    def __init__(self):
        self.sum, self.n = 0.0, 0

    def add(self, got, want):
        err = np.abs(got.astype(np.float64) - want.astype(np.float64))
        self.sum += float(err.sum())
        self.n += err.size

    def mean(self):
        return self.sum / max(self.n, 1)


def ulp(x):
    # a BF16 ulp at |x|: 8 bits of significand
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x != 0 else 2.0 ** -133


def head(model, device):
    import torch
    from safetensors import safe_open
    index = json.loads((Path(model) / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(Path(model) / index["lm_head.weight"]), framework="pt") as f:
        return f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)


def top2(w, row):
    # transformers' BF16 head on both contenders' rows: the check measures the rows, and Clef's decisions read the
    # rows through the joint head, never through the LM head
    import torch
    with torch.inference_mode():
        logits = torch.nn.functional.linear(torch.from_numpy(row).to(w.device, torch.bfloat16)[None], w)[0].float()
    v, i = torch.topk(logits, 2)
    return [int(i[0]), int(i[1])], [float(v[0]), float(v[1])]


def cmd_compare(a):
    import torch
    index = chosen(a.records, a.only)
    if not index:
        sys.exit("no records selected")
    missing = [f"{d}/{r['name']}.npy" for r in index for d in (a.native, a.bf16, a.fp32)
               if not (Path(d) / f"{r['name']}.npy").exists()]
    if missing:
        sys.exit(f"{len(missing)} row files missing, first {missing[0]}: every selected record must be compared")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    w = head(a.model, device)
    report = {"sources": SOURCES, "buckets": [], "top1": []}
    for b, (lo, hi) in enumerate(BUCKETS):
        recs = [r for r in index if r["bucket"] == b]
        if not recs:
            continue
        native, bf16, rows = Errors(), Errors(), 0
        for r in recs:
            want = widen(Path(a.fp32) / f"{r['name']}.npy")
            got = widen(Path(a.native) / f"{r['name']}.npy")
            ref = widen(Path(a.bf16) / f"{r['name']}.npy")
            if got.shape != want.shape or ref.shape != want.shape:
                sys.exit(f"{r['name']}: shapes {got.shape}, {ref.shape}, {want.shape}")
            if not (np.isfinite(want).all() and np.isfinite(ref).all()):
                sys.exit(f"{r['name']}: a non-finite reference value voids the run")
            if not np.isfinite(got).all():
                print(f"{r['name']}: native has a non-finite value")
                return 1
            native.add(got, want)
            bf16.add(ref, want)
            rows += want.shape[0]
            ni, nv = top2(w, got[-1])
            ri, rv = top2(w, ref[-1])
            if ni[0] != ri[0]:
                same_pair = set(ni) == set(ri)
                ok = same_pair and nv[0] - nv[1] <= ulp(nv[0]) and rv[0] - rv[1] <= ulp(rv[0])
                report["top1"].append({"record": r["name"], "native": ni, "native_logits": nv, "bf16": ri,
                                       "bf16_logits": rv, "passes": ok})
        n, f = native.mean(), bf16.mean()
        passed = n <= f
        report["buckets"].append({"bucket": f"{lo}-{hi}", "records": [r["name"] for r in recs], "rows": rows,
                                  "native_mean": n, "bf16_mean": f, "passes": passed})
        print(f"bucket {lo}-{hi}: {len(recs)} records, {rows} rows, mean |error| native {n:.6e} "
              f"bf16 {f:.6e}: {'PASS' if passed else 'FAIL'}")
    diffs = report["top1"]
    print(f"teacher-forced last-row top-1: {len(diffs)} differences, {sum(not d['passes'] for d in diffs)} failing")
    for d in diffs:
        print(f"  {d['record']}: native {d['native']} {d['native_logits']}, bf16 {d['bf16']} {d['bf16_logits']}")
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    ok = all(b["passes"] for b in report["buckets"]) and all(d["passes"] for d in diffs)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("model")
    r.add_argument("out")
    f = sub.add_parser("reference")
    f.add_argument("model")
    f.add_argument("records")
    f.add_argument("out")
    f.add_argument("--dtype", choices=["bf16", "fp32"], required=True)
    f.add_argument("--device")
    f.add_argument("--only", nargs="*", default=[])
    n = sub.add_parser("native")
    n.add_argument("binary")
    n.add_argument("model")
    n.add_argument("records")
    n.add_argument("out")
    n.add_argument("--cut")
    n.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    n.add_argument("--only", nargs="*", default=[])
    c = sub.add_parser("compare")
    for name in ("model", "records", "native", "bf16", "fp32"):
        c.add_argument(name)
    c.add_argument("--report")
    c.add_argument("--device")
    c.add_argument("--only", nargs="*", default=[])
    a = p.parse_args()
    return {"records": cmd_records, "reference": cmd_reference, "native": cmd_native, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
