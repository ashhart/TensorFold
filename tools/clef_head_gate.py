#!/usr/bin/env python3
"""Gate 3 for Clef's joint schema head: the same BF16 hidden rows through the native, reference BF16 and FP64 heads."""
# usage:
#   clef_head_gate.py cases <model dir> <records> <rows> <out>        each record's spans, ids and rows paths (case.json)
#   clef_head_gate.py reference <model dir> <cases> <out> --dtype bf16|fp64 [--device cuda|cpu]
#   clef_head_gate.py native <tf-clef-test|tf-cuda-test> <model> <cases> <out> [--piece N|whole] [--backend hip|cuda]
#   clef_head_gate.py compare <cases> <native> <bf16> <fp64> [--report FILE] [--rule strict|margin]
# The strict rule: per question type, native mean and worst error against FP64 no larger than the reference BF16
# head's. The margin rule keeps the worst and lets the mean sit up to 10% above the reference's.
# The records are clef_backbone_rows.py's; their schema text comes again from its seeded generator, so each field's
# instruction and each option's description give the spans, as token ranges of the record by character offsets.
import argparse, importlib.util, json, math, os, subprocess, sys
from pathlib import Path

import numpy as np

SOURCES = {
    "torch": "2.15.0a0+rocm10.1.0a20260822",
    "transformers": "5.22.1",
    "checkpoint": "Cloudflare/clef-flash@fb4417d31c1d0f9e7e77f1141b40aaede7491e00",
}
TYPES = {"noul": 0, "choice": 1, "score": 2}
# the margin rule's bound on the native mean error, as a multiple of the reference BF16 head's
MEAN_MARGIN = 1.10


def rows_tool():
    spec = importlib.util.spec_from_file_location("clef_backbone_rows", Path(__file__).with_name("clef_backbone_rows.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fields(text):
    """Each field of a schema text: its type, its instruction's character range, and each option's JSON range."""
    out, at = [], 0
    while True:
        f = text.find("\nFIELD ", at)
        if f < 0:
            return out
        kind = text[text.index("TYPE: ", f) + 6:text.index("\n", text.index("TYPE: ", f))]
        i0 = text.index("INSTRUCTION: ", f) + len("INSTRUCTION: ")
        i1 = text.index("\nALLOWED OPTIONS:", i0)
        end = text.index("END FIELD\n", i1)
        opts, o = [], i1
        while True:
            o = text.find("OPTION ", o, end)
            if o < 0:
                break
            j0 = text.index("{", o)
            j1 = text.index("}\n", j0) + 1
            opts.append((j0, j1))
            o = j1
        out.append({"type": TYPES[kind], "instruction": (i0, i1), "options": opts})
        at = end


def tokens_of(offsets, lo, hi, base):
    """The token range [start, end) whose characters meet [lo, hi), shifted to the record's rows."""
    hit = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    return [base + hit[0], base + hit[-1] + 1]


def cmd_cases(a):
    rt = rows_tool()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    index = json.loads((Path(a.records) / "index.json").read_text())["records"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = [(b, j) for b, sizes in enumerate(rt.lengths()) for j in range(len(sizes))]
    if len(index) != len(flat):
        sys.exit(f"{len(index)} records in the index, {len(flat)} from the generator: not clef_backbone_rows.py's records")
    for r, (b, j) in zip(index, flat):
        rng = rt.random.Random(rt.SEED * 1000 + b * 100 + j)
        schema = rt.schema_text(rng)
        enc = tok(schema + rt.SUFFIX, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.load(Path(a.records) / f"{r['name']}.npy")
        base = ids.size - len(enc["input_ids"])
        if list(ids[base:]) != list(enc["input_ids"]):
            sys.exit(f"{r['name']}: the regenerated schema does not give the record's tail")
        qs, types, os_, owner = [], [], [], []
        for k, f in enumerate(fields(schema)):
            qs.append(tokens_of(enc["offset_mapping"], *f["instruction"], base))
            types.append(f["type"])
            for lo, hi in f["options"]:
                os_.append(tokens_of(enc["offset_mapping"], lo, hi, base))
                owner.append(k)
        case = {"name": r["name"], "bucket": r["bucket"], "tokens": int(ids.size),
                "ids": str((Path(a.records) / f"{r['name']}.npy").resolve()), "rows": str((Path(a.rows) / f"{r['name']}.npy").resolve()),
                "question_spans": qs, "question_types": types, "option_spans": os_, "option_question": owner}
        (out / f"{r['name']}.json").write_text(json.dumps(case))
    (out / "index.json").write_text(json.dumps({"sources": SOURCES, "cases": [r["name"] for r in index]}))
    print(f"{len(index)} cases into {out}")


def cmd_reference(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    spec = importlib.util.spec_from_file_location("joint_schema_model", Path(a.model) / "joint_schema_model.py")
    jsm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jsm
    spec.loader.exec_module(jsm)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float64
    device = a.device or ("cuda" if a.dtype == "bf16" else "cpu")
    head = jsm.JointSchemaHead(**json.loads((Path(a.model) / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(Path(a.model) / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=dtype).eval()
    index = json.loads((Path(a.model) / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(Path(a.model) / index["lm_head.weight"]), framework="pt") as f:
        table = f.get_tensor("lm_head.weight")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in json.loads((Path(a.cases) / "index.json").read_text())["cases"]:
        c = json.loads((Path(a.cases) / f"{name}.json").read_text())
        ids = np.load(c["ids"]).astype(np.int64)
        rows = torch.from_numpy(np.load(c["rows"]).view(np.int16)).view(torch.bfloat16)
        # the head reads lm_head rows only at the options' tokens: a table of the record's own ids, remapped
        uniq, local = np.unique(ids, return_inverse=True)
        sub = table[torch.from_numpy(uniq)].to(device=device, dtype=dtype)
        qs = []
        for k, (span, t) in enumerate(zip(c["question_spans"], c["question_types"])):
            opts = [tuple(s) for s, q in zip(c["option_spans"], c["option_question"]) if q == k]
            qs.append(jsm.EncodedQuestion(f"q{k}", t, tuple(span), tuple(opts), tuple(str(i) for i in range(len(opts)))))
        record = jsm.EncodedRecord(tuple(int(x) for x in local), tuple(qs), name)
        x = rows.to(device=device, dtype=dtype)[None]
        with torch.inference_mode():
            logits = head(x, torch.from_numpy(local)[None].to(device), torch.ones(1, ids.size, dtype=torch.long, device=device), [record], sub)[0]
        flat = torch.cat(logits).float().cpu() if a.dtype == "bf16" else torch.cat(logits).cpu()
        (out / f"{name}.json").write_text(json.dumps({"logits": [float(v) for v in flat.tolist()]}))
        print(f"{name}: {flat.numel()} logits", flush=True)
    (out / "meta.json").write_text(json.dumps({"sources": SOURCES, "torch": torch.__version__, "dtype": a.dtype, "device": device}))


def cmd_native(a):
    head = ["clef", "head"] if a.backend == "cuda" else ["head"]
    cmd = [a.binary, *head, a.model, a.cases, a.out] + (["--piece", a.piece] if a.piece else [])
    subprocess.run(cmd, check=True)
    if a.backend == "cuda":
        check_pieces(a)


def check_pieces(a):
    """Each case's consumed pieces, as the runner recorded them: in order, covering the prompt, at the asked size."""
    for name in json.loads((Path(a.cases) / "index.json").read_text())["cases"]:
        c = json.loads((Path(a.cases) / f"{name}.json").read_text())
        rows = int(np.load(c["ids"]).shape[0])
        size = rows if a.piece in (None, "whole") else int(a.piece)
        want = [[at, min(size, rows - at)] for at in range(0, rows, size)]
        got = json.loads((Path(a.out) / f"{name}.json").read_text())["pieces"]
        if got != want:
            sys.exit(f"{name}: consumed pieces {got[:3]}... are not {size}-row pieces of {rows} rows")
    print(f"pieces: every case consumed in {a.piece or 'whole'}-row pieces")


def errors(got, want):
    e = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64))
    return e


def cmd_compare(a):
    names = json.loads((Path(a.cases) / "index.json").read_text())["cases"]
    if not names:
        sys.exit("no cases: gate 3 needs every record")
    pooled = {t: {"native": [], "bf16": []} for t in TYPES.values()}
    same = nearer = farther = 0
    for name in names:
        c = json.loads((Path(a.cases) / f"{name}.json").read_text())
        want = json.loads((Path(a.fp64) / f"{name}.json").read_text())["logits"]
        got = json.loads((Path(a.native) / f"{name}.json").read_text())["logits"]
        ref = json.loads((Path(a.bf16) / f"{name}.json").read_text())["logits"]
        if not (len(want) == len(got) == len(ref) == len(c["option_question"])):
            sys.exit(f"{name}: logit counts differ")
        if not (np.isfinite(want).all() and np.isfinite(ref).all()):
            sys.exit(f"{name}: a non-finite reference logit voids the run")
        if not np.isfinite(got).all():
            print(f"{name}: native has a non-finite logit")
            return 1
        for x, y, z in zip(got, ref, want):
            same += x == y
            nearer += x != y and abs(x - z) < abs(y - z)
            farther += x != y and abs(x - z) > abs(y - z)
        kinds = [c["question_types"][q] for q in c["option_question"]]
        for e_n, e_r, t in zip(errors(got, want), errors(ref, want), kinds):
            pooled[t]["native"].append(e_n)
            pooled[t]["bf16"].append(e_r)
    names_of = {v: k for k, v in TYPES.items()}
    table, ok = [], True
    margin = MEAN_MARGIN if a.rule == "margin" else 1.0
    print("| type | logits | native mean | native worst | reference BF16 mean | reference BF16 worst | strict | margin |")
    print("|---|---|---|---|---|---|---|---|")
    for t, p in pooled.items():
        if not p["native"]:
            continue
        n, r = np.array(p["native"]), np.array(p["bf16"])
        row = {"type": names_of[t], "logits": int(n.size), "native_mean": float(n.mean()), "native_worst": float(n.max()),
               "bf16_mean": float(r.mean()), "bf16_worst": float(r.max())}
        worst_ok = row["native_worst"] <= row["bf16_worst"]
        row["strict"] = row["native_mean"] <= row["bf16_mean"] and worst_ok
        row["margin"] = row["native_mean"] <= MEAN_MARGIN * row["bf16_mean"] and worst_ok
        row["pass"] = row["native_mean"] <= margin * row["bf16_mean"] and worst_ok
        ok &= row["pass"]
        table.append(row)
        print(f"| {row['type']} | {row['logits']} | {row['native_mean']:.3e} | {row['native_worst']:.3e} | "
              f"{row['bf16_mean']:.3e} | {row['bf16_worst']:.3e} | {'yes' if row['strict'] else 'NO'} | "
              f"{'yes' if row['margin'] else 'NO'} |")
    total = sum(len(p["native"]) for p in pooled.values())
    print(f"paired: {total} logits, {same} identical to the reference BF16 head's; of the rest native is nearer FP64 in "
          f"{nearer}, farther in {farther}, as near in {total - same - nearer - farther}")
    if a.report:
        paired = {"logits": total, "identical": same, "native_nearer": nearer, "native_farther": farther}
        Path(a.report).write_text(json.dumps({"sources": SOURCES, "rule": a.rule, "types": table, "paired": paired, "pass": ok}, indent=1))
    print(f"PASS gate 3 ({a.rule} rule)" if ok else f"FAIL gate 3 ({a.rule} rule)")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    for x in ("model", "records", "rows", "out"):
        c.add_argument(x)
    r = sub.add_parser("reference")
    for x in ("model", "cases", "out"):
        r.add_argument(x)
    r.add_argument("--dtype", choices=["bf16", "fp64"], required=True)
    r.add_argument("--device")
    n = sub.add_parser("native")
    for x in ("binary", "model", "cases", "out"):
        n.add_argument(x)
    n.add_argument("--piece")
    n.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    m = sub.add_parser("compare")
    for x in ("cases", "native", "bf16", "fp64"):
        m.add_argument(x)
    m.add_argument("--report")
    m.add_argument("--rule", choices=["strict", "margin"], default="strict")
    a = p.parse_args()
    return {"cases": cmd_cases, "reference": cmd_reference, "native": cmd_native, "compare": cmd_compare}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
