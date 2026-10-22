"""Staged references for qwen3_5_moe: engine stage dumps against a float64 forward of the same checkpoint.

Each engine directory holds what families/qwen3_5/stages.zig writes (or qwen36_mlx_stages.py): one bf16 file per
stage (embed, lNN.mixer, lNN.block, final, logits) and index.json with every row's cache position. Two references:
  local    float64 of one stage fed the engine's own bf16 input to it: that stage's error alone.
  chained  float64 from the token ids: the accumulated error.
Per stage and engine: the worst row's relative error and its worst element in bf16 steps of the row's largest value.
At the asked teacher positions: every contender's logit in float64, chained and fed each engine's final norm.

  python tools/zig/qwen36_stage_ref.py --snapshot SNAP --ids ids.json --reference ref.json \\
      --engine metal=DIR --engine mlx=DIR --positions 24,39,44,54 --out stages.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen36_f64 import Checkpoint, Model  # noqa: E402


def stage_names(layers):
    names = ["embed"]
    for l in range(layers):
        names += [f"l{l:02d}.mixer", f"l{l:02d}.block"]
    return names + ["final", "logits"]


def bf16(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class Dump:
    """One engine's stage directory; refuses a missing stage, a wrong shape or rows at other positions."""

    def __init__(self, name, path: Path, layers, hidden, vocab, positions, logit_positions):
        self.name, self.path = name, path
        index = json.loads((path / "index.json").read_text())
        if (index["layers"], index["hidden"], index["vocab"]) != (layers, hidden, vocab):
            raise SystemExit(f"{name}: shape {index['layers']}/{index['hidden']}/{index['vocab']} is not the model's")
        # Logits may cover more rows (a one-row-step engine has every row's); the teacher rows are taken from them.
        have = {p: i for i, p in enumerate(index["logit_positions"])}
        if index["positions"] != positions or any(p not in have for p in logit_positions):
            raise SystemExit(f"{name}: rows are not positions {positions[0]}..{positions[-1]} / logits {logit_positions[0]}..")
        self.logit_rows = np.array([have[p] for p in logit_positions])
        listed = {s["name"]: s for s in index["stages"]}
        if sorted(listed) != sorted(stage_names(layers)):
            raise SystemExit(f"{name}: stages differ from {len(stage_names(layers))} expected")
        self.rows = {n: s["rows"] for n, s in listed.items()}
        self.width = {n: s["width"] for n, s in listed.items()}
        # An engine whose own pick is not its logits' first maximum (mlx_lm argmaxes its log-probabilities) lists it.
        self.picks = np.array(index["picks"]) if "picks" in index else None

    def get(self, stage):
        raw = np.fromfile(self.path / f"{stage}.bf16", dtype=np.uint16)
        if raw.size != self.rows[stage] * self.width[stage]:
            raise SystemExit(f"{self.name}: {stage}.bf16 has {raw.size} values")
        rows = bf16(raw.reshape(self.rows[stage], self.width[stage]))
        return rows[self.logit_rows] if stage == "logits" else rows


def step(v):
    """A bf16 step at |v|: 2^(exponent - 7)."""
    v = np.maximum(np.abs(v), np.finfo(np.float32).tiny)
    return np.exp2(np.floor(np.log2(v)) - 7)


def measure(engine, ref, rows_of_interest):
    """Per row: ||e - r|| / ||r|| and max|e - r| in bf16 steps of max|r|; the worst row and the asked rows."""
    diff = engine - ref
    rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-300)
    steps = np.abs(diff).max(axis=1) / step(np.abs(ref).max(axis=1))
    worst = int(np.argmax(rel))
    return {
        "max_rel": float(rel.max()), "median_rel": float(np.median(rel)), "worst_row": worst,
        "max_steps": float(steps.max()), "median_steps": float(np.median(steps)),
        "rows": {str(k): {"rel": float(rel[r]), "steps": float(steps[r])} for k, r in rows_of_interest.items()},
    }


def outliers(engine, local, gap, factor=5.0, most=8):
    """Rows whose MLP-stage error is `factor` times the median, with the float64 router's k-th/(k+1)-th logit gap
    on the engine's input: a small gap means the engine may have routed to the other expert."""
    rel = np.linalg.norm(engine - local, axis=1) / np.linalg.norm(local, axis=1)
    rows = np.nonzero(rel > factor * np.median(rel))[0]
    rows = rows[np.argsort(-rel[rows])][:most]
    return [[int(r), float(rel[r]), float(gap[r])] for r in rows]


def top1_summary(chained, picks):
    """Every teacher position's float64 top two and margin; per engine, the positions it picks another token at,
    with how far below float64's top its pick is (in bf16 steps at that logit)."""
    order = np.argsort(-chained, axis=1, kind="stable")[:, :2]
    top = np.take_along_axis(chained, order, axis=1)
    margin = top[:, 0] - top[:, 1]
    out = {"f64_margin_steps": [float(x) for x in margin / step(top[:, 0])], "engines": {}}
    for name, pick in picks.items():
        pick = np.asarray(pick)
        wrong = np.nonzero(pick != order[:, 0])[0]
        out["engines"][name] = {
            "matches_f64": int(len(pick) - len(wrong)), "of": len(pick),
            "other": [[int(p), int(pick[p]), int(order[p, 0]),
                       float((top[p, 0] - chained[p, pick[p]]) / step(top[p, 0]))] for p in wrong]}
    return out


def contenders(teacher, dumps, logits, reference_token):
    ids = {int(reference_token)}
    for d in dumps:
        ids.update(int(i) for i in np.argsort(-logits[d.name][teacher], kind="stable")[:2])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="teacher tokens; every one but the last is fed")
    ap.add_argument("--engine", action="append", default=[], help="NAME=DIR, repeatable")
    ap.add_argument("--logits", action="append", default=[],
                    help="NAME=FILE: an engine's bf16 logits for every fed row (tf-qwen35-cuda step), top-1 only")
    ap.add_argument("--positions", default="24,39,44,54", help="teacher positions to report in full")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (a quick check)")
    args = ap.parse_args()

    prompt = json.loads(args.ids.read_text())
    reference = json.loads(args.reference.read_text())
    tokens = prompt + reference[:-1]
    positions = list(range(len(tokens)))
    logit_positions = positions[len(prompt) - 1:]
    asked = [int(p) for p in args.positions.split(",") if p]
    asked_rows = {p: len(prompt) - 1 + p for p in asked}

    t0 = time.time()
    model = Model(Checkpoint(args.snapshot))
    layers = args.layers or model.layers
    dumps = []
    for spec in args.engine:
        name, _, path = spec.partition("=")
        dumps.append(Dump(name, Path(path), model.layers, model.hidden, model.vocab, positions, logit_positions))
    if not dumps:
        raise SystemExit("no --engine")

    result = {"tokens": len(tokens), "prompt": len(prompt), "engines": [d.name for d in dumps], "stages": {},
              "routing": {}, "teacher": {}}

    def record(stage, local, chained):
        rows = {p: p for p in asked} if stage == "logits" else asked_rows
        result["stages"][stage] = {
            d.name: {"local": measure(d.get(stage), local[i], rows),
                     "chained": measure(d.get(stage), chained, rows)} for i, d in enumerate(dumps)}

    x = model.embed(tokens)
    record("embed", [x] * len(dumps), x)
    prev = {d.name: d.get("embed") for d in dumps}
    for layer in range(layers):
        mixed = model.mixer(layer, [x] + [prev[d.name] for d in dumps], positions)
        h = x + mixed[0]
        local = [prev[d.name] + mixed[1 + i] for i, d in enumerate(dumps)]
        name = f"l{layer:02d}.mixer"
        record(name, local, h)
        after = {d.name: d.get(name) for d in dumps}
        y, picks, gap = model.moe(layer, [h] + [after[d.name] for d in dumps])
        x = h + y[0]
        local = [after[d.name] + y[1 + i] for i, d in enumerate(dumps)]
        name = f"l{layer:02d}.block"
        record(name, local, x)
        result["routing"][f"l{layer:02d}"] = {
            "chained_gap": {str(p): float(gap[0][r]) for p, r in asked_rows.items()},
            **{d.name: {"rows_with_other_picks": int((np.sort(picks[1 + i], 1) != np.sort(picks[0], 1)).any(1).sum()),
                        "asked_other_picks": {str(p): bool((np.sort(picks[1 + i][r]) != np.sort(picks[0][r])).any())
                                              for p, r in asked_rows.items()},
                        "outliers": outliers(d.get(name), local[i], gap[1 + i])}
               for i, d in enumerate(dumps)}}
        prev = {d.name: d.get(name) for d in dumps}
        print(f"layer {layer} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    if layers == model.layers:
        final = model.final(x)
        record("final", [model.final(prev[d.name]) for d in dumps], final)
        engine_final = {d.name: d.get("final") for d in dumps}
        lp = np.array(logit_positions)
        heads = model.head([final[lp]] + [engine_final[d.name][lp] for d in dumps])
        record("logits", heads[1:], heads[0])
        logits = {d.name: d.get("logits") for d in dumps}
        for p in asked:
            ids = contenders(p, dumps, logits, reference[p])
            chained = heads[0][p]
            best = np.argsort(-chained, kind="stable")[:3]
            result["teacher"][str(p)] = {
                "reference_token": int(reference[p]),
                "f64_chained_top3": [[int(i), float(chained[i])] for i in best],
                "contenders": {str(i): {
                    "f64_chained": float(chained[i]),
                    **{f"f64_from_{d.name}_final": float(heads[1 + k][p][i]) for k, d in enumerate(dumps)},
                    **{d.name: float(logits[d.name][p][i]) for d in dumps}} for i in ids},
                "top2": {d.name: [int(i) for i in np.argsort(-logits[d.name][p], kind="stable")[:2]] for d in dumps},
            }
        picks = {d.name: d.picks if d.picks is not None else np.argmax(logits[d.name], 1) for d in dumps}
        for spec in args.logits:
            name, _, path = spec.partition("=")
            raw = np.fromfile(path, dtype=np.uint16)
            if raw.size != len(tokens) * model.vocab:
                raise SystemExit(f"{name}: {path} is not {len(tokens)} rows of {model.vocab} logits")
            picks[name] = np.argmax(bf16(raw.reshape(len(tokens), model.vocab)[lp]), 1)
        picks["reference"] = np.array(reference)
        result["top1"] = top1_summary(heads[0], picks)

    result["seconds"] = time.time() - t0
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out} in {result['seconds']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
