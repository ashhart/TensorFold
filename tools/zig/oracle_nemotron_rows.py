"""MLX oracle for the Nemotron expert kernels: random inputs at the real shapes and MLX's outputs (GPU lock)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_nemotron_metal as gen  # noqa: E402

from tensorfold.kernels.inputs import padded  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import rows  # noqa: E402

ALIGN = 256                         # per-trial strides: buffer offsets stay 256-byte aligned


class Linear(dict):
    """A stacked 4-bit expert projection as rows.experts reads it."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> None:
        super().__init__(weight=weight, scales=scales, biases=biases)
        self.group_size = group_size


class Table:
    def __init__(self, fc1: Linear, fc2: Linear) -> None:
        self.fc1, self.fc2 = fc1, fc2


def quantized(experts: int, n: int, k: int, seed: int) -> Linear:
    """Random bf16 weights [experts, n, k] (std 0.02) quantized to 4 bits in groups of gen.GROUP, one expert at a time."""

    parts = []
    for e in range(experts):
        w = (mx.random.normal((n, k), key=mx.random.key(seed * 1000 + e)) * 0.02).astype(mx.bfloat16)
        parts.append(mx.quantize(w, group_size=gen.GROUP, bits=4))
        mx.eval(parts[-1])
    return Linear(*(mx.stack([p[i] for p in parts]) for i in range(3)), group_size=gen.GROUP)


def stride(nbytes: int) -> int:
    return -(-nbytes // ALIGN) * ALIGN


def write_stacked(path: Path, trials: list[np.ndarray]) -> int:
    """Each trial's bytes at a multiple of one aligned stride; returns the stride."""

    step = stride(max(t.nbytes for t in trials))
    with open(path, "wb") as f:
        for t in trials:
            raw = np.ascontiguousarray(t).tobytes()
            f.write(raw + bytes(step - len(raw)))
    return step


def raw(a: mx.array) -> np.ndarray:
    """An mx.array's bytes (bf16 as uint16)."""

    mx.eval(a)
    return np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a, copy=True)


def mlx_text(kernel, **call) -> str:
    """The kernel text MLX generates for this call (its verbose print, captured from fd 1)."""

    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            mx.eval(kernel(**call, verbose=True))
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        printed = tmp.read().decode()
    body = printed.split("```\n", 1)[1]
    return body[: body.rindex("\n```")]


def run_case(table: Table, rows_count: int, grouped: bool, trials: int, seed: int, kinds: dict) -> dict:
    """rows.experts' two launches, as it makes them, for `trials` random inputs; returns per-kind trial arrays."""

    rng = np.random.default_rng(seed)
    fc1, fc2 = table.fc1, table.fc2
    experts, hidden = int(fc1["weight"].shape[0]), int(fc1["weight"].shape[1])
    dims, out = int(fc1["weight"].shape[2]) * 8, int(fc2["weight"].shape[1])
    block = rows.RPS * gen.SIMDGROUPS
    got: dict = {"up": [], "down": []}
    for t in range(trials):
        x = (mx.random.normal((rows_count, dims), key=mx.random.key(seed + t))).astype(mx.bfloat16)
        ids_np = np.stack([rng.permutation(experts)[: gen.TOP_K] for _ in range(rows_count)]).astype(np.uint32)
        ids = mx.array(ids_np.reshape(-1))
        pairs = rows_count * gen.TOP_K
        uids, start, counts, members, used, groups = rows._grouping(ids, rows_count, experts, grouped)
        tables = [uids, start, counts, members, used]
        up_call = dict(inputs=[x, *tables, fc1["weight"], fc1["scales"], fc1["biases"]],
                       template=kinds["up"]["template"], grid=(32 * gen.SIMDGROUPS, hidden // block, groups),
                       threadgroup=(32 * gen.SIMDGROUPS, 1, 1), output_shapes=[(pairs, hidden)],
                       output_dtypes=[mx.bfloat16])
        act = kinds["up"]["mlx"](**up_call)[0]
        down_call = dict(inputs=[act, *tables, fc2["weight"], fc2["scales"], fc2["biases"]],
                         template=kinds["down"]["template"], grid=(32 * gen.SIMDGROUPS, out // block, groups),
                         threadgroup=(32 * gen.SIMDGROUPS, 1, 1), output_shapes=[(pairs, out)],
                         output_dtypes=[mx.bfloat16])
        y = kinds["down"]["mlx"](**down_call)[0]
        # the production entry point must give the same bits as the launches recorded here
        same = mx.array_equal(rows.experts(table, x, mx.array(ids_np), grouped=grouped).reshape(pairs, out), y)
        if not bool(same.item()):
            raise SystemExit("rows.experts differs from the oracle's launches")
        if t == 0:
            for kind, call in (("up", up_call), ("down", down_call)):
                kinds[kind]["text_ok"] = mlx_text(kinds[kind]["mlx"], **call) == kinds[kind]["gen"]["text"]
        tabs = [raw(a) for a in tables]
        got["up"].append({"inputs": [raw(x), *tabs], "out": raw(act), "grid": up_call["grid"]})
        got["down"].append({"inputs": [raw(act), *tabs], "out": raw(y), "grid": down_call["grid"]})
    return got


def write_case(root: Path, name: str, kind: dict, trials: list[dict], weights: list[str]) -> None:
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    grids = {tuple(t["grid"]) for t in trials}
    assert len(grids) == 1, "one grid a case"
    lines = [f"function {kind['gen']['function']}", f"source zig/kernels/metal/{kind['gen']['file']}",
             "grid {} {} {}".format(*grids.pop()), f"group {32 * gen.SIMDGROUPS} 1 1", f"trials {len(trials)}"]
    names = ["X", "UIDS", "START", "COUNT", "MEMBERS", "UCOUNT"]
    for i, n in enumerate(names):
        step = write_stacked(case / f"{n}.bin", [t["inputs"][i] for t in trials])
        lines.append(f"buffer {i} {n}.bin {step}")
    for i, w in enumerate(weights):
        lines.append(f"buffer {6 + i} ../{w} 0")
    step = write_stacked(case / "expected.bin", [t["out"] for t in trials])
    lines.append(f"output 9 expected.bin {step} {trials[0]['out'].nbytes}")
    (case / "manifest.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--trials", type=int, default=96, help="1-row trials (grouped 16-row cases run trials / 12)")
    args = ap.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    kinds = {}
    for k in gen.kernels():
        mlx_kernel = rows._kernel(f"nemotron_rows_expert_{k['kind']}", rows._EXPERT_UP if k["kind"] == "up"
                                  else rows._EXPERT_DOWN, ["X", "UIDS", "START", "COUNT", "MEMBERS", "UCOUNT", "W",
                                                           "S", "B"], ["ACT" if k["kind"] == "up" else "Y"])
        kinds[k["kind"]] = {"gen": k, "mlx": mlx_kernel, "template": k["template"]}

    table = Table(quantized(gen.EXPERTS, gen.EXPERT, gen.HIDDEN, 1), quantized(gen.EXPERTS, gen.HIDDEN, gen.EXPERT, 2))
    for kind, fc in (("up", table.fc1), ("down", table.fc2)):
        names = []
        for part in ("weight", "scales", "biases"):
            fname = f"{kind}_{part}.bin"
            (root / fname).write_bytes(raw(fc[part]).tobytes())
            names.append(fname)
        kinds[kind]["weights"] = names

    summary = {}
    for case, rows_count, grouped, trials in (("row1", 1, False, args.trials), ("rows16", 16, True, args.trials // 12)):
        got = run_case(table, rows_count, grouped, trials, 7 + rows_count, kinds)
        for kind in ("up", "down"):
            name = f"{kind}_{case}"
            write_case(root, name, kinds[kind], got[kind], kinds[kind]["weights"])
            summary[name] = {"trials": trials, "outputs": int(sum(t["out"].size for t in got[kind]))}
    summary["mlx_text_equals_generated"] = {k: v.get("text_ok") for k, v in kinds.items()}
    summary["mlx"] = mx.__version__
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if all(summary["mlx_text_equals_generated"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
