"""Check our prefill kernels against MLX 0.32.3's at Nemotron's prefill shapes: bits, time, entry points, launch JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402

import prefill_cases as pc  # noqa: E402
import prefill_kernels as pk  # noqa: E402
import prefill_glue as pg  # noqa: E402
import prefill_launch as pl  # noqa: E402

CHUNKS = (17, 35, 64, 100, 128, 192, 256)            # SSD steps are at most 256 rows
ROWS = (17, 35, 64, 100, 256, 300, 896, 1024, 1025, 1100, 2048, 4096, 8192)
ATTENTION = ((17, 0), (35, 0), (64, 0), (96, 0), (100, 0), (256, 0), (300, 0), (1024, 0), (1100, 0), (2048, 0),
             (17, 35), (100, 300), (64, 1), (64, 2048), (300, 2048), (2048, 1100), (2048, 2048), (4096, 0),
             (8192, 0), (4096, 4096))
PROJECTIONS = {"in_proj": "0.mixer.in_proj", "out_proj": "0.mixer.out_proj", "shared_up": "1.mixer.shared_experts.up_proj",
               "shared_down": "1.mixer.shared_experts.down_proj", "q_proj": "5.mixer.q_proj", "k_proj": "5.mixer.k_proj",
               "o_proj": "5.mixer.o_proj", "fc1": "1.mixer.switch_mlp.fc1", "fc2": "1.mixer.switch_mlp.fc2"}
QMM_ROWS = (129, 300, 1025, 2048, 4096, 8192)
EXPERT_ROWS = (86, 100, 300, 1100, 1400, 2048, 4096, 8192)        # MLX sorts experts into the NAX gather from 512 rows (86 tokens)
SPLIT_ROWS = 1024                                    # k/v (N 256) run MLX's split-K qmm up to here
FILES = ("gemm_nax.metal", "attention_nax.metal", "scan.metal", "conv.metal", "gemv.metal", "qmm_nax.metal",
         "glue.metal", "sort.metal")
JSON = pk.DIR / "kernels.json"


def weights(model: Path) -> dict:
    """Real tensors the kernels read: the router and conv weights and 4-bit projections of layers 0, 1 and 5."""

    from prefill_layers import tensors

    names = {"gate": "backbone.layers.1.mixer.gate.weight", "conv": "backbone.layers.0.mixer.conv1d.weight"}
    for key, path in PROJECTIONS.items():
        for part in ("weight", "scales", "biases"):
            names[f"{key}.{part}"] = f"backbone.layers.{path}.{part}"
    t = tensors(model, tuple(names.values()))
    return {k: t[v] for k, v in names.items()}


def cases(w: dict) -> list[pc.Case]:
    out = []
    for s in CHUNKS:
        out += [pc.ssd_cb(s), pc.ssd_y(s), pc.ssd_state(s), pc.scan(pc.HEADS, s, s, f"segsum s={s}"),
                pc.scan(1, s, pc.HEADS, f"dtA s={s}"), pc.gemv(s)]
    out += [pc.router(L, w["gate"]) for L in ROWS]
    out += [pc.conv(L, w["conv"]) for L in ROWS]
    out += [pc.attention(L, off) for L, off in ATTENTION]
    out += [pc.qmm_splitk(L, w["k_proj.weight"], w["k_proj.scales"], w["k_proj.biases"], "k_proj")
            for L in (129, 300, 600, 1024)]
    for L in (17, 35, 64, 100, 300, 1100, 4096, 8192):
        out += pc.expert_order(L)
    for key in ("fc1", "fc2"):
        out += [pc.experts(L, w[f"{key}.weight"], w[f"{key}.scales"], w[f"{key}.biases"], key) for L in EXPERT_ROWS]
    for key in [k for k in PROJECTIONS if not k.startswith("fc")]:
        rows = [L for L in QMM_ROWS if key != "k_proj" or L > SPLIT_ROWS]
        out += [pc.qmm(L, w[f"{key}.weight"], w[f"{key}.scales"], w[f"{key}.biases"], key) for L in rows]
    return out


def differ(a: mx.array, b: mx.array) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        raise SystemExit(f"shape or dtype mismatch: {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
    view = {2: mx.uint16, 4: mx.uint32}[a.itemsize]
    return int(mx.sum(a.view(view) != b.view(view)).item())


def timed(fn, reps: int) -> float:
    """Median microseconds a call over 3 rounds of `reps` calls in one evaluation each, after a warm call."""

    mx.eval(fn())
    rounds = []
    for _ in range(3):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval([fn() for _ in range(reps)])
        mx.synchronize()
        rounds.append((time.perf_counter() - start) * 1e6 / reps)
    return sorted(rounds)[1]


def launch_json(case: pc.Case) -> dict:
    rows = [{"function": v.function(), "file": v.family.file, "grid": list(grid), "threadgroup": list(group),
             "params": dict(zip({**pl.PARAM_FIELDS, **pg.PARAM_FIELDS}[v.family.name], params))}
            for v, _, grid, group, _, params in case.launches]
    return {"case": case.name, "mlx": case.mlx, "launches": rows}


def variants(all_cases: list[pc.Case], file: str) -> list[pk.Variant]:
    found = {v.function(): v for c in all_cases for v, *_ in c.launches if v.family.file == file}
    found.update({v.function(): v for v in pg.VARIANTS if v.family.file == file})
    return sorted(found.values(), key=lambda v: v.function())


def write(all_cases: list[pc.Case]) -> None:
    for file in FILES:
        found = variants(all_cases, file)
        print(f"{'wrote' if pk.write(file, found) else 'kept'} {file}: {len(found)} entry points")
    doc = {"compile": {"api": "newLibraryWithSource at run time, local #include \"nax.h\" inlined first",
                       "language": "metal4.0 (MLX's for this macOS)", "math_mode": "safe", "fp32_functions": "fast",
                       "trap": "an offline xcrun metallib (toolchain 32023.883) gives wrong bf16 tensor-op results here"},
           "dispatch": "dispatchThreads(grid, threadgroup); grid in threads, as mx.fast.metal_kernel launches",
           "params": {k: list(v) for k, v in pl.PARAM_FIELDS.items()},
           "cases": [launch_json(c) for c in all_cases]}
    JSON.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {JSON.relative_to(pk.ROOT)}: {len(all_cases)} cases")


def picked(case: pc.Case, only: str) -> bool:
    return any(part and part in case.name for part in only.split(","))


def page_bytes(a: mx.array) -> bytes:
    """An array's bytes padded to whole 16 KB pages, as tf-kernel-exact maps them."""

    raw = bytes(memoryview(a.view(mx.uint8) if a.dtype == mx.bfloat16 else a).cast("B"))
    return raw + bytes(-len(raw) % 16384)


def oracle(case: pc.Case, runner: pk.Runner, root: Path) -> None:
    """One tf-kernel-exact case directory a launch: inputs, the MLX-checked output, a manifest."""

    prev = None
    for n, (v, inputs, grid, group, shapes, _) in enumerate(case.launches):
        inputs = [prev if a is None else a for a in inputs]
        prev = runner(v, inputs, grid, group, shapes)[0]
        mx.eval(prev, *inputs)
        out = root / f"{case.name.replace(' ', '_').replace('=', '')}_{n}"
        out.mkdir(parents=True, exist_ok=True)
        lines = [f"function {v.function()}", "grid {} {} {}".format(*grid), "group {} {} {}".format(*group), "trials 1"]
        for i, a in enumerate(inputs):
            data = page_bytes(mx.contiguous(a))
            (out / f"in{i}.bin").write_bytes(data)
            lines.append(f"buffer {i} in{i}.bin {len(data)}")
        data = page_bytes(prev)
        (out / "expected.bin").write_bytes(data)
        lines.append(f"output {len(inputs)} expected.bin {len(data)} {prev.nbytes}")
        (out / "manifest.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="the Nemotron 3.5 Lightning MLX 4-bit folder")
    ap.add_argument("--write", action="store_true", help="regenerate entry points and kernels.json, then check")
    ap.add_argument("--only", default="", help="cases whose name contains any of these, comma separated")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", default="", help="write the results here")
    ap.add_argument("--oracle", type=Path, default=None, help="write tf-kernel-exact cases of the --only cases here")
    args = ap.parse_args()
    all_cases = cases(weights(args.model))
    if args.write:
        write(all_cases)
    stale = [f for f in FILES if pk.stale(f, variants(all_cases, f))]
    if stale:
        raise SystemExit(f"stale entry points in {stale}: run with --write")
    runner = pk.Runner(mx)
    if args.oracle:
        for case in [c for c in all_cases if picked(c, args.only)]:
            oracle(case, runner, args.oracle)
        return 0
    results, bad = [], 0
    for case in all_cases:
        if args.only and not picked(case, args.only):
            continue
        ref, ours = case.ref(), pl.run(runner, case.launches, pick=case.pick)
        mx.eval(ref, ours)
        n = differ(ref, ours)
        bad += n != 0
        t_mlx, t_ours = timed(case.ref, args.reps), timed(lambda: pl.run(runner, case.launches, False, case.pick), args.reps)
        results.append({"case": case.name, "mlx": case.mlx, "values": int(ref.size), "differ": n,
                        "mlx_us": round(t_mlx, 1), "ours_us": round(t_ours, 1)})
        print(f"{case.name:34s} {'OK ' if n == 0 else 'BAD'} {n:>8d}/{ref.size:<9d} mlx {t_mlx:9.1f} us  "
              f"ours {t_ours:9.1f} us  {t_ours / t_mlx:5.2f}x", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"mlx": mx.__version__, "results": results}, indent=1) + "\n")
    print(f"{len(results)} cases, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
