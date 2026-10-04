"""Check zig/kernels/metal/ops bit for bit against MLX 0.32.3's own kernels, three ways, and time both."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = ROOT / "zig" / "kernels" / "metal" / "ops"
BUILD = ROOT / "build" / "zig-ops"
METALLIB = Path(mx.__file__).parent / "lib" / "mlx.metallib"
MLX_ATTRIBUTES = [  # (name, type) in the order mx.fast.metal_kernel writes them
    ("dispatch_quadgroups_per_threadgroup", "uint"), ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"), ("grid_origin", "uint3"), ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"), ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"), ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"), ("thread_index_in_quadgroup", "uint"), ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"), ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"), ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"), ("threads_per_grid", "uint3"), ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3")]
TYPE_NAMES = {mx.float32: "float", mx.float16: "float16_t", mx.bfloat16: "bfloat16_t", mx.uint8: "uint8_t",
              mx.uint16: "uint16_t", mx.uint32: "uint32_t", mx.int32: "int32_t", mx.int64: "int64_t",
              mx.bool_: "bool"}
BITS_VIEW = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


class OpFile:
    """A .metal file: the header between `// tf:header` and the first kernel, and each `// tf:kernel` entry point."""

    def __init__(self, path: Path) -> None:
        text = with_includes(path)
        self.path = path
        start = text.index("// tf:header\n") + len("// tf:header\n")
        first = text.index("// tf:kernel ")
        self.header = text[start:first]
        self.kernels: dict[str, dict[str, Any]] = {}
        marker = r"// tf:kernel (\S+) inputs=(\S+) outputs=(\S+)[^\n]*\n(.*?\) \{\n)(.*?)\n\}\n"
        for m in re.finditer(marker, text[first:], re.S):
            name, ins, outs, signature, body = m.groups()
            self.kernels[name] = {"inputs": ins.split(","), "outputs": outs.split(","), "signature": signature,
                                  "body": body}

    def run(self, name: str, inputs: list[mx.array], outputs: list[tuple[tuple[int, ...], Any]],
            grid: tuple[int, int, int], threadgroup: tuple[int, int, int]) -> list[mx.array]:
        k = self.kernels[name]
        if "mlx" not in k:
            expected = mlx_signature(name, k, inputs, [d for _, d in outputs])
            if " ".join(expected.split()) != " ".join(k["signature"].split()):
                raise AssertionError(f"{self.path.name}:{name}: signature is not MLX's\n{expected}\n{k['signature']}")
            k["mlx"] = mx.fast.metal_kernel(name=name, input_names=k["inputs"], output_names=k["outputs"],
                                            source=k["body"], header=self.header)
        return k["mlx"](inputs=inputs, output_shapes=[s for s, _ in outputs], output_dtypes=[d for _, d in outputs],
                        grid=grid, threadgroup=threadgroup)


def with_includes(path: Path) -> str:
    """A .metal file's text with its local includes inlined, as the run-time compiler must be given it."""

    return "".join((path.parent / ln.split('"')[1]).read_text() if ln.startswith('#include "') else ln
                   for ln in path.read_text().splitlines(keepends=True))


def inlined(path: Path, into: Path) -> str:
    """with_includes(path) written into `into`; the copy's path."""

    out = into / path.name
    out.write_text(with_includes(path))
    return str(out)


def mlx_signature(name: str, k: dict[str, Any], inputs: list[mx.array], out_dtypes: list[Any]) -> str:
    """The signature mx.fast.metal_kernel writes for this body and these inputs (its write_signature rules)."""

    lines = [f"[[kernel]] void {name}("]
    for i, (n, a) in enumerate(zip(k["inputs"], inputs)):
        where = "constant" if a.size < 8 else "device"
        lines.append(f"  const {where} {TYPE_NAMES[a.dtype]}{'&' if a.ndim == 0 else '*'} {n} [[buffer({i})]],")
    for j, (n, d) in enumerate(zip(k["outputs"], out_dtypes)):
        lines.append(f"  device {TYPE_NAMES[d]}* {n} [[buffer({len(inputs) + j})]],")
    lines += [f"  {t} {a} [[{a}]]," for a, t in MLX_ATTRIBUTES if a in k["body"]]
    return "\n".join(lines)[:-1] + ") {\n"


_files: dict[str, OpFile] = {}


def op_file(stem: str) -> OpFile:
    if stem not in _files:
        _files[stem] = OpFile(OPS_DIR / f"{stem}.metal")
    return _files[stem]


@dataclass
class Buf:
    """One bound buffer: input data (an array, a small list, or an earlier launch's output) or an output."""

    slot: int
    array: mx.array | None = None
    i32: list[int] | None = None
    u64: list[int] | None = None
    f32: list[float] | None = None
    src: tuple[int, int] | None = None          # (launch, output) of an earlier launch; a negative launch counts back
    out: tuple[tuple[int, ...], Any] | None = None
    copies: int = 1                             # rotated copies while timing (weights past the cache)

    def value(self, done: list[list[mx.array]]) -> mx.array:
        if self.src is not None:
            return done[self.src[0]][self.src[1]]
        if self.array is not None:
            return self.array
        if self.i32 is not None:
            return mx.array(self.i32, dtype=mx.int32)
        if self.u64 is not None:
            return mx.array(self.u64, dtype=mx.int64)
        return mx.array(self.f32, dtype=mx.float32)


@dataclass
class Launch:
    function: str
    buffers: list[Buf]
    grid: tuple[int, int, int]              # threads (dispatchThreads), or threadgroups when ``groups``
    threadgroup: tuple[int, int, int]
    groups: bool = False
    constants: list[tuple[int, str, float]] = field(default_factory=list)
    stem: str | None = None                 # its .metal file when not the case's

    def threads(self) -> tuple[int, int, int]:
        if not self.groups:
            return self.grid
        return tuple(g * t for g, t in zip(self.grid, self.threadgroup))   # type: ignore[return-value]


@dataclass
class Case:
    op: str
    label: str
    variant: str                            # the MLX kernel this shape runs
    stem: str                               # our .metal file
    ours: list[Launch]
    mlx: list[Launch] | None
    reference: list[mx.array]               # the MLX op's results, matched to the last launch's outputs in order
    time: bool = True
    chain: int = 32


def run_ours(case: Case) -> list[mx.array]:
    """Our launches through mx.fast.metal_kernel, the last launch's outputs."""

    done: list[list[mx.array]] = []
    for launch in case.ours:
        ins = [b.value(done) for b in launch.buffers if b.out is None]
        outs = [b.out for b in launch.buffers if b.out is not None]
        f = op_file(launch.stem or case.stem)
        done.append(f.run(launch.function, ins, outs, launch.threads(), launch.threadgroup))
    return done[-1]


def same_bits(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(mx.array_equal(a.view(BITS_VIEW[a.dtype.size]), b.view(BITS_VIEW[a.dtype.size])).item())


def differing(a: mx.array, b: mx.array) -> int:
    if a.shape != b.shape:
        return -1
    return int((a.view(BITS_VIEW[a.dtype.size]) != b.view(BITS_VIEW[a.dtype.size])).sum().item())


def runner() -> Path:
    exe, src = BUILD / "metal_bench", ROOT / "tools" / "zig" / "metal_bench.swift"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        BUILD.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", "-o", str(exe), str(src)], check=True)
    return exe


class Batch:
    """Runner jobs for many cases in one process: files for inputs, outputs read back after."""

    def __init__(self, tag: str) -> None:
        self.dir = BUILD / "io" / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs: list[dict[str, Any]] = []
        self.count = 0
        self.saved: dict[int, str] = {}

    def path(self, stem: str) -> str:
        self.count += 1
        return str(self.dir / f"{self.count}_{stem}.bin")

    def file_for(self, a: mx.array) -> str:
        key = id(a)
        if key not in self.saved:
            p = self.path("in")
            a = mx.contiguous(a)
            mx.eval(a)
            np.array(a.view(BITS_VIEW[a.dtype.size])).tofile(p)
            self.saved[key] = p
        return self.saved[key]

    def add(self, label: str, library: dict[str, str], launch: Launch, done_files: list[list[str]], reps: int,
            chain: int, timing: bool) -> list[str]:
        bufs, outs = [], []
        for b in launch.buffers:
            entry: dict[str, Any] = {"slot": b.slot}
            if b.out is not None:
                shape, dtype = b.out
                entry["bytes"] = int(np.prod(shape)) * dtype.size
                entry["out"] = self.path("out")
                outs.append(entry["out"])
            elif b.src is not None:
                entry["file"] = done_files[b.src[0]][b.src[1]]
            elif b.array is not None:
                entry["file"] = self.file_for(b.array)
                if timing and b.copies > 1:
                    entry["copies"] = b.copies
            elif b.i32 is not None:
                entry["i32"] = b.i32
            elif b.u64 is not None:
                entry["u64"] = b.u64
            else:
                entry["f32"] = b.f32
            bufs.append(entry)
        run = {"label": label, "library": library, "function": launch.function, "buffers": bufs,
               "threadgroup": list(launch.threadgroup), "reps": reps, "chain": chain,
               "constants": [{"index": i, "type": t, "value": v} for i, t, v in launch.constants]}
        run["threadgroups" if launch.groups else "threads"] = list(launch.grid)
        self.runs.append(run)
        return outs

    def execute(self) -> dict[str, list[float]]:
        spec = self.dir / "spec.json"
        spec.write_text(json.dumps({"runs": self.runs}))
        done = subprocess.run([str(runner()), str(spec)], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"metal_bench failed: {done.stderr[-2000:]}")
        times: dict[str, list[float]] = {}
        for line in done.stdout.splitlines():
            label, med, _lo, _hi = line.split()
            times.setdefault(label, []).append(float(med))
        return times


def read_out(path: str, like: mx.array) -> mx.array:
    raw = np.fromfile(path, dtype={1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[like.dtype.size])
    return mx.array(raw).view(like.dtype).reshape(like.shape)


RESULTS: list[dict[str, Any]] = []


def check(cases: list[Case], tag: str, timing: bool) -> None:
    """Bit checks (metal_kernel, direct compile, MLX's metallib) and the GPU-time A/B for a list of cases."""

    batch = Batch(tag)
    plan = []
    for n, case in enumerate(cases):
        refs = case.reference
        mx.eval(refs)
        mine = run_ours(case)
        mx.eval(mine)
        via_kernel = all(same_bits(r, o) for r, o in zip(refs, mine))
        worst = max((differing(r, o) for r, o in zip(refs, mine)), default=0) if not via_kernel else 0
        timed = timing and case.time
        lib = {"metallib": str(METALLIB)}
        entries: dict[str, list[list[str]]] = {}
        orders = (("mlx", "ours") * 3 if case.mlx else ("ours",) * 3) if timed else \
            (("ours", "mlx") if case.mlx else ("ours",))
        for k, who in enumerate(orders):
            launches = case.ours if who == "ours" else case.mlx
            files: list[list[str]] = []
            for s, launch in enumerate(launches or []):
                source = {"source": inlined(OPS_DIR / f"{launch.stem or case.stem}.metal", batch.dir)}
                files.append(batch.add(f"c{n}:{who}:{s}:{k}", source if who == "ours" else lib, launch, files,
                                       reps=12 if timed else 1, chain=case.chain if timed else 1, timing=timed))
            entries.setdefault(who, files)
        plan.append((case, refs, via_kernel, worst, entries))
    times = batch.execute()
    for n, (case, refs, via_kernel, worst, entries) in enumerate(plan):
        direct = [read_out(p, r) for p, r in zip(entries["ours"][-1], refs)]
        via_direct = all(same_bits(r, o) for r, o in zip(refs, direct))
        theirs_ok = None
        if "mlx" in entries:
            theirs = [read_out(p, r) for p, r in zip(entries["mlx"][-1], refs)]
            theirs_ok = all(same_bits(r, o) for r, o in zip(refs, theirs))

        def stage_us(who: str) -> float | None:
            stages = len(case.ours if who == "ours" else (case.mlx or []))
            if not stages:
                return None
            per = [statistics.mean(times.get(f"c{n}:{who}:{s}:{k}", [0.0])[0]
                                   for k in range(6) if f"c{n}:{who}:{s}:{k}" in times) for s in range(stages)]
            return sum(per)

        mlx_us = stage_us("mlx") if timing and case.time and case.mlx else None
        ours_us = stage_us("ours") if timing and case.time else None
        equal = via_kernel and via_direct
        detail = []
        if not via_kernel:
            detail.append(f"metal_kernel: {worst} values differ")
        if not via_direct:
            detail.append("direct compile differs")
        if theirs_ok is False:
            detail.append("MLX metallib replay differs from the MLX op (dispatch rule or bindings)")
        record(case, equal, "; ".join(detail), mlx_us, ours_us, theirs_ok)
    shutil.rmtree(batch.dir, ignore_errors=True)       # rotated weight copies are large


def record(case: Case, equal: bool, detail: str, mlx_us: float | None, ours_us: float | None,
           replay: bool | None) -> None:
    launches = [{"function": go.function, "grid": list(go.grid), "threadgroup": list(go.threadgroup),
                 "grid_is": "threadgroups" if go.groups else "threads"} for go in case.ours]
    RESULTS.append({"op": case.op, "case": case.label, "mlx_kernel": case.variant, "equal": equal,
                    "mlx_replay_equal": replay, "detail": detail, "mlx_us": mlx_us, "ours_us": ours_us,
                    "launches": launches})
    t = ""
    if mlx_us is not None and ours_us is not None:
        t = f"  mlx {mlx_us:8.2f} us  ours {ours_us:8.2f} us  ({ours_us / mlx_us:5.3f}x)"
    elif ours_us is not None:
        t = f"  ours {ours_us:8.2f} us"
    print(f"{'OK  ' if equal else 'DIFF'} {case.op:9s} {case.label:40s} {case.variant:34s}{t} {detail}", flush=True)


def main() -> None:
    sys.modules.setdefault("check_mlx_ops", sys.modules[__name__])     # the case modules share these classes
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mlx_op_cases as cases
    import mtp_op_cases

    builders = {**cases.BUILDERS, **mtp_op_cases.BUILDERS}

    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default=",".join(builders))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    mx.set_cache_limit(4 << 30)
    ck = cases.Checkpoint(cases.MODEL)
    for op in args.ops.split(","):
        for k, group in enumerate(builders[op](ck, args.quick)):
            check(group, f"{op}{k}", not args.no_time)
    bad = [r for r in RESULTS if not r["equal"]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} cases bit-identical", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(RESULTS, indent=1))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
