"""Measure the EXL3 linear against ExLlamaV3's own matmul on a real checkpoint's tensors, on an idle GPU.

    python tools/bench_exl3.py --model MODEL_DIR --rows 1,2,4,8,16

It cycles over enough copies of a layer's weights to exceed L2 (24 MB on GB10), so the numbers are DRAM figures,
and times CUDA graphs of back-to-back calls so launch overhead does not flatter either side. ExLlamaV3 is
imported for the comparison only (TensorFold never depends on it).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3.linear import Exl3Linear

DEFAULT_TENSORS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.2.self_attn.o_proj",
    "model.layers.41.self_attn.o_proj",
    "lm_head",
)


def _index(model_dir: Path) -> dict[str, str]:
    """The weight map, when the checkpoint ships one (a group absent from it is read from the shards instead)."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    return {}


def _find_shard(model_dir: Path, prefix: str) -> Path:
    """The safetensors file that holds ``prefix``.trellis, for a group the index does not name."""
    names = set()
    for part in sorted(model_dir.glob("*.safetensors")):
        try:
            from safetensors import safe_open
            with safe_open(str(part), framework="pt") as f:
                names = set(f.keys())
        except Exception:
            continue
        if prefix + ".trellis" in names:
            return part
    raise SystemExit(f"no safetensors file of {model_dir} holds {prefix}.trellis")


def weight_map(model_dir: Path) -> dict[str, str]:
    return _index(model_dir)


def read_group(model_dir: Path, prefix: str, index: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    rel = index.get(prefix + ".trellis")
    file = model_dir / rel if rel else _find_shard(model_dir, prefix)
    with safe_open(str(file), framework="pt") as f:
        names = set(f.keys())

        def get(part: str) -> torch.Tensor | None:
            return f.get_tensor(f"{prefix}.{part}") if f"{prefix}.{part}" in names else None

        trellis = get("trellis")
        suh = get("suh") if get("suh") is not None else get("su")
        svh = get("svh") if get("svh") is not None else get("sv")
    if trellis is None:
        raise SystemExit(f"{prefix} is not an EXL3 group in {model_dir}")
    return trellis, suh, svh


def graph_us(fns: list, per_graph: int, reps: int) -> float:
    for f in fns:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(per_graph):
            for f in fns:
                f()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / (per_graph * len(fns)) * 1e6)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensors", default=",".join(DEFAULT_TENSORS))
    parser.add_argument("--rows", default="1,2,4,8,16")
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--copy-mb", type=int, default=512, help="weight bytes cycled per call")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser()
    rows = [int(r) for r in args.rows.split(",")]
    index = weight_map(model_dir)
    exllamav3 = None
    try:
        from exllamav3.modules.quant.exl3 import LinearEXL3

        exllamav3 = LinearEXL3
    except ImportError:
        print("exllamav3 is not importable: TensorFold numbers only", flush=True)

    print(f"# {model_dir}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("| tensor | bits | bytes | rows | TensorFold us | GB/s | exllamav3 us | GB/s |", flush=True)
    for prefix in args.tensors.split(","):
        trellis, suh, svh = read_group(model_dir, prefix, index)
        group = fmt.scan(model_dir).groups[prefix]
        base = Exl3Linear.from_tensors(trellis, suh, svh, group.codebook)
        copies = max(1, -(-(args.copy_mb << 20) // base.nbytes()))
        layers = [base] + [Exl3Linear.from_tensors(trellis, suh, svh, group.codebook) for _ in range(copies - 1)]
        theirs = []
        if exllamav3 is not None:
            marker = torch.tensor(0, dtype=torch.int32, device="cuda")
            theirs = [exllamav3(None, base.k, base.n, suh=suh.cuda(), svh=svh.cuda(), trellis=trellis.cuda(),
                                mcg=marker if group.codebook == "mcg" else None,
                                mul1=marker if group.codebook == "mul1" else None) for _ in range(copies)]
        per_graph = copies if copies <= 32 else 1
        for m in rows:
            x = torch.randn((m, base.k), device="cuda").half()
            xh = torch.empty((m, base.k), device="cuda", dtype=torch.half)
            y = torch.empty((m, base.n), device="cuda", dtype=torch.half)
            z = torch.empty((32 * m * base.n,), device="cuda", dtype=torch.float32)
            mine = graph_us([lambda L=L: L(x, out=y, xh=xh, z=z) for L in layers], per_graph, args.reps)
            if theirs:
                ye = torch.empty((m, base.n), device="cuda", dtype=torch.half)
                ref = graph_us([lambda E=E: E.bc.run(x, ye) for E in theirs], max(1, per_graph // 4), args.reps)
                cell = f" {ref:.1f} | {base.nbytes() / ref / 1e3:.0f} |"
            else:
                cell = " - | - |"
            print(f"| {prefix} | {base.bits} | {base.nbytes() / 1e6:.1f} MB | {m} | {mine:.1f} | "
                  f"{base.nbytes() / mine / 1e3:.0f} |{cell}", flush=True)
        del layers, theirs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
