"""Command-line interface for SmartTensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from smarttensor.manifest import SmartTensorManifest
from smarttensor.planner import build_streaming_plan, format_bytes, parse_bytes
from smarttensor.runtime import SmartTensorRuntime


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "smarttensor",
    command_aliases: dict[str, str] | None = None,
    error_prefix: str = "smarttensor",
) -> int:
    parser = build_parser(prog=prog, command_aliases=command_aliases)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"{error_prefix}: error: {exc}", file=sys.stderr)
        return 1


def build_parser(
    *,
    prog: str = "smarttensor",
    command_aliases: dict[str, str] | None = None,
) -> argparse.ArgumentParser:
    command_aliases = command_aliases or {}
    parser = argparse.ArgumentParser(prog=prog)
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser("inspect", help="Inspect safetensors shards.")
    inspect.add_argument("paths", nargs="+", type=Path)
    inspect.add_argument("--json", action="store_true", help="Print full manifest JSON.")
    inspect.set_defaults(func=cmd_inspect)

    manifest = subcommands.add_parser("manifest", help="Write a SmartTensor manifest JSON file.")
    manifest.add_argument("paths", nargs="+", type=Path)
    manifest.add_argument("-o", "--output", required=True, type=Path)
    manifest.set_defaults(func=cmd_manifest)

    tensor = subcommands.add_parser("tensor", help="Read bytes from one tensor lazily.")
    tensor.add_argument("path", type=Path)
    tensor.add_argument("name")
    tensor.add_argument("--bytes", type=int, default=64, dest="byte_count")
    tensor.set_defaults(func=cmd_tensor)

    plan = subcommands.add_parser("plan", help="Build a memory-budget layer streaming plan.")
    plan.add_argument("paths", nargs="+", type=Path)
    plan.add_argument("--budget", required=True, help="Memory budget, e.g. 8GiB or 1200MB.")
    plan.add_argument("--prefetch-window", type=int, default=1)
    plan.add_argument("--json", action="store_true", help="Print plan JSON.")
    plan.set_defaults(func=cmd_plan)

    prefetch = subcommands.add_parser("prefetch-layer", help="Touch all tensor pages for one layer.")
    prefetch.add_argument("path", type=Path)
    prefetch.add_argument("layer", type=int)
    prefetch.set_defaults(func=cmd_prefetch_layer)

    pack = subcommands.add_parser(
        command_aliases.get("pack-experts", "pack-experts"),
        help="Build contiguous per-shard expert packs for a model (one .pack per shard with expert tensors).",
    )
    pack.add_argument("model_dir", type=Path)
    pack.add_argument("-o", "--out", required=True, type=Path, help="Output directory for the .pack files.")
    pack.add_argument(
        "--layers",
        help="Optional comma-separated layer ids/ranges to pack, e.g. 3,8-12.",
    )
    pack.set_defaults(func=cmd_pack_experts)

    serve = subcommands.add_parser(
        "serve",
        help="Serve an OpenAI-compatible chat endpoint with streamed low-resident hosting.",
    )
    serve.add_argument("model_dir", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8421)
    serve.add_argument("--served-name", help="Model id reported to clients; defaults to the directory name.")
    serve.add_argument("--resident-budget", default="2GiB", help="Resident weight budget, e.g. 2GiB or 4GB.")
    serve.add_argument("--retain-layers", type=int, help="Override: retain the first N base layers instead of using the budget.")
    serve.add_argument("--pin-policy", choices=["all", "phase"], default="phase")
    serve.add_argument("--loader-backend", choices=["native", "safetensors"], default="native")
    serve.add_argument("--max-tokens-default", type=int, default=512)
    serve.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum compatible non-streaming chat requests to decode together.",
    )
    serve.add_argument(
        "--batch-wait-ms",
        type=float,
        default=10.0,
        help="Milliseconds to wait for compatible requests before dispatching a batch.",
    )
    serve.add_argument(
        "--pack-read-workers",
        type=int,
        default=1,
        help=(
            "Maximum worker threads for concurrent expert pack reads. "
            "Values above 1 are experimental and model/storage dependent."
        ),
    )
    serve.add_argument(
        "--pack-dir",
        type=Path,
        help="Directory created by `tensorfold pack` for contiguous expert-pack reads.",
    )
    serve.add_argument(
        "--expert-hot-set",
        type=int,
        default=0,
        help="GPT-OSS only: resident expert rows per layer for hot-set speed mode.",
    )
    serve.add_argument(
        "--native-layers",
        type=int,
        default=0,
        help="GPT-OSS only: first N layers run the MLX-native fused forward (full bank resident). With --pin-policy all + greedy this enables the overlapped ~native-speed path.",
    )
    serve.add_argument(
        "--weight-page-budget",
        help="GPT-OSS/DeepSeek only: resident row-page cache budget for streamed expert rows, e.g. 3GiB.",
    )
    serve.add_argument(
        "--weight-page-policy",
        choices=["auto", "lru", "frequency", "scan_resistant", "two_queue"],
        default="auto",
        help=(
            "GPT-OSS/DeepSeek only: eviction policy for --weight-page-budget. "
            "auto keeps GPT-OSS on LRU and uses frequency for DeepSeek row paging."
        ),
    )
    serve.add_argument(
        "--weight-page-rows",
        type=int,
        default=1,
        help="GPT-OSS/DeepSeek only: leading-axis expert rows per cached weight page.",
    )
    serve.add_argument(
        "--expert-prefetch",
        choices=["off", "previous", "previous_table"],
        default="off",
        help=(
            "Predictive expert prefetch. GPT-OSS supports previous row-page "
            "warming; DeepSeek also supports previous_table."
        ),
    )
    serve.add_argument(
        "--expert-prefetch-cap",
        type=int,
        default=32,
        help="DeepSeek only: maximum predicted expert rows per layer to prefetch.",
    )
    serve.add_argument(
        "--expert-compute-mode",
        choices=[
            "table",
            "direct_qmm",
            "table_overlap_shared",
            "split_overlap_down",
            "split_overlap_shared_down",
            "slot_arena_direct_qmm",
            "per_expert",
        ],
        default="table",
        help=(
            "Selected-expert compute path. GPT-OSS supports table or "
            "direct_qmm; DeepSeek also supports the split/slot experimental "
            "paths."
        ),
    )
    serve.add_argument(
        "--expert-slot-capacity",
        type=int,
        help="DeepSeek only: slots per routed layer for slot_arena_direct_qmm. Defaults to model top-k.",
    )
    serve.add_argument(
        "--decode-scheduler",
        choices=["auto", "serial", "async-lookahead"],
        default="auto",
        help="Decode scheduling mode. auto only enables async lookahead on overlap-safe resident paths.",
    )
    serve.add_argument(
        "--sliding-cache",
        choices=["rotating", "kv", "temporal"],
        default="rotating",
        help="GPT-OSS only: cache backend for sliding-attention layers.",
    )
    serve.add_argument(
        "--page-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "NemotronH only: page MoE expert weights on demand instead of "
            "holding them resident (default on; use --no-page-experts to disable)."
        ),
    )
    serve.add_argument(
        "--exact-mode",
        choices=["target-verified", "exact-strict"],
        default="target-verified",
        help=(
            "Exactness contract. target-verified (default): emitted tokens are "
            "target-verified, rare near-ties vs the single-token baseline are "
            "counted in telemetry. exact-strict: bitwise-identical to single-token "
            "greedy — GPT-OSS runs speculation on a temporal sliding cache; "
            "Qwen and others run speculation off (SSM chunk-exactness unproven)."
        ),
    )
    serve.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Let thinking-mode chat templates emit reasoning blocks (slow at streamed token rates).",
    )
    serve.add_argument(
        "--draft",
        choices=["off", "prompt-lookup", "model", "external"],
        default="prompt-lookup",
        help="Speculative decoding drafter for greedy requests.",
    )
    serve.add_argument(
        "--draft-model",
        type=Path,
        help="Small same-tokenizer MLX model directory for --draft model.",
    )
    serve.add_argument(
        "--draft-command",
        help=(
            "JSON-lines helper command for --draft external. The helper can run "
            "outside the MLX Python process, for example a Core ML/ANE sidecar."
        ),
    )
    serve.add_argument("--max-draft", type=int, default=8)
    serve.add_argument(
        "--speculative-scheduler",
        choices=["linear", "tree"],
        default="linear",
        help=(
            "Speculative verifier shape for greedy single requests. "
            "'tree' verifies multiple prompt-lookup branches for non-streaming "
            "requests; streaming falls back to linear until tree supports callbacks."
        ),
    )
    serve.add_argument(
        "--max-branches",
        type=int,
        default=16,
        help="Maximum branches for --speculative-scheduler tree.",
    )
    serve.add_argument(
        "--draft-margin",
        type=float,
        default=0.5,
        help="Reject draft tokens whose verify top-1/top-2 logit gap is below this margin.",
    )
    serve.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort for harmony-template models (GPT-OSS).",
    )
    serve.set_defaults(func=cmd_serve)

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest = SmartTensorManifest.from_safetensors(args.paths)
    if args.json:
        print(manifest.to_json())
        return 0

    print_summary(manifest)
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    manifest = SmartTensorManifest.from_safetensors(args.paths)
    args.output.write_text(manifest.to_json() + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def cmd_tensor(args: argparse.Namespace) -> int:
    if args.byte_count < 0:
        raise ValueError("--bytes must be non-negative")
    with SmartTensorRuntime([args.path]) as runtime:
        record = runtime.tensor_record(args.name)
        data = runtime.read_tensor_bytes(args.name, limit=args.byte_count)
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
        print(data.hex())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    manifest = SmartTensorManifest.from_safetensors(args.paths)
    plan = build_streaming_plan(
        manifest,
        budget_bytes=parse_bytes(args.budget),
        prefetch_window=args.prefetch_window,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    print("SmartTensor streaming plan")
    print(f"  budget:          {format_bytes(plan.budget_bytes)}")
    print(f"  pinned:          {format_bytes(plan.pinned_bytes)}")
    print(f"  largest layer:   {format_bytes(plan.largest_layer_bytes)}")
    print(f"  peak estimate:   {format_bytes(plan.peak_estimated_bytes)}")
    print(f"  prefetch window: {plan.prefetch_window}")
    print(f"  fits budget:     {'yes' if plan.fits_budget else 'no'}")
    print(f"  steps:           {len(plan.steps)}")
    return 0


def cmd_prefetch_layer(args: argparse.Namespace) -> int:
    with SmartTensorRuntime([args.path]) as runtime:
        timings = runtime.prefetch_layer(args.layer)
        print(json.dumps(timings, indent=2, sort_keys=True))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from smarttensor.server import run_server

    return run_server(args)


def cmd_pack_experts(args: argparse.Namespace) -> int:
    from smarttensor.packstore import build_model_packs

    layers = parse_layer_filter(args.layers) if getattr(args, "layers", None) else None
    mapping = build_model_packs(args.model_dir, args.out, layers=layers)
    if not mapping:
        print(f"no expert-axis tensors found in {args.model_dir}")
        return 1
    total = 0
    for shard, pack in sorted(mapping.items()):
        size = Path(pack).stat().st_size
        total += size
        print(f"{Path(shard).name} -> {pack} ({size / 1e9:.2f} GB)")
    print(f"wrote {len(mapping)} pack(s), {total / 1e9:.2f} GB total, to {args.out}")
    return 0


def parse_layer_filter(raw: str) -> set[int]:
    layers: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, stop_raw = item.split("-", 1)
            start = int(start_raw)
            stop = int(stop_raw)
            if start < 0 or stop < 0:
                raise ValueError("--layers values must be non-negative")
            if stop < start:
                raise ValueError("--layers ranges must be ascending")
            layers.update(range(start, stop + 1))
        else:
            value = int(item)
            if value < 0:
                raise ValueError("--layers values must be non-negative")
            layers.add(value)
    if not layers:
        raise ValueError("--layers must include at least one layer id")
    return layers


def print_summary(manifest: SmartTensorManifest) -> None:
    print("SmartTensor manifest")
    print(f"  format:       {manifest.format}")
    print(f"  files:        {len(manifest.files)}")
    print(f"  tensors:      {len(manifest.tensors)}")
    print(f"  layers:       {len(manifest.layers)}")
    print(f"  total bytes:  {manifest.total_bytes:,}")
    print("  dtype bytes:")
    for dtype, nbytes in manifest.bytes_by_dtype().items():
        print(f"    {dtype:<8} {nbytes:>14,}")

    if manifest.layers:
        print("  layer bytes:")
        for _, layer in sorted(manifest.layers.items()):
            print(f"    layer {layer.index:<4} {layer.nbytes:>14,} bytes  {len(layer.tensor_names)} tensors")

    unlayered = manifest.unlayered_tensors()
    if unlayered:
        print(f"  unlayered tensors: {len(unlayered)}")
