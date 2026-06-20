"""Public command-line interface for TensorFold Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import tempfile

from smarttensor import __version__ as smarttensor_version
from smarttensor.manifest import SmartTensorManifest
from smarttensor.planner import format_bytes


DELEGATE_COMMANDS = {
    "inspect": "inspect",
    "manifest": "manifest",
    "pack": "pack-experts",
    "serve": "serve",
}
SMARTTENSOR_COMMAND_ALIASES = {
    "pack-experts": "pack",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in DELEGATE_COMMANDS:
        return _delegate_to_smarttensor(argv[0], argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorfold",
        description=(
            "TensorFold Runtime: run MoE models too large to fit by folding "
            "execution into a bounded memory budget."
        ),
    )
    parser.add_argument("--version", action="version", version=f"TensorFold Runtime {smarttensor_version}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="Check the local runtime without loading a model.")
    doctor.add_argument("--model-dir", type=Path, help="Optional model directory to check for existence.")
    doctor.add_argument("--pack-dir", type=Path, help="Optional expert pack directory to check for existence.")
    doctor.set_defaults(func=cmd_doctor)

    selftest = subcommands.add_parser("selftest", help="Run the no-model TensorFold install smoke test.")
    selftest.set_defaults(func=cmd_selftest)

    demo = subcommands.add_parser("demo", help="Create and inspect no-model TensorFold demo fixtures.")
    demo_subcommands = demo.add_subparsers(dest="demo_command", required=True)
    demo_create = demo_subcommands.add_parser("create", help="Create a tiny safetensors and MoE demo model.")
    demo_create.add_argument("output_dir", type=Path)
    demo_create.add_argument("--force", action="store_true", help="Overwrite files in an existing demo directory.")
    demo_create.set_defaults(func=cmd_demo_create)

    _add_delegate_parser(subcommands, "inspect", "Inspect safetensors shards.")
    _add_delegate_parser(subcommands, "manifest", "Write a TensorFold manifest JSON file.")
    _add_delegate_parser(subcommands, "pack", "Build contiguous expert packs for supported MoE models.")
    _add_delegate_parser(subcommands, "serve", "Serve an OpenAI-compatible endpoint with low-resident hosting.")

    bench = subcommands.add_parser("bench", help="Show benchmark guidance for the current TensorFold build.")
    bench.set_defaults(func=cmd_bench)

    canary = subcommands.add_parser("canary", help="Run release canaries for supported model families.")
    canary.add_argument("target", choices=["qwen-frontier"])
    canary.add_argument("args", nargs=argparse.REMAINDER)
    canary.set_defaults(func=cmd_canary)

    return parser


def _add_delegate_parser(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> None:
    parser = subcommands.add_parser(name, add_help=False, help=help_text)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.set_defaults(func=cmd_delegate)


def cmd_doctor(args: argparse.Namespace) -> int:
    print("TensorFold Runtime doctor")
    print(f"python: {platform.python_version()} ({sys.executable})")
    print(f"smarttensor: {smarttensor_version}")
    print("mlx: " + _mlx_status())

    if args.model_dir is not None:
        _print_path_status("model_dir", args.model_dir)
    if args.pack_dir is not None:
        _print_path_status("pack_dir", args.pack_dir)

    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from smarttensor.cli import main as smarttensor_main
    from tensorfold.demo import create_demo

    print("TensorFold selftest")
    with tempfile.TemporaryDirectory(prefix="tensorfold-selftest-") as tmp:
        root = Path(tmp)
        paths = create_demo(root / "demo", force=False)
        print("demo create: ok")

        toy_manifest = SmartTensorManifest.from_safetensors([paths.toy_safetensors])
        moe_manifest = SmartTensorManifest.from_safetensors([paths.toy_moe_shard])
        if not toy_manifest.tensors or not moe_manifest.tensors:
            print("inspect: failed", file=sys.stderr)
            return 1
        print("inspect: ok")

        pack_dir = root / "packs"
        pack_code = smarttensor_main(["pack-experts", str(paths.toy_moe_dir), "--out", str(pack_dir)])
        if pack_code != 0:
            print("pack: failed", file=sys.stderr)
            return pack_code or 1
        packs = sorted(pack_dir.glob("*.pack"))
        if len(packs) != 1:
            print(f"pack: expected 1 pack, found {len(packs)}", file=sys.stderr)
            return 1
        print("pack: ok")

    return 0


def cmd_demo_create(args: argparse.Namespace) -> int:
    from tensorfold.demo import create_demo

    try:
        paths = create_demo(args.output_dir, force=args.force)
    except FileExistsError as exc:
        print(f"tensorfold: {exc}", file=sys.stderr)
        return 2

    print(f"created TensorFold demo in {paths.root}")
    print("")
    print("Try:")
    print(f"  tensorfold inspect {paths.toy_safetensors}")
    print(f"  tensorfold inspect {paths.toy_moe_shard}")
    print(f"  tensorfold pack {paths.toy_moe_dir} --out {paths.root / 'toy-packs'}")
    return 0


def cmd_delegate(args: argparse.Namespace) -> int:
    return _delegate_to_smarttensor(args.command, args.args)


def _delegate_to_smarttensor(command: str, command_args: list[str]) -> int:
    from smarttensor.cli import main as smarttensor_main

    smart_command = DELEGATE_COMMANDS[command]
    display_command = SMARTTENSOR_COMMAND_ALIASES.get(smart_command, smart_command)
    return smarttensor_main(
        [display_command, *command_args],
        prog="tensorfold",
        command_aliases=SMARTTENSOR_COMMAND_ALIASES,
        error_prefix="tensorfold",
    )


def cmd_bench(args: argparse.Namespace) -> int:
    print("TensorFold Runtime bench")
    print("Use `tensorfold serve` for live serving and the benchmarks/ tools for measured runs.")
    print("No model was loaded by this command.")
    return 0


def cmd_canary(args: argparse.Namespace) -> int:
    if args.target == "qwen-frontier":
        print(
            "tensorfold: qwen-frontier canary is not bundled in this public runtime release. "
            "Use doctor, selftest, inspect, pack, and serve for supported workflows.",
            file=sys.stderr,
        )
        return 2
    raise AssertionError(f"unhandled canary target: {args.target}")


def _mlx_status() -> str:
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:
        return f"not importable ({exc.__class__.__name__}: {exc})"
    return f"available, metal={mx.metal.is_available()}"


def _print_path_status(label: str, path: Path) -> None:
    exists = path.exists()
    suffix = ""
    if exists and path.is_file():
        try:
            manifest = SmartTensorManifest.from_paths([path])
        except Exception:
            suffix = ""
        else:
            suffix = f", tensors={len(manifest.tensors)}, bytes={format_bytes(manifest.total_bytes)}"
    print(f"{label}: {path} ({'exists' if exists else 'missing'}{suffix})")
