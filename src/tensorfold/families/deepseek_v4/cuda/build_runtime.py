"""Offline candidate build/install and header-only preflight for the existing GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tensorfold.families.deepseek_v4.gguf import (
    CHECKPOINT_0731,
    EOS_TOKENS_0731,
    TOKENIZER_TYPE_0731,
    VOCAB_SIZE_0731,
    prepare_candidate,
    validate_deepseek_v4_gguf_or_raise,
)
from tensorfold.gguf import parse_gguf_tensors

from .build import PACKAGE, PIN, VENDOR, build, verify_sources


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev, "inode": stat.st_ino}


def aligned_artifact_extra_bytes(tensors):
    """Pinned ds4_repack.cu byte geometry: MoE replaces raw; dense Q8 adds.

    Full raw-file residency is already budgeted, so charge only additional
    Q8 copies and expert alignment padding, not a second set of MoE weights.
    F16 prebuilding is disabled; aligned out_a uses the donor fused kernel.
    """
    extra = 0
    for t in tensors:
        dims, size = t.shape, t.size
        if (
            t.type_id == 8
            and len(dims) == 2
            and all(dims)
            and dims[0] % 1024 == 0
            and size >= 2 * 1024**2
            and size % 34 == 0
            and "token_embd" not in t.name
        ):
            blocks = size // 34
            extra += ((blocks * 2 + 63) // 64) * 64 + blocks * 32
        elif (
            t.type_id == 16
            and len(dims) == 3
            and all(dims)
            and dims[0] % 1024 == 0
            and dims[2] <= 2**32 - 1
            and size % 66 == 0
            and t.name.endswith((".ffn_gate_exps.weight", ".ffn_up_exps.weight"))
        ):
            blocks = size // 66
            extra += ((blocks * 2 + 63) // 64) * 64 + blocks * 64 - size
        elif (
            t.type_id == 10
            and len(dims) == 3
            and all(dims)
            and dims[0] % 256 == 0
            and dims[1] % 2 == 0
            and dims[2] <= 2**32 - 1
            and size % 84 == 0
            and t.name.endswith(".ffn_down_exps.weight")
        ):
            pairs = size // 84 // 2
            extra += ((pairs * 8 + 63) // 64) * 64 + ((pairs * 32 + 63) // 64) * 64 + pairs * 128 - size
    return extra


def inspect_inputs(args):
    """Map the file read-only; visit only header/table pages, never tensor payload."""
    source = args.gguf.expanduser().resolve(strict=True)
    before = stamp(source)
    with source.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        inventory = parse_gguf_tensors(mapped)
        md = inventory.info.metadata
        validate_deepseek_v4_gguf_or_raise(inventory, md)
        if md.get("general.architecture") != "deepseek4":
            raise ValueError("GGUF architecture must be deepseek4")
        tokens = md.get("tokenizer.ggml.tokens", [])
        eos = md.get("tokenizer.ggml.eos_token_id")
        if (
            len(tokens) != VOCAB_SIZE_0731
            or md.get("tokenizer.ggml.pre") != TOKENIZER_TYPE_0731
            or not isinstance(eos, int)
            or not 0 <= eos < len(tokens)
            or tokens[eos] != EOS_TOKENS_0731[0]
        ):
            raise ValueError("GGUF embedded tokenizer vocabulary/pre/EOS is incompatible with 0731")
        header_hash = hashlib.sha256(mapped[: inventory.data_offset]).hexdigest()
        arch = {k: v for k, v in md.items() if k.startswith("deepseek4.")}
        count, header_bytes = len(inventory.tensors), inventory.data_offset
        artifact_extra = aligned_artifact_extra_bytes(inventory.tensors)
    if stamp(source) != before:
        raise ValueError("GGUF changed during header inspection")
    external = {}
    if args.tokenizer_dir is not None:
        directory = args.tokenizer_dir.expanduser().resolve(strict=True)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            path = directory / name
            json.loads(path.read_text())
            external[name] = {"source": str(path), "sha256": digest(path)}
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if tokenizer.get_vocab_size() != len(tokens) or tokenizer.token_to_id(tokens[eos]) != eos:
            raise ValueError("optional external tokenizer vocabulary/EOS differs from GGUF")
    provenance = {
        "checkpoint": CHECKPOINT_0731,
        "vocab_size": len(tokens),
        "tokenizer_type": TOKENIZER_TYPE_0731,
        "eos_tokens": list(EOS_TOKENS_0731),
        "provenance": {"source": str(source), "sha256": header_hash, "sha256_scope": "gguf-header", "eos_id": eos},
    }
    # This is explicitly a HEADER hash, not a claimed hash of all 81 GiB of weights.
    identity = {**before, "header_bytes": header_bytes, "sha256_scope": "gguf-header"}
    return {
        "artifact_valid": True,
        "source": str(source),
        "source_size": before["size"],
        "source_identity": identity,
        "header_sha256": header_hash,
        "tensor_count": count,
        "aligned_artifact_extra_bytes": artifact_extra,
        "arch": arch,
        "tokenizer": provenance,
        "external_tokenizer": external,
        "context_window": args.context,
        "companion_reserve_gib": args.companion_reserve_gib,
        "current_memory_fit": {
            "status": "not_evaluated",
            "reason": "runtime capacity admission runs before model loading",
        },
    }


def native_library(args):
    directory = (args.native_library.parent if args.native_library else args.build_dir).expanduser().resolve()
    output = args.native_library.expanduser().resolve() if args.native_library else directory / "libtensorfold_ds4.so"
    manifest = verify_sources(VENDOR)
    receipt_path = directory / "native-build.json"
    if output.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("abi") == 1
            and receipt.get("revision") == PIN
            and receipt.get("backend") == "cuda"
            and receipt.get("cuda_arch") == "sm_121a"
            and receipt.get("source_sha256") == manifest["sha256"]
            and receipt.get("shim_sha256") == digest(PACKAGE / "native/shim.c")
            and receipt.get("library_sha256") == digest(output)
        ):
            return output
    if args.native_library:
        raise ValueError("selected native library lacks a matching source/shim/library build receipt")
    return build(directory, backend="cuda", jobs=args.jobs)


def make_wheel(source, library, destination):
    """Build from an isolated copy: never place a compiled .so in the checkout."""
    source, destination = Path(source), Path(destination)
    if not (source / "pyproject.toml").is_file():
        raise ValueError("building a wheel requires the TensorFold source checkout")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tensorfold-wheel-") as temporary:
        stage = Path(temporary)
        shutil.copytree(source / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"))
        for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(source / name, stage / name)
        shutil.copytree(source / "LICENSES", stage / "LICENSES")
        target = stage / "src/tensorfold/families/deepseek_v4/cuda/native/libtensorfold_ds4.so"
        shutil.copy2(library, target)
        # ctypes ABI is Python-version independent but the ELF is Linux/architecture specific.
        (stage / "setup.py").write_text("""from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
    def get_tag(self):
        return "py3", "none", super().get_tag()[2]
setup(cmdclass={"bdist_wheel": NativeWheel})
""")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                str(destination),
                str(stage),
            ],
            check=True,
        )
    wheels = list(destination.glob("tensorfold-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("isolated build did not produce exactly one TensorFold wheel")
    return wheels[0]


def install_candidate(args, report):
    if sys.prefix == sys.base_prefix:
        raise ValueError("build/install must run in the candidate virtual environment")
    source = Path(__file__).resolve().parents[5]
    library = native_library(args)
    with tempfile.TemporaryDirectory(prefix="tensorfold-artifact-") as temporary:
        wheel = make_wheel(source, library, Path(temporary))
        wheel_hash = digest(wheel)
        # The invoking candidate interpreter owns the install; no download or base-env mutation.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)],
            check=True,
        )
        wheel_dir = args.model_dir.expanduser().resolve().parent / "artifacts"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        saved = wheel_dir / wheel.name
        shutil.copy2(wheel, saved)
    code = (
        "from pathlib import Path; import tensorfold.families.deepseek_v4.cuda as c; "
        'print(Path(c.__file__).parent / "native/libtensorfold_ds4.so")'
    )
    installed = Path(subprocess.check_output([sys.executable, "-I", "-c", code], text=True).strip())
    from .native import NativeLibrary

    if NativeLibrary(installed).backend() != 1:
        raise RuntimeError("installed native library is not CUDA")
    cli = Path(sys.executable).parent / "tensorfold"
    if not cli.is_file():
        raise RuntimeError("candidate wheel did not install the TensorFold console script")
    return installed, {"path": str(saved), "sha256": wheel_hash, "donor_revision": PIN}


def execute(args):
    if (
        not math.isfinite(args.companion_reserve_gib)
        or args.companion_reserve_gib < 0
        or not 1 <= args.jobs <= 4
        or args.context < 1
    ):
        raise ValueError("reserve must be finite/nonnegative; jobs 1..4; context positive")
    args.model_dir = args.model_dir.expanduser().resolve()
    report = inspect_inputs(args)
    if args.preflight_only:
        return report
    library, wheel = install_candidate(args, report)
    if stamp(report["source"]) != {k: report["source_identity"][k] for k in ("size", "mtime_ns", "device", "inode")}:
        raise ValueError("GGUF changed during build; refusing candidate preparation")
    prepare_candidate(
        args.model_dir,
        arch=report["arch"],
        tokenizer=report["tokenizer"],
        source=report["source"],
        source_size=report["source_size"],
        source_sha256=report["header_sha256"],
        source_identity=report["source_identity"],
        reserve_gib=args.companion_reserve_gib,
        native_library=library,
        replace=True,
    )
    model = args.model_dir.expanduser().resolve()
    # Sidecars retain the embedded default even when optional local tokenizer files are supplied.
    for name, data in report["external_tokenizer"].items():
        if digest(data["source"]) != data["sha256"]:
            raise ValueError("optional tokenizer changed during build")
        shutil.copy2(data["source"], model / name)
    report.update(native_library=str(library), wheel=wheel, model_dir=str(model))
    temporary = model / "build-ready.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(model / "build-ready.json")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument(
        "--companion-reserve-gib",
        type=float,
        required=True,
        help="additional measured companion growth; existing occupancy is already in MemAvailable",
    )
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--build-dir", type=Path, default=Path.home() / ".cache/tensorfold/deepseek-native-cuda")
    p.add_argument("--native-library", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    return p


def main():
    try:
        report = execute(parser().parse_args())
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"DeepSeek build: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
