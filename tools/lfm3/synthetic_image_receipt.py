#!/usr/bin/env python3
"""Measure the synthetic image through the native tower and decoder and the independent references, then write one receipt.

Needs NumPy, the official LFM3-VL-2B MLX checkpoints, the built `tf-lfm3-vision` and `tf-lfm3-forward` tools and a Mac with
Metal. Every value in the receipt is read back from a file a tool wrote; nothing is carried over from the photograph receipts.
Run from the checkout root; the commands the receipt records are the ones executed here.
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import synthetic_image as fixture  # noqa: E402
from compare_logits import compare, read_bf16  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
VOCABULARY, HIDDEN, IMAGE_START, SUFFIX_START, GRID = 128000, 2048, 4, 74, (14, 20)
THREADS = {"OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
COMMANDS: list[dict] = []


def run(command: list, env: dict | None = None) -> None:
    """Execute and record one command; any failure stops the receipt."""
    text = [str(c) for c in command]
    COMMANDS.append({"argv": text, "environment": env or {}})
    print("+", " ".join(text), flush=True)
    subprocess.run(text, check=True, cwd=ROOT, env={**os.environ, **(env or {})})


def output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def words(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2")


def floats(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def save_rows(path: pathlib.Path, bf16: pathlib.Path) -> None:
    np.save(path, floats(words(bf16)).reshape(-1, HIDDEN))


def vision_weight_identity(model: pathlib.Path) -> dict:
    """A hash over every vision and projector tensor's name, dtype, shape and bytes, in name order."""
    with open(model / "model.safetensors", "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
        names = sorted(n for n in header if n.startswith(("vision_tower.", "multi_modal_projector.")))
        digest, total = hashlib.sha256(), 0
        for name in names:
            entry = header[name]
            begin, end = entry["data_offsets"]
            f.seek(8 + length + begin)
            digest.update(json.dumps([name, entry["dtype"], entry["shape"]]).encode() + f.read(end - begin))
            total += end - begin
    return {"tensors": len(names), "tensor_bytes": total, "named_tensor_sha256": digest.hexdigest(),
            "algorithm": "sha256 over, per tensor in name order, the JSON [name, dtype, shape] then its bytes"}


def vision_stats(native: np.ndarray, reference: np.ndarray) -> dict:
    """The projected rows' differences, as the photograph receipt reported its own."""
    a, b = floats(native).reshape(-1, HIDDEN).astype(np.float64), floats(reference).reshape(-1, HIDDEN).astype(np.float64)
    if a.shape != b.shape or a.shape[0] != fixture.IMAGE_ROWS:
        raise ValueError(f"expected {fixture.IMAGE_ROWS} rows of {HIDDEN}, got {a.shape} and {b.shape}")
    error = a - b
    columns = np.argsort((error ** 2).sum(axis=0))[-20:][::-1]
    return {"values": int(a.size), "equal_values": int((native == reference).sum()),
            "relative_rms": float(np.sqrt((error ** 2).sum() / (b ** 2).sum())),
            "maximum_absolute_error": float(np.abs(error).max()),
            "worst_columns": [{"column": int(c), "rms": float(np.sqrt((error[:, c] ** 2).mean())),
                               "max_abs": float(np.abs(error[:, c]).max())} for c in columns],
            "rows": [{"row": i, "rms": float(np.sqrt((error[i] ** 2).mean())), "max_abs": float(np.abs(error[i]).max())}
                     for i in range(len(a))]}


def decoder(native: pathlib.Path, reference: pathlib.Path, tokens: pathlib.Path) -> dict:
    """Teacher-forced logits over all 90 rows and over the 16 suffix rows alone, never conflated."""
    a, b = read_bf16(native, VOCABULARY), read_bf16(reference, VOCABULARY)
    result = compare(a, b)
    result["text_suffix"] = compare(a[SUFFIX_START:], b[SUFFIX_START:], SUFFIX_START)
    result.update({"token_fixture_sha256": file_sha256(tokens), "native_logits_sha256": file_sha256(native),
                   "reference_logits_sha256": file_sha256(reference)})
    return result


def measure(bits: int, args, tokens: pathlib.Path, shared: dict) -> dict:
    model = args.model_root / f"LFM3-VL-2B-MLX-{bits}bit"
    work = args.work / f"{bits}bit"
    work.mkdir(parents=True, exist_ok=True)
    native_rows, patches_f32 = work / "native-image.bf16", work / "native-image.bf16.patches.f32"
    run([args.bin_dir / "tf-lfm3-vision", model, "--image", fixture.PNG, native_rows])
    patches = np.fromfile(patches_f32, dtype="<f4")
    if patches.size != 4 * fixture.IMAGE_ROWS * 1536:
        raise ValueError(f"expected {4 * fixture.IMAGE_ROWS} patches of 1536 values, got {patches.size}")
    patches_npy = work / "patches.npy"
    np.save(patches_npy, patches.reshape(-1, 1536))
    identity = vision_weight_identity(model)
    shared_with = next((w for w, other in shared.items() if other["identity"] == identity), None)
    if shared_with is None:
        fp64_rows = work / "fp64-image.bf16"
        run([sys.executable, "tools/lfm3/vision_fp64_reference.py", "--model-dir", model, "--patches", patches_npy,
             "--grid-h", GRID[0], "--grid-w", GRID[1], "--output", fp64_rows])
    else:
        if file_sha256(patches_f32) != shared[shared_with]["patches_sha256"]:
            raise ValueError("equal vision weights but different patches")
        fp64_rows = shared[shared_with]["fp64_rows"]
    shared.setdefault(bits, {"identity": identity, "patches_sha256": file_sha256(patches_f32), "fp64_rows": fp64_rows})
    native_npy, fp64_npy = work / "native-rows.npy", work / "fp64-rows.npy"
    save_rows(native_npy, native_rows)
    save_rows(fp64_npy, fp64_rows)
    forward = [args.bin_dir / "tf-lfm3-forward", model, tokens]
    run(forward + [work / "native-on-native.bf16", "images", native_rows, IMAGE_START])
    run(forward + [work / "native-on-fp64.bf16", "images", fp64_rows, IMAGE_START])
    reference = [sys.executable, "tools/lfm3/fp64_reference.py", "--model-dir", model, "--tokens", tokens, "--image-start", IMAGE_START]
    run(reference + ["--image-rows", native_npy, "--output", work / "fp64-on-native.bf16"])
    run(reference + ["--image-rows", fp64_npy, "--output", work / "fp64-on-fp64.bf16"])
    result = {
        "checkpoint": model.name, "model_safetensors_sha256": file_sha256(model / "model.safetensors"),
        "vision_weight_identity": identity, "vision_fp64_rows_shared_with_width": shared_with,
        "patches_sha256": file_sha256(patches_f32), "native_projected_rows_sha256": file_sha256(native_rows),
        "fp64_projected_rows_sha256": file_sha256(fp64_rows),
        "vision": vision_stats(words(native_rows), words(fp64_rows)),
        "decoder": {
            "same_native_projected_rows": decoder(work / "native-on-native.bf16", work / "fp64-on-native.bf16", tokens),
            "same_fp64_projected_rows": decoder(work / "native-on-fp64.bf16", work / "fp64-on-fp64.bf16", tokens),
            "end_to_end_own_vision_rows": decoder(work / "native-on-native.bf16", work / "fp64-on-fp64.bf16", tokens),
        },
    }
    if "fp32" in args.stages:
        result["independent_fp32"] = fp32_stage(model, work, tokens, patches_npy, work / "native-on-native.bf16",
                                                work / "fp64-on-fp64.bf16")
    if "rounded" in args.stages:
        result["rounded_replay"] = rounded_stage(model, work, tokens, patches_npy, native_rows, work / "native-on-native.bf16")
    return result


def fp32_stage(model, work, tokens, patches, native_logits, fp64_logits) -> dict:
    """NumPy FP32 graph with and without the BF16 boundary sites, each from its own tower rows, against native and FP64."""
    result = {}
    for mode in ("fp32", "fp32-bf16"):
        out = work / f"independent-{mode}.npy"
        run([sys.executable, "tools/lfm3/fp32_graph_reference.py", "--model-dir", model, "--tokens", tokens, "--patches", patches,
             "--grid-h", GRID[0], "--grid-w", GRID[1], "--image-start", IMAGE_START, "--boundaries", mode, "--output", out],
            THREADS)
        logits = np.load(out, allow_pickle=False).astype(np.float32)
        view = {}
        for name, other in (("native", native_logits), ("ideal_fp64", fp64_logits)):
            reference = read_bf16(other, VOCABULARY)
            view[name] = {"all90": compare(reference, logits), "suffix16": compare(reference[SUFFIX_START:], logits[SUFFIX_START:], SUFFIX_START)}
        result[mode] = {"logits_sha256": file_sha256(out), "against": view}
    result["numpy"] = np.__version__
    return result


def rounded_stage(model, work, tokens, patches, native_rows, native_logits) -> dict:
    """The CPU replay of production's FP32 sites with Metal scalar elementary functions: how many BF16 words still differ."""
    library = work / "row-replay.so"
    run(["cc", "-std=c11", "-O3", "-ffp-contract=off", "-shared", "-fPIC", "tools/lfm3/row_projection_replay.c", "-o", library])
    env = {"LFM3_ROW_REPLAY_LIBRARY": str(library)}
    rows, rows_npy, logits = work / "rounded-image.bf16", work / "rounded-image.npy", work / "rounded-image-text.bf16"
    run([sys.executable, "tools/lfm3/vision_fp64_reference.py", "--model-dir", model, "--patches", patches, "--grid-h", GRID[0],
         "--grid-w", GRID[1], "--output", rows, "--dense-rounding", "metal-row", "--operator-rounding", "metal-fp32",
         "--elementary-backend", "metal"], env)
    save_rows(rows_npy, rows)
    run([sys.executable, "tools/lfm3/fp64_reference.py", "--model-dir", model, "--tokens", tokens, "--image-rows", rows_npy,
         "--image-start", IMAGE_START, "--output", logits, "--projection-rounding", "metal-row", "--norm-conv-rounding", "metal",
         "--attention-rounding", "metal-fp32", "--elementary-backend", "metal"], env)
    native, replay = words(native_rows), words(rows)
    a, b = words(native_logits), words(logits)
    return {"projected_image": {"values": int(native.size), "different_bf16": int((native != replay).sum()), "sha256": file_sha256(rows)},
            "decoder_logits": {"values": int(a.size), "different_bf16": int((a != b).sum()), "sha256": file_sha256(logits),
                               "suffix_top1_differences": compare(read_bf16(native_logits, VOCABULARY)[SUFFIX_START:],
                                                                   read_bf16(logits, VOCABULARY)[SUFFIX_START:], SUFFIX_START)["top1_differences"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=pathlib.Path, required=True, help="folder holding LFM3-VL-2B-MLX-{4,6,8}bit")
    parser.add_argument("--bin-dir", type=pathlib.Path, default=pathlib.Path("zig-out/bin"))
    parser.add_argument("--work", type=pathlib.Path, required=True, help="scratch folder for rows, logits and traces")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="the receipt to write")
    parser.add_argument("--widths", type=int, nargs="+", choices=(4, 6, 8), default=[4, 6, 8])
    parser.add_argument("--stages", nargs="*", choices=("fp32", "rounded"), default=[],
                        help="optional extra comparisons beyond the ideal FP64 one")
    args = parser.parse_args()
    manifest = json.loads(fixture.MANIFEST.read_text())
    problems = fixture.check_png(manifest)
    for bits in args.widths:
        problems += fixture.check_tokens(manifest, args.model_root / f"LFM3-VL-2B-MLX-{bits}bit")
    if problems:
        print("\n".join(problems))
        return 1
    args.work.mkdir(parents=True, exist_ok=True)
    tokens = args.work / "image-tokens.npy"
    tokens.write_bytes(fixture.npy_u4(fixture.load_tokenizer(args.model_root / f"LFM3-VL-2B-MLX-{args.widths[0]}bit").prompt(fixture.ANSWER)))
    shared: dict = {}
    widths = {str(bits): measure(bits, args, tokens, shared) for bits in args.widths}
    receipt = {
        "scope": ("Synthetic-image replacement for the photograph-era numerical receipts: the same native tower, decoder and "
                  "independent references on a generated image. Nothing here is transferred from, or a relabelling of, a "
                  "photograph receipt, whose values stay in their own files."),
        "image": {k: manifest["image"][k] for k in ("generator", "png_file", "png_sha256", "pixels_sha256", "width", "height")},
        "grid": list(GRID), "image_start": IMAGE_START, "image_rows": fixture.IMAGE_ROWS, "suffix_start": SUFFIX_START,
        "teacher_forced_answer": fixture.ANSWER, "token_fixture": {**manifest["token_fixture"], "file_sha256": file_sha256(tokens)},
        "source": {"commit": output(["git", "rev-parse", "HEAD"]), "dirty": bool(output(["git", "status", "--porcelain"])),
                   "tools": {n: file_sha256(args.bin_dir / n) for n in ("tf-lfm3-vision", "tf-lfm3-forward")},
                   "python": platform.python_version(), "numpy": np.__version__},
        "host": {"chip": output(["sysctl", "-n", "machdep.cpu.brand_string"]), "model": output(["sysctl", "-n", "hw.model"]),
                 "memory_bytes": int(output(["sysctl", "-n", "hw.memsize"])), "macos": output(["sw_vers", "-productVersion"]),
                 "macos_build": output(["sw_vers", "-buildVersion"]), "architecture": platform.machine()},
        "commands": COMMANDS, "widths": widths, "accepted_threshold": None, "fidelity_pass": None,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
