"""Compare buffered and concatenated caches through actual native CLI decoding.

Runs checkpoints sequentially. Fixed draft depths keep measured calibration costs
from changing the schedule being compared. Full logits and cache contents are
checked separately by test-kv-buffers and test-cache-stress.
"""
import argparse
import json
from pathlib import Path
import subprocess


MODELS = {
    "qwen": "Qwen3.8-27B-MLX-4bit",
    "nemotron": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",
    "flash": "Qwen3.8-Flash-Next-MLX-4bit-MTP",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("build/models"))
    parser.add_argument("--output", type=Path, default=Path("build/native-checks/kv-runtime"))
    parser.add_argument("--family", choices=MODELS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checked = 0
    for family, name in MODELS.items():
        if args.family and family != args.family:
            continue
        serial_options = [] if family == "qwen" else ["--no-drafts"]
        modes = [("serial-sync", serial_options + ["--no-serial-pipeline"])]
        if family != "flash":
            modes.append(("serial-pipeline", serial_options))
        if family == "qwen":
            modes.append(("dflash", ["--drafter", str(args.model_root / "Qwen3.8-27B-DFlash2")]))
        else:
            for depth in (1, 3, 15):
                modes.append((f"mtp-{depth}", ["--mtp-drafts", str(depth), "--fixed-drafts"]))
            modes.append(("mtp-15-late", ["--mtp-drafts", "15", "--fixed-drafts", "--no-early-mtp"]))
        for backend in (["tensor"] if family == "flash" else ["tensor", "simd"]):
            for mode, options in modes:
                reference = None
                for buffered in (False, True):
                    label = f"{family}-{backend}-{mode}-{'buffered' if buffered else 'concat'}"
                    report = args.output / f"{label}.json"
                    command = [str(args.binary.resolve()), "run", str(args.model_root / name),
                               "--prompt", "Write a short Python function that computes the Fibonacci sequence.",
                               "--no-copy", "--metal-sampling", "--temperature", "0.7", "--seed", "5678",
                               "--top-k", "12", "--top-p", "0.8", "--max-tokens", "32",
                               "--report", str(report), *options]
                    if backend == "simd":
                        command.append("--metal-simd")
                    if not buffered:
                        command.append("--no-kv-buffers")
                    with (args.output / f"{label}.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                    result = json.loads(report.read_text())
                    assert result["kv_buffers"] is buffered, label
                    assert len(result["tokens"]) == 32 and result["rounds"] > 0, label
                    assert result["context_copy"] is False, label
                    if reference is None:
                        reference = result
                    else:
                        for field in ("prompt_tokens", "tokens", "token_sha256", "rounds", "accepted_drafts",
                                      "serial_pipeline", "queued_serial_steps"):
                            assert result[field] == reference[field], (label, field)
                        if family != "qwen":
                            for field in ("proposal_sha256", "gpu_handoff_rounds", "early_mtp"):
                                assert result[field] == reference[field], (label, field)
                        checked += 1
                        print(f"PASS {label}: every target ID and scheduling counter matches concatenation", flush=True)
    print(f"PASS: {checked} buffered/concatenated CLI comparisons", flush=True)


if __name__ == "__main__":
    main()
