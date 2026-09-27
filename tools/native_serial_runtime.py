"""Compare synchronous and pipelined native CLI completions and output budgets.

Real checkpoint/cache arithmetic and synthetic EOS ownership checks run separately
in test-serial-pipeline. This checks the actual CLI integration and report counters.
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
    parser.add_argument("--output", type=Path, default=Path("build/native-checks/serial-runtime"))
    parser.add_argument("--family", choices=MODELS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checked = 0
    for family, name in MODELS.items():
        if args.family and family != args.family:
            continue
        for backend in (["tensor"] if family == "flash" else ["tensor", "simd"]):
            for temperature in (0, 0.7):
                for limit in (0, 1, 2, 32):
                    common = [str(args.binary.resolve()), "run", str(args.model_root / name),
                              "--prompt", "Write a short Python function that computes the Fibonacci sequence.",
                              "--metal-sampling", "--temperature", str(temperature), "--seed", "5678",
                              "--top-k", "12", "--top-p", "0.8", "--max-tokens", str(limit)]
                    if family != "qwen":
                        common.append("--no-drafts")
                    if family == "flash":
                        common.append("--resident-ple")
                    if backend == "simd":
                        common.append("--metal-simd")
                    reference = None
                    for pipeline in (False, True):
                        label = f"{family}-{backend}-{temperature}-{limit}-{'pipeline' if pipeline else 'sync'}"
                        report = args.output / f"{label}.json"
                        options = [] if pipeline else ["--no-serial-pipeline"]
                        with (args.output / f"{label}.log").open("w") as log:
                            subprocess.run(common + options + ["--report", str(report)], stdout=log,
                                           stderr=subprocess.STDOUT, check=True, timeout=300)
                        result = json.loads(report.read_text())
                        assert len(result["tokens"]) == limit, (label, "early EOS hides budget coverage")
                        assert result["serial_pipeline"] is pipeline, label
                        assert result["queued_serial_steps"] == (max(0, limit - 2) if pipeline else 0), label
                        assert result["rounds"] == max(0, limit - 1), label
                        if reference is None:
                            reference = result
                        else:
                            for field in ("prompt_tokens", "tokens", "token_sha256", "accepted_drafts", "rounds"):
                                assert result[field] == reference[field], (label, field)
                            if family != "qwen":
                                assert result["proposal_sha256"] == reference["proposal_sha256"], label
                            checked += 1
                            print(f"PASS {label}: {limit} IDs, rounds and queued-step counts exact", flush=True)
    print(f"PASS: {checked} synchronous/pipelined CLI comparisons", flush=True)


if __name__ == "__main__":
    main()
