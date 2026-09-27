"""Verify full/cut draft vocabularies and queued/unqueued MTP with real checkpoints.

Target IDs must match serial in every mode. Queuing must also preserve the complete
proposal stream and acceptance counts, so target rejection cannot conceal an error.
Runs one full model process at a time. This is correctness coverage, not a benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess


MODELS = {
    "nemotron": ("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", 131072, 32768),
    "flash": ("Qwen3.8-Flash-Next-MLX-4bit-MTP", 248320, 79592),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("build/models"))
    parser.add_argument("--output", type=Path, default=Path("build/native-checks/mtp-runtime"))
    parser.add_argument("--family", choices=MODELS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checked, queued_pairs = 0, 0

    def run(label, common, options):
        report = args.output / (label + ".json")
        command = [str(args.binary.resolve()), *common, *options, "--report", str(report)]
        with (args.output / (label + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        result = json.loads(report.read_text())
        assert result["rounds"] > 0, (label, "no decode rounds")
        return result

    for family, (model, full_size, cut_size) in MODELS.items():
        if args.family and family != args.family:
            continue
        for backend in (["tensor", "simd"] if family == "nemotron" else ["tensor"]):
            for sampler in ("greedy", "metal", "cpu"):
                count = 17 if sampler == "greedy" else 32
                prefix = f"{family}-{backend}-{sampler}"
                common = ["run", str(args.model_root / model), "--no-copy",
                          "--prompt", "Write a short Python function that computes the Fibonacci sequence.",
                          "--max-tokens", str(count), "--seed", "5678",
                          "--temperature", "0" if sampler == "greedy" else "0.7",
                          "--top-k", "12", "--top-p", "0.8"]
                if backend == "simd":
                    common.append("--metal-simd")
                if sampler != "cpu":
                    common.append("--metal-sampling")
                serial = run(prefix + "-serial", common, ["--no-drafts"])
                assert len(serial["tokens"]) == count, (prefix, "early EOS prevents the requested coverage")
                for budget in (1, 3, 15):
                    for reduced in (False, True):
                        vocabulary = "cut" if reduced else "full"
                        options = ["--mtp-drafts", str(budget)]
                        if not reduced:
                            options.append("--full-draft-vocab")
                        plain = None
                        for queued in ([False, True] if sampler != "cpu" else [False]):
                            label = f"{prefix}-{budget}-{vocabulary}-{'queued' if queued else 'host'}"
                            result = run(label, common, options + ([] if queued else ["--no-queued-drafts"]))
                            assert result["prompt_tokens"] == serial["prompt_tokens"], label
                            assert result["tokens"] == serial["tokens"], label
                            assert result["draft_vocab_size"] == (cut_size if reduced else full_size), label
                            assert result["queued_drafts"] == queued, label
                            assert result["context_copy"] is False, label
                            checked += 1
                            if queued:
                                for field in ("proposal_sha256", "accepted_drafts", "rounds"):
                                    assert result[field] == plain[field], (label, field)
                                queued_pairs += 1
                            else:
                                plain = result
                            print(f"PASS {label}: {len(result['tokens'])} target IDs"
                                  + (" and exact proposal stream" if queued else ""), flush=True)
    print(f"PASS: {checked} serial/MTP comparisons; {queued_pairs} queued proposal comparisons", flush=True)


if __name__ == "__main__":
    main()
