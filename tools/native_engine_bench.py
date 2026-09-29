"""Measure original LaneEngine or native CLI in sequential fresh processes.

This uses the original production loaders and engine, without correctness-oracle
substitutions. Reports retain actual draft availability and token IDs. Cold process
does not imply cold filesystem or Metal compiler caches. Run one benchmark at a time.
"""
import argparse
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time


MODELS = {
    "qwen": "Qwen3.8-27B-MLX-4bit",
    "nemotron": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",
    "flash": "Qwen3.8-Flash-Next-MLX-4bit-MTP",
}
PROMPT = "Write a short Python function that computes the Fibonacci sequence."
RUNTIME_ENV = ("TF_LANE_TILE", "TF_FLASH_MTP", "TF_FLASH_DRAFT_VOCAB", "TF_FLASH_QUEUED",
               "TF_NEMOTRON_FOLD_SHARED", "TF_NEMOTRON_LANE_QMM", "TF_NEMOTRON_ROWS",
               "TF_NEMOTRON_ROW_EXPERTS")


def python_worker(args):
    from native_runtime import require_mlx
    require_mlx()
    started = time.perf_counter()
    import mlx.core as mx
    from tensorfold import __version__
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream

    calibration = []

    def time_method(cls, name):
        original = getattr(cls, name)

        @wraps(original)
        def measured(*positional, **keywords):
            before = time.perf_counter()
            try:
                return original(*positional, **keywords)
            finally:
                calibration.append({"method": f"{cls.__name__}.{name}",
                                    "seconds": time.perf_counter() - before})

        setattr(cls, name, measured)

    model_dir = args.model_root / MODELS[args.family]
    load_started = time.perf_counter()
    if args.family == "qwen":
        from tensorfold.families import qwen3_5
        from tensorfold.families.qwen3_5.family import Qwen35Family
        time_method(Qwen35Family, "check_windows")
        drafter = str(args.model_root / "Qwen3.8-27B-DFlash2") if args.drafts else ""
        model, tokenizer = qwen3_5.load(model_dir, drafter=drafter)
        engine_options = qwen3_5.engine_settings(model)
    elif args.family == "nemotron":
        from tensorfold.families.nemotron_h import model as implementation
        time_method(implementation.NemotronH, "check_windows")
        time_method(implementation.NemotronH, "_time_mtp_step")
        model, tokenizer = implementation.load(model_dir, mtp_drafts=args.drafts)
        engine_options = {"max_rows": 16, "max_draft": 15}
    else:
        from tensorfold.families.qwen4_exp import runtime as implementation
        time_method(implementation.FlashNext, "check_windows")
        time_method(implementation.FlashNext, "_time_mtp_step")
        model, tokenizer = implementation.load(model_dir, drafts=args.drafts)
        engine_options = {"max_rows": 16, "max_draft": 15}
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started
    sampling = Sampling(args.seed, args.temperature, args.top_k, args.top_p) if args.temperature else None
    prompt = ([1000 + (i % 4) * 37 for i in range(args.prompt_tokens)] if args.prompt_tokens else
              tokenizer.encode(args.prompt, add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        eos = [tokenizer.eos_token_id]
    engine = LaneEngine(model, **engine_options)
    stream = LaneStream("benchmark", prompt, args.max_tokens,
                        eos_ids=frozenset(int(t) for t in eos if t is not None),
                        sampling=sampling, drafts=bool(args.drafts))
    startup_seconds = time.perf_counter() - started
    before = time.perf_counter()
    engine.add_stream(stream)
    # Preserve the production overlap: add_stream may queue the next serial pass.
    # An extra synchronization here would change the engine's scheduling.
    prefill_seconds = time.perf_counter() - before
    before = time.perf_counter()
    engine.run()
    mx.synchronize()
    decode_seconds = time.perf_counter() - before
    result = {
        "prompt_tokens": prompt, "tokens": stream.emitted,
        "text": tokenizer.decode(stream.emitted),
        "token_sha256": hashlib.sha256(struct.pack(f"<{len(stream.emitted)}I", *stream.emitted)).hexdigest(),
        "seed": args.seed, "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        "metal_sampling": args.family != "qwen",
        "context_copy": False, "startup_seconds": startup_seconds, "load_seconds": load_seconds,
        "calibration_seconds": sum(item["seconds"] for item in calibration),
        "calibration_calls": calibration, "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "engine_seconds": time.perf_counter() - started,
        "peak_mlx_bytes": mx.get_peak_memory(), "active_mlx_bytes": mx.get_active_memory(),
        "rounds": stream.rounds, "accepted_drafts": stream.accepted,
        "drafted": stream.drafted, "finish_reason": stream.finish_reason,
        "mtp_enabled": args.family != "qwen" and getattr(model, "mtp", None) is not None,
        "dflash_enabled": args.family == "qwen" and getattr(model, "head_drafts", None) is not None,
        "exact_width": getattr(model, "exact_width", None),
        "engine_options": engine_options, "engine_summary": engine.summary(),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "python_version": platform.python_version(),
        "tensorfold_version": __version__,
        "installed_tensorfold_version": importlib.metadata.version("tensorfold"),
        "prefill_method": "production family prefill",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("python", "zig"), required=True)
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--drafts", type=int, default=0, choices=(0, 3, 15))
    parser.add_argument("--resident-ple", action="store_true")
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/tensorfold"))
    parser.add_argument("--model-root", type=Path, default=Path("build/models"))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="Use a repeating four-token prompt of this length (0 uses --prompt)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.resident_ple and (args.engine != "zig" or args.family != "flash"):
        parser.error("--resident-ple applies only to native Flash")
    if args.family == "qwen" and args.drafts not in (0, 15):
        parser.error("Qwen uses the original 15-node DFlash2 configuration")
    if args.repetitions < 1 or args.max_tokens < 2:
        parser.error("positive repetitions and at least two output tokens required")
    if not 0 <= args.prompt_tokens <= 32768:
        parser.error("--prompt-tokens must be 0..32768")
    if args.worker:
        return python_worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for repetition in range(args.repetitions):
        report = args.output / f"run-{repetition}.json"
        log = args.output / f"run-{repetition}.log"
        shared = ["--prompt", args.prompt, "--max-tokens", str(args.max_tokens),
                  "--seed", str(args.seed), "--temperature", str(args.temperature),
                  "--top-k", str(args.top_k), "--top-p", str(args.top_p)]
        if args.engine == "python":
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--engine", "python",
                       "--family", args.family, "--drafts", str(args.drafts),
                       "--model-root", str(args.model_root), "--output", str(report),
                       "--prompt-tokens", str(args.prompt_tokens), *shared]
        else:
            command = [str(args.binary.resolve()), "run", str(args.model_root / MODELS[args.family]),
                       *shared, "--no-copy", "--warmup", "--report", str(report)]
            if args.prompt_tokens:
                command += ["--tokens", ",".join(str(1000 + (i % 4) * 37) for i in range(args.prompt_tokens))]
            # The production Qwen LaneEngine uses exact_sampling (CPU f64).
            # FamilyRounds uses the fp32 GPU sampler for Nemotron and Flash.
            if args.family != "qwen":
                command.append("--metal-sampling")
            if args.family == "qwen" and args.drafts:
                command += ["--drafter", str(args.model_root / "Qwen3.8-27B-DFlash2")]
            elif args.family != "qwen":
                command += ["--mtp-drafts", str(args.drafts)] if args.drafts else ["--no-drafts"]
            if args.resident_ple:
                command.append("--resident-ple")
        before = time.perf_counter()
        with log.open("w") as handle:
            try:
                process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "ok" if process.returncode == 0 else f"exit {process.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
        elapsed = time.perf_counter() - before
        result = json.loads(report.read_text()) if status == "ok" else {}
        result.update(engine=args.engine, family=args.family, requested_drafts=args.drafts,
                      repetition=repetition, process_seconds=elapsed, status=status, command=command,
                      platform=platform.platform(), environment={k: os.environ[k] for k in RUNTIME_ENV if k in os.environ})
        report.write_text(json.dumps(result, indent=2) + "\n")
        reports.append(result)
        (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"{args.engine}/{args.family}/{args.drafts} run {repetition}: {status}, {elapsed:.3f}s process", flush=True)
        if status != "ok":
            raise SystemExit(f"Benchmark failed; see {log}")
        if len(result["tokens"]) != args.max_tokens:
            raise SystemExit("Early EOS prevents the requested token-count comparison; retain report and choose another prompt")
        if repetition and result["tokens"] != reports[0]["tokens"]:
            raise SystemExit("Output changed between repetitions")


if __name__ == "__main__":
    main()
