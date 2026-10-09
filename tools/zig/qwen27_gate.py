"""Qwen27 token oracle (Python 0.6.6) and Zig comparison under the GPU wrapper; the two never coexist."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = {
    "copy": "Reply with exactly: The quick brown fox jumps over the lazy dog.",
    "arithmetic": "Compute 37 times 19 and explain the calculation briefly.",
    "story": "Write a short story about a lighthouse keeper finding a message in a bottle.",
    "code": "Write a Python function that returns the first n Fibonacci numbers.",
}


def first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def oracle(args: argparse.Namespace) -> None:
    import tensorfold

    if tensorfold.__version__ != "0.6.6":
        raise SystemExit(f"Need the Python engine 0.6.6, got {tensorfold.__version__}")
    import mlx.core as mx
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.qwen3_5 import engine_settings, load

    from qwen27_margins import capture

    class CapturingEngine(LaneEngine):
        def __init__(self, *values, **options):
            super().__init__(*values, **options)
            self.top2 = []

        def _draw(self, logits, sampling, positions):
            self.top2.extend(capture(logits, positions))
            return super()._draw(logits, sampling, positions)

    family, tokenizer = load(args.model, drafter="", vision=False)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    prompts = json.loads(args.ids.read_text()) if args.ids else {
        name: list(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )) for name, text in PROMPTS.items()
    }
    if any(not name or not name.replace("-", "").replace("_", "").isalnum() for name in prompts):
        raise SystemExit("Prompt names must contain only letters, digits, hyphens or underscores")
    if len(prompts) != 4:
        raise SystemExit("The first gate needs exactly four prompts")
    if any(not ids or len(ids) > 128 for ids in prompts.values()):
        raise SystemExit("Use four prompts of 1-128 tokens for a literal single-pass Zig chunk comparison")
    results = {}
    for name, ids in prompts.items():
        (out / f"{name}.ids.json").write_text(json.dumps(ids) + "\n")
        engine = CapturingEngine(family, **engine_settings(family))
        stream = LaneStream(name, ids, 256, eos_ids=frozenset(), drafts=False, sampling=None)
        engine.add_stream(stream)
        started = time.perf_counter()
        engine.run()
        elapsed = time.perf_counter() - started
        emitted = [int(t) for t in stream.emitted]
        if stream.error or len(emitted) != 256:
            raise RuntimeError(f"{name}: expected 256 tokens, got {len(emitted)}, error {stream.error}")
        (out / f"{name}.reference.json").write_text(json.dumps(emitted) + "\n")
        if len(engine.top2) != len(emitted):
            raise RuntimeError(f"{name}: missing generated-position margins")
        results[name] = {"tokens": emitted, "elapsed_s": elapsed, "timing_includes_margin_capture": True, "top2": engine.top2}
        (out / f"{name}.top2.json").write_text(json.dumps(engine.top2) + "\n")
        engine.release_rounds()
    # Warm once before every size's measurements. All requested lengths are exact token counts.
    paragraph = tokenizer.encode("A lighthouse keeper writes in the log as the boats cross the bay. ")
    if not paragraph:
        raise RuntimeError("empty benchmark tokenizer output")
    for size in (2048, 8192, 32768):
        ids = (paragraph * ((size + len(paragraph) - 1) // len(paragraph)))[:size]
        (out / f"prompt-{size}.ids.json").write_text(json.dumps(ids) + "\n")
        if args.no_speed:
            continue
        durations = []
        for repeat in range(4):
            engine = LaneEngine(family, **engine_settings(family))
            stream = LaneStream(f"bench-{size}-{repeat}", ids, 1, eos_ids=frozenset(), drafts=False, sampling=None)
            started = time.perf_counter()
            engine.add_stream(stream)
            engine.run()
            mx.synchronize()
            duration = time.perf_counter() - started
            if stream.error or len(stream.emitted) != 1:
                raise RuntimeError(f"benchmark {size} failed: {stream.error}")
            if repeat:
                durations.append(duration)
            engine.release_rounds()
            engine.reset()
            del engine, stream
            gc.collect()
            mx.clear_cache()
        results[f"prompt-{size}"] = {"seconds": durations, "best_tok_s": size / min(durations)}
    (out / "oracle.json").write_text(json.dumps({
        "version": tensorfold.__version__, "module": tensorfold.__file__,
        "drafts": False, "sampling": "greedy", "force_length": True,
        "prompts": list(prompts), "results": results,
    }, indent=2) + "\n")


def check(args: argparse.Namespace) -> None:
    record = json.loads((args.out / "oracle.json").read_text())
    if record["version"] != "0.6.6" or record["drafts"]:
        raise SystemExit("Wrong Python oracle")
    from qwen27_margins import verdict
    receipts = {}
    failed = False
    for name in record["prompts"]:
        command = [str(args.binary), "--run", "--model", str(args.model),
                   "--tokens", str(args.out / f"{name}.ids.json"), "--max-tokens", "256",
                   "--reference", str(args.out / f"{name}.reference.json"),
                   "--chunk", "128", "--compare-chunk", "16", "--force-length"]
        proc = subprocess.run(command, capture_output=True, text=True)
        (args.out / f"{name}.zig.log").write_text(proc.stdout + proc.stderr)
        try:
            got = json.loads(proc.stdout)
            difference = first_difference(got["tokens"], record["results"][name]["tokens"])
            if difference is not None:
                failed = True
                native = got.get("first_difference")
                python = record["results"][name].get("top2", [])
                diagnostic = verdict(native["top2"], python[difference],
                                     record["results"][name]["tokens"][difference], got["tokens"][difference]) \
                    if native and difference < len(python) else {"verdict": "margins missing"}
                got["diagnostic"] = {"first_mismatch": difference, **diagnostic}
                print(f"{name}: first mismatch {difference}, {diagnostic}")
            elif proc.returncode or not got["prompt_state_byte_equal"]:
                failed = True
                got["diagnostic"] = {"verdict": "native gate failed", "returncode": proc.returncode}
                print(f"{name}: native/chunk gate failed")
            else:
                got["diagnostic"] = {"verdict": "equal"}
                print(f"{name}: 256/256 equal, prompt bytes equal")
            receipts[name] = got
        except (ValueError, KeyError, IndexError) as error:
            failed = True
            receipts[name] = {"diagnostic": {"verdict": "native gate error", "error": str(error),
                                            "returncode": proc.returncode}}
            print(f"{name}: native gate error {error}")
    (args.out / "token-gate.json").write_text(json.dumps(receipts, indent=2) + "\n")
    if failed:
        raise RuntimeError("All four prompts attempted; strict token/chunk gate failed, see token-gate.json")
    if args.no_speed:
        return
    # The speed stage is deliberately after all four correctness gates.
    for size in (2048, 8192, 32768):
        samples = []
        for repeat in range(3):
            command = [str(args.binary), "--run", "--model", str(args.model),
                       "--tokens", str(args.out / f"prompt-{size}.ids.json"),
                       "--max-tokens", "0", "--chunk", "128"]
            proc = subprocess.run(command, capture_output=True, text=True)
            (args.out / f"prompt-{size}-{repeat}.zig.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"Zig benchmark {size} failed")
            samples.append(json.loads(proc.stdout))
        receipts[f"prompt-{size}"] = {
            "samples": samples,
            "best_tok_s": max(item["prompt_tok_s"] for item in samples),
            "python_best_tok_s": record["results"][f"prompt-{size}"]["best_tok_s"],
        }
    (args.out / "gate.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print("Qwen27 gate: 4 x 256 tokens equal Python 0.6.6; prompt state/KV/logits equal one pass; timings recorded")



def served(args: argparse.Namespace) -> None:
    record = json.loads((args.out / "oracle.json").read_text())
    native_file = args.out / "token-gate.json"
    native = json.loads(native_file.read_text()) if native_file.exists() else None
    receipts = {}
    for name in record["prompts"]:
        ids = json.loads((args.out / f"{name}.ids.json").read_text())
        wanted = native[name]["tokens"] if native is not None else record["results"][name]["tokens"]
        import hashlib
        digest = hashlib.sha256(",".join(str(token) for token in wanted).encode()).hexdigest()[:12]
        request = Request(args.url.rstrip("/") + "/v1/completions", data=json.dumps({
            "model": args.name, "prompt": ids, "max_tokens": 256,
            "temperature": 0, "draft": False, "ignore_eos": True,
        }).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            got = json.load(response)
        (args.out / f"{name}.served.json").write_text(json.dumps(got, indent=2) + "\n")
        runtime = got.get("tensorfold", {})
        if runtime.get("token_sha") != digest or got["usage"]["completion_tokens"] != 256:
            raise RuntimeError(f"{name}: served token SHA/length differs from selected native CLI receipt")
        receipts[name] = runtime
    (args.out / "served-gate.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print("Qwen27 served: four prompts x256 greedy token SHA equal the selected CLI reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("oracle", "check", "served"):
        command = sub.add_parser(mode)
        if mode != "served":
            command.add_argument("--model", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if mode == "oracle":
            command.add_argument("--no-speed", action="store_true")
            command.add_argument("--ids", type=Path, help="optional JSON mapping of exactly four names to ID arrays")
        elif mode == "check":
            command.add_argument("--binary", type=Path, required=True)
            command.add_argument("--no-speed", action="store_true")
        else:
            command.add_argument("--url", required=True)
            command.add_argument("--name", default="qwen27")
    args = parser.parse_args()
    if args.mode == "oracle":
        oracle(args)
    elif args.mode == "check":
        check(args)
    else:
        served(args)


if __name__ == "__main__":
    main()
