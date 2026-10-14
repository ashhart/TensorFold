"""Gemma 4 dense token oracle (gilby's gemma4-dense Python engine, lane kernels) and the native engine's checks."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

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


def environment() -> dict:
    import mlx.core as mx
    import mlx_lm
    import tensorfold

    return {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "tensorfold": tensorfold.__version__,
            "device": mx.metal.device_info().get("device_name", ""), "macos": platform.mac_ver()[0]}


def chat_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False))


def greedy(family, ids: list[int], count: int, prompt: str) -> tuple[list[int], list[dict]]:
    """`count` greedy tokens after `ids`; the prompt through mlx_lm's forward ("mlx") or one decode row a token."""
    import mlx.core as mx

    from qwen27_margins import capture

    cache = family.make_cache()
    if prompt == "mlx":
        hidden = family.prefill(mx.array([ids]), cache)[:, -1:]
    else:
        for token in ids[:-1]:
            mx.eval(family.hidden(mx.array([token]), cache))
        hidden = family.hidden(mx.array([ids[-1]]), cache)
    tokens, top2 = [], []
    for step in range(count):
        logits = family.head(hidden)
        mx.eval(logits)
        top2.extend(capture(logits, [len(ids) + step]))
        token = top2[-1]["ids"][0]
        tokens.append(token)
        hidden = family.hidden(mx.array([token]), cache)
    return tokens, top2


def oracle(args: argparse.Namespace) -> None:
    import tensorfold

    if tensorfold.__version__ != "0.6.5":
        raise SystemExit(f"Need gilby's gemma4-dense branch (tensorfold 0.6.5), got {tensorfold.__version__}")
    from tensorfold.families.gemma4.model import load

    family, tokenizer = load(args.model, backend="lane", check=False)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    prompts = json.loads(args.ids.read_text()) if args.ids else {n: chat_ids(tokenizer, t) for n, t in PROMPTS.items()}
    results = {}
    for name, ids in prompts.items():
        (out / f"{name}.ids.json").write_text(json.dumps(ids) + "\n")
        started = time.perf_counter()
        tokens, top2 = greedy(family, ids, args.tokens, args.prompt)
        results[name] = {"tokens": tokens, "elapsed_s": round(time.perf_counter() - started, 3)}
        (out / f"{name}.reference.json").write_text(json.dumps(tokens) + "\n")
        (out / f"{name}.top2.json").write_text(json.dumps(top2) + "\n")
        print(f"{name}: {len(ids)} prompt tokens, {tokenizer.decode(tokens)[:80]!r}", flush=True)
    receipt = {"environment": environment(), "model": str(args.model), "prompt_path": args.prompt,
               "decode": family.decode.backend, "kinds": sorted({k for p in family.decode.qkv for k in p.kinds}),
               "tokens": args.tokens, "results": results}
    (out / "oracle.json").write_text(json.dumps(receipt, indent=1) + "\n")


def check(args: argparse.Namespace) -> None:
    """Compare a native run's tokens and top-two margins with the oracle's, prompt by prompt."""
    from qwen27_margins import verdict

    oracle_dir, native_dir = args.oracle, args.native
    report, failed = {}, False
    for ref in sorted(oracle_dir.glob("*.reference.json")):
        name = ref.name.removesuffix(".reference.json")
        want = json.loads(ref.read_text())
        got = json.loads((native_dir / f"{name}.tokens.json").read_text())
        at = first_difference(want, got)
        entry = {"equal": at is None, "first_difference": at}
        if at is not None and at < min(len(want), len(got)):
            python = json.loads((oracle_dir / f"{name}.top2.json").read_text())[at]
            native = json.loads((native_dir / f"{name}.top2.json").read_text())[at]
            entry["margins"] = verdict(native, python, want[at], got[at])
        failed |= at is not None
        report[name] = entry
        print(f"{name}: {'equal' if at is None else f'differs at {at}'}", flush=True)
    (native_dir / "token-gate.json").write_text(json.dumps(report, indent=1) + "\n")
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--model", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--ids", type=Path, help="JSON {name: [ids]} in place of the four chat prompts")
    o.add_argument("--tokens", type=int, default=128)
    o.add_argument("--prompt", choices=("decode", "mlx"), default="decode")
    c = sub.add_parser("check")
    c.add_argument("--oracle", type=Path, required=True)
    c.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    {"oracle": oracle, "check": check}[args.mode](args)


if __name__ == "__main__":
    main()
