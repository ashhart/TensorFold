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
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)  # transformers 5 returns an encoding


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
        if step == 0:
            family.last_row = logits.reshape(-1).astype(mx.float32)
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
        import numpy as np
        np.asarray(family.last_row).astype(np.float32).tofile(out / f"{name}.last.f32")
        print(f"{name}: {len(ids)} prompt tokens, {tokenizer.decode(tokens)[:80]!r}", flush=True)
    receipt = {"environment": environment(), "model": str(args.model), "prompt_path": args.prompt,
               "decode": family.decode.backend, "kinds": sorted({k for p in family.decode.qkv for k in p.kinds}),
               "tokens": args.tokens, "results": results}
    (out / "oracle.json").write_text(json.dumps(receipt, indent=1) + "\n")


def prompts(args: argparse.Namespace) -> None:
    """Long chat prompts of about the requested sizes from the repo's own text, written as {name: ids} JSON."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    root = Path(__file__).resolve().parents[2]
    text = "\n\n".join((root / f).read_text() for f in ("CHANGELOG.md", "README.md", "CONTRIBUTING.md", "docs/api.md"))
    words = tokenizer.encode(text, add_special_tokens=False)
    out = {}
    for size in args.sizes:
        body = tokenizer.decode(words[:size])
        ask = "\n\nIn one paragraph, what does the text above say about prompt caching?"
        out[f"long{size // 1024}k"] = chat_ids(tokenizer, body + ask)
    args.out.write_text(json.dumps(out) + "\n")
    print({k: len(v) for k, v in out.items()})


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


def post(url: str, path: str, body: dict) -> dict:
    from urllib.request import Request, urlopen

    request = Request(url.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def served(args: argparse.Namespace) -> None:
    """The served model's greedy replies equal the oracle's, alone and four at once; thinking and tool calls parse."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    names = sorted(p.name.removesuffix(".reference.json") for p in args.oracle.glob("*.reference.json"))
    # the native CLI's tokens when given (its prompt path), else the oracle's
    src = args.native or args.oracle
    want = {n: json.loads((src / f"{n}.{'tokens' if args.native else 'reference'}.json").read_text()) for n in names}
    ids = {n: json.loads((args.oracle / f"{n}.ids.json").read_text()) for n in names}

    def one(name: str, draft: bool) -> tuple[str, bool]:
        got = post(args.url, "/v1/completions", {"model": "gemma", "prompt": ids[name], "max_tokens": len(want[name]),
                                                 "temperature": 0, "draft": draft, "ignore_eos": True})
        digest = hashlib.sha256(",".join(str(t) for t in want[name]).encode()).hexdigest()[:12]
        return got.get("tensorfold", {}).get("token_sha", ""), got.get("tensorfold", {}).get("token_sha") == digest

    failed = 0
    for draft in (False, True):
        for name in names:
            sha, ok = one(name, draft)
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} alone, drafts {'on ' if draft else 'off'}: {name} {sha}", flush=True)
    with ThreadPoolExecutor(len(names)) as pool:
        for name, (sha, ok) in zip(names, pool.map(lambda n: one(n, True), names)):
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} four at once: {name} {sha}", flush=True)
    think = post(args.url, "/v1/chat/completions", {"model": "gemma", "max_tokens": 400, "temperature": 0,
                                                    "chat_template_kwargs": {"enable_thinking": True},
                                                    "messages": [{"role": "user", "content": "What is 17 * 23? Think first."}]})
    msg = think["choices"][0]["message"]
    ok = bool(msg.get("reasoning_content")) and "391" in (msg.get("content") or "") and "<|channel>" not in (msg.get("content") or "")
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} thinking apart from the answer: {len(msg.get('reasoning_content') or '')} reasoning chars, answer {(msg.get('content') or '')[:60]!r}")
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    call = post(args.url, "/v1/chat/completions", {"model": "gemma", "max_tokens": 200, "temperature": 0, "tools": tools,
                                                   "messages": [{"role": "user", "content": "What's the weather in Lisbon?"}]})
    calls = call["choices"][0]["message"].get("tool_calls") or []
    ok = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and "Lisbon" in calls[0]["function"]["arguments"]
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} tool call: {[c['function'] for c in calls]}")
    print("PASS" if not failed else f"FAIL: {failed}")
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
    p = sub.add_parser("prompts")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768])
    v = sub.add_parser("served")
    v.add_argument("--url", default="http://127.0.0.1:8888")
    v.add_argument("--oracle", type=Path, required=True)
    v.add_argument("--native", type=Path, help="tf-gemma4-run's output (prompts in 512-row chunks): the reference")
    c = sub.add_parser("check")
    c.add_argument("--oracle", type=Path, required=True)
    c.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    {"oracle": oracle, "prompts": prompts, "served": served, "check": check}[args.mode](args)


if __name__ == "__main__":
    main()
