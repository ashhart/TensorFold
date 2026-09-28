"""Gemma 4 decode: one-row steps and verify windows timed, and the decode path scored against mlx_lm's forward."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy one-row steps from a chat prompt, each evaluated before the next "
                                 "(median ms after 20 warm-ups), then one forward of each window width from the same "
                                 "state; --nll-tokens scores the decode path teacher-forced against mlx_lm's forward.")
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--windows", default="2,4,8,16", help="window widths to time (comma-separated)")
    ap.add_argument("--nll-tokens", type=int, default=0, help="answer tokens to score (0: none)")
    ap.add_argument("--backend", choices=("lane", "rows", "mlx"), default=None,
                    help="the decode matmul (default: by GPU), or mlx: mlx_lm's own forward")
    ap.add_argument("--eval-file", default="/tmp/gemma4-eval-set.json")
    args = ap.parse_args()

    import mlx.core as mx

    from tensorfold.families.gemma4.model import load

    ours = args.backend != "mlx"
    model, tokenizer = load(Path(args.model), backend=args.backend if ours else None, check=ours)
    step = Stepper(model, ours)
    print(f"decode path: {model.decode.backend if ours else 'mlx_lm forward'}", flush=True)

    prompt = _chat(tokenizer, "Write a Python function that parses ISO-8601 durations and explain it.")
    cache = step.start(prompt)
    tok = mx.argmax(step.last[0, -1]).reshape(1, 1)
    mx.eval(tok)
    times, out = [], []
    for i in range(args.steps + 20):
        t0 = time.perf_counter()
        tok = mx.argmax(step(tok, cache)[0, -1]).reshape(1, 1)
        mx.eval(tok)
        if i >= 20:
            times.append(time.perf_counter() - t0)
        out.append(int(tok.item()))
    times.sort()
    med = times[len(times) // 2]
    print(f"decode: median {med * 1e3:.2f} ms/step = {1 / med:.1f} tok/s  (p10 {times[len(times) // 10] * 1e3:.2f}, "
          f"p90 {times[len(times) * 9 // 10] * 1e3:.2f})", flush=True)
    print("greedy head:", repr(tokenizer.decode(out[:40])), flush=True)
    if ours:
        from tensorfold.engine.lane_engine import LaneEngine

        for width in (int(w) for w in args.windows.split(",") if w):
            best = float("inf")
            for _ in range(5):
                work = LaneEngine.copy_single_cache(cache)
                t0 = time.perf_counter()
                mx.eval(step(mx.array([out[:width]], dtype=mx.uint32), work))
                best = min(best, time.perf_counter() - t0)
            print(f"window of {width} rows: {best * 1e3:.2f} ms", flush=True)

    if args.nll_tokens:
        quality(step, tokenizer, args.nll_tokens, Path(args.eval_file))
    return 0


class Stepper:
    """Prompt then decode rows through the family's decode path, or through mlx_lm's forward (``ours`` False)."""

    def __init__(self, model, ours: bool) -> None:
        self.model, self.ours, self.last = model, ours, None

    def start(self, prompt: list[int]):
        import mlx.core as mx

        cache = self.model.make_cache() if self.ours else self.model.text.make_cache()
        hidden = self.model.prefill(mx.array([prompt]), cache) if self.ours else None
        self.last = self.model.head(hidden[:, -1:]) if self.ours else self.model.text(mx.array([prompt]), cache=cache)
        return cache

    def __call__(self, tokens, cache):
        return self.model(tokens, cache) if self.ours else self.model.text(tokens, cache=cache)


def _chat(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def eval_set(model, tokenizer, total: int, path: Path) -> list[tuple[list[int], list[int]]]:
    """The model's own sampled answers to fixed prompts (mlx_lm, temperature 1, seed 0), kept in ``path``."""

    import json

    import mlx.core as mx

    if path.is_file():
        return [tuple(pair) for pair in json.loads(path.read_text())]
    mx.random.seed(0)
    out, have = [], 0
    for prompt in PROMPTS:
        ids = _chat(tokenizer, prompt)
        cache = model.model.make_cache()
        lg = model.model(mx.array([ids]), cache=cache)[0, -1]
        answer: list[int] = []
        while len(answer) < total // len(PROMPTS):
            t = int(mx.random.categorical(lg.astype(mx.float32)).item())
            answer.append(t)
            if t in (1, 106):                      # <eos>, <turn|>
                break
            lg = model.model(mx.array([[t]]), cache=cache)[0, -1]
        out.append((ids, answer))
        have += len(answer)
    path.write_text(json.dumps(out))
    return out


PROMPTS = (
    "Write a thread-safe LRU cache in Python with unit tests.",
    "Explain how a sliding-window rate limiter works and when to prefer a token bucket.",
    "Compute the expected queue length of an M/M/4 queue at 500 requests/s with 3 ms mean service time.",
    "Write a shell pipeline that lists the five largest files in a folder as a Markdown table.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Go HTTP handler that streams server-sent events and explain its error handling.",
    "Explain the CAP theorem with a concrete example for each trade-off.",
    "Translate this to French and explain two idioms: 'It is raining cats and dogs; break a leg.'",
    "Write a SQL query that finds the top three customers by revenue per month, with window functions.",
    "Describe how photosynthesis works to a ten-year-old.",
    "Review this Python for bugs: def avg(xs): return sum(xs) / len(xs) if xs else 0 / 0",
)


def quality(step: Stepper, tokenizer, total: int, path: Path) -> None:
    """Teacher-forced answers through the decode path against mlx_lm's whole forward: NLL, KL, argmax agreement."""

    import mlx.core as mx

    model = step.model
    pairs = eval_set(model, tokenizer, total, path)
    ref_nll, dec_nll, kls, agree, n = 0.0, 0.0, 0.0, 0.0, 0
    for prompt, answer in pairs:
        seq = prompt + answer
        k = len(prompt)
        ref = model.model(mx.array([seq]))[0, k - 1:-1].astype(mx.float32)   # predicts answer[0:]
        ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
        cache = step.start(prompt)
        rows = [step.last[0, -1]]
        for t in answer[:-1]:
            rows.append(step(mx.array([[t]]), cache)[0, -1])
            if len(rows) % 64 == 0:
                mx.eval(rows[-64:])
        dec = mx.stack(rows).astype(mx.float32)
        dec = dec - mx.logsumexp(dec, axis=-1, keepdims=True)
        tgt = mx.array(answer)[:, None]
        ref_nll += -mx.take_along_axis(ref, tgt, axis=-1).sum().item()
        dec_nll += -mx.take_along_axis(dec, tgt, axis=-1).sum().item()
        kls += (mx.exp(ref) * (ref - dec)).sum().item()
        agree += (mx.argmax(ref, axis=-1) == mx.argmax(dec, axis=-1)).sum().item()
        n += len(answer)
    print(f"quality over {n} answer tokens ({len(pairs)} prompts): nll mlx_lm forward {ref_nll / n:.4f}, "
          f"decode path {dec_nll / n:.4f} (diff {(dec_nll - ref_nll) / n:+.4f} nats); "
          f"KL(ref||decode) {kls / n:.5f} nats/token; argmax agreement {agree / n * 100:.2f}%", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
