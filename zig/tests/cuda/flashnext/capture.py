#!/usr/bin/env python3
"""Flash Next's Python CUDA engine (qwen4_exp, drafts off, greedy) as the Zig port's oracle: every Triton launch,
prompt ids, and two greedy references per prompt (the served path: chunked prefill then serial decode; the teacher
path: every prompt token as its own decode row, the way zig's flashnext-prompt runs it)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROMPTS = {
    "short": "What is the capital of France?",
    "mid": ("Explain in plain words why the sky looks blue during the day but turns red and orange at sunset. "
            "Mention the role of the atmosphere and the wavelength of light, and keep it under five sentences."),
}
TOP_STEPS = 4              # sampled steps whose five best logits the teacher paths keep
LONG_TOKENS = 600            # the long prompt: module source cut to make the whole prompt about this many tokens


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompts(model: Path) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    import textwrap

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids      # noqa: E731
    out = {name: enc(chat(text)) for name, text in PROMPTS.items()}
    head = enc("<|im_start|>user\nSummarize what this Python module does, then list its public functions:\n\n")
    tail = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    body = enc(Path(textwrap.__file__).read_text())[:LONG_TOKENS - len(head) - len(tail)]
    out["long"] = head + body + tail
    return out


def specialization() -> dict:
    """Each JIT function's parameter names and which ones Triton never specializes (aot_pack.py's --jit)."""

    from triton.runtime.jit import JITFunction
    import gc

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            out[name] = {"params": [p.name for p in fn.params],
                         "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                         "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the checkpoint's snapshot dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="", help="folder of triton_aot_manifest.py: record every Triton launch")
    ap.add_argument("--tokens", type=int, default=32, help="greedy tokens per prompt")
    ap.add_argument("--context", type=int, default=1536,
                    help="the cache window; 1025..1536 gives the port's NCH=3, QSA=0 attention specialization")
    ap.add_argument("--only", default="", help="comma list of prompt names (default all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()

    import torch
    from contextlib import nullcontext

    def phase(name: str, detail: bool = False):
        return rec.scope(name, detail) if rec is not None else nullcontext()

    model = Path(a.model)
    ids = prompts(model)
    if a.only:
        ids = {k: v for k, v in ids.items() if k in a.only.split(",")}
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    for name, p in ids.items():
        (out / f"{name}.ids").write_text(" ".join(map(str, p)) + "\n")

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.forward import commit

    t0 = time.perf_counter()
    with phase("startup"):
        eng = FlashNextEngine(model, depth=0, max_len=a.context, context_explicit=True, streams=1,
                              ple_on_ssd=True, kv_dtype="bf16", graphs=False)
    e = eng.e
    info = {"startup_s": round(time.perf_counter() - t0, 2), "max_len": eng.max_len, "prefill_rows": eng.prefill_rows,
            "nch": e.buf.attn.nch, "qsa": bool(e.buf.attn.qsa), "idw": e.buf.attn.idw, "eos": list(eng.eos),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "prompt_lengths": {k: len(v) for k, v in ids.items()}}
    print(json.dumps(info), flush=True)

    results: dict[str, dict] = {}
    from functools import partial
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    teach = partial(_teach, phase, results, a, commit, nullcontext)
    fp32 = None
    for name, prompt in ids.items():
        # the served path: chunked prefill, then one token a round (draft=False is FlashNextEngine._serial)
        toks: list[int] = []
        with phase(f"serial-{name}"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            stats = eng.generate(prompt, a.tokens, None, lambda new: toks.extend(map(int, new)) and False,
                                 draft=False, stop_eos=False)
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
        results[f"serial/{name}"] = {"tokens": toks, "wall_s": round(wall, 4), **stats}
        print(f"serial {name} ({len(prompt)} prompt): {toks} {stats}", flush=True)
        if fp32 is None:
            # teacher32: the decode-form experts (fp32 slot outputs, Buffers' default and the MTP head's), the
            # arithmetic zig's flashnext-prompt runs; the serial engine's buffers take the prompt-form experts
            fp32 = eng.serial.twin()
            fp32.buf = Buffers(eng.e.w, eng.e.rows, eng.e.capacity)

        for label, t in (("teacher", eng.serial), ("teacher32", fp32)):
            teach(label, name, prompt, t, toks)

    (out / "results.json").write_text(json.dumps({"info": info, "results": results}, indent=1) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    eng.close()
    print("done", flush=True)
    return 0


def _teach(phase, results, a, commit, nullcontext, label, name, prompt, t, served):
    """Every prompt token one decode row, as zig's flashnext-prompt feeds them, then greedy rows."""

    import torch

    t.reset()
    got: list[int] = []
    top: list = []
    steps = 0
    with phase(f"{label}-{name}"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        feed = list(prompt)
        dec_start = None
        while len(got) < a.tokens:
            tok = feed.pop(0) if feed else got[-1]
            with (phase(f"{label}-{name}-detail", detail=True) if name == "short" and steps < 2
                  else nullcontext()):
                logits = t.forward([tok])
                if not feed:
                    got.append(t.sample(logits[:1], [t.st.pos + 1], None)[0])
                    if len(got) <= TOP_STEPS:
                        v, i = logits[0].float().topk(5)
                        top.append([[int(x), round(float(y), 4)] for x, y in zip(i.tolist(), v.tolist())])
                    if dec_start is None:
                        torch.cuda.synchronize()
                        dec_start = time.perf_counter()
                commit(t.w, t.st, t.buf, 1, 1)
            steps += 1
        torch.cuda.synchronize()
        end = time.perf_counter()
    ms = (end - dec_start) / max(1, a.tokens - 1) * 1e3
    results[f"{label}/{name}"] = {"tokens": got, "wall_s": round(end - start, 4),
                                  "decode_ms_per_token": round(ms, 3), "top5": top}
    print(f"{label} {name}: {got} ({ms:.2f} ms/token) same as serial: {got == served}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
