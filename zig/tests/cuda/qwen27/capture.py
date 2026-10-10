#!/usr/bin/env python3
"""The 27B's Python CUDA engine as the Zig engine's oracle: Triton launches, per-layer dumps, tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TEXTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "facts": "Explain how a refrigerator moves heat out of its cabinet, step by step.",
}
LONG_BODY = 5000          # tokens of module source in the long prompt: two 4096-row prompt chunks
TEACHER = 32              # teacher-forced decode steps (one row each, from an empty state)
DUMP_STEPS = (0, 1, TEACHER - 1)


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompts(model: Path) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    import textwrap

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    out = {name: tok.encode(chat(text), add_special_tokens=False).ids for name, text in TEXTS.items()}
    words = Path(textwrap.__file__).read_text()
    body = tok.encode(words * (1 + LONG_BODY // max(1, len(words) // 3)), add_special_tokens=False).ids[:LONG_BODY]
    head = tok.encode("<|im_start|>user\nReview this module and list its public functions:\n\n",
                      add_special_tokens=False).ids
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    out["long"] = head + body + tail
    return out


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def digest(t) -> str:
    import torch

    raw = t.detach().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class Dumps:
    """Raw little-endian tensors by name under the current directory; None turns dumping off."""

    def __init__(self) -> None:
        self.dir: Path | None = None
        self.i = 0

    def at(self, d: Path | None) -> None:
        self.dir, self.i = d, 0
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, t) -> None:
        import torch

        if self.dir is None:
            return
        torch.cuda.synchronize()
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tofile(self.dir / f"{name}.bin")


def hook(dumps: Dumps) -> None:
    """Every add_rmsnorm call's (h, y, xs), in call order: two a layer, then the final norm."""

    from tensorfold.families.qwen3_5.cuda import glue

    norm = glue.add_rmsnorm

    def traced(x, r, w, eps):
        h, y, xs = norm(x, r, w, eps)
        if dumps.dir is not None:
            i = dumps.i
            dumps.i += 1
            dumps.save(f"{i:03d}_h", h)
            dumps.save(f"{i:03d}_y", y)
            dumps.save(f"{i:03d}_xs", xs)
        return h, y, xs

    glue.add_rmsnorm = traced


def specialization() -> dict:
    """Each launched JIT function's parameter names and which ones Triton never specializes."""

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
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="", help="folder of triton_aot_manifest.py: record every Triton launch")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--context", type=int, default=16384)
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
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    dumps = Dumps()
    hook(dumps)
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    t0 = time.perf_counter()
    with phase("startup"):
        eng = Qwen27Engine(model, None, streams=1, context=a.context, context_explicit=True)
    if rec is not None:
        from tensorfold.cuda.kernels import gdn, prefill_attention, qmm

        rec.wrap(qmm._ext(), ("qmm", "qmm_group", "qmm_prefill"), "qmm")
        rec.wrap(gdn._ext(), ("tree", "replay", "prefill"), "gdn")
        rec.wrap(prefill_attention._ext(), ("prefill_attention",), "prefill_attention")
    info = {"startup_s": round(time.perf_counter() - t0, 2), "context": eng.context_window, "eos": list(eng.eos),
            "prompt_rows": eng.w.prompt_rows, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(info), flush=True)

    results: dict[str, dict] = {}
    for name, prompt in ids.items():
        runs = []
        for _ in range(a.repeats):
            toks: list[int] = []
            with phase(f"serial-{name}"):
                torch.cuda.synchronize()
                start = time.perf_counter()
                stats = eng.generate(prompt, a.max_tokens, None, lambda new: toks.extend(map(int, new)) and False,
                                     draft=False, stop_eos=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - start
            runs.append({"tokens": toks, "sha": sha12(toks), "wall_s": round(wall, 4), **stats})
        last = runs[-1]
        same = all(r["tokens"] == last["tokens"] for r in runs)
        step = last.get("decode_s", 0.0) / max(1, len(last["tokens"]) - 1) * 1e3
        results[name] = {**last, "repeats_same": same, "ms_per_token": round(step, 4)}
        print(f"serial {name}: {len(last['tokens'])} tokens sha {last['sha']} prefill {last.get('prefill_s'):.3f}s "
              f"{step:.3f} ms/token same {same}", flush=True)

    from tensorfold.families.qwen3_5.cuda.decode import prefill
    from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward

    w = eng.w
    tf = ids["code"][:TEACHER]
    sampled, logit_sha = [], []
    st = State(w)
    with phase("teacher", detail=True):
        for i, t in enumerate(tf):
            dumps.at(out / "teacher" / f"step{i:03d}" if i in DUMP_STEPS else None)
            logits, record = tree_forward(w, torch.tensor([t], dtype=torch.int32, device="cuda"), [-1], st)
            dumps.save("logits", logits)
            sampled.append(int(logits.argmax(dim=-1)[0]))
            logit_sha.append(digest(logits))
            commit(st, record, [0])
    dumps.at(None)
    pre_out = {}
    for name in ("code", "long"):
        dumps.at(out / "prefill" / name if name == "code" else None)
        with phase(f"prefill-{name}", detail=name == "code"):
            _, pending = prefill(w, ids[name], None, limit=a.context)
        dumps.at(None)
        pre_out[name] = {"pending": pending}
    (out / "teacher.json").write_text(json.dumps({"tokens": tf, "sampled": sampled, "logits_sha256": logit_sha,
                                                  "dump_steps": list(DUMP_STEPS), "prefill": pre_out}) + "\n")
    (out / "results.json").write_text(json.dumps({"info": info, "results": results}, indent=1) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
