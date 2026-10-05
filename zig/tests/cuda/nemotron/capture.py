#!/usr/bin/env python3
"""Nemotron's Python CUDA engine as the Zig engine's oracle: Triton launches, weight digests, per-block dumps, tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SYSTEM = "You are a helpful assistant."
TEXTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "facts": "Explain how a refrigerator moves heat out of its cabinet, step by step.",
}
LONG_BODY = 2900          # tokens of module source in the long prompt: past one 2048-row prompt chunk
TEACHER = 48              # teacher-forced decode steps (one row each, from an empty state)
DUMP_STEPS = (0, 1, TEACHER - 1)


def chat(user: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>")


SECOND = {
    "math": "A train leaves at 9:40 and travels 212 km at 84 km/h. When does it arrive? Show the arithmetic.",
    "json": "Return a JSON object listing three European capitals with their population and one landmark each.",
    "translate": "Translate into French, then explain two choices you made: 'The meeting moved to Thursday afternoon.'",
}
HUGE_BODY = 5200          # tokens of module source in the second set's long prompt: three prompt chunks


def prompts(model: Path, second: bool = False) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    import json.decoder as decoder
    import textwrap

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    texts, source, size = (SECOND, decoder, HUGE_BODY) if second else (TEXTS, textwrap, LONG_BODY)
    out = {name: tok.encode(chat(text), add_special_tokens=False).ids for name, text in texts.items()}
    words = Path(source.__file__).read_text()
    body = tok.encode(words * (1 + size // max(1, len(words) // 3)), add_special_tokens=False).ids[:size]
    head = tok.encode(chat("Review this module and list its public functions:\n\n").split("<|im_end|>\n<|im_start|>a")[0],
                      add_special_tokens=False).ids
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think></think>", add_special_tokens=False).ids
    out["huge" if second else "long"] = head + body + tail
    return out


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def digest(t) -> str:
    import torch

    raw = t.detach().contiguous()
    if raw.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(raw.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class Dumps:
    """Raw little-endian tensors by name under the current directory; None turns dumping off."""

    def __init__(self) -> None:
        self.dir: Path | None = None
        self.i = 0
        self.blocks = True

    def at(self, d: Path | None, blocks: bool = True) -> None:
        self.dir, self.i, self.blocks = d, 0, blocks
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, t) -> None:
        import torch

        if self.dir is None:
            return
        torch.cuda.synchronize()
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tofile(self.dir / f"{name}.bin")


def hook(dumps: Dumps) -> None:
    """Every block norm (h, y, xs) and every sampler call's logits, in call order."""

    from tensorfold.families.nemotron_h.cuda import engine as E

    norm = E.Engine.norm

    def traced_norm(self, x, delta, weight):
        h, y, xs = norm(self, x, delta, weight)
        if dumps.dir is not None and dumps.blocks:
            i = dumps.i
            dumps.i += 1
            dumps.save(f"{i:03d}_h", h)
            dumps.save(f"{i:03d}_y", y)
            dumps.save(f"{i:03d}_xs", xs)
        return h, y, xs

    E.Engine.norm = traced_norm
    sample = E.S.sample

    def traced_sample(logits, meta, params, out, **kw):
        dumps.save("logits", logits)
        res = sample(logits, meta, params, out, **kw)
        dumps.save("sampled", out)
        return res

    E.S.sample = traced_sample


def weight_digests(eng) -> dict[str, str]:
    """sha256 of every device tensor the engine holds as weights, by a dotted name."""

    import torch

    out: dict[str, str] = {}

    def walk(prefix: str, v) -> None:
        if isinstance(v, torch.Tensor):
            out[prefix] = digest(v)
            out[prefix + ".shape"] = "x".join(map(str, v.shape)) + ":" + str(v.dtype).replace("torch.", "")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(f"{prefix}.{i}", x)
        elif hasattr(v, "__dataclass_fields__"):
            for name in v.__dataclass_fields__:
                walk(f"{prefix}.{name}" if prefix else name, getattr(v, name))

    w = eng.e.w
    walk("", w)
    if eng.mtp is not None:
        walk("draft_head", eng.mtp.head)
        out["draft_ids"] = digest(eng.mtp.id_map)
    return {k: v for k, v in out.items() if not k.startswith("extra")}


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
    ap.add_argument("--bench", action="store_true", help="tokens and timings only: no recorder, no dumps")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--context", type=int, default=16384, help="as serve sizes it without --context")
    ap.add_argument("--second", action="store_true", help="the second prompt set (a three-chunk prompt among them)")
    ap.add_argument("--sampling", default="", help="SEED,TEMPERATURE,TOP_K,TOP_P,MIN_P for every run (default greedy)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools and not a.bench:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()

    import torch
    from contextlib import nullcontext

    def phase(name: str, detail: bool = False):
        return rec.scope(name, detail) if rec is not None else nullcontext()

    model = Path(a.model)
    ids = prompts(model, a.second)
    sampling = None
    if a.sampling:
        from tensorfold.engine.exact_sampling import Sampling

        seed, temperature, top_k, top_p, min_p = a.sampling.split(",")
        sampling = Sampling(int(seed), float(temperature), int(top_k), float(top_p), float(min_p))
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    dumps = Dumps()
    if not a.bench:
        hook(dumps)
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine

    t0 = time.perf_counter()
    with phase("startup"):
        eng = NemotronEngine(model, context=a.context, context_explicit=False)
    if rec is not None:
        from tensorfold.cuda import experts
        from tensorfold.cuda.kernels import prefill_attention, qmm
        from tensorfold.families.nemotron_h.cuda import mamba

        rec.wrap(qmm._ext(), ("qmm", "qmm_group", "qmm_prefill"), "qmm")
        rec.wrap(experts._ext(), ("plan", "run", "prefill", "pack"), "experts")
        rec.wrap(prefill_attention._ext(), ("prefill_attention",), "prefill_attention")
        rec.wrap(mamba._ext(), ("scan_rows",), "scan_rows")
    costs = eng.rules[False].costs if eng.rules else None
    info = {"startup_s": round(time.perf_counter() - t0, 2), "max_len": eng.max_len, "drafts": eng.drafts,
            "max_rows": eng.e.max_rows, "prefill_rows": eng.e.prefill_rows, "eos": list(eng.eos),
            "verify_ms": list(costs.verify) if costs else None, "level_ms": costs.level if costs else None,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "sampling": a.sampling or None}
    print(json.dumps(info), flush=True)
    if not a.bench:
        (out / "weights.json").write_text(json.dumps(weight_digests(eng), indent=0, sort_keys=True) + "\n")

    results: dict[str, dict] = {}
    for draft in (False, True):
        label = "drafted" if draft else "serial"
        for name, prompt in ids.items():
            runs = []
            for r in range(a.repeats):
                toks: list[int] = []
                with phase(f"{label}-{name}"):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    stats = eng.generate(prompt, a.max_tokens, sampling,
                                         lambda new: toks.extend(map(int, new)) and False, draft=draft, stop_eos=True)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                runs.append({"tokens": toks, "sha": sha12(toks), "wall_s": round(wall, 4), **stats})
            last = runs[-1]
            same = all(r["tokens"] == last["tokens"] for r in runs)
            step = last.get("decode_s", 0.0) / max(1, len(last["tokens"]) - 1) * 1e3
            results[f"{label}/{name}"] = {**last, "repeats_same": same, "ms_per_token": round(step, 4)}
            print(f"{label} {name}: {len(last['tokens'])} tokens sha {last['sha']} decode {last.get('decode_s')}s "
                  f"prefill {last.get('prefill_s')}s {step:.3f} ms/token rounds {last.get('rounds')} "
                  f"accepted {last.get('accepted')} same {same}", flush=True)
    for name in ids:
        s, d = results[f"serial/{name}"], results[f"drafted/{name}"]
        print(f"drafted == serial {name}: {s['tokens'] == d['tokens']}", flush=True)

    if not a.bench:
        e = eng.e
        e.use_graphs = False
        short, longest = ("math", "huge") if a.second else ("code", "long")    # the set's teacher prompt, its longest
        tf = ids[short][:TEACHER]
        sampled, logit_sha = [], []
        e.reset()
        e.set_sampling(None)
        with phase("teacher", detail=True):
            for i, t in enumerate(tf):
                dumps.at(out / "teacher" / f"step{i:03d}" if i in DUMP_STEPS else None)
                e.forward([t])
                sampled.append(e.tokens()[0])
                logit_sha.append(digest(e.logits[0]))
                e.commit(1)
        dumps.at(None)
        e.use_graphs = True
        from tensorfold.families.nemotron_h.cuda.decode import prefill

        pre_out = {}
        for name in (short, longest):
            dumps.at(out / "prefill" / name, blocks=name == short)
            with phase(f"prefill-{name}", detail=name == short):
                pre = prefill(e, None, ids[name], None)
            dumps.save("p_hidden_last", pre.last_hidden)
            dumps.at(None)
            pre_out[name] = {"pending": pre.pending, "last_hidden": digest(pre.last_hidden)}
        (out / "teacher.json").write_text(json.dumps({"tokens": tf, "sampled": sampled, "logits_sha256": logit_sha,
                                                      "dump_steps": list(DUMP_STEPS), "prefill": pre_out}) + "\n")
    (out / ("bench.json" if a.bench else "results.json")).write_text(
        json.dumps({"info": info, "results": results}, indent=1) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
