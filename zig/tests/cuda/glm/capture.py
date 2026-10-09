#!/usr/bin/env python3
"""GLM-5.3-Flash's Python CUDA engine as the Zig engine's oracle: Triton + extension launches, weight digests, tokens.

Two ranks, two machines: start rank 1 first (`--rank 1 --master <rank 0's address>` on the peer),
then rank 0 (`--rank 0`); rank 1 blocks in `follow()` while rank 0 drives the prompt matrix. Both
ranks install the recorder, so each writes its own launches.json; pack them together with
`tools/zig/aot_pack.py` (--manifest/--cache repeatable pairs). The box script
`zig/tests/cuda/glm/box/capture.sh` orchestrates both ranks, the caches and the pack.

The peer's `follow()` raises (`DistNetworkError`, connection closed) when rank 0's process exits —
the pair's normal shutdown — so rank 1 catches it and still writes its launch record.

Scope: prompt matrix (serial/drafted x prompts x repeats, greedy), weight digests, per-rank launch
records. Teacher-forced per-step and prefill dumps are not in v1 (nemotron's inner-engine hooks do
not transfer to glm5_next); add them when a replay needs finer fixtures than end-to-end tokens.

Environment: the mounted tree needs both this repository's zig/ and the Python engine's src/
(the 1.0.0 split moved the Python line to the python-0.6 branch; PYTHONPATH=/tensorfold/src), the
pinned container with --network host (the NCCL bootstrap crosses machines), --gpus all. Checkpoint:
the EXL3 TR3-4bpw GLM-5.3-Flash (Brandon M. Music's, re-hosted by Mia-AiLab; ShapleyMCG 1.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

SYSTEM = "You are a helpful assistant."
PROMPTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a string is a palindrome, with a few tests.",
    "facts": "Explain how a refrigerator moves heat out of its inside, in three short paragraphs.",
}


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def digest(t) -> str:
    import torch

    raw = t.detach().contiguous()
    if raw.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(raw.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


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

    walk("", eng.__dict__)
    return {k: v for k, v in out.items() if not k.startswith("_")}


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


def prompt_ids(model: Path, tokenizer_cache: dict = {}) -> dict[str, list[int]]:
    """The chat-templated prompt ids: the checkpoint's chat_template.jinja rendered by jinja2 (as
    transformers would), encoded with the tokenizers library — no transformers needed."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    import datetime
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    tpl = env.from_string((model / "chat_template.jinja").read_text())
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))

    def render(messages: list[dict]) -> list[int]:
        text = tpl.render(messages=messages, add_generation_prompt=True,
                          bos_token="", eos_token="", pad_token="")
        return tok.encode(text, add_special_tokens=False).ids

    out = {name: render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
           for name, text in PROMPTS.items()}

    words = Path(__file__).read_text(errors="replace")          # a public source: this script itself
    body = (tok.encode(words, add_special_tokens=False).ids * (1 + 2900 // max(1, len(tok.encode(words, add_special_tokens=False).ids))))[:2900]
    out["long"] = render([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": "Review this module and list its public functions:\n\n"
                           + tok.decode(body)}])
    return out


def install_recorder(tools: str):
    sys.path.insert(0, tools)
    import triton_aot_manifest as aot

    rec = aot.Recorder().install()
    from tensorfold.cuda import experts
    from tensorfold.cuda.kernels import prefill_attention, qmm
    from tensorfold.cuda.exl3 import experts as exl3_experts, linear as exl3_linear
    from tensorfold.families.glm5_next.cuda import kda

    rec.wrap(qmm._ext(), ("qmm", "qmm_group", "qmm_prefill"), "qmm")
    rec.wrap(experts._ext(), ("plan", "run", "prefill", "pack"), "experts")
    rec.wrap(prefill_attention._ext(), ("prefill_attention",), "prefill_attention")
    rec.wrap(exl3_experts._ext(), ("run", "prefill", "plan", "pack", "dequant", "unpack"), "exl3_experts")
    rec.wrap(exl3_linear._ext(), ("unpack", "dequant", "mm", "prefill"), "exl3_linear")
    rec.wrap(kda._ext(), ("replay_layers", "chain_wide", "chain", "replay"), "kda")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="", help="folder of triton_aot_manifest.py: record every Triton launch")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--context", type=int, default=8192)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = install_recorder(a.tools) if a.tools else None

    def phase(name: str, detail: bool = False):
        return rec.scope(name, detail) if rec is not None else nullcontext()

    import torch
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    model = Path(a.model)
    t0 = time.perf_counter()
    with phase("startup"):
        eng = GlmEngine(model, rank=a.rank, master=a.master, port=a.port,
                        context=a.context, context_explicit=True)
    info = {"startup_s": round(time.perf_counter() - t0, 2), "rank": a.rank,
            "policy": eng.policy, "mtp_on": eng.mtp_on,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(torch.cuda.current_device())}
    print(json.dumps(info), flush=True)
    if not rec:
        return 0
    if a.rank != 0:
        try:
            eng.follow()                  # the follower serves its rank until rank 0 closes the pair
        except Exception as e:            # the peer's shutdown surfaces here; the record is complete
            print(f"[capture] follow() ended: {type(e).__name__}: {e}", flush=True)
    else:
        (out / "weights.json").write_text(json.dumps(weight_digests(eng), indent=0, sort_keys=True) + "\n")
        ids = prompt_ids(model, {})
        (out / "prompts.json").write_text(json.dumps(ids) + "\n")
        results: dict[str, dict] = {}
        for draft in (False, True):
            label = "drafted" if draft else "serial"
            for name, prompt in ids.items():
                toks: list[int] = []
                with phase(f"{label}-{name}"):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    stats = eng.generate(prompt, a.max_tokens, None,
                                         lambda new: toks.extend(map(int, new)) and False, draft=draft)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                step = stats.get("decode_s", 0.0) / max(1, len(toks) - 1) * 1e3
                results[f"{label}/{name}"] = {"tokens": toks, "sha": sha12(toks), "wall_s": round(wall, 4),
                                              "ms_per_token": round(step, 4), **stats}
                print(f"{label} {name}: {len(toks)} tokens sha {sha12(toks)} {step:.3f} ms/token "
                      f"rounds {stats.get('rounds')} accepted {stats.get('accepted')}", flush=True)
        for name in ids:
            s, d = results[f"serial/{name}"], results[f"drafted/{name}"]
            print(f"drafted == serial {name}: {s['tokens'] == d['tokens']}", flush=True)
        (out / "results.json").write_text(json.dumps({"info": info, "results": results}, indent=1) + "\n")
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
