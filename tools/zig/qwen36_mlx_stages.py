"""mlx_lm's staged references for qwen3_5_moe, in the layout families/qwen3_5/stages.zig writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache

RECORD: list | None = None


def decoder_call(self, x, mask=None, cache=None):
    if self.is_linear:
        r = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    out = h + self.mlp(self.post_attention_layernorm(h))
    if RECORD is not None:
        RECORD.append((h, out))
    return out


def raw(x):
    x = x.astype(mx.bfloat16)
    mx.eval(x)
    return np.array(x.view(mx.uint16)).reshape(-1, x.shape[-1])


def main():
    global RECORD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prompt = json.loads(args.ids.read_text())
    reference = json.loads(args.reference.read_text())

    qwen3_5.DecoderLayer.__call__ = decoder_call
    model, _ = load(str(args.snapshot))
    text = model.language_model.model
    layers = len(text.layers)
    names = ["embed"] + [f"l{l:02d}.{k}" for l in range(layers) for k in ("mixer", "block")] + ["final"]
    planes = {n: [] for n in names + ["logits"]}
    cache = make_prompt_cache(model)
    if len(prompt) < 2:
        raise SystemExit("the prompt needs two tokens: one prefill call, then the steps")
    calls = [prompt[:-1], [prompt[-1]]] + [[t] for t in reference[:-1]]
    picks = []
    for step, ids in enumerate(calls):
        tokens = mx.array(ids)[None]
        RECORD = []
        logits = model(tokens, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        planes["embed"].append(raw(text.embed_tokens(tokens)))
        for l, (h, out) in enumerate(RECORD):
            planes[f"l{l:02d}.mixer"].append(raw(h))
            planes[f"l{l:02d}.block"].append(raw(out))
        planes["final"].append(raw(text.norm(RECORD[-1][1])))
        if step > 0:
            last = logits[:, -1, :]
            planes["logits"].append(raw(last))
            logprobs = last - mx.logsumexp(last, keepdims=True)
            picks.append(int(mx.argmax(logprobs, axis=-1).item()))
        RECORD = None
    args.out.mkdir(parents=True, exist_ok=True)
    stages = []
    for n, chunks in planes.items():
        data = np.concatenate(chunks)
        data.tofile(args.out / f"{n}.bf16")
        stages.append({"name": n, "width": int(data.shape[1]), "rows": int(data.shape[0])})
    total = len(prompt) + len(reference) - 1
    index = {"version": 1, "dtype": "bf16", "layers": layers, "hidden": stages[0]["width"],
             "vocab": stages[-1]["width"], "stages": stages, "positions": list(range(total)),
             "logit_positions": list(range(len(prompt) - 1, total)),
             "mlx": mx.__version__, "greedy_picks_equal_reference": picks == reference, "picks": picks}
    (args.out / "index.json").write_text(json.dumps(index))
    print(json.dumps({"picks_equal_reference": picks == reference,
                      "differ_at": [i for i, (a, b) in enumerate(zip(picks, reference)) if a != b]}))


if __name__ == "__main__":
    main()
