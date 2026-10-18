#!/usr/bin/env python3
"""Check step5_ref.py against StepFun's own model (inference/model.py, Triton stubbed) on a bf16 slice, on the CPU."""
# Usage: step5_check_ref.py --model SLICE --stepfun SRC/inference (SLICE from step5_fp8_to_mlx.py --unquantized)

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def stub_triton():
    def anything(name):
        def attr(key):
            if key.startswith("__"):
                raise AttributeError(key)
            return lambda *a, **k: a[0] if a else None
        m = types.ModuleType(name)
        m.__getattr__ = attr
        return m

    tr = anything("triton")
    tr.jit = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    tr.autotune = lambda *a, **k: (lambda f: f)
    tr.heuristics = lambda *a, **k: (lambda f: f)
    tr.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    tr.cdiv = lambda a, b: -(-a // b)
    tl = anything("triton.language")
    extra = anything("triton.language.extra")
    libdevice = anything("triton.language.extra.libdevice")
    tr.language, tl.extra, extra.libdevice = tl, extra, libdevice
    sys.modules.update({"triton": tr, "triton.language": tl, "triton.language.extra": extra,
                        "triton.language.extra.libdevice": libdevice})


def stepfun_rows(model_dir: Path, stepfun: Path, ids: list[int]):
    import torch
    from safetensors.torch import load_file

    stub_triton()
    sys.path.insert(0, str(stepfun))
    import model as sf

    tc = json.loads((model_dir / "config.json").read_text())["text_config"]
    tc["sparse_config"]["enabled"] = False
    tc["max_position_embeddings"] = max(8192, len(ids) + 1)
    cfg = sf.Step5Config(**tc)
    with torch.device("meta"):
        m = sf.Step5ForCausalLM(cfg)
    sd = {}
    for f in sorted(set(json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"].values())):
        sd.update(load_file(str(model_dir / f)))
    missing, _ = m.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith(("_qkv_weight", "rope_cos", "rope_sin"))]
    if missing:
        raise SystemExit(f"StepFun's model wants weights the slice lacks: {missing[:8]}")
    m.eval()
    rows = []
    for layer in m.model.layers:
        layer.register_forward_hook(lambda _m, _i, out: rows.append(out[-1].float().numpy().copy()))
    cache = m.allocate_cache(batch=1, max_tokens=len(ids) + 8, device=torch.device("cpu"))
    with torch.inference_mode():
        out = m(torch.tensor(ids), cache, torch.tensor([len(ids)], dtype=torch.int32))
    return rows, out.logits[0].float().numpy()


def mlx_rows(model_dir: Path, ids: list[int]):
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).parent))
    from step5_ref import Step5

    m = Step5(model_dir)
    hidden: list = []
    logits = m.forward(ids, 0, {}, hidden)
    return [np.array(h.astype(mx.float32)) for h in hidden], np.array(logits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="an --unquantized slice from step5_fp8_to_mlx.py")
    ap.add_argument("--stepfun", type=Path, required=True, help="the release's inference/ directory")
    ap.add_argument("--tokens", type=int, default=700, help="prompt length (past the 512 window)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model))
    text = (Path(__file__).read_text() * 8)
    ids = tok.encode(text)[: args.tokens]
    a, la = mlx_rows(args.model, ids)
    b, lb = stepfun_rows(args.model, args.stepfun, ids)
    worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        rel = float(np.abs(x - y).max() / (np.abs(y).max() + 1e-30))
        worst = max(worst, rel)
        print(f"layer {i:2d}: max |mlx - stepfun| / max |stepfun| = {rel:.2e}")
    rel = float(np.abs(la - lb).max() / (np.abs(lb).max() + 1e-30))
    top_a, top_b = np.argsort(-la)[:5], np.argsort(-lb)[:5]
    print(f"logits: rel {rel:.2e}; top-5 mlx {top_a.tolist()} stepfun {top_b.tolist()}")
    ok = top_a[0] == top_b[0] and worst < 2e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
