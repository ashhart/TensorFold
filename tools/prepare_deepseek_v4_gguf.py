#!/usr/bin/env python3
"""Prepare native TensorFold sidecars from a local GGUF; packed weights remain in place."""

from __future__ import annotations

import argparse
import json
import mmap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfold.cuda.gguf.linear import FORMATS
from tensorfold.gguf import parse_gguf_tensors
from tensorfold.gguf_tokenizer import tokenizer


def prepare(path, output, *, context=163840, drafter="", retained_prefix=131072):
    path = Path(path).resolve(strict=True)
    output = Path(output)
    if not 0 <= retained_prefix < context:
        raise ValueError("retained prefix must be nonnegative and below the context")
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        inv = parse_gguf_tensors(m)
        meta = inv.info.metadata
        if meta.get("general.architecture") != "deepseek4":
            raise ValueError("expected deepseek4 architecture")
        bad = [t for t in inv.tensors if t.type_name not in {"F32", "F16", "I32", *FORMATS}]
        if bad:
            raise ValueError(f"unsupported quantization {bad[0].type_name}: {bad[0].name}")
        from tensorfold.families.deepseek_v4.gguf import validate_cuda_shapes

        validate_cuda_shapes(inv, meta)
        tok = tokenizer(meta)
        cfg = {
            "model_type": "deepseek_v4",
            "max_position_embeddings": int(context),
            "quantization_config": {"quant_method": "gguf"},
            "gguf": {
                "path": str(path),
                "context": int(context),
                "retained_prefix": int(retained_prefix),
                "drafter": str(Path(drafter).resolve(strict=True)) if drafter else "",
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    tok.save(str(output / "tokenizer.json"))
    (output / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{{ messages }}",
                "bos_token": "<｜begin▁of▁sentence｜>",
                "eos_token": "<｜end▁of▁sentence｜>",
            }
        )
        + "\n"
    )
    tmp = output / "config.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    tmp.replace(output / "config.json")
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--drafter", default="")
    p.add_argument("--context", type=int, default=163840)
    p.add_argument("--retained-prefix", type=int, default=131072)
    a = p.parse_args()
    print(
        json.dumps(
            prepare(a.gguf, a.model_dir, context=a.context, drafter=a.drafter, retained_prefix=a.retained_prefix),
            indent=2,
        )
    )
