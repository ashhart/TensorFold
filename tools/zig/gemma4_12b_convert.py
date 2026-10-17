"""Gemma 4 12B (gemma4_unified) to the native checkpoint: text weights MLX affine 4-bit g64, the image embedder bf16."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm import convert

# the image embedder's tensors, kept in bf16 in their own shard; the audio input's are dropped
VISION = ("model.vision_embedder.", "model.embed_vision.")
DROPPED = VISION + ("model.embed_audio.",)
VISION_SHARD = "model-vision.safetensors"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="google/gemma-4-12B-it as downloaded (config.json, model*.safetensors)")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    config = json.loads((args.source / "config.json").read_text())
    if config.get("model_type") != "gemma4_unified":
        raise SystemExit("expected a gemma4_unified checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for f in args.source.iterdir():
            weights_file = f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json"
            if f.name != "config.json" and not weights_file:
                (staged / f.name).symlink_to(f.resolve())
        weights = {}
        for f in sorted(args.source.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        kept = {k: v for k, v in weights.items() if not k.startswith(DROPPED)}
        if any(not k.startswith("model.language_model.") for k in kept):
            raise SystemExit("unexpected tensors beside the language model")
        print(f"{len(weights)} tensors, {len(kept)} kept (the language model)")
        mx.save_safetensors(str(staged / "model.safetensors"), kept, metadata={"format": "mlx"})
        vision = {k.removeprefix("model."): v for k, v in weights.items() if k.startswith(VISION)}
        if len(vision) != 10 or any(v.dtype != mx.bfloat16 for v in vision.values()):
            raise SystemExit("expected the 10 bf16 tensors of the image embedder")
        # mlx-lm's gemma4 module reads the unified text model as it reads the dense 31B's
        remapped = dict(config, model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"])
        remapped["text_config"] = dict(config["text_config"], model_type="gemma4_text")
        (staged / "config.json").write_text(json.dumps(remapped, indent=2))
        convert(str(staged), str(args.output), quantize=True, q_bits=4, q_group_size=64)
    out = json.loads((args.output / "config.json").read_text())
    for key in ("model_type", "architectures"):
        out[key] = config[key]
    out["text_config"]["model_type"] = config["text_config"]["model_type"]
    out["vision_config"] = config["vision_config"]
    out.pop("audio_config", None)
    (args.output / "config.json").write_text(json.dumps(out, indent=2) + "\n")
    mx.save_safetensors(str(args.output / VISION_SHARD), vision, metadata={"format": "pt"})
    index_path = args.output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({k: VISION_SHARD for k in sorted(vision)})
    index_path.write_text(json.dumps(index, indent=4) + "\n")
    for name in ("chat_template.jinja", "processor_config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if (args.source / name).exists() and not (args.output / name).exists():
            shutil.copy(args.source / name, args.output / name)
    print(f"wrote {args.output}: {out['model_type']}, quantization {out.get('quantization')}")


if __name__ == "__main__":
    main()
