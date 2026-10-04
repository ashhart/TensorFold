#!/usr/bin/env python3
"""Development render: Qwen-Image-2.1 through TensorFold's transformer, sampler and image decoder.

The prompt encoder (Qwen3-VL language model and tokenizer) is borrowed from mflux, which must be importable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def encode_prompt(model_dir: str, prompt: str):
    """Prompt embeddings (1, tokens, 4096) from mflux's Qwen-Image-2.1 prompt encoder."""

    import mlx.core as mx
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    class TextOnly(Qwen21WeightDefinition):
        @staticmethod
        def get_components():
            return [c for c in Qwen21WeightDefinition.get_components() if c.name == "text_encoder"]

    class Holder:
        pass

    holder = Holder()
    holder.text_encoder = Qwen21TextEncoder()
    tokenizers = TokenizerLoader.load_all(definitions=Qwen21WeightDefinition.get_tokenizers(), model_path=model_dir)
    Qwen21Initializer.load_components(holder, Path(model_dir), TextOnly, None)
    embeds, mask = Qwen21PromptEncoder.encode_prompt(prompt=prompt, prompt_cache={}, tokenizer=tokenizers["qwen21"],
                                                     text_encoder=holder.text_encoder)
    if int(mx.sum(mask).item()) != mask.shape[1]:
        raise ValueError("a single prompt should carry no padding")
    mx.eval(embeds)
    del holder
    mx.clear_cache()
    return embeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", help="comma-separated raw noise levels for a distilled adapter")
    parser.add_argument("--lora", help="adapter safetensors to merge before quantizing")
    parser.add_argument("--no-int8", action="store_true", help="keep every projection in bfloat16")
    parser.add_argument("--vae-dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--save-latents")
    args = parser.parse_args()
    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else args.prompt
    if not prompt:
        parser.error("give --prompt or --prompt-file")

    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from tensorfold.families.qwen_image import lora, sampler, vae, weights

    started = time.perf_counter()
    text = encode_prompt(args.model_dir, prompt)
    text_s = time.perf_counter() - started

    mark = time.perf_counter()
    dit = weights.load_dit(args.model_dir)
    if args.lora:
        print("[tensorfold] " + json.dumps({"adapter": Path(args.lora).name, **lora.merge(dit, args.lora)}))
    changed = {} if args.no_int8 else weights.int8(dit)
    if args.lora:
        lora.settle(dit)
    load_s = time.perf_counter() - mark

    mark = time.perf_counter()
    nodes = tuple(float(v) for v in args.nodes.split(",")) if args.nodes else None
    latents = sampler.denoise(dit, text, args.width, args.height, args.steps, args.seed, nodes,
                              on_step=lambda i, n: print(f"[tensorfold] step {i + 1}/{n}", file=sys.stderr))
    mx.eval(latents)
    denoise_s = time.perf_counter() - mark
    forwards = len(nodes) if nodes else args.steps
    if args.save_latents:
        mx.save_safetensors(args.save_latents, {"latents": latents})
    del dit
    mx.clear_cache()

    mark = time.perf_counter()
    decoder = vae.load_decoder(args.model_dir, getattr(mx, args.vae_dtype))
    image = decoder.decode(latents)
    mx.eval(image)
    decode_s = time.perf_counter() - mark
    pixels = (np.asarray(image[0]) * 255.0 + 0.5).astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print("[tensorfold] " + json.dumps({
        "output": args.output, "width": args.width, "height": args.height, "text_tokens": int(text.shape[1]),
        "forwards": forwards, "int8": changed, "text_s": round(text_s, 1), "load_s": round(load_s, 1),
        "denoise_s": round(denoise_s, 1), "per_forward_s": round(denoise_s / forwards, 3),
        "decode_s": round(decode_s, 1), "total_s": round(time.perf_counter() - started, 1),
        "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
