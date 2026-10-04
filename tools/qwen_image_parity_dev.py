#!/usr/bin/env python3
"""Parity of TensorFold's Qwen-Image-2.1 transformer, schedule and decoder against mflux on real weights."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--text-len", type=int, default=77)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.config.config import Config
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    from tensorfold.families.qwen_image import sampler, schedule, vae, weights
    from tensorfold.families.qwen_image.config import latent_grid

    def compare(name, ours, theirs):
        a, b = np.asarray(ours.astype(mx.float32)).ravel(), np.asarray(theirs.astype(mx.float32)).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"{name}: max abs diff {np.abs(a - b).max():.5f}, reference max {np.abs(b).max():.3f}, cosine {cos:.6f}")
        return cos

    reference = QwenImage21(model_path=args.model_dir, model_config=ModelConfig.qwen_image_21())
    config = Config(width=args.width, height=args.height, guidance=1.0, scheduler="linear",
                    model_config=reference.model_config, num_inference_steps=40)
    ours_sigmas = schedule.sigmas(40, args.width, args.height)
    theirs_sigmas = np.asarray(config.scheduler.sigmas)
    print(f"schedule: max abs diff {np.abs(ours_sigmas - theirs_sigmas).max():.2e}")

    rows, columns = latent_grid(args.width, args.height)
    text = (mx.random.normal((1, args.text_len, 4096), key=mx.random.key(1)) * 4).astype(mx.bfloat16)
    latents = sampler.start_noise(7, args.width, args.height)
    exact = 1.0
    if not args.decoder_only:
        dit = weights.load_dit(args.model_dir)
        worst = 1.0
        for step in (0, 20, 39):
            theirs = reference.transformer(t=step, config=config, hidden_states=latents, encoder_hidden_states=text,
                                           encoder_hidden_states_mask=None)
            ours = dit(latents, float(ours_sigmas[step]), dit.prefix(text, rows, columns))
            mx.eval(ours, theirs)
            worst = min(worst, compare(f"transformer bf16, step {step}", ours, theirs))
            reference.transformer.clear_text_cache()
        exact = worst
        changed = weights.int8(dit)
        if changed["mlp"]:
            for step in (0, 20, 39):
                theirs = reference.transformer(t=step, config=config, hidden_states=latents, encoder_hidden_states=text,
                                               encoder_hidden_states_mask=None)
                ours = dit(latents, float(ours_sigmas[step]), dit.prefix(text, rows, columns))
                mx.eval(ours, theirs)
                compare(f"transformer int8 {changed}, step {step}", ours, theirs)
                reference.transformer.clear_text_cache()

    z = mx.random.normal((1, 64, rows, columns), key=mx.random.key(3))
    theirs = reference.vae.decode(z)                       # (1, 3, H, W) in [-1, 1]
    ours = vae.load_decoder(args.model_dir).decode(z)      # (1, H, W, 3) in [0, 1]
    theirs = mx.clip(theirs.transpose(0, 2, 3, 1).astype(mx.float32) / 2 + 0.5, 0, 1)
    mx.eval(ours, theirs)
    decoded = compare("decoder", ours, theirs)
    return 0 if exact > 0.999 and decoded > 0.9999 else 1


if __name__ == "__main__":
    sys.exit(main())
