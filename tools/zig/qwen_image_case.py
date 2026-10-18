#!/usr/bin/env python3
"""Qwen-Image-2.1 reference cases for tf-qwen-image from mflux (MLX): prompt encoder, transformer and decoder."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

USAGE = """\
  case MODEL_DIR OUT_DIR --prompt-file F   prompt rows, noise levels, start latents, each step's input and velocity
  decode MODEL_DIR LATENTS.f32 PNG         raw float32 latents (rows x columns x 64) from tf-qwen-image to a picture
  compare A.png B.png [SIDE.png]           PSNR and mean absolute difference of two pictures, optionally side by side
MODEL_DIR is a diffusers checkpoint folder (Qwen/Qwen-Image-2.1 or -Turbo). The noise levels are the checkpoint's
sample_sigmas (model_index.json) with the resolution shift and no terminal stretch, else even steps (--steps).
"""


def load_parts(model_dir: str, *names: str):
    """mflux's Qwen-Image-2.1 components `names` (text_encoder, transformer, vae) from a checkpoint folder."""
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
    from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    class Only(Qwen21WeightDefinition):
        @staticmethod
        def get_components():
            return [c for c in Qwen21WeightDefinition.get_components() if c.name in names]

    class Parts:
        pass

    parts = Parts()
    made = {"text_encoder": Qwen21TextEncoder, "transformer": Qwen21Transformer, "vae": Qwen21VAE}
    for name in names:
        setattr(parts, name, made[name]())
    Qwen21Initializer.load_components(parts, Path(model_dir), Only, None)
    return parts


def encode_prompt(model_dir: str, prompt: str):
    """The text encoder's rows (1, tokens, 4096) bf16 for one prompt, as mflux's pipeline makes them."""
    import mlx.core as mx
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    parts = load_parts(model_dir, "text_encoder")
    tokenizers = TokenizerLoader.load_all(definitions=Qwen21WeightDefinition.get_tokenizers(), model_path=model_dir)
    embeds, mask = Qwen21PromptEncoder.encode_prompt(prompt=prompt, prompt_cache={}, tokenizer=tokenizers["qwen21"],
                                                     text_encoder=parts.text_encoder)
    if int(mx.sum(mask).item()) != mask.shape[1]:
        raise ValueError("a single prompt should carry no padding")
    mx.eval(embeds)
    del parts
    mx.clear_cache()
    return embeds


def sigmas(model_dir: str, width: int, height: int, steps: int):
    """Noise levels down to 0: the saved nodes (or even steps) under the scheduler's exponential resolution shift."""
    import numpy as np

    root = Path(model_dir)
    sched = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
    nodes = json.loads((root / "model_index.json").read_text()).get("sample_sigmas")
    raw = np.asarray(nodes if nodes else np.linspace(1.0, 1.0 / steps, steps), dtype=np.float32)
    slope = (sched["max_shift"] - sched["base_shift"]) / (sched["max_image_seq_len"] - sched["base_image_seq_len"])
    mu = slope * (width * height / 256) + sched["base_shift"] - slope * sched["base_image_seq_len"]
    scale = np.float32(math.exp(mu))
    shifted = scale / (scale + (1.0 / raw - 1.0))
    if not nodes and sched.get("shift_terminal"):
        rest = 1.0 - shifted
        shifted = 1.0 - rest / (rest[-1] / (1.0 - sched["shift_terminal"]))
    return np.concatenate([shifted, np.zeros(1, dtype=np.float32)]).astype(np.float32)


def to_png(model_dir: str, latents, rows: int, columns: int, path: str) -> None:
    """Latents (rows * columns, 64) through mflux's decoder in float32 to a PNG."""
    import mlx.core as mx
    from mlx.utils import tree_map
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.utils.image_util import ImageUtil

    vae = load_parts(model_dir, "vae").vae
    vae.update(tree_map(lambda v: v.astype(mx.float32), vae.parameters()))
    z = mx.array(latents, dtype=mx.float32).reshape(1, rows, columns, 64).transpose(0, 3, 1, 2)[:, :, None]
    ImageUtil.to_pil(VAEUtil.decode(vae=vae, latent=z)).save(path)


def case(args) -> int:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_map
    from mflux.models.common.config import ModelConfig

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, columns = args.height // 16, args.width // 16
    mark = time.perf_counter()
    embeds = encode_prompt(args.model_dir, Path(args.prompt_file).read_text().strip())
    encode_s = time.perf_counter() - mark
    levels = sigmas(args.model_dir, args.width, args.height, args.steps)
    dit = load_parts(args.model_dir, "transformer").transformer
    geometry = type("Geometry", (), {"width": args.width, "height": args.height})()
    noise = mx.random.normal(shape=[1, rows * columns, 64], key=mx.random.key(args.seed)).astype(mx.bfloat16)
    x = noise.astype(mx.float32)
    inputs, velocity, forward_s = [], [], []
    for s in range(len(levels) - 1):
        given = x.astype(mx.bfloat16)
        mark = time.perf_counter()
        v = dit(float(levels[s]), geometry, given, embeds)
        mx.eval(v)
        forward_s.append(time.perf_counter() - mark)
        inputs.append(given[0].astype(mx.float32))
        velocity.append(v[0].astype(mx.float32))
        x = x + v.astype(mx.float32) * float(levels[s + 1] - levels[s])
        mx.eval(x)
    start = noise[0].astype(mx.float32).reshape(rows, columns, 64)
    tensors = {"prompt": embeds[0], "sigmas": mx.array(levels), "latents": start, "inputs": mx.stack(inputs),
               "velocity": mx.stack(velocity)}
    final = np.asarray(x[0])
    if args.fp32_steps:
        # the same transformer and inputs in float32 throughout: the bf16 reference's own distance from exact arithmetic
        ModelConfig.precision = mx.float32
        dit.update(tree_map(lambda v: v.astype(mx.float32), dit.parameters()))
        for block in dit.transformer_blocks:
            block.attn.use_fused_prologue = False
        dit._image_step_fn = None
        dit.clear_text_cache()
        wide = embeds.astype(mx.float32)
        exact = []
        for s in range(min(args.fp32_steps, len(levels) - 1)):
            v = dit(float(levels[s]), geometry, inputs[s][None], wide)
            mx.eval(v)
            exact.append(v[0])
        tensors["velocity_f32"] = mx.stack(exact)
        ModelConfig.precision = mx.bfloat16
    mx.save_safetensors(str(out / "case.safetensors"), tensors)
    del dit, tensors
    mx.clear_cache()
    final.astype(np.float32).tofile(out / "reference.latents.f32")
    to_png(args.model_dir, final, rows, columns, str(out / "reference.png"))
    meta = {"width": args.width, "height": args.height, "seed": args.seed, "prompt_rows": int(embeds.shape[1]),
            "sigmas": [round(float(v), 6) for v in levels], "encode_s": round(encode_s, 2),
            "mflux_bf16_forward_s": [round(t, 3) for t in forward_s], "mlx": mx.__version__}
    (out / "reference.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


def decode(args) -> int:
    import numpy as np

    raw = np.fromfile(args.latents, dtype=np.float32)
    rows, columns = args.height // 16, args.width // 16
    if raw.size != rows * columns * 64:
        sys.exit(f"{args.latents} holds {raw.size} values; {args.width}x{args.height} needs {rows * columns * 64}")
    to_png(args.model_dir, raw, rows, columns, args.png)
    return 0


def compare(args) -> int:
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(args.a).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(args.b).convert("RGB"), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
    print(json.dumps({"psnr_db": round(psnr, 2), "mean_abs": round(float(np.mean(np.abs(a - b))), 3),
                      "max_abs": int(np.max(np.abs(a - b)))}))
    if args.side:
        side = np.concatenate([a, b], axis=1).astype(np.uint8)
        Image.fromarray(side).save(args.side)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("case")
    c.add_argument("model_dir")
    c.add_argument("out_dir")
    c.add_argument("--prompt-file", required=True)
    c.add_argument("--width", type=int, default=1344)
    c.add_argument("--height", type=int, default=768)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--steps", type=int, default=40, help="even steps when the checkpoint saves no nodes")
    c.add_argument("--fp32-steps", type=int, default=1, help="steps also run in float32 (0: none)")
    d = sub.add_parser("decode")
    d.add_argument("model_dir")
    d.add_argument("latents")
    d.add_argument("png")
    d.add_argument("--width", type=int, default=1344)
    d.add_argument("--height", type=int, default=768)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("side", nargs="?")
    args = parser.parse_args()
    return {"case": case, "decode": decode, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
