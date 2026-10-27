#!/usr/bin/env python3
"""MOSS-Image reference cases for tf-moss-image from the publisher's own pipeline (pipeline_moss_image.py, torch)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

USAGE = """\
  case MODEL_DIR OUT_DIR --prompt-file F   prompt rows, timesteps, start latents, each step's inputs and both velocities
  start MODEL_DIR OUT_DIR --prompt-file F  only what a picture needs: both prompts' rows, the schedule, the noise
  exact MODEL_DIR CASE_DIR                 the case's schedule with a float32 transformer: CASE_DIR/exact.png
  decode MODEL_DIR LATENTS.f32 PNG         raw float32 latents (rows x columns x 128) from tf-moss-image to a picture
  compare A.png B.png [SIDE.png]           PSNR and mean absolute difference of two pictures, optionally side by side
MODEL_DIR is the OpenMOSS-Team/MOSS-Image-Base folder. The case runs the card's defaults: 50 Euler steps, CFG 3, an
empty negative prompt whose branch skips block 22, the bf16 transformer and text encoder, the float32 decoder.
"""


# pipeline_moss_image.py at OpenMOSS-Team/MOSS-Image-Base revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3
PIPELINE_SHA256 = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"


def module(model_dir: str):
    """The checkpoint's pipeline_moss_image.py, imported from its folder once its sha256 is the pinned revision's."""
    path = Path(model_dir) / "pipeline_moss_image.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PIPELINE_SHA256:
        sys.exit(f"{path} has sha256 {digest}; this tool runs only the pinned revision's ({PIPELINE_SHA256})")
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pipeline(model_dir: str, device: str, text: bool = True, dit: bool = True):
    """The publisher's MOSSImagePipeline from the folder's components (the encoder and transformer optional)."""
    import torch
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer

    m = module(model_dir)
    root = Path(model_dir)
    transformer = m.MOSSImageTransformer.from_pretrained(root / "transformer", torch_dtype=torch.bfloat16).eval() if dit else None
    if dit and device == "mps":
        # MPS has no float64: the publisher's float64 time sinusoid runs on the host; all else runs as published.
        class HostTimes:
            def __getattr__(self, name):
                return getattr(torch, name)

            @staticmethod
            def as_tensor(data, device=None, dtype=None):
                return torch.as_tensor(data, dtype=dtype).cpu()

        m.torch = HostTimes()
        transformer.time_embedding.register_forward_pre_hook(lambda mod, a: tuple(x.to(mod[0].weight.device) for x in a))
    scheduler = m.MOSSImageScheduler.from_pretrained(root / "scheduler")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer") if text else None
    encoder = AutoModel.from_pretrained(root / "text_encoder", dtype=torch.bfloat16).eval() if text else None
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.float32).eval()
    return m.MOSSImagePipeline(transformer, scheduler, encoder, tokenizer, vae).to(device), m


def rows(x):
    """[1, C, H, W] to (H, W, C) float32 on the host: tf-moss-image's row order."""
    return x[0].float().permute(1, 2, 0).contiguous().cpu()


def to_png(pipe, latents, path: str) -> None:
    """Latents [1, 128, H, W] through the pipeline's own decode path (its batch-norm undo, unpack, VAE) to a PNG."""
    import torch

    with torch.no_grad():
        z = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
        bn = pipe.vae.bn
        eps = pipe.vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        image = pipe.vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        pipe.image_processor.postprocess(image, output_type="pil")[0].save(path)


def case(args) -> int:
    import torch
    from safetensors.torch import save_file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe, _ = pipeline(args.model_dir, args.device)
    mark = time.perf_counter()
    positive, _ = pipe.encode_prompt(Path(args.prompt_file).read_text().strip())
    negative, _ = pipe.encode_prompt("")
    encode_s = time.perf_counter() - mark
    # the encoder is done; the transformer and decoder stay
    pipe.text_encoder = None
    if args.device == "mps":
        torch.mps.empty_cache()
    h, w = args.height // 16, args.width // 16
    noise = torch.randn((1, 128, h, w), generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32)
    calls = []
    forward = pipe.transformer.forward

    def recorded(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, return_dict=True,
                 skip_layers=None):
        mark = time.perf_counter()
        result = forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, return_dict,
                         skip_layers)
        sample = result.sample if return_dict else result[0]
        sample.float().sum().item()
        calls.append({"input": rows(hidden_states), "t": float(timestep), "skip": sorted(skip_layers or ()),
                      "velocity": rows(sample), "s": time.perf_counter() - mark})
        return result

    pipe.transformer.forward = recorded
    image = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, height=args.height, width=args.width,
                 num_inference_steps=args.steps, latents=noise.to(args.device), output_type="latent").images
    pipe.transformer.forward = forward
    cond, uncond = calls[0::2], calls[1::2]
    if not all(c["skip"] == [] for c in cond) or not all(u["skip"] == [22] for u in uncond):
        sys.exit("the pipeline's calls are not (prompt, negative skipping 22) pairs")
    sched = pipe.scheduler
    tensors = {"prompt": positive[0].cpu().contiguous(), "negative": negative[0].cpu().contiguous(),
               "timesteps": sched.timesteps.float().cpu().contiguous(), "sigmas": sched.sigmas.float().cpu().contiguous(),
               "latents": noise[0].permute(1, 2, 0).contiguous(), "inputs": torch.stack([c["input"] for c in cond]),
               "velocity": torch.stack([c["velocity"] for c in cond]),
               "velocity_negative": torch.stack([u["velocity"] for u in uncond])}
    if args.fp32_steps:
        # the same transformer and inputs in float32 throughout: the bf16 reference's own distance from exact arithmetic
        pipe.transformer.float()
        exact, exact_negative = [], []
        with torch.no_grad():
            for s in range(min(args.fp32_steps, len(cond))):
                x = cond[s]["input"].permute(2, 0, 1)[None].to(args.device)
                t = sched.timesteps[s]
                exact.append(rows(pipe.transformer(x, t, positive.float()).sample))
                exact_negative.append(rows(pipe.transformer(x, t, negative.float(), skip_layers=[22]).sample))
        tensors["velocity_f32"] = torch.stack(exact)
        tensors["velocity_negative_f32"] = torch.stack(exact_negative)
        pipe.transformer.to(torch.bfloat16)
    save_file(tensors, str(out / "case.safetensors"))
    image.float().cpu().permute(0, 2, 3, 1).contiguous().numpy().tofile(out / "reference.latents.f32")
    to_png(pipe, image, str(out / "reference.png"))
    meta = {"width": args.width, "height": args.height, "seed": args.seed, "steps": len(cond),
            "prompt_rows": int(positive.shape[1]), "negative_rows": int(negative.shape[1]), "device": args.device,
            "encode_s": round(encode_s, 2), "torch_bf16_forward_s": [round(c["s"], 3) for c in calls],
            "torch": torch.__version__}
    (out / "reference.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "torch_bf16_forward_s"}))
    return 0


def start(args) -> int:
    """What tf-moss-image needs to make a picture: both prompts' encoder rows, the schedule and the seeded noise."""
    import torch
    from safetensors.torch import save_file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe, _ = pipeline(args.model_dir, args.device, dit=False)
    positive, _ = pipe.encode_prompt(Path(args.prompt_file).read_text().strip())
    negative, _ = pipe.encode_prompt(args.negative)
    h, w = args.height // 16, args.width // 16
    noise = torch.randn((1, 128, h, w), generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32)
    pipe.scheduler.set_timesteps(args.steps)
    save_file({"prompt": positive[0].cpu().contiguous(), "negative": negative[0].cpu().contiguous(),
               "timesteps": pipe.scheduler.timesteps.float().cpu().contiguous(),
               "sigmas": pipe.scheduler.sigmas.float().cpu().contiguous(),
               "latents": noise[0].permute(1, 2, 0).contiguous()}, str(out / "case.safetensors"))
    print(json.dumps({"width": args.width, "height": args.height, "seed": args.seed, "steps": args.steps,
                      "prompt_rows": int(positive.shape[1]), "negative_rows": int(negative.shape[1])}))
    return 0


def exact(args) -> int:
    """The case's whole guided schedule with the transformer in float32: where exact arithmetic lands."""
    import torch
    from safetensors.torch import load_file

    case_dir = Path(args.case_dir)
    case = load_file(str(case_dir / "case.safetensors"))
    pipe, _ = pipeline(args.model_dir, args.device, text=False)
    pipe.transformer.float()
    h, w, _ = case["latents"].shape
    image = pipe(prompt_embeds=case["prompt"][None].float().to(args.device),
                 negative_prompt_embeds=case["negative"][None].float().to(args.device), height=16 * h, width=16 * w,
                 num_inference_steps=args.steps, latents=case["latents"].permute(2, 0, 1)[None].to(args.device),
                 output_type="latent").images
    image.float().cpu().permute(0, 2, 3, 1).contiguous().numpy().tofile(case_dir / "exact.latents.f32")
    to_png(pipe, image, str(case_dir / "exact.png"))
    return 0


def decode(args) -> int:
    import numpy as np
    import torch

    raw = np.fromfile(args.latents, dtype=np.float32)
    h, w = args.height // 16, args.width // 16
    if raw.size != h * w * 128:
        sys.exit(f"{args.latents} holds {raw.size} values; {args.width}x{args.height} needs {h * w * 128}")
    pipe, _ = pipeline(args.model_dir, args.device, text=False, dit=False)
    to_png(pipe, torch.from_numpy(raw).reshape(1, h, w, 128).permute(0, 3, 1, 2), args.png)
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
    c.add_argument("--width", type=int, default=1024)
    c.add_argument("--height", type=int, default=1024)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--steps", type=int, default=50)
    c.add_argument("--fp32-steps", type=int, default=1, help="steps also run in float32 (0: none)")
    c.add_argument("--device", default="mps")
    s = sub.add_parser("start")
    s.add_argument("model_dir")
    s.add_argument("out_dir")
    s.add_argument("--prompt-file", required=True)
    s.add_argument("--negative", default="", help="the negative prompt (the card's default: empty)")
    s.add_argument("--width", type=int, default=1024)
    s.add_argument("--height", type=int, default=1024)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--device", default="mps")
    x = sub.add_parser("exact")
    x.add_argument("model_dir")
    x.add_argument("case_dir")
    x.add_argument("--steps", type=int, default=50)
    x.add_argument("--device", default="mps")
    d = sub.add_parser("decode")
    d.add_argument("model_dir")
    d.add_argument("latents")
    d.add_argument("png")
    d.add_argument("--width", type=int, default=1024)
    d.add_argument("--height", type=int, default=1024)
    d.add_argument("--device", default="mps")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("side", nargs="?")
    args = parser.parse_args()
    return {"case": case, "start": start, "exact": exact, "decode": decode, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
