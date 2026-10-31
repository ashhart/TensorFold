#!/usr/bin/env python3
"""Idol reference cases for tf-idol from the publisher's idol-sdk (torch): the denoiser's schedule and its blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

USAGE = """\
  case WEIGHTS SDK OUT --features F [F]    noise, each step's input and both velocities, the final sample
  blocks WEIGHTS SDK CASE [--step S]       every block's rows at step S, and each block in float32 on its input
  exact WEIGHTS SDK CASE [--steps N]       both velocities in float32 on the case's first N inputs (the ruler)
  vae VAE SDK IMAGE OUT [--stages]         the FLUX.2 VAE rows of a preprocessed image: the SDK's, then float64's
WEIGHTS is a speridlabs/idol snapshot; SDK an idol-sdk checkout at the pinned commit. F holds feature1 (DINOv3 rows
[1, 4101, 1280]) and feature2 (FLUX.2 VAE rows [1, 4101, 128]), in one file or two, as the SDK's ImageEncoder returns.
VAE is the vae/ folder of black-forest-labs/FLUX.2-klein-4B at the pinned revision (FLUX.2-dev's gives the same bf16
weights); IMAGE the 1024 x 1024 RGB picture IdolPipeline.preprocess returns.
"""

SDK_COMMIT = "4702a7f00850e34b6e83a2170cc123c3e9570744"
SDK_SHA256 = {
    "idol/models/dit.py": "fbe725cc478b0acd245d9404ac8d9df327ebfbf79fe51c954d94b005c5bd6af4",
    "idol/models/layers.py": "2dbf24107396194636c9a2e89e7acee1bcf91b14dba1b4cde0318cb85b3f482f",
    "idol/sampler.py": "532987b5599be678361c2fcf566335bb3e1b0171696f985818d10b7460bfc8b0",
}
WEIGHTS_REVISION = "2731c2291fc65a797c5cb627a5e5526fc1490902"
CONDITIONING_SHA256 = "8c9cc50d327df3a5269cda392ef46cf4ed9c1e603f42f46050beda83f355f5ab"
DIFFUSERS_VERSION = "0.42.1"
DIFFUSERS_SHA256 = {
    "models/autoencoders/autoencoder_kl_flux2.py": "1afe08d76a93e994a453bd9036b57d173516211c2e6b9794ea64f485bea57f57",
    "models/autoencoders/vae.py": "59b51fdf2d951da45d697569271e0c99adf4a959ec0f91885f3d9314e925a12c",
    "models/unets/unet_2d_blocks.py": "545d158c9d98fb8971c501303117fd3e32d27ae28506b89005c65548404f7437",
    "models/resnet.py": "77ad799719fa51bf0fc703f0486915b791c9b69673983abeb3e32c6edcc76650",
    "models/downsampling.py": "bce174aa8007b2c40787331e7f46b31eef1eab7785f16b4e54733553b3d14724",
    "models/attention_processor.py": "59871fd4e66fc14a26f0e0464706ab24680e13dcd93161c55123678a5e671772",
    "utils/torch_utils.py": "bce0a1a4488557e8f80dc02de5782c2b510e91ba2e07f7ae8d26168cee9afc2a",
}
# FLUX.2-klein-4B's vae/ (Apache 2.0, ungated) at its pinned revision, and FLUX.2-dev's: the same weights once in bf16
VAE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
VAE_SHA256 = {"ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04": "FLUX.2-klein-4B",
              "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5": "FLUX.2-dev"}
DENOISER_ARGS = {"q_token_length": 8192, "in_channels": 16, "cam_channels": 5, "model_channels": 1024,
                 "cond_channels": 1280, "cond2_channels": 128, "num_blocks": 24, "num_refiner_blocks": 2,
                 "num_heads": 16, "mlp_ratio": 4}


def sdk(path: str):
    """The SDK's denoiser and sampler, imported once each file used has the pinned commit's sha256."""
    root = Path(path)
    for name, want in SDK_SHA256.items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != want:
            sys.exit(f"{root / name} has sha256 {digest}; this tool runs only idol-sdk {SDK_COMMIT[:12]}'s ({want})")
    sys.path.insert(0, str(root))
    from idol.models.dit import LatentSeqMMFlowModel
    from idol.sampler import sample_flow
    return LatentSeqMMFlowModel, sample_flow


def denoiser(weights: str, sdk_dir: str, device: str):
    """The SDK's denoiser with the snapshot's weights, strictly loaded, as IdolPipeline builds it."""
    import torch
    from safetensors.torch import load_file

    model_cls, _ = sdk(sdk_dir)
    config = json.loads((Path(weights) / "config.json").read_text())
    args = config["denoiser"]["args"]
    if config["denoiser"]["name"] != "LatentSeqMMFlowModel" or args != DENOISER_ARGS:
        sys.exit(f"{weights}/config.json is not speridlabs/idol {WEIGHTS_REVISION[:12]}'s denoiser")
    model = model_cls(**args)
    model.load_state_dict(load_file(str(Path(weights) / "denoiser.safetensors")), strict=True)
    return model.eval().requires_grad_(False).to(torch.device(device))


def features(paths: list[str]):
    """feature1 and feature2 [1, 4101, C] float32 from one or two safetensors files."""
    from safetensors.torch import load_file

    found = {}
    for path in paths:
        found.update(load_file(path))
    for key, width in (("feature1", 1280), ("feature2", 128)):
        if key not in found or tuple(found[key].shape) != (1, 4101, width):
            sys.exit(f"the features need {key} [1, 4101, {width}]")
    return found["feature1"].float(), found["feature2"].float()


def condition(f1, f2, device: str):
    """The sampler's cond and neg_cond (both streams zeroed), as IdolPipeline passes them."""
    from tensordict import TensorDict

    cond = TensorDict({"feature1": f1, "feature2": f2}, batch_size=1, device=device)
    return cond, cond.apply(lambda x: x * 0)


def sample(latent, camera, device: str):
    """The sampler's TensorDict for one latent [8192, 16] and camera [5]."""
    from tensordict import TensorDict

    return TensorDict({"latent": latent[None].to(device), "camera": camera.view(1, 1, 5).to(device)}, batch_size=1,
                      device=device)


def flush(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()


def case(args) -> int:
    import torch
    from safetensors.torch import save_file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = denoiser(args.weights, args.sdk, args.device)
    _, sample_flow = sdk(args.sdk)
    f1, f2 = features(args.features)
    cond, neg = condition(f1, f2, args.device)
    gen = torch.Generator("cpu").manual_seed(args.seed)
    latent = torch.randn((8192, 16), generator=gen, dtype=torch.float32)
    camera = torch.randn((5,), generator=gen, dtype=torch.float32)
    calls = []
    forward = model.forward

    def recorded(x, t, c):
        mark = time.perf_counter()
        v = forward(x, t, c)
        flush(args.device)
        calls.append({"latent": x["latent"][0].float().cpu(), "camera": x["camera"][0, 0].float().cpu(),
                      "t": float(t[0]), "negative": bool(c["feature1"].abs().max() == 0),
                      "v": v["latent"][0].float().cpu(), "v_camera": v["camera"][0, 0].float().cpu(),
                      "s": time.perf_counter() - mark})
        return v

    model.forward = recorded
    with torch.no_grad():
        final = sample_flow(model, sample(latent, camera, args.device), cond=cond, neg_cond=neg, steps=args.steps,
                            rescale_t=args.rescale_t, cfg_strength=args.cfg, verbose=False, batch_cfg=False)
    model.forward = forward
    pos, negs = calls[0::2], calls[1::2]
    if args.cfg <= 0 or any(c["negative"] for c in pos) or not all(c["negative"] for c in negs):
        sys.exit("the sampler's calls are not (conditional, negative) pairs")
    stack = lambda rows, key: torch.stack([r[key] for r in rows]).contiguous()
    tensors = {"feature1": f1[0].contiguous(), "feature2": f2[0].contiguous(), "latent": latent, "camera": camera,
               "timesteps": torch.tensor([c["t"] for c in pos], dtype=torch.float32),
               "inputs": stack(pos, "latent"), "inputs_camera": stack(pos, "camera"),
               "velocity": stack(pos, "v"), "velocity_camera": stack(pos, "v_camera"),
               "velocity_negative": stack(negs, "v"), "velocity_negative_camera": stack(negs, "v_camera"),
               "sample": final["latent"][0].float().cpu().contiguous(),
               "sample_camera": final["camera"][0, 0].float().cpu().contiguous()}
    save_file(tensors, str(out / "case.safetensors"))
    meta = {"seed": args.seed, "steps": args.steps, "cfg": args.cfg, "rescale_t": args.rescale_t,
            "device": args.device, "torch": torch.__version__, "sdk": SDK_COMMIT, "weights": WEIGHTS_REVISION,
            "torch_forward_s": [round(c["s"], 3) for c in calls]}
    (out / "reference.json").write_text(json.dumps(meta, indent=1) + "\n")
    times = sorted(meta["torch_forward_s"])
    print(json.dumps({k: v for k, v in meta.items() if k != "torch_forward_s"} | {"median_forward_s": times[len(times) // 2]}))
    return 0


def blocks(args) -> int:
    """Each block's rows in the conditional forward at one step, and each block again in float32 on the same input."""
    import copy

    import torch
    from safetensors.torch import load_file, save_file

    case_dir = Path(args.case_dir)
    data = load_file(str(case_dir / "case.safetensors"))
    model = denoiser(args.weights, args.sdk, args.device)
    cond, _ = condition(data["feature1"][None], data["feature2"][None], args.device)
    seen = {}

    def keep(name):
        def hook(_module, _inputs, output):
            seen[name] = output.detach().contiguous()
        return hook

    stages = [("noise_refiner", model.noise_refiner), ("context_refiner", model.context_refiner), ("block", model.blocks)]
    hooks = [m.register_forward_hook(keep(f"{name}.{i}")) for name, layers in stages for i, m in enumerate(layers)]
    x = sample(data["inputs"][args.step], data["inputs_camera"][args.step], args.device)
    t = data["timesteps"][args.step : args.step + 1].to(args.device)
    with torch.no_grad():
        v = model(x, t, cond)
        for h in hooks:
            h.remove()
        # the ruler: each joint block in float32 on the fp16 rows the SDK fed it, with an unrounded modulation
        t_emb = model.t_embedder(t)
        t_mod = model.adaLN_modulation(t_emb).float()
        exact = {}
        for i in range(1, len(model.blocks)):
            block = copy.deepcopy(model.blocks[i]).float()
            h = seen[f"block.{i - 1}"].float()
            exact[f"block.{i}"] = block(h, mod=t_mod, rotary_emb=model.repo_layers[i](h))[0].cpu().contiguous()
            del block
    tensors = {k: x[0].cpu().contiguous() for k, x in seen.items()}
    tensors["velocity"] = v["latent"][0].float().cpu().contiguous()
    tensors["velocity_camera"] = v["camera"][0, 0].float().cpu().contiguous()
    save_file(tensors, str(case_dir / f"blocks{args.step}.safetensors"))
    save_file(exact, str(case_dir / f"blocks{args.step}_f32.safetensors"))
    drift = (tensors["velocity"] - data["velocity"][args.step]).abs().max().item()
    print(json.dumps({"step": args.step, "tensors": len(tensors), "velocity_max_abs_vs_case": drift}))
    return 0


def exact(args) -> int:
    """Both branches in float32 throughout on the case's own inputs: the fp16 reference's distance from exact."""
    import torch
    from safetensors.torch import load_file, save_file

    case_dir = Path(args.case_dir)
    data = load_file(str(case_dir / "case.safetensors"))
    model = denoiser(args.weights, args.sdk, args.device).float()
    model.dtype = torch.float32
    cond, neg = condition(data["feature1"][None], data["feature2"][None], args.device)
    keys = ("velocity", "velocity_camera", "velocity_negative", "velocity_negative_camera")
    rows = {k: [] for k in keys}
    with torch.no_grad():
        for s in range(min(args.steps, data["inputs"].shape[0])):
            x = sample(data["inputs"][s], data["inputs_camera"][s], args.device)
            t = data["timesteps"][s : s + 1].to(args.device)
            for prefix, c in (("velocity", cond), ("velocity_negative", neg)):
                v = model(x, t, c)
                rows[prefix].append(v["latent"][0].float().cpu())
                rows[prefix + "_camera"].append(v["camera"][0, 0].float().cpu())
    save_file({k + "_f32": torch.stack(v).contiguous() for k, v in rows.items()}, str(case_dir / "exact.safetensors"))
    print(json.dumps({"steps": len(rows["velocity"])}))
    return 0


def flux2_vae(vae_dir: str, sdk_dir: str):
    """The SDK's ImageEncoder and diffusers' FLUX.2 VAE (bf16, as the SDK loads it), each file used checked first."""
    import diffusers
    import torch

    def pinned(path: Path, want: str, what: str) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != want:
            sys.exit(f"{path} has sha256 {digest}; this tool runs only {what}'s ({want})")

    pinned(Path(sdk_dir) / "idol/conditioning.py", CONDITIONING_SHA256, f"idol-sdk {SDK_COMMIT[:12]}")
    if diffusers.__version__ != DIFFUSERS_VERSION:
        sys.exit(f"diffusers is {diffusers.__version__}; this tool runs only {DIFFUSERS_VERSION}")
    root = Path(diffusers.__file__).parent
    for name, want in DIFFUSERS_SHA256.items():
        pinned(root / name, want, f"diffusers {DIFFUSERS_VERSION}")
    weights = Path(vae_dir) / "diffusion_pytorch_model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest not in VAE_SHA256:
        sys.exit(f"{weights} is neither FLUX.2-klein-4B {VAE_REVISION[:12]}'s vae/ nor FLUX.2-dev's")
    sys.path.insert(0, str(Path(sdk_dir)))
    from diffusers.models import AutoencoderKLFlux2
    from idol.conditioning import ImageEncoder

    vae = AutoencoderKLFlux2.from_pretrained(vae_dir, torch_dtype=torch.bfloat16)
    return ImageEncoder, vae.eval().requires_grad_(False), VAE_SHA256[digest]


def tokens64(vae, pixels, noise):
    """ImageEncoder._flux2_tokens's steps in float64 with the SDK's noise: the ruler, and each stage when asked."""
    import torch

    latents = vae.encode(pixels * 2 - 1).latent_dist
    latents = latents.mean + latents.std * noise
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    rows = ((latents - mean) / std).flatten(2).transpose(1, 2)
    return torch.cat([torch.zeros(batch, 5, rows.shape[2], dtype=rows.dtype), rows], dim=1)


def distance(ours, ruler) -> dict:
    """1 - cosine and the worst row's relative L2 error against the ruler, the rows of 128 as tf-idol-vae reads them."""
    a, b = ours.double().reshape(-1, 128), ruler.double().reshape(-1, 128)
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    rows = ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).max()
    return {"one_minus_cos": 1 - cos, "worst_row": float(rows), "max_abs": float((a - b).abs().max())}


def vae_case(args) -> int:
    """The SDK's feature2 for one image on MPS (seeded as IdolPipeline seeds it), its noise, and the float64 ruler."""
    import types

    import diffusers.models.autoencoders.vae as vae_module
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_encoder, vae, source = flux2_vae(args.vae, args.sdk)
    picture = Image.open(args.image).convert("RGB")
    if picture.size != (1024, 1024):
        sys.exit(f"{args.image} is {picture.size}; the encoders see the 1024 x 1024 picture preprocess returns")
    pixels = torch.from_numpy(np.array(picture)).permute(2, 0, 1).float().div(255.0)[None]
    encoder = types.SimpleNamespace(flux2_vae=vae.to(args.device))
    drawn = []
    randn = vae_module.randn_tensor

    def recorded(*a, **k):
        drawn.append(randn(*a, **k))
        return drawn[-1]

    vae_module.randn_tensor = recorded
    times = []
    with torch.no_grad():
        for i in range(args.repeats + 1):
            torch.manual_seed(args.seed)
            mark = time.perf_counter()
            rows = image_encoder._flux2_tokens(encoder, pixels.to(args.device))
            flush(args.device)
            times.append(time.perf_counter() - mark)
            if i == 0:
                sdk_rows, noise = rows[0].cpu(), drawn[0].float().cpu()
    vae_module.randn_tensor = randn
    if any(not torch.equal(d.float().cpu(), noise) for d in drawn):
        sys.exit("the seeded noise differs between encodes")
    stages = {}
    hooks = []
    if args.stages:
        named = [(f"down.{i}", b) for i, b in enumerate(vae.encoder.down_blocks)] + [("mid", vae.encoder.mid_block)]
        named.append(("moments", vae.quant_conv))
        for name, module in named:
            def keep(_m, _i, output, name=name):
                stages[name] = output[0].permute(1, 2, 0).float().contiguous()
            hooks.append(module.register_forward_hook(keep))
    vae64 = vae.to("cpu").double()
    with torch.no_grad():
        mark = time.perf_counter()
        ruler = tokens64(vae64, pixels.double(), noise.double())[0]
        ruler_s = time.perf_counter() - mark
    for h in hooks:
        h.remove()
    tensors = {"image": pixels[0].permute(1, 2, 0).contiguous(), "noise": noise[0].permute(1, 2, 0).contiguous(),
               "sdk": sdk_rows.float().contiguous(), "ruler": ruler.float().contiguous()}
    save_file(tensors, str(out / "vae.safetensors"))
    if stages:
        save_file(stages, str(out / "vae_stages.safetensors"))
    sdk = distance(sdk_rows, ruler)
    meta = {"seed": args.seed, "device": args.device, "torch": torch.__version__, "sdk": SDK_COMMIT,
            "diffusers": DIFFUSERS_VERSION, "vae": source, "vae_revision": VAE_REVISION,
            "sdk_vs_float64": sdk, "float64_s": round(ruler_s, 2), "encode_s": [round(t, 4) for t in times]}
    (out / "vae.json").write_text(json.dumps(meta, indent=1) + "\n")
    warm = sorted(times[1:]) or times
    print(json.dumps(meta | {"median_encode_s": warm[len(warm) // 2]}))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vae")
    for name in ("vae", "sdk", "image", "out_dir"):
        p.add_argument(name)
    p.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=5, help="timed encodes after the first")
    p.add_argument("--stages", action="store_true", help="float64's down blocks, mid block and moments too")
    for name in ("case", "blocks", "exact"):
        p = sub.add_parser(name)
        p.add_argument("weights")
        p.add_argument("sdk")
        p.add_argument("out_dir" if name == "case" else "case_dir")
        p.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
        if name == "case":
            p.add_argument("--features", nargs="+", required=True)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--steps", type=int, default=50)
            p.add_argument("--cfg", type=float, default=3.0)
            p.add_argument("--rescale-t", type=float, default=3.0)
        elif name == "blocks":
            p.add_argument("--step", type=int, default=0)
        else:
            p.add_argument("--steps", type=int, default=50)
    args = ap.parse_args(argv)
    return {"case": case, "blocks": blocks, "exact": exact, "vae": vae_case}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
