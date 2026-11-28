"""idol_case.py vae: the SDK's FLUX.2 VAE rows for a preprocessed picture, its posterior noise and a float64 ruler."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from idol_case import SDK_COMMIT, flush

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
