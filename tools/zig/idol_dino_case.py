"""idol_case.py dino: DINOv3 ViT-H+/16's feature1 as idol-sdk's ImageEncoder makes it, its ruler and its blocks."""

# timm's ungated weights in the transformers graph the SDK builds from Meta's gated ones; --official proves them one

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

SDK_FILES = {"idol/conditioning.py": "8c9cc50d327df3a5269cda392ef46cf4ed9c1e603f42f46050beda83f355f5ab"}
TRANSFORMERS_VERSION = "5.22.1"
TRANSFORMERS_FILES = {
    "models/dinov3_vit/modeling_dinov3_vit.py": "28c72d5adedc7398a7988d2568dce3b1e399f277e853836cb8bce8c9515be1e3",
    "models/dinov3_vit/configuration_dinov3_vit.py": "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc",
}
TIMM_REVISION = "dd0addc09788111fa893d24799a44744ba022eee"
TIMM_SHA256 = "0d54131c846c91a78ec7641482f32e45b27cb4895f83653c038f305f18c7e61c"
OFFICIAL_REVISION = "c807c9eeea853df70aec4069e6f56b28ddc82acc"
OFFICIAL_SHA256 = "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"
# facebook/dinov3-vith16plus-pretrain-lvd1689m's config.json at OFFICIAL_REVISION, the graph the SDK builds
CONFIG = {"hidden_act": "silu", "hidden_size": 1280, "image_size": 224, "intermediate_size": 5120, "key_bias": False,
          "layer_norm_eps": 1e-05, "layerscale_value": 1.0, "mlp_bias": True, "num_attention_heads": 20,
          "num_channels": 3, "num_hidden_layers": 32, "num_register_tokens": 4, "patch_size": 16,
          "pos_embed_jitter": None, "pos_embed_rescale": 2.0, "pos_embed_shift": None, "proj_bias": True,
          "query_bias": True, "rope_theta": 100.0, "use_gated_mlp": True, "value_bias": True,
          "attention_dropout": 0.0, "drop_path_rate": 0.0}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned(root: Path, files: dict[str, str], what: str) -> None:
    for name, want in files.items():
        digest = sha256(root / name)
        if digest != want:
            sys.exit(f"{root / name} has sha256 {digest}; this tool runs only {what}'s ({want})")


def official_name(k: str) -> str | None:
    """A timm tensor's name in transformers' DINOv3ViTModel (checkpoint names, no live `model.` prefix)."""
    direct = {"cls_token": "embeddings.cls_token", "reg_token": "embeddings.register_tokens",
              "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
              "patch_embed.proj.bias": "embeddings.patch_embeddings.bias", "norm.weight": "norm.weight",
              "norm.bias": "norm.bias"}
    if k in direct:
        return direct[k]
    m = re.fullmatch(r"blocks\.(\d+)\.(.+)", k)
    if not m:
        return None
    i, s = m.groups()
    subs = {"attn.proj.": "attention.o_proj.", "gamma_1": "layer_scale1.lambda1", "gamma_2": "layer_scale2.lambda1",
            "mlp.fc1_g.": "mlp.gate_proj.", "mlp.fc1_x.": "mlp.up_proj.", "mlp.fc2.": "mlp.down_proj.",
            "norm1.": "norm1.", "norm2.": "norm2."}
    for a, b in subs.items():
        if s.startswith(a):
            return f"layer.{i}.{b}{s[len(a):]}"
    return None


def mapped(path: Path):
    """(official name, fp32 tensor) for every timm tensor; a fused qkv weight as its q, k and v thirds."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            x = f.get_tensor(k)
            if k.endswith(".attn.qkv.weight"):
                layer = k.split(".")[1]
                for q, part in zip("qkv", x.chunk(3, dim=0)):
                    yield f"layer.{layer}.attention.{q}_proj.weight", part.contiguous()
                continue
            name = official_name(k)
            if name is None:
                sys.exit(f"{path} holds {k}, which no DINOv3ViTModel tensor takes")
            yield name, x


def build(weights: Path, dtype):
    """DINOv3ViTModel on the official config with timm's tensors; q and v biases and the mask token zero."""
    import torch
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    model = DINOv3ViTModel(DINOv3ViTConfig(**CONFIG))
    model.config._attn_implementation = "sdpa"
    sd = model.state_dict()
    live = lambda k: k if k in sd else "model." + k
    used = set()
    with torch.no_grad():
        for name, x in mapped(weights / "model.safetensors"):
            key = live(name)
            if key not in sd or tuple(sd[key].shape) != tuple(x.shape):
                sys.exit(f"timm's {name} {tuple(x.shape)} has no place in DINOv3ViTModel")
            sd[key].copy_(x)
            used.add(key)
        rest = [k for k in sd if k not in used]
        for k in rest:
            if k != "embeddings.mask_token" and not k.endswith(("attention.q_proj.bias", "attention.v_proj.bias")):
                sys.exit(f"DINOv3ViTModel's {k} has no timm tensor")
            sd[k].zero_()
    # as from_pretrained(dtype=...) does: the parameters in `dtype`, the generated RoPE inv_freq kept in float32
    inv_freq = model.rope_embeddings.inv_freq.clone()
    model = model.to(dtype)
    model.rope_embeddings.inv_freq = inv_freq
    return model.eval().requires_grad_(False)


def identity(weights: Path, official: Path) -> dict:
    """Every timm tensor against its official one, bitwise; the official q and v biases must be zeros."""
    import torch
    from safetensors import safe_open

    digest = sha256(official / "model.safetensors")
    if digest != OFFICIAL_SHA256:
        sys.exit(f"{official}/model.safetensors has sha256 {digest}, not {OFFICIAL_REVISION[:12]}'s")
    equal, compared, used = True, 0, set()
    with safe_open(str(official / "model.safetensors"), framework="pt", device="cpu") as f:
        names = set(f.keys())
        for name, x in mapped(weights / "model.safetensors"):
            y = f.get_tensor(name)
            same = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.view(torch.int32), y.view(torch.int32))
            equal &= same
            compared += 1
            used.add(name)
        zero = all(not f.get_tensor(n).any() for n in names if n.endswith(("q_proj.bias", "v_proj.bias")))
        left = sorted(names - used)
    allowed = all(n == "embeddings.mask_token" or n.endswith(("q_proj.bias", "v_proj.bias")) for n in left)
    return {"tensors_compared": compared, "bitwise_equal": bool(equal), "official_q_v_biases_zero": bool(zero),
            "official_only": len(left), "official_only_allowed": bool(allowed), "official_sha256": digest}


def encoder(model, device: str):
    """The SDK's ImageEncoder holding `model` as its dinov3, without loading the FLUX.2 VAE it never touches here."""
    import torch
    from idol.conditioning import ImageEncoder

    enc = object.__new__(ImageEncoder)
    enc.device = torch.device(device)
    enc.dinov3 = model.to(enc.device)
    return enc


def flush(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()


def dino(args, sdk_root: str) -> int:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from safetensors.torch import load_file, save_file

    pinned(Path(sdk_root), SDK_FILES, "idol-sdk 4702a7f00850's")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        sys.exit(f"transformers {transformers.__version__}: this tool runs only {TRANSFORMERS_VERSION}")
    pinned(Path(transformers.__file__).parent, TRANSFORMERS_FILES, f"transformers {TRANSFORMERS_VERSION}")
    sys.path.insert(0, sdk_root)
    weights, out = Path(args.weights), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256(weights / "model.safetensors")
    if digest != TIMM_SHA256:
        sys.exit(f"{weights}/model.safetensors has sha256 {digest}; not timm's {TIMM_REVISION[:12]}")
    image = Image.open(args.image).convert("RGB")
    if image.size != (1024, 1024):
        sys.exit(f"{args.image} is {image.size}; the encoder takes the SDK's 1024 x 1024 preprocessed image")
    # IdolPipeline.__call__'s pixels: [1, 3, 1024, 1024] in [0, 1]
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0)[None]
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.2)

    ruler = build(weights, torch.float32)
    model = build(weights, torch.bfloat16)
    enc = encoder(model, args.device)
    seen, rope = {}, {}

    def keep(i):
        def hook(_m, a, kw):
            seen[f"block.{i}"] = a[0][0].detach().cpu().contiguous()
            if i == 0:
                cos, sin = kw["position_embeddings"]
                rope["cos"], rope["sin"] = cos.cpu().contiguous(), sin.cpu().contiguous()
        return hook

    layers = enc.dinov3.model.layer
    hooks = [m.register_forward_pre_hook(keep(i), with_kwargs=True) for i, m in enumerate(layers)]
    hooks.append(enc.dinov3.norm.register_forward_pre_hook(lambda _m, a: seen.__setitem__("block.32", a[0][0].cpu())))
    hooks.append(enc.dinov3.norm.register_forward_hook(lambda _m, _a, o: seen.__setitem__("hidden", o[0].cpu())))
    times = []
    with torch.no_grad():
        for _ in range(2):
            mark = time.perf_counter()
            feature1 = enc._dinov3_tokens(pixels.to(args.device))
            flush(args.device)
            times.append(time.perf_counter() - mark)
        for h in hooks:
            h.remove()
        feature1 = feature1[0].cpu().contiguous()
        enc.dinov3.to("cpu")
        meta = {"sdk_encode_s": [round(t, 4) for t in times], "device": args.device, "torch": torch.__version__,
                "transformers": transformers.__version__, "timm_revision": TIMM_REVISION, "timm_sha256": digest,
                "image_sha256": sha256(Path(args.image)), "attention": enc.dinov3.config._attn_implementation}
        # the ruler: the same graph in float32 on the CPU (the M5's fp32 GEMM on its tensor units is ~fp16-accurate)
        mark = time.perf_counter()
        f32 = encoder(ruler, "cpu")
        feature1_f32 = f32._dinov3_tokens(pixels)[0].contiguous()
        meta["ruler_s"] = round(time.perf_counter() - mark, 2)
        tensors = {"pixels": pixels[0].contiguous(), "feature1": feature1, "feature1_f32": feature1_f32,
                   "hidden": seen["hidden"].contiguous(), "cos": rope["cos"], "sin": rope["sin"]}
        exact = {}
        if not args.no_blocks:
            # each block in float32 on the bf16 rows torch fed it, with float32 cos and sin
            norm = ((pixels - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
                    / torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
            exact["embed"] = ruler.embeddings(norm)[0].contiguous()
            pos = ruler.rope_embeddings(norm)
            for i, layer in enumerate(ruler.model.layer):
                exact[f"block.{i + 1}"] = layer(seen[f"block.{i}"][None].float(), position_embeddings=pos)[0].contiguous()
            last = ruler.norm(seen["block.32"][None].float())
            exact["feature1"] = F.layer_norm(last, last.shape[-1:])[0].contiguous()
            save_file({k: v for k, v in seen.items() if k.startswith("block.")}, str(out / "dino_blocks.safetensors"))
            save_file(exact, str(out / "dino_blocks_f32.safetensors"))
    save_file(tensors, str(out / "dino.safetensors"))
    d = (feature1.double() - feature1_f32.double())
    meta["sdk_vs_ruler"] = {"max_abs": float(d.abs().max()), "cosine": float(F.cosine_similarity(
        feature1.double().flatten(), feature1_f32.double().flatten(), dim=0)),
        "worst_row": float((d.norm(dim=-1) / feature1_f32.double().norm(dim=-1)).max())}
    if args.sdk_features:
        theirs = load_file(args.sdk_features)["feature1"][0]
        meta["sdk_features"] = {"sha256": sha256(Path(args.sdk_features)), "bitwise_equal": bool(torch.equal(theirs, feature1)),
                                "max_abs": float((theirs - feature1).abs().max())}
    if args.official:
        meta["identity"] = identity(weights, Path(args.official))
        from transformers import AutoModel

        off = AutoModel.from_pretrained(args.official, dtype=torch.bfloat16).eval().requires_grad_(False)
        with torch.no_grad():
            theirs = encoder(off, args.device)._dinov3_tokens(pixels.to(args.device))[0].cpu()
        meta["identity"]["feature1_bitwise_equal"] = bool(torch.equal(theirs, feature1))
        meta["identity"]["feature1_max_abs"] = float((theirs - feature1).abs().max())
    (out / "dino.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    ok = meta.get("identity", {}).get("feature1_bitwise_equal", True) and meta.get("identity", {}).get("bitwise_equal", True)
    return 0 if ok else 1
