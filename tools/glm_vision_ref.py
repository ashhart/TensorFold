#!/usr/bin/env python3
"""GLM-5.3-Flash's vision tower as mlx-vlm's glm5_next (vision.py, processing.py) defines it, in torch on the CPU
(fp32 math on the checkpoint's weights: bf16, or FP4-Local's MXFP8 and NVFP4 projections as the values their codes and
scales hold): the reference the native tower is checked against.

usage: glm_vision_ref.py [--bf16] <model dir> <out dir> <image or video frames dir>...
  --bf16: the same tower in bfloat16 (tf-cuda-test glm-vision bounds the native tower's error by this run's)
  an image: its patches (the GLM processor's canvas, padding, normalization and merge order) -> features
  a folder of frame_*.png: a video's sampled frames (already sampled) -> one [t, h, w] grid
writes <out>/case<i>/{patches.f32, grid.json, features.f32, block0.f32}
"""
import json, math, struct, sys
from pathlib import Path

import numpy as np
import torch

DT = torch.float32
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
PATCH, MERGE, TEMPORAL = 14, 2, 2


def smart_resize(num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels):
    ppt = temporal_factor * factor ** 2
    min_pixels *= ppt
    max_pixels *= ppt

    def align(v, f):
        return math.ceil(v / f) * f

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)

    def fit():
        low, high, best = 1, height, (factor, factor)
        while low <= high:
            ch = (low + high) // 2
            cw = max(1, math.floor(width * ch / height))
            cand = (align(ch, factor), align(cw, factor))
            if aligned_frames * cand[0] * cand[1] <= max_pixels:
                best, low = cand, ch + 1
            else:
                high = ch - 1
        return best

    ah, aw = align(height, factor), align(width, factor)
    pixels = aligned_frames * ah * aw
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        ah, aw = align(max(1, math.ceil(height * scale)), factor), align(max(1, math.ceil(width * scale)), factor)
        pixels = aligned_frames * ah * aw
    if pixels > max_pixels:
        ah, aw = fit()
    return ah, aw


def geometry(num_frames, height, width, min_tokens, max_tokens):
    factor = PATCH * MERGE
    ppt = TEMPORAL * factor ** 2
    th, tw = smart_resize(num_frames, height, width, TEMPORAL, factor, min_tokens, max_tokens)
    scale = min(th / height, tw / width)
    if num_frames * height * width >= min_tokens * ppt:
        scale = min(1.0, scale)
    return th, tw, max(1, min(th, math.floor(height * scale))), max(1, min(tw, math.floor(width * scale)))


def frames_to_patches(frames, min_tokens, max_tokens):
    """frames: list of RGB PIL images (1 for an image) -> (patches [n, 1176] fp32, grid [t, h, w])."""
    n = len(frames)
    w0, h0 = frames[0].size
    th, tw, ch, cw = geometry(TEMPORAL if n == 1 else n, h0, w0, min_tokens, max_tokens)
    arr = []
    for f in frames:
        f = f if f.size == (cw, ch) else f.resize((cw, ch), Image.BICUBIC)
        a = np.asarray(f, np.uint8).transpose(2, 0, 1)
        a = np.pad(a, ((0, 0), (0, th - ch), (0, tw - cw))).astype(np.float32) / np.float32(255.0)
        arr.append((a - MEAN[:, None, None]) / STD[:, None, None])
    gh, gw = th // PATCH, tw // PATCH
    if n == 1:  # an image: its one frame twice (the temporal patch)
        a = arr[0]
        p = a.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).transpose(1, 4, 2, 5, 0, 3, 6)
        p = np.broadcast_to(p[:, :, :, :, :, None], (*p.shape[:5], TEMPORAL, *p.shape[5:]))
        return np.ascontiguousarray(p.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)), [1, gh, gw]
    f = np.stack(arr)
    if n % TEMPORAL:
        f = np.concatenate([f, f[-1:]])
    gt = f.shape[0] // TEMPORAL
    p = f.reshape(gt, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(p.reshape(gt * gh * gw, 3 * TEMPORAL * PATCH * PATCH)), [gt, gh, gw]


DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.uint8}
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant(w, scale, scale2):
    """A projection stored MXFP8 (e4m3, a UE8M0 byte a 32 inputs) or NVFP4 (two e2m1 codes a byte, the low nibble
    first, an e4m3 scale a 16 inputs and a global scale) as the float32 values its codes and scales hold."""
    if scale2 is None:
        n, k = w.shape
        return (w.view(torch.float8_e4m3fn).float().reshape(n, k // 32, 32) * torch.exp2(scale.float() - 127)[..., None]).reshape(n, k)
    n, half = w.shape
    codes = torch.stack([w & 15, w >> 4], -1).reshape(n, 2 * half).long()
    s = scale.view(torch.float8_e4m3fn).float()
    return (E2M1[codes].reshape(n, -1, 16) * s[..., None]).reshape(n, 2 * half) * scale2.float()


def load_visual(model_dir):
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for k, f in idx.items() if k.startswith("model.visual.")})
    out = {}
    for name in files:
        raw = (model_dir / name).read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        head = json.loads(raw[8:8 + n])
        for k, v in head.items():
            if not k.startswith("model.visual."):
                continue
            a, b = v["data_offsets"]
            t = torch.frombuffer(bytearray(raw[8 + n + a:8 + n + b]), dtype=DTYPES[v["dtype"]]).reshape(v["shape"])
            out[k[len("model.visual."):]] = t
    for k in [k for k in out if k.endswith(".weight_scale")]:  # FP4-Local's MXFP8 and W4A16_NVFP4 projections
        m = k[: -len(".weight_scale")]
        out[m + ".weight"] = dequant(out[m + ".weight"], out.pop(k), out.pop(m + ".weight_scale_2", None))
        out.pop(m + ".input_scale", None)
    return {k: v.float() for k, v in out.items()}


def rms(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def swiglu(g, u, limit=10.0):
    g = torch.clamp(g, max=limit)
    u = torch.clamp(u, -limit, limit)
    return torch.nn.functional.silu(g) * u


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def tower(W, patches, grid, keep=None):
    W = {k: v.to(DT) for k, v in W.items()}
    x = torch.from_numpy(patches).to(DT) @ W["patch_embed.proj.weight"].reshape(1024, -1).T + W["patch_embed.proj.bias"]
    if keep is not None:
        keep["embed"] = x.clone()
    t, h, w = grid
    # rotary: (row, col) in merge-block order, repeated for each temporal group
    hi = torch.arange(h)[:, None].expand(h, w).reshape(h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 1, 3).flatten()
    wi = torch.arange(w)[None, :].expand(h, w).reshape(h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 1, 3).flatten()
    pos = torch.stack([hi, wi], -1).repeat(t, 1)
    dim = 32
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    table = torch.outer(torch.arange(max(h, w), dtype=torch.float32), inv)
    rot = table[pos].reshape(pos.shape[0], -1)
    emb = torch.cat([rot, rot], -1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]  # fp32 (the rotary runs in fp32 either way)
    seg = h * w
    for i in range(24):
        p = f"blocks.{i}."
        y = rms(x, W[p + "norm1.weight"])
        qkv = (y @ W[p + "attn.qkv.weight"].T + W[p + "attn.qkv.bias"]).reshape(-1, 3, 16, 64)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = rms(q, W[p + "attn.q_norm.weight"]), rms(k, W[p + "attn.k_norm.weight"])
        qf, kf = q.float(), k.float()
        q, k = (qf * cos + rotate_half(qf) * sin).to(DT), (kf * cos + rotate_half(kf) * sin).to(DT)
        outs = []
        for s in range(t):
            sl = slice(s * seg, (s + 1) * seg)
            qs, ks, vs = q[sl].transpose(0, 1), k[sl].transpose(0, 1), v[sl].transpose(0, 1)
            a = (torch.softmax((qs.float() @ ks.float().transpose(1, 2)) * 0.125, -1) @ vs.float()).to(DT)
            outs.append(a.transpose(0, 1).reshape(seg, 1024))
        x = x + torch.cat(outs) @ W[p + "attn.proj.weight"].T + W[p + "attn.proj.bias"]
        y = rms(x, W[p + "norm2.weight"])
        g = y @ W[p + "mlp.gate_proj.weight"].T + W[p + "mlp.gate_proj.bias"]
        u = y @ W[p + "mlp.up_proj.weight"].T + W[p + "mlp.up_proj.bias"]
        x = x + swiglu(g, u) @ W[p + "mlp.down_proj.weight"].T + W[p + "mlp.down_proj.bias"]
        if keep is not None and i == 0:
            keep["block0"] = x.clone()
        if keep is not None:
            keep.setdefault("all", []).append(x.clone())
    if keep is not None:
        keep["blocks"] = x.clone()
    x = rms(x, W["post_layernorm.weight"])
    # the 2x2 conv over each merge block's four patches (in row, col order)
    x = x.reshape(-1, MERGE, MERGE, 1024).permute(0, 3, 1, 2)
    x = torch.nn.functional.conv2d(x, W["downsample.weight"], W["downsample.bias"], stride=MERGE).reshape(-1, 4096)
    if keep is not None:
        keep["down"] = x.clone()
    x = x @ W["merger.proj.weight"].T
    x = torch.nn.functional.layer_norm(x, (4096,), W["merger.post_projection_norm.weight"], W["merger.post_projection_norm.bias"], 1e-5)
    x = torch.nn.functional.gelu(x)
    return swiglu(x @ W["merger.gate_proj.weight"].T, x @ W["merger.up_proj.weight"].T) @ W["merger.down_proj.weight"].T


def main():
    global DT
    if sys.argv[1] == "--bf16":  # mlx-vlm's arithmetic: bf16 tensors (each op rounds), its products accumulating in fp32
        DT = torch.bfloat16
        del sys.argv[1]
    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    pp = json.loads((model / "processor_config.json").read_text())["image_processor"]
    W = load_visual(model)
    torch.set_num_threads(16)
    for i, src in enumerate(sys.argv[3:]):
        src = Path(src)
        if src.is_dir():
            frames = [Image.open(f).convert("RGB") for f in sorted(src.glob("frame_*.png"))]
            mn, mx = pp.get("min_image_tokens", 16), 240000
        else:
            frames = [Image.open(src).convert("RGB")]
            mn, mx = pp.get("min_image_tokens", 16), pp.get("max_image_tokens", 8000)
        patches, grid = frames_to_patches(frames, mn, mx)
        keep = {}
        with torch.no_grad():
            feats = tower(W, patches, grid, keep)
        d = out / f"case{i}"
        d.mkdir(parents=True, exist_ok=True)
        patches.astype(np.float32).tofile(d / "patches.f32")
        for name, v in (("features", feats), ("block0", keep["block0"]), ("embed", keep["embed"]), ("blocks", keep["blocks"]), ("down", keep["down"])):
            v.float().numpy().astype(np.float32).tofile(d / f"{name}.f32")
        torch.stack(keep["all"]).float().numpy().astype(np.float32).tofile(d / "all.f32")
        (d / "grid.json").write_text(json.dumps({"grid": grid, "source": src.name, "tokens": int(feats.shape[0])}))
        print(f"case{i}: {src.name} grid {grid} -> {feats.shape[0]} tokens, |features| rms {feats.float().pow(2).mean().sqrt():.4f}", flush=True)


if __name__ == "__main__":
    main()
