#!/usr/bin/env python3
"""tf-idol-birefnet's case: the SDK's preprocess of a photo, a float64 CPU ruler, and with --stages each stage's."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idol_case import SDK_COMMIT, flush, pin  # noqa: E402

USAGE = """\
  idol_birefnet_case.py WEIGHTS SDK IMAGE OUT [--mode synthetic|real] [--crop-margin X] [--stages] [--repeats N]
WEIGHTS is a speridlabs/idol snapshot (birefnet.safetensors, config.json); SDK an idol-sdk checkout at the pinned
commit; IMAGE a PNG, JPEG or WebP photo. Writes OUT/birefnet.safetensors (the resized photo, BiRefNet's input, logits
and alpha, torch's and float64's, and both preprocessed pictures), OUT/birefnet.json and, with --stages,
OUT/birefnet_stages.safetensors.
"""

PREPROCESS_SHA256 = {
    "idol/models/birefnet.py": "7c8e5e1021978225740bc9e0a94beef2fba1545a2ffc739235fab6aadc244068",
    "idol/preprocess.py": "b6ef41020760b1d1ffbee2f82a2eb22752797d8c474f7b3eaa25f2a6413a1ffe",
    "idol/pipeline.py": "091fc0fe7d8f66f5f9f63795a91f9b654e12cf879d94bf03624f39c360ddd4a1",
}
# speridlabs/idol's birefnet.safetensors: ZhengPeng7/BiRefNet e2bf8e4460fc's tensors in fp16, less 12 training-only ones
BIREFNET_SHA256 = "44bbc97b041f879975d0d6d538b8e66a25eb04d1f49aab40697544cdd97eaa8e"
PILLOW_VERSION = "13.0.0"


def rows(t):
    """[1, C, H, W] (or [1, L, C] tokens) as float32 rows [H * W, C], the layout tf-idol-birefnet keeps."""
    t = t.detach().float().cpu()
    if t.dim() == 4:
        return t[0].permute(1, 2, 0).reshape(-1, t.shape[1]).contiguous()
    return t[0].contiguous()


def distance(ours, ruler, width: int) -> dict:
    """1 - cosine and the worst row's relative L2 error against the ruler, rows of `width` as tf-idol-birefnet reads."""
    a, b = ours.double().reshape(-1, width), ruler.double().reshape(-1, width)
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    worst = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
    return {"one_minus_cos": 1 - cos, "worst_row": float(worst.max()), "worst_at": int(worst.argmax()),
            "max_abs": float((a - b).abs().max())}


def differ(a, b) -> dict:
    d = (a.astype("int32") - b.astype("int32"))
    d = abs(d).max(axis=-1) if d.ndim == 3 else abs(d)
    return {"pixels": int((d > 0).sum()), "max": int(d.max())}


def birefnet(weights: str, sdk_dir: str, device: str):
    """The SDK's BiRefNet with the snapshot's weights, strictly loaded as IdolPipeline loads it (fp16 on CUDA only)."""
    import PIL
    import torch
    from safetensors.torch import load_file

    pin(Path(sdk_dir), PREPROCESS_SHA256)
    if PIL.__version__ != PILLOW_VERSION:
        sys.exit(f"Pillow is {PIL.__version__}; this tool runs only {PILLOW_VERSION}")
    path = Path(weights) / "birefnet.safetensors"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BIREFNET_SHA256:
        sys.exit(f"{path} has sha256 {digest}; this tool runs only speridlabs/idol's ({BIREFNET_SHA256})")
    from idol.models import BiRefNet

    model = BiRefNet()
    model.load_state_dict(load_file(str(path)), strict=True)
    model = model.eval().requires_grad_(False).to(torch.device(device))
    if device == "cuda":
        model.half()
    return model


class Recorder:
    """The background remover preprocess_image calls: the SDK's own, its input and output kept."""

    def __init__(self, model, device):
        self.model, self.device, self.calls = model, device, []

    def remove_background(self, image):
        import numpy as np

        flush(self.device)
        mark = time.perf_counter()
        out = self.model.remove_background(image)
        flush(self.device)
        # preprocess erodes the alpha in place next, so it is copied now
        self.calls.append({"resized": image, "alpha": np.array(out.getchannel(3)), "seconds": time.perf_counter() - mark})
        return out


class Given:
    """A background remover that puts a given alpha on the photo: preprocess's later steps on that alpha."""

    def __init__(self, alpha):
        self.alpha = alpha

    def remove_background(self, image):
        from PIL import Image

        rgba = image.copy()
        rgba.putalpha(Image.fromarray(self.alpha, mode="L"))
        return rgba


def ruler_input(resized):
    """remove_background's tensor in float64 from the resized photo: bilinear (align_corners) to 1024, ImageNet."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.array(resized, dtype=np.float64) / 255.0).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(1024, 1024), mode="bilinear", align_corners=True)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float64).view(1, 3, 1, 1)
    return (t - mean) / std


def to_alpha(logits, size):
    """remove_background's alpha from logits [1, 1, 1024, 1024]: sigmoid, bilinear to (H, W), clamp * 255, truncated."""
    import torch
    import torch.nn.functional as F

    a = F.interpolate(torch.sigmoid(logits), size=size, mode="bilinear", align_corners=True)[0, 0]
    return (a.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def stage_hooks(model, keep):
    """Hooks keeping each stage's torch input and output: both passes' layers, the encoder, squeeze and decoder."""
    hooks, seen = [], {"bb": 0}

    def on(name, fn):
        def hook(_m, args, out):
            fn(name, args, out)
        return hook

    def layer(name, args, out):
        p = seen["bb"]
        x, H, W = args
        keep[f"{name}.{p}.in"] = rows(x)
        keep[f"{name}.{p}.out"] = rows(out[0])
        keep[f"{name}.{p}.down"] = rows(out[3])
        keep[f"{name}.{p}.hw"] = (int(H), int(W))
        if name.endswith(".3"):
            seen["bb"] += 1

    for i, lay in enumerate(model.bb.layers):
        hooks.append(lay.register_forward_hook(on(f"layer.{i}", layer)))

    def block(name, args, out):
        keep[f"{name}.in"], keep[f"{name}.out"] = rows(args[0]), rows(out)

    hooks.append(model.squeeze_module.register_forward_hook(on("squeeze", block)))
    for n in (4, 3, 2, 1):
        hooks.append(getattr(model.decoder, f"decoder_block{n}").register_forward_hook(on(f"dec{n}", block)))

    def decoder(name, args, out):
        for k, v in zip(("x", "x1", "x2", "x3", "x4"), args):
            keep[f"decoder.in.{k}"] = rows(v)
        keep["decoder.out"] = rows(out)

    hooks.append(model.decoder.register_forward_hook(on("decoder", decoder)))
    forward_enc = model._forward_enc

    def enc(x):
        out = forward_enc(x)
        keep["enc.in"] = rows(x)
        for k, v in zip(("x1", "x2", "x3", "x4"), out):
            keep[f"enc.out.{k}"] = rows(v)
        return out

    model._forward_enc = enc
    return hooks, forward_enc


def nchw(r, h, w):
    """Rows [H * W, C] back to [1, C, H, W] float64."""
    return r.double().reshape(h, w, -1).permute(2, 0, 1)[None].contiguous()


def stage_rulers(m64, keep, gap) -> tuple[dict, dict]:
    """Each stage in float64 on torch's own input; torch's distance to it per stage."""
    import torch

    out, dist = {}, {}
    with torch.no_grad():
        for p in (0, 1):
            for i in range(4):
                key = f"layer.{i}.{p}"
                H, W = keep.pop(f"{key}.hw")
                x_out, _, _, x_down, _, _ = m64.bb.layers[i](keep[f"{key}.in"].double()[None], H, W)
                out[f"{key}.out"], out[f"{key}.down"] = rows(x_out), rows(x_down)
        side = 1024
        x = nchw(keep["enc.in"], side, side)
        for k, v in zip(("x1", "x2", "x3", "x4"), m64._forward_enc(x)):
            out[f"enc.out.{k}"] = rows(v)
        s4 = 32
        out["squeeze.out"] = rows(m64.squeeze_module(nchw(keep["squeeze.in"], s4, s4)))
        for n, s in ((4, 32), (3, 64), (2, 128), (1, 256)):
            out[f"dec{n}.out"] = rows(getattr(m64.decoder, f"decoder_block{n}")(nchw(keep[f"dec{n}.in"], s, s)))
        ins = [nchw(keep[f"decoder.in.{k}"], s, s) for k, s in (("x", 1024), ("x1", 256), ("x2", 128), ("x3", 64),
                                                                   ("x4", 32))]
        out["decoder.out"] = rows(m64.decoder(*ins))
    for k, v in out.items():
        dist[k] = distance(keep[k], v, gap(k, v))
    return out, dist


def width_of(key, v):
    """A stage's row: a pixel's channels; the logits' row: an image row of 1,024."""
    return 1024 if key == "decoder.out" else v.shape[-1]


def case(args) -> int:
    import copy

    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = birefnet(args.weights, args.sdk, args.device)
    from idol.preprocess import CROP_MARGIN, has_real_alpha, preprocess_image

    config = json.loads((Path(args.weights) / "config.json").read_text())
    size = config["image_size"]
    margin = args.crop_margin if args.crop_margin is not None else CROP_MARGIN[args.mode]
    photo = Image.open(args.image)
    meta = {"image": Path(args.image).name, "format": photo.format, "mode": photo.mode, "size": list(photo.size),
            "preprocess_mode": args.mode, "crop_margin": margin, "device": args.device, "torch": torch.__version__,
            "sdk": SDK_COMMIT, "pillow": PILLOW_VERSION}
    keep, hooks, forward_enc = {}, [], None
    logits = []
    hooks.append(model.decoder.register_forward_hook(lambda _m, _a, o: logits.append(o.detach().float().cpu())))
    inputs = []
    # remove_background calls forward directly, so the input is the backbone's first
    hooks.append(model.bb.register_forward_pre_hook(lambda _m, a: inputs.append(a[0].detach().float().cpu())))
    if args.stages:
        more, forward_enc = stage_hooks(model, keep)
        hooks += more
    rec = Recorder(model, args.device)
    with torch.no_grad():
        mark = time.perf_counter()
        picture = preprocess_image(Image.open(args.image), background_remover=rec, image_size=size,
                                   crop_margin=margin)
        flush(args.device)
        first_s = time.perf_counter() - mark
    for h in hooks:
        h.remove()
    if forward_enc is not None:
        model._forward_enc = forward_enc
    times = []
    with torch.no_grad():
        for _ in range(args.repeats):
            mark = time.perf_counter()
            again = preprocess_image(Image.open(args.image), background_remover=Recorder(model, args.device),
                                     image_size=size, crop_margin=margin)
            flush(args.device)
            times.append(time.perf_counter() - mark)
            if not np.array_equal(np.array(again), np.array(picture)):
                sys.exit("the SDK's preprocessed picture differs between runs")
    tensors = {"picture": torch.from_numpy(np.array(picture))}
    meta |= {"has_real_alpha": not rec.calls, "preprocess_s": round(first_s, 4),
             "preprocess_repeat_s": [round(t, 4) for t in times]}
    if rec.calls:
        call = rec.calls[0]
        resized = call["resized"]
        alpha = call["alpha"]
        sdk_logits = logits[0]
        tensors |= {"resized": torch.from_numpy(np.array(resized)), "input": rows(inputs[0]),
                    "logits": sdk_logits[0, 0].contiguous(), "alpha": torch.from_numpy(alpha)}
        m64 = copy.deepcopy(model).to("cpu").double()
        with torch.no_grad():
            mark = time.perf_counter()
            x64 = ruler_input(resized)
            x1, x2, x3, x4 = m64._forward_enc(x64)
            ruler = m64.decoder(x64, x1, x2, x3, m64.squeeze_module(x4))
            ruler_s = time.perf_counter() - mark
        # the ruler's logits through torch's own fp32 alpha steps: float64's sigmoid would round differently
        ruler_alpha = to_alpha(ruler.float().to(args.device), alpha.shape)
        ruler_picture = preprocess_image(Image.open(args.image), background_remover=Given(ruler_alpha),
                                         image_size=size, crop_margin=margin)
        tensors |= {"input_ruler": rows(x64), "logits_ruler": ruler[0, 0].float().contiguous(),
                    "alpha_ruler": torch.from_numpy(ruler_alpha),
                    "picture_ruler": torch.from_numpy(np.array(ruler_picture))}
        sdk_alpha_again = to_alpha(sdk_logits.to(args.device), alpha.shape)
        meta |= {"resized_size": list(resized.size), "remove_background_s": round(call["seconds"], 4),
                 "float64_s": round(ruler_s, 2),
                 "input_vs_ruler_max_abs": float((rows(inputs[0]).double() - rows(x64)).abs().max()),
                 "logits_vs_ruler": distance(sdk_logits[0, 0], ruler[0, 0], 1024),
                 "alpha_vs_ruler": differ(alpha, ruler_alpha), "picture_vs_ruler": differ(np.array(picture),
                                                                                         np.array(ruler_picture)),
                 "alpha_from_recorded_logits_equal": bool(np.array_equal(sdk_alpha_again, alpha))}
        if args.stages:
            ruled, dist = stage_rulers(m64, keep, width_of)
            meta["stages_vs_float64"] = dist
            stages = {k: v for k, v in keep.items()} | {f"ruler.{k}": v for k, v in ruled.items()}
            save_file({k: v.float().contiguous() for k, v in stages.items()}, str(out / "birefnet_stages.safetensors"))
    else:
        meta["real_alpha"] = bool(has_real_alpha(Image.open(args.image)))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "birefnet.safetensors"))
    (out / "birefnet.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("weights", "sdk", "image", "out_dir"):
        ap.add_argument(name)
    ap.add_argument("--mode", default="synthetic", choices=("synthetic", "real"))
    ap.add_argument("--crop-margin", type=float)
    ap.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    ap.add_argument("--stages", action="store_true", help="each stage's torch rows and float64 on them")
    ap.add_argument("--repeats", type=int, default=3, help="timed preprocesses after the recorded one")
    return case(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
