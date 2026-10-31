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
  decode WEIGHTS SDK CASE OUT [--gaussians N] [--ruler] [--time-only]
                                           the SDK's decode of CASE's final sample: each octree level's parents,
                                           probabilities, draws and counts, the anchors, the Gaussian decoder's raw
                                           rows, the activated Gaussians, the PLY and the camera (OUT/decode.*)
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
DECODE_SHA256 = {
    "idol/models/decoder.py": "ffaf304f0924360782f902a126efac72084ff907b23db8721b4937f0c3175bbd",
    "idol/gaussian.py": "f181ae2eaf5e44fee3055bd032d57b5545ad2e38e4d2315ae3c21a2fb4f85217",
    "idol/camera.py": "5576043593454e492403b821d1e355433e71f515a2d4406a51c53c30fc097935",
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


def pin(root: Path, files: dict) -> None:
    """Exits unless each file has the pinned commit's sha256, then puts the checkout on the import path."""
    for name, want in files.items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != want:
            sys.exit(f"{root / name} has sha256 {digest}; this tool runs only idol-sdk {SDK_COMMIT[:12]}'s ({want})")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def sdk(path: str):
    """The SDK's denoiser and sampler, imported once each file used has the pinned commit's sha256."""
    pin(Path(path), SDK_SHA256)
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


DECODER_ARGS = {"model_channels": 1024, "cond_channels": 16, "num_blocks": 4, "num_heads": 16, "mlp_ratio": 4}
GS_ARGS = {"in_channels": 3, "model_channels": 1024, "cond_channels": 16, "num_blocks": 16, "num_heads": 16,
           "mlp_ratio": 4}


def decoders(weights: str, sdk_dir: str, device: str):
    """The SDK's octree and Gaussian decoders with the snapshot's weights, strictly loaded, as IdolPipeline does."""
    import torch
    from safetensors.torch import load_file

    pin(Path(sdk_dir), SDK_SHA256 | DECODE_SHA256)
    from idol.models.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
    config = json.loads((Path(weights) / "config.json").read_text())
    d, g = config["decoder"], config["decoder_gs"]
    gs_args = {k: v for k, v in g["args"].items() if k != "representation_config"}
    if (d["name"], d["args"], g["name"], gs_args, config["max_voxel_level"]) != (
            "OctreeProbabilityFixedlenDecoder", DECODER_ARGS, "GaussianFixedlenDecoder", GS_ARGS, 8):
        sys.exit(f"{weights}/config.json is not speridlabs/idol {WEIGHTS_REVISION[:12]}'s decoders")
    built = []
    for cls, entry, name in ((OctreeProbabilityFixedlenDecoder, d, "decoder"), (GaussianFixedlenDecoder, g, "decoder_gs")):
        model = cls(**entry["args"])
        model.load_state_dict(load_file(str(Path(weights) / f"{name}.safetensors")), strict=True)
        built.append(model.eval().requires_grad_(False).to(torch.device(device)))
    return built[0], built[1], config


def replay_counts(original, probs, counts, draws):
    """systematic_sample again on the CPU, its torch.rand calls answered from the recorded per-row draws."""
    import torch

    c = counts.reshape(-1)
    unique_n, inv = c.unique(sorted=False, return_inverse=True)
    queue = [draws[(inv == i).nonzero(as_tuple=False).squeeze(1)].view(-1, 1) for i, n in enumerate(unique_n.tolist())
             if n != 0]
    real = torch.rand
    torch.rand = lambda *a, **k: queue.pop(0)
    try:
        return original(probs[None].cpu(), c[None].cpu())[0]
    finally:
        torch.rand = real


NUDGES = (1, 2, 3)


def nudged(model, seed: int):
    """A pre-hook on the first block: 1% of its fp16 input values one ulp off, seeded (the SDK's own sensitivity)."""
    import torch

    def hook(_module, args):
        x = args[0]
        g = torch.Generator().manual_seed(seed)
        h = x.cpu()
        pick = torch.rand(h.shape, generator=g) < 0.01
        up = torch.rand(h.shape, generator=g) < 0.5
        moved = torch.nextafter(h, torch.where(up, torch.full_like(h, float("inf")), torch.full_like(h, -float("inf"))))
        return (torch.where(pick, moved, h).to(x.device),) + tuple(args[1:])

    return model.blocks[0].register_forward_pre_hook(hook)


def sensitivity(octree, gs, xs, anchors, latent, device) -> dict:
    """Each decoder again in fp16 on the SDK's inputs with nudged first-block inputs, once a seed in NUDGES."""
    import torch

    found = {}
    z = latent[None].to(device)
    with torch.no_grad():
        for seed in NUDGES:
            for model in (octree, gs):
                hook = nudged(model, seed)
                if model is octree:
                    for lv, (x, res) in enumerate(xs, start=1):
                        lvl = torch.full((1,), res, dtype=torch.long, device=device)
                        found[f"probs16n{seed}.{lv}"] = torch.softmax(octree(x[None].to(device), lvl, z), -1)[0].float().cpu()
                else:
                    found[f"raw16n{seed}"] = gs(anchors.to(device), z)[0].float().cpu()
                hook.remove()
    return found


def rulers(octree, gs, xs, anchors, latent, device, dev) -> dict:
    """Both decoders in float32, again with the last rows rounded to fp16 once, and levels 1 to 3 block by block."""
    # rounded once: the floor no fp16 stream gets under; block by block: the SDK's rows, each block in float32
    import copy

    import torch

    found = {}
    z = latent[None]
    o32 = copy.deepcopy(octree).to(dev).float()
    o32.dtype = torch.float32
    g32 = copy.deepcopy(gs).to(dev).float()
    g32.dtype = torch.float32
    round16 = lambda _m, _i, o: o.half().float()
    with torch.no_grad():
        for model, key in ((o32, "probs32"), (g32, "raw32")):
            for suffix in ("", "h"):
                hook = model.blocks[-1].register_forward_hook(round16) if suffix else None
                if model is o32:
                    for lv, (x, res) in enumerate(xs, start=1):
                        logits = o32(x[None].to(dev), torch.full((1,), res, dtype=torch.long, device=dev), z.to(dev))
                        found[f"{key}{suffix}.{lv}"] = torch.softmax(logits, dim=-1)[0].float().cpu()
                else:
                    found[f"{key}{suffix}"] = g32(anchors.float().to(dev), z.to(dev))[0].float().cpu()
                if hook:
                    hook.remove()
        for lv, (x, res) in enumerate(xs[:3], start=1):
            seen = []
            hooks = [b.register_forward_hook(lambda _m, i, o: seen.append((i[0], o))) for b in octree.blocks]
            lvl = torch.full((1,), res, dtype=torch.long, device=device)
            octree(x[None].to(device), lvl, z.to(device))
            for h in hooks:
                h.remove()
            emb = o32.adaLN_modulation(o32.l_embedder(lvl.to(dev).float()))
            found[f"stem.{lv}"] = seen[0][0][0].float().cpu()
            for i, (inp, o) in enumerate(seen):
                found[f"block{i}.{lv}"] = o[0].float().cpu()
                found[f"block{i}_f32.{lv}"] = o32.blocks[i](inp.float().to(dev), emb, z.to(dev))[0].float().cpu()
    return found


def decode(args) -> int:
    """The SDK's decode of a case's final sample, timed clean (cold, then warm), then recorded on a third run."""
    import torch
    from safetensors.torch import load_file, save_file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_file(str(Path(args.case_dir) / "case.safetensors"))
    latent, camera = data["sample"].float().contiguous(), data["sample_camera"].float().contiguous()
    octree, gs, config = decoders(args.weights, args.sdk, args.device)
    import idol.models.decoder as dec
    from idol.camera import decode_camera

    points = max(1, args.gaussians // gs.rep_config["num_gaussians"])
    level = config["max_voxel_level"]
    z = latent[None].to(args.device)

    def run():
        torch.manual_seed(args.seed)
        flush(args.device)
        marks = [time.perf_counter()]
        anchors = dec.OctreeProbabilityFixedlenDecoder.sample(octree, z, num_points=points, level=level)
        flush(args.device)
        marks.append(time.perf_counter())
        raw = gs(anchors, z)
        flush(args.device)
        marks.append(time.perf_counter())
        g = gs.to_representation(anchors, raw)[0]
        flush(args.device)
        marks.append(time.perf_counter())
        return anchors, raw, g, [round(b - a, 4) for a, b in zip(marks, marks[1:])]

    timings = {}
    with torch.no_grad():
        for name in ("cold", "warm"):
            clean, _, g, spent = run()
            timings[name] = {"octree_s": spent[0], "gaussian_decoder_s": spent[1], "activation_s": spent[2],
                             "decode_s": round(sum(spent), 4)}
        mark = time.perf_counter()
        g.save_ply(str(out / "timed.ply"))
        timings["save_ply_s"] = round(time.perf_counter() - mark, 4)
        (out / "timed.ply").unlink()
    if args.device == "mps":
        timings["mps_driver_bytes_after"] = torch.mps.driver_allocated_memory()
    meta = {"seed": args.seed, "gaussians": points * gs.rep_config["num_gaussians"], "anchors": points,
            "level": level, "device": args.device, "torch": torch.__version__, "sdk": SDK_COMMIT,
            "weights": WEIGHTS_REVISION, "sdk_time": timings, "camera": decode_camera(camera)}
    if args.time_only:
        (out / "decode.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps(meta))
        return 0

    levels, xs, jitter = [], [], []
    forward, original, real_like = octree.forward, dec.systematic_sample, torch.rand_like

    def recorded_forward(x, lvl, c):
        xs.append((x[0].float().cpu(), int(lvl[0])))
        return forward(x, lvl, c)

    def recorded_systematic(probs, counts):
        draws, real = [], torch.rand

        def rand(*a, **k):
            v = real(*a, **k)
            draws.append(v.float().cpu().reshape(-1))
            return v

        torch.rand = rand
        try:
            result = original(probs, counts)
        finally:
            torch.rand = real
        c = counts.reshape(-1)
        unique_n, inv = c.unique(sorted=False, return_inverse=True)
        per_row, k = torch.full((c.numel(),), float("nan")), 0
        for index, n in enumerate(unique_n.tolist()):
            if n == 0:
                continue
            rows = (inv == index).nonzero(as_tuple=False).squeeze(1).cpu()
            if k >= len(draws) or draws[k].numel() != rows.numel():
                sys.exit("systematic_sample's draws do not follow its count groups")
            per_row[rows] = draws[k]
            k += 1
        if k != len(draws):
            sys.exit("systematic_sample drew more often than it has count groups")
        levels.append({"probs": probs[0].float().cpu(), "counts_in": c.cpu(), "draws": per_row,
                       "counts": result[0].cpu()})
        return result

    def recorded_like(*a, **k):
        v = real_like(*a, **k)
        jitter.append(v.float().cpu())
        return v

    octree.forward, dec.systematic_sample, torch.rand_like = recorded_forward, recorded_systematic, recorded_like
    try:
        with torch.no_grad():
            anchors, raw, g, _ = run()
            mods = [octree.adaLN_modulation(octree.l_embedder(torch.full((1,), float(1 << lv), device=args.device)))
                    .half()[0].float().cpu() for lv in range(1, level + 1)]
    finally:
        octree.forward, dec.systematic_sample, torch.rand_like = forward, original, real_like
    g.save_ply(str(out / "reference.ply"))
    offsets = torch.tensor([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=torch.long)
    tensors = {"latent": latent, "camera": camera, "freqs": octree.pos_embedder.freqs.float().cpu(),
               "freqs_gs": gs.pos_embedder.freqs.float().cpu()}
    mismatch = []
    for lv, ((x, res), rec) in enumerate(zip(xs, levels), start=1):
        if res != 1 << lv:
            sys.exit(f"octree level {lv} ran at resolution {res}")
        parents = torch.round(x * (1 << (lv - 1)) - 0.5).long()
        again = replay_counts(original, rec["probs"], rec["counts_in"], rec["draws"])
        mismatch.append(int((again != rec["counts"]).any(dim=1).sum()))
        tensors |= {f"parents.{lv}": parents.int(), f"probs.{lv}": rec["probs"], f"draws.{lv}": rec["draws"],
                    f"counts_in.{lv}": rec["counts_in"].int(), f"counts.{lv}": rec["counts"].int(),
                    f"mods.{lv}": mods[lv - 1]}
    last = levels[-1]["counts"].reshape(-1)
    children = (tensors[f"parents.{level}"].long()[:, None, :] * 2 + offsets[None]).reshape(-1, 3)
    tensors |= {"leaves": children[last > 0].int(), "leaf_counts": last[last > 0].int(),
                "jitter": jitter[0].reshape(-1, 3), "anchors": anchors[0].float().cpu(),
                "raw": raw[0].float().cpu(), "xyz": g.xyz.float().cpu(), "features_dc": g.features_dc.reshape(-1, 3).cpu(),
                "scaling": g.scaling.cpu(), "rotation": g.rotation.cpu(), "opacity": g.opacity.cpu()}
    if args.ruler:
        tensors |= rulers(octree, gs, xs, anchors, latent, args.device, torch.device(args.ruler_device))
        tensors |= sensitivity(octree, gs, xs, anchors, latent, args.device)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "decode.safetensors"))
    same = bool(torch.equal(clean.cpu(), anchors.cpu()))
    meta |= {"parents": [int(x.shape[0]) for x, _ in xs], "leaves": int(tensors["leaves"].shape[0]),
             "cpu_replay_count_mismatch_rows": mismatch, "recorded_anchors_equal_timed": same,
             "ruler_device": args.ruler_device if args.ruler else None}
    (out / "decode.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
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
    d = sub.add_parser("decode")
    for name in ("weights", "sdk", "case_dir", "out_dir"):
        d.add_argument(name)
    d.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    d.add_argument("--gaussians", type=int, default=262144)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--ruler", action="store_true", help="both decoders again in float32 on the SDK's own inputs")
    d.add_argument("--ruler-device", default="cpu", choices=("mps", "cpu", "cuda"))
    d.add_argument("--time-only", action="store_true", help="the timed decode alone, nothing recorded")
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
    return {"case": case, "blocks": blocks, "exact": exact, "decode": decode, "vae": vae_case}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
