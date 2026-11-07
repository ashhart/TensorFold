"""Three Library actions on a DGX Spark, each through its author's own ComfyUI nodes (spark/setup-workflows.sh)."""
import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import spark_generate as sg
from workflow_comfy import (NODES, PACK, PATCHED, SAM, comfy_pid, execute, follow, listed, loaders,  # noqa: F401
                            lora_name, mask_nodes, remover_nodes, say, server, still_vae, take)
from workflow_frames import (TAIL, again, cut, green_left, green_of, grown, mend, prepared_source,  # noqa: F401
                             read_frames, write_frames)

UPSCALER = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
# the caption the h3upscale LoRA was trained on (the node's own default)
UPSCALE_PROMPT = "h3upscale, increase resolution and restore fine details while preserving the original content and visual style."
REMOVE_PROMPT = ("Remove the green-masked person and reconstruct the background. Preserve the rest of the video, "
                 "including its camera motion and frame timing.")
# the LoRA's own wording sometimes reads as a roll of the picture; a level orbit, subject upright, held every time
TURN_PROMPT = "the camera orbits horizontally around the subject of <Picture 1> at eye level, ninety degrees clockwise, the subject stays upright on the ground"
WINDOWS = tuple(range(22, 363, 17))


# ---------------------------------------------------------------------------------------------------------- turnaround
def turnaround(args) -> None:
    if args.size % 32 or not 512 <= args.size <= 2048:
        sys.exit("--size is a multiple of 32 between 512 and 2048")
    if "<Picture 1>" not in args.prompt:
        sys.exit("the prompt names the picture as <Picture 1>")
    if not Path(args.ref).is_file():
        sys.exit(f"no picture at {args.ref}")
    server("turnaround")
    name, strength = lora_name(args.lora)
    prefix = f"studio/turn_{uuid.uuid4().hex[:8]}"
    w = {}
    loaders(w, still_vae())
    w.update({
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": name, "strength_model": strength}},
        "6": {"class_type": "LoadImage", "inputs": {"image": sg.upload(args.ref)}},
        "7": {"class_type": "H3ContactSheet", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": args.prompt,
                                                         "ref_image": ["6", 0], "size": args.size}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        # the settings the LoRA was trained against (the node's README): res_multistep / simple, 28 passes
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": args.steps, "denoise": 1.0}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["7", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0],
                                                                 "sigmas": ["10", 0], "latent_image": ["7", 1]}},
        "13": {"class_type": "H3ContactSheetDecode", "inputs": {"vae": ["4", 0], "samples": ["12", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix + "_view"}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["13", 1], "filename_prefix": prefix + "_sheet"}},
    })
    say(f"engine: cuda (ComfyUI, MiniMax H3 Ref2VA bf16 with the Turnaround LoRA at {strength:g}), five views of "
        f"{args.size}x{args.size}, {args.steps} passes")
    try:
        saved, seconds = execute(w, "12")
    finally:
        sg.forget_uploads()
    views, sheet = sorted(saved.get("14", []), key=lambda i: i["filename"]), saved.get("15", [])
    if len(views) != 5 or not sheet:
        sys.exit("ComfyUI finished without the five views")
    base = Path(args.output)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [str(base.parent / f"{base.name}_sheet.png")] + [str(base.parent / f"{base.name}_view{i + 1}.png") for i in range(5)]
    take(sheet[0], Path(outputs[0]))
    for item, target in zip(views, outputs[1:]):
        take(item, Path(target))
    say(json.dumps({"outputs": outputs, "engine": "cuda", "action": "turnaround", "size": args.size, "forwards": args.steps,
                    "total_s": round(seconds, 1)}))


# ------------------------------------------------------------------------------------------------------------- upscale
def upscale(args) -> None:
    probe = sg.probe_media(args.video)
    if probe["video"] is None:
        sys.exit(f"{args.video} has no picture")
    if not 0.1 <= args.megapixels <= 8:
        sys.exit("--megapixels is between 0.1 and 8")
    server("upscale")
    if UPSCALER not in listed("H3VideoUpsampler", "upscale_model"):
        sys.exit(f"{UPSCALER} is not installed: bash spark/setup-workflows.sh")
    name, strength = lora_name(args.lora)
    w = {}
    loaders(w)
    w.update({
        "6": {"class_type": "LoadVideo", "inputs": {"file": sg.upload(args.video)}},
        "7": {"class_type": "GetVideoComponents", "inputs": {"video": ["6", 0]}},
        # the plain transformer: the LoRA was trained on it, and the node merges the LoRA itself
        "8": {"class_type": "H3VideoUpsampler", "inputs": {
            "model": ["1", 0], "clip": ["3", 0], "vae": ["4", 0], "images": ["7", 0], "prompt": args.prompt,
            "upscale_model": UPSCALER, "megapixels": args.megapixels, "denoise": args.denoise, "lora": name,
            "lora_strength": strength, "tile_tokens": args.tile_tokens, "sampler_name": "euler"}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": ["7", 2]}},
        "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": f"studio/up_{uuid.uuid4().hex[:8]}",
                                                     "format": "auto", "codec": "auto"}},
    })
    if probe["audio"]:  # the re-draw hears the clip's sound, and the node hands the same sound back
        w["8"]["inputs"].update(audio=["7", 1], audio_vae=["5", 0])
        w["10"]["inputs"]["audio"] = ["8", 1]
    width, height = probe["video"]
    say(f"engine: cuda (ComfyUI, H3 Video Upsampler on MiniMax H3 Ref2VA bf16 with the h3upscale LoRA), {width}x{height} "
        f"to {args.megapixels:g} megapixels, re-draw {args.denoise:g}, tiles of {args.tile_tokens} tokens")
    try:
        saved, seconds = execute(w, "8")
    finally:
        sg.forget_uploads()
    clips = saved.get("11", [])
    if not clips:
        sys.exit("ComfyUI finished without writing the clip")
    target = Path(args.output)
    take(clips[0], target)
    after = sg.probe_media(str(target))
    say(json.dumps({"output": str(target), "engine": "cuda", "action": "upscale", "from": [width, height],
                    "size": list(after["video"] or ()), "total_s": round(seconds, 1)}))


# --------------------------------------------------------------------------------------------------------- mask, remove


def plate(source: Path, matte: Path, size: dict, target: Path) -> float:
    """A first frame without the person: masked points take other frames' median, the rest filled from the edges."""

    import numpy as np
    from PIL import Image

    frames = read_frames(source, size["width"], size["height"])
    mask = read_frames(matte, size["width"], size["height"])[..., 0] > 127
    count = min(len(frames), len(mask))
    frames, mask = frames[:count], mask[:count]
    grown = mask.copy()                                     # the person's soft edge and a little shadow: 8 points wider
    for axis in (1, 2):
        wide = grown.copy()
        for step in range(1, 9):
            wide |= np.roll(grown, step, axis) | np.roll(grown, -step, axis)
        grown = wide
    mask = grown
    first, hole = frames[0].astype(np.float32), mask[0]
    if not hole.any():
        Image.fromarray(frames[0]).save(target)
        return 1.0
    seen = (~mask).sum(0)                                   # frames in which a point is background
    stack = np.where(mask[..., None], np.nan, frames.astype(np.float32))
    ys, xs = np.nonzero(hole & (seen >= 3))
    out = first.copy()
    if len(ys):
        out[ys, xs] = np.nanmedian(stack[:, ys, xs], axis=0)
    # a moving camera: trust other frames' median only where it matches frame 0 on a ring around the mask
    left = hole & (seen < 3)
    ring = np.zeros_like(hole)
    for dy, dx in ((0, 4), (0, -4), (4, 0), (-4, 0)):
        ring |= np.roll(hole, (dy, dx), (0, 1))
    ring &= ~hole
    if ring.any():
        still = np.abs(np.nanmedian(stack[:, ring], axis=0) - first[ring]).mean()
        if still > 18:                                       # the scene moves: do not trust other frames
            left, out = hole.copy(), first.copy()
    filled = 1.0 - left.sum() / hole.sum()
    if left.any():                                           # fill from the edges inwards, then smooth what was filled
        known = ~left
        for _ in range(4000):
            if known.all():
                break
            total, weight = np.zeros_like(out), np.zeros(hole.shape, np.float32)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                k = np.roll(known, (dy, dx), (0, 1))
                total += np.where(k[..., None], np.roll(out, (dy, dx), (0, 1)), 0)
                weight += k
            grow = ~known & (weight > 0)
            out[grow] = total[grow] / weight[grow][:, None]
            known |= grow
        for _ in range(60):
            blur = sum(np.roll(out, s, (0, 1)) for s in ((0, 1), (0, -1), (1, 0), (-1, 0))) / 4
            out[left] = blur[left]
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(target)
    return float(filled)


def find_person(args, scratch: Path) -> tuple[str, dict, Path, Path]:
    """Upload the prepared clip and run the mask alone; returns (uploaded name, sizes, green clip, matte clip)."""

    source, size = prepared_source(args.video, scratch)
    name = sg.upload(str(source))
    w = {}
    mask_nodes(w, name, args.person, args.expand)
    prefix = f"studio/mask_{uuid.uuid4().hex[:8]}"
    w.update({
        "40": {"class_type": "CreateVideo", "inputs": {"images": ["28", 0], "fps": ["21", 2]}},
        "41": {"class_type": "SaveVideo", "inputs": {"video": ["40", 0], "filename_prefix": prefix + "_green", "format": "auto", "codec": "auto"}},
        "42": {"class_type": "MaskToImage", "inputs": {"mask": ["27", 0]}},
        "43": {"class_type": "CreateVideo", "inputs": {"images": ["42", 0], "fps": ["21", 2]}},
        "44": {"class_type": "SaveVideo", "inputs": {"video": ["43", 0], "filename_prefix": prefix + "_matte", "format": "auto", "codec": "auto"}},
    })
    say(f"engine: cuda (ComfyUI, SAM 3.1), finding \"{args.person}\" in {size['frames']} frames of {size['width']}x{size['height']}")
    saved, seconds = execute(w, "26")
    if not saved.get("41") or not saved.get("44"):
        sys.exit("ComfyUI finished without the mask")
    green, matte = scratch / "green.mp4", scratch / "matte.mp4"
    for item, target in ((saved["41"][0], green), (saved["44"][0], matte)):   # without the copies of the last frame
        take(item, scratch / "long.mp4")
        if not cut(scratch / "long.mp4", target, size["frames"]):
            (scratch / "long.mp4").replace(target)
    import numpy as np

    covered = read_frames(matte, size["width"], size["height"])[..., 0] > 127
    share = covered.reshape(len(covered), -1).mean(1)
    size.update(mask_s=round(seconds, 1), mask_first=round(float(share[0]), 4), mask_mean=round(float(share.mean()), 4),
                mask_frames=int((share > 0).sum()))
    if share[0] == 0:
        sys.exit(f"nobody matching \"{args.person}\" was found in the first frame: describe the person differently "
                 "(clothes, position), or start the clip where the person is in the picture")
    say(f"mask: {share[0] * 100:.1f}% of the first frame, on {size['mask_frames']} of {len(share)} frames, {seconds:.0f} s")
    return name, size, green, matte


def mask(args) -> None:
    server("mask")
    if SAM not in listed("CheckpointLoaderSimple", "ckpt_name"):
        sys.exit(f"{SAM} is not installed: bash spark/setup-workflows.sh")
    base = Path(args.output)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="studio-mask-") as scratch:
            _, size, green, matte = find_person(args, Path(scratch))
            outputs = [str(base.parent / f"{base.name}_mask.mp4"), str(base.parent / f"{base.name}_matte.mp4"),
                       str(base.parent / f"{base.name}_plate.png")]
            size["plate_from_clip"] = round(plate(Path(scratch) / "source.mp4", matte, size, Path(outputs[2])), 3)
            green.replace(outputs[0]); matte.replace(outputs[1])
    finally:
        sg.forget_uploads()
    say(json.dumps({"outputs": outputs, "engine": "cuda", "action": "mask", **size}))


def remove(args) -> None:
    if args.window not in WINDOWS:
        sys.exit("--window is 22, 39, 56, ... (17n + 5)")
    server("remove")
    if SAM not in listed("CheckpointLoaderSimple", "ckpt_name"):
        sys.exit(f"{SAM} is not installed: bash spark/setup-workflows.sh")
    lora = lora_name(args.lora)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    try:
        with tempfile.TemporaryDirectory(prefix="studio-remove-") as scratch:
            scratch = Path(scratch)
            name, size, green, matte = find_person(args, scratch)
            window = args.window if size["frames"] >= args.window else 22
            if args.clean_frame:
                from PIL import Image

                clean = scratch / "clean.png"
                Image.open(args.clean_frame).convert("RGB").resize((size["width"], size["height"]), Image.LANCZOS).save(clean)
                report["clean_frame"] = "given"
            else:
                rough = scratch / "plate.png"
                share = plate(scratch / "source.mp4", matte, size, rough)
                report.update(clean_frame="built", plate_from_clip=round(share, 3))
                # one remover window from that plate: its redrawn first frame is the clean frame the run starts from
                w = {}
                mask_nodes(w, name, args.person, args.expand)
                w["34"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["28", 0], "batch_index": 0, "length": 22}}
                remover_nodes(w, lora, sg.upload(str(rough)), ["34", 0], args, 22, args.seed + 1)
                w["35"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["31", 0], "batch_index": 0, "length": 1}}
                w["36"] = {"class_type": "SaveImage", "inputs": {"images": ["35", 0], "filename_prefix": f"studio/clean_{uuid.uuid4().hex[:8]}"}}
                say("clean first frame: built from the clip, then drawn again by one window of the remover")
                saved, seconds = execute(w, "31")
                if not saved.get("36"):
                    sys.exit("ComfyUI finished without the clean frame")
                clean = scratch / "clean.png"
                take(saved["36"][0], clean)
                report["clean_frame_s"] = round(seconds, 1)
            starts, long = [0], size["frames"] + TAIL     # the relay's own windows over the lengthened clip
            while starts[-1] + window < long:
                starts.append(max(window - 1, min(starts[-1] + window - 1, long - window)))
            w = {}
            mask_nodes(w, name, args.person, args.expand)
            remover_nodes(w, lora, sg.upload(str(clean)), ["28", 0], args, window, args.seed)
            w["37"] = {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "fps": ["21", 2]}}
            w["38"] = {"class_type": "SaveVideo", "inputs": {"video": ["37", 0], "filename_prefix": f"studio/clean_{uuid.uuid4().hex[:8]}",
                                                             "format": "auto", "codec": "auto"}}
            say(f"engine: cuda (ComfyUI, MiniMax H3 Ref2VA bf16 with the Person Remover LoRA at {lora[1]:g}), "
                f"{size['frames']} frames of {size['width']}x{size['height']} in {len(starts)} windows of {window}, {args.steps} passes each")
            saved, seconds = execute(w, "31", len(starts))
            if not saved.get("38"):
                sys.exit("ComfyUI finished without writing the clip")
            silent = scratch / "silent.mp4"
            take(saved["38"][0], silent)
            # cut back to the clip's own length, then green that flashed back for a frame or two painted over
            trimmed = scratch / "trimmed.mp4"
            if not cut(silent, trimmed, size["frames"]):
                silent.replace(trimmed)
            stretches, repaired, extra = again(trimmed, green, size, lora, args, scratch, args.again)
            if stretches:
                seconds += extra
                report.update(second_look=stretches, second_look_frames=repaired, second_look_s=round(extra, 1))
                say(f"check: the first pass left green on some frames; {stretches} stretch{'es' if stretches > 1 else ''} of 22 frames "
                    f"went through the remover again and {repaired} frame{'s' if repaired != 1 else ''} came back clean")
            mended, left = mend(trimmed, size)
            report.update(mended_frames=mended, green_frames=left)
            if mended:
                say(f"check: green had come back on {mended} of {size['frames']} frames; painted over from the frames beside them")
            if left:
                say(f"check: {left} of {size['frames']} frames still show green: that part was not removed. Another seed "
                    "draws those windows again.")
            # under the clip's own sound, as the author's guide has it: the removed person's voice stays in it
            if size["sound"] and not args.silent and subprocess.run(
                    [sg._tool("ffmpeg"), "-v", "error", "-y", "-i", str(trimmed), "-i", args.video, "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", sg.SOUND_BITRATE, "-t", f"{size['frames'] / sg.FPS:.6f}",
                     "-movflags", "+faststart", str(target)], capture_output=True).returncode == 0:
                report["sound"] = "the clip's own"
            else:
                trimmed.replace(target)
                report["sound"] = "none"
            keep = Path(args.keep) if args.keep else None
            if keep:
                keep.parent.mkdir(parents=True, exist_ok=True)
                green.replace(keep.parent / f"{keep.name}_mask.mp4")
                clean.replace(keep.parent / f"{keep.name}_clean.png")
                report["kept"] = [str(keep.parent / f"{keep.name}_mask.mp4"), str(keep.parent / f"{keep.name}_clean.png")]
    finally:
        sg.forget_uploads()
    say(json.dumps({"output": str(target), "engine": "cuda", "action": "remove", "windows": len(starts), "window": window,
                    "forwards": args.steps, "total_s": round(seconds, 1), **size, **report}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    t = sub.add_parser("turnaround")
    t.add_argument("--ref", required=True); t.add_argument("--lora", required=True); t.add_argument("-o", "--output", required=True)
    t.add_argument("--size", type=int, default=1024, help="each view's side: 512 (what the LoRA was trained at), 1024 or 2048")
    t.add_argument("--prompt", default=TURN_PROMPT); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=28)
    u = sub.add_parser("upscale")
    u.add_argument("--video", required=True); u.add_argument("--lora", required=True); u.add_argument("-o", "--output", required=True)
    u.add_argument("--megapixels", type=float, default=2.0, help="the result's area, sides kept in proportion (2 is about 1920x1080)")
    u.add_argument("--denoise", type=float, default=0.1, help="how hard the one re-draw works: 0.1 is the author's tested setting")
    u.add_argument("--tile-tokens", type=int, default=70000, help="memory: the longest row of one tile (the author's default)")
    u.add_argument("--prompt", default=UPSCALE_PROMPT)
    for name in ("mask", "remove"):
        m = sub.add_parser(name)
        m.add_argument("--video", required=True); m.add_argument("--person", required=True, help="a few words: man in a gray shirt")
        m.add_argument("-o", "--output", required=True)
        m.add_argument("--expand", type=int, default=5, help="points the mask grows by (the author's 5)")
        if name == "remove":
            m.add_argument("--lora", required=True); m.add_argument("--clean-frame", default="")
            m.add_argument("--window", type=int, default=22); m.add_argument("--steps", type=int, default=12)
            m.add_argument("--seed", type=int, default=904234); m.add_argument("--prompt", default=REMOVE_PROMPT)
            m.add_argument("--silent", action="store_true", help="leave the clip's own sound out")
            m.add_argument("--again", type=int, default=3, help="stretches of 22 frames to run a second time where green was left (0: none)")
            m.add_argument("--keep", default="", help="folder/name: keep the green clip and the clean frame beside the result")
    args = parser.parse_args()
    {"turnaround": turnaround, "upscale": upscale, "mask": mask, "remove": remove}[args.action](args)


if __name__ == "__main__":
    main()
