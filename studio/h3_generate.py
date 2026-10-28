"""Render one MiniMax H3 clip with TensorFold's H3 family (copy of tools/h3_generate_dev.py at the pinned commit)."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from tensorfold.families.h3 import config as h3
from tensorfold.families.h3.lora import merge, settle
from tensorfold.families.h3.sampler import denoise
from tensorfold.families.h3.schedule import parse_subset
from tensorfold.families.h3.weights import int8_attention, int8_mlp, load_dit

sys.path.append(str(Path(__file__).resolve().parent))  # the h3_*.py helpers beside the real file, also through a link
from h3_condition import (encode_first_frame, encode_reference_rows, encode_reference_text,  # noqa: E402,F401
                          encode_references, reference_layout, reference_request, video_patches)
from h3_decode import AUDIO_PEAK, SHEET_AUDIO, SHEET_SLOTS, decode, unclipped_audio, write_sheet  # noqa: E402,F401
from h3_media import (ASPECT_LIMIT, AUDIO_LIMIT, CLIP_SECONDS, CLIP_TOLERANCE, REFERENCE_LIMIT,  # noqa: E402,F401
                      REFERENCE_MULTIPLE, TOTAL_LIMIT, TOTAL_SECONDS, VIDEO_LIMIT, check_clip_lengths, encoded_frames,
                      prepare_reference_media, prepare_references, probe_media, read_video_frames, read_waveform,
                      reference_size, reference_video_size, sample_video)
from h3_text import (MLX_VISION_STATS, VISION_STATS, comfy_int8_weights, encode_text, vision_stats)  # noqa: E402,F401


def _lazy_fasth3():
    """Make the engine export's `fasth3.load_fasth3` lazy: skip its final eval so the 65 GB checkpoint stays on disk."""

    from tensorfold.families.h3 import fasth3

    load = fasth3.load_fasth3
    if getattr(load, "lazy", False):
        return

    def load_fasth3(*args, **kwargs):
        evaluate = mx.eval
        mx.eval = lambda *_: None
        try:
            return load(*args, **kwargs)
        finally:
            mx.eval = evaluate

    load_fasth3.lazy = True
    fasth3.load_fasth3 = load_fasth3


if __name__ == "h3_generate_dev":  # imported by the native engine's export, not run as the MLX renderer
    _lazy_fasth3()
    mx.set_cache_limit(2 * 2**30)  # freed buffers are otherwise kept for reuse, tens of GiB of them


def parity(root: Path, dit, args, text, tags):
    """Largest and relative difference between this transformer and the reference on one real forward."""

    from minimax_h3_mlx.load import load_dit as load_reference

    from tensorfold.families.h3.packing import layout, timestep_plan
    from tensorfold.families.h3.sampler import start_noise
    from tensorfold.families.h3.schedule import AUDIO_SHIFT, VIDEO_SHIFT, Schedule

    config = dit.config
    frames, lat_h, lat_w = h3.latent_frames(args.frames), args.height // 16, args.width // 16
    packed = layout(tags, frames, lat_h, lat_w, h3.audio_latents(args.frames), config.patch_size)
    video, audio = start_noise(config, frames, lat_h, lat_w, h3.audio_latents(args.frames), args.seed)
    table, plan = timestep_plan(packed, Schedule(VIDEO_SHIFT, args.points).timesteps,
                                Schedule(AUDIO_SHIFT, args.points).timesteps)
    # the reference rounds the latent rows to bfloat16 on the way in; do the same so the comparison is like for like
    call = (video[None].astype(mx.bfloat16), audio[None].astype(mx.bfloat16), text.astype(mx.bfloat16), table,
            plan[0], packed.tags,
            packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows)
    ours = dit(*call)
    mx.eval(*ours)
    reference = load_reference(root / "transformer")
    theirs = reference(*call)
    mx.eval(*theirs)
    for name, a, b in zip(("video", "audio"), ours, theirs, strict=True):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        diff = float(mx.max(mx.abs(a - b)).item())
        rel = float((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item())
        print(f"[tensorfold] parity {name}: max abs diff {diff:.3e}, relative {rel:.3e}, "
              f"reference rms {float(mx.sqrt(mx.mean(b * b)).item()):.3f}", flush=True)
    del reference
    gc.collect()
    mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--points", type=int, default=21, help="sigma points; one fewer forwards")
    parser.add_argument("--subset", default=None, help="N:i0,i1,... keeps those points of an N-point grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="PATH or PATH:STRENGTH, merged at load")
    parser.add_argument("--int8-mlp", action="store_true", help="run the block MLPs through the int8 kernel")
    parser.add_argument("--int8-qkv", action="store_true", help="int8 QKV projection")
    parser.add_argument("--int8-out", action="store_true", help="int8 attention-output projection")
    parser.add_argument("--unfused-qkv", action="store_true", help="int8 QKV without the fused norm and rotation")
    parser.add_argument("--keep-adaln", action="store_true", help="keep the AdaLN projection weights loaded")
    parser.add_argument("--audio-shift", type=float, default=None, help="sigma shift of the audio schedule (released: 3)")
    parser.add_argument("--upscale-vae", help="safetensors of a packed-head (2x) video decoder; frames come out larger")
    parser.add_argument("--crop", help="WxH: centre-crop the decoded frames before the MP4 is written")
    parser.add_argument("--fasth3", help="a FastH3 checkpoint folder: its transformer, schedule and routed attention")
    parser.add_argument("--fasth3-q8", action="store_true",
                        help="with --fasth3: the 8-bit weights written by fasth3_q8.py, for chips without tensor units")
    parser.add_argument("--vsa-impl", default="tensor", choices=("tensor", "reference", "simd"),
                        help="routed attention: our int8 tile kernel, or FastVideo's reference or SIMD-group forms")
    parser.add_argument("--fasth3-steps", type=int, default=0, metavar="N",
                        help="with --fasth3: N forwards instead of the trained count; rungs are spread evenly from "
                             "the checkpoint's first rung, which keeps the trained rungs when N divides their number")
    parser.add_argument("--dense", action="store_true", help="with --fasth3: dense attention instead of routed")
    parser.add_argument("--step-cache", type=float, default=0.0,
                        help="reuse the last velocity while the summed relative move stays under this (0.05)")
    parser.add_argument("--attention-every", type=int, default=0,
                        help="compute attention every N-th middle step and reuse it otherwise (2)")
    parser.add_argument("--fast-gates", help="A,B,C: opening and closing steps that always compute attention, and "
                                             "steps before the velocity cache may skip")
    parser.add_argument("--save-frames", help="also write the decoded frames to this .npy file, for comparisons")
    parser.add_argument("--revoice", type=int, default=0, metavar="STEPS",
                        help="after the run, denoise the audio again in STEPS steps with the model without adapters")
    parser.add_argument("--revoice-exact", action="store_true", help="re-voice with whole-sequence forwards")
    parser.add_argument("--audio-hard-clip", action="store_true",
                        help="clip the decoded audio at full scale, as the reference decoder does (for comparisons)")
    parser.add_argument("--float-vae", action="store_true", help="video decoder in float32, without int8 kernels")
    parser.add_argument("--first-frame", default=None, help="image the clip starts from (image to video)")
    parser.add_argument("--last-frame", default=None,
                        help="image the clip ends on (MiniMax H3's first-and-last-frame mode; with or without "
                             "--first-frame). Beside a first frame it is fitted to the canvas by a centre crop")
    parser.add_argument("--ref-image", action="append", default=[], metavar="PATH",
                        help="a reference picture (repeatable, up to 9): <Picture 1>, <Picture 2>, ... in the "
                             "prompt. Needs the Ref2VA transformer; not with --first-frame or --fasth3")
    parser.add_argument("--ref-video", action="append", default=[], metavar="PATH",
                        help="a reference video (repeatable, up to 3, each 2 to 15 s, 15 s in all): <Video 1>, ... in "
                             "the prompt, after the pictures. Read at 24 fps up to the clip's own length")
    parser.add_argument("--ref-audio", action="append", default=[], metavar="PATH",
                        help="a reference audio clip (repeatable, up to 3, each 2 to 15 s, 15 s in all): <Audio 1>, "
                             "... in the prompt. Only beside a reference picture or video")
    parser.add_argument("--ref-video-sound", action="store_true",
                        help="also use each reference video's own soundtrack as a reference; those sounds are then "
                             "<Audio 1>, ... ahead of the --ref-audio clips")
    parser.add_argument("--ref-video-size", default="match", choices=("match", "full"),
                        help="match: a reference video is brought down to the clip's pixel area (default); full: "
                             "the model's own 768-pixel canvas, several times slower")
    parser.add_argument("--vision-stats", default="checkpoint", choices=("checkpoint", "mlx"),
                        help="how a first frame or a reference picture is normalised for the text encoder's vision "
                             "tower: the checkpoint's own processor values (default) or minimax-h3-mlx's fixed "
                             "ones, which releases up to 2.5 used (for comparisons)")
    parser.add_argument("--text-encoder", default=None, metavar="PATH",
                        help="a ComfyUI int8 (int8_tensorwise, ConvRot) safetensors of the Qwen3-VL text encoder, "
                             "decoded to bfloat16 as it loads; the tokenizer stays the model folder's")
    parser.add_argument("--sheet", default=None, metavar="PREFIX",
                        help="a turnaround sheet instead of a clip: five views of the subject of the one --ref-image, "
                             "written as PREFIX_sheet.png and PREFIX_view1..5.png. Needs the Turnaround LoRA (--lora) "
                             "and a square size; -o is not written")
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    if args.sheet:
        if len(args.ref_image) != 1 or args.ref_video or args.ref_audio or args.first_frame or args.fasth3:
            raise SystemExit("--sheet takes exactly one --ref-image and nothing else to start from")
        if args.width != args.height or args.width % 32:
            raise SystemExit("--sheet makes square views: --width and --height the same multiple of 32")
        if not args.lora:
            raise SystemExit("--sheet needs the Turnaround LoRA: --lora PATH")
        # five independent pictures on the time axis: not a frame count the clip grid has, so it is set by hand
        h3.latent_frames, h3.audio_latents = (lambda frames: SHEET_SLOTS), (lambda frames: SHEET_AUDIO)
    if args.text_encoder and not (args.text_encoder.endswith(".safetensors") and Path(args.text_encoder).is_file()):
        raise SystemExit(f"--text-encoder takes a .safetensors file, and {args.text_encoder} is not one")

    references = None
    wanted = bool(args.ref_image or args.ref_video or args.ref_audio)
    if wanted:
        if args.first_frame or args.last_frame or args.fasth3:
            raise SystemExit("references cannot be combined with --first-frame, --last-frame or --fasth3")
        if len(args.ref_image) > REFERENCE_LIMIT:
            raise SystemExit(f"at most {REFERENCE_LIMIT} reference pictures, got {len(args.ref_image)}")
        if len(args.ref_video) > VIDEO_LIMIT:
            raise SystemExit(f"at most {VIDEO_LIMIT} reference videos, got {len(args.ref_video)}")
        if len(args.ref_audio) > AUDIO_LIMIT:
            raise SystemExit(f"at most {AUDIO_LIMIT} reference audio clips, got {len(args.ref_audio)}")
        total = len(args.ref_image) + len(args.ref_video) + len(args.ref_audio)
        if total > TOTAL_LIMIT:
            raise SystemExit(f"at most {TOTAL_LIMIT} references in all, got {total}")
        if args.ref_audio and not (args.ref_image or args.ref_video):
            raise SystemExit("a reference audio clip needs a reference picture or video beside it: the model does "
                             "not take sound alone")
        if (Path(args.model_dir) / "Ref2VA" / "transformer").is_dir():
            args.model_dir = str(Path(args.model_dir) / "Ref2VA")
    root = h3.pipeline_root(args.model_dir)
    if root is None:
        raise SystemExit(f"{args.model_dir} is not a MiniMax H3 pipeline folder")
    media = None
    if wanted:
        with open(root / "model_index.json") as handle:
            partition = json.load(handle).get("_minimax_h3", {}).get("partition")
        if partition != "ref2va":
            raise SystemExit(f"references need the Ref2VA transformer; {root} is {partition or 'not marked'} "
                             "(oneshot-setup.sh --ref2va fetches it)")
        missing = [name for name in ("text_encoder", "tokenizer", "video_vae", "audio_vae") if not (root / name).exists()]
        if missing:
            raise SystemExit(f"{root} lacks {', '.join(missing)}: link them from FL2VA (oneshot-setup.sh --ref2va)")
        if args.ref_video or args.ref_audio:
            from tensorfold.families.h3 import packing

            if not hasattr(packing, "reference_layout"):
                raise SystemExit("reference videos and audio need a TensorFold whose H3 family packs references "
                                 "(packing.reference_layout); this one takes reference pictures only")
            if not (root / "processor" / "video_preprocessor_config.json").is_file():
                raise SystemExit(f"{root} lacks processor/video_preprocessor_config.json: link processor from FL2VA "
                                 "(oneshot-setup.sh --ref2va)")
            with open(root / "audio_vae" / "metadata.json") as handle:
                rate = int(json.load(handle)["metadata"]["kwargs"]["sample_rate"])
            media = prepare_reference_media(args.ref_image, args.ref_video, args.ref_audio, args.width, args.height,
                                            args.frames, rate, args.ref_video_sound, args.ref_video_size)
            for item in media:
                if item["kind"] == "image":
                    what = "{}x{}".format(*item["image"].size)
                elif item["kind"] == "video":
                    count, tall, wide = item["frames"].shape[:3]
                    what = f"{wide}x{tall}, {count} frames ({count / h3.FPS:.1f} s)"
                    if item["sound"] is not None:
                        what += f", its sound is {item['sound_name']} ({item['sound'].shape[1] / rate:.1f} s)"
                else:
                    what = f"{item['sound'].shape[1] / rate:.1f} s"
                print(f"[tensorfold] reference {item['name']}: {item['path']} ({what})", flush=True)
        else:
            references = prepare_references(args.ref_image, args.width, args.height)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    started = time.perf_counter()
    image = None
    if args.first_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        image = prepare_keyframe_image(Image.open(args.first_frame).convert("RGB"), args.height, args.width,
                                       stretch=True)
    # keyframes in packed order (first, then last): the first is stretched onto the canvas, a second centre-cropped
    frames, anchors = ([image], ["first"]) if image is not None else ([], [])
    if args.last_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        frames.append(prepare_keyframe_image(Image.open(args.last_frame).convert("RGB"), args.height, args.width,
                                             stretch=not frames))
        anchors.append("last")
        image = frames if len(frames) > 1 else frames[0]
        print(f"[tensorfold] keyframes: {', '.join(anchors)}", flush=True)
    stats = args.vision_stats == "checkpoint"
    condition = audio_condition = geometry = None
    patch = h3.DiTConfig.from_checkpoint(args.model_dir).patch_size
    if media is not None:
        text, tags = encode_reference_text(root, prompt, media, args.text_encoder, stats)
        condition, audio_condition, geometry = encode_reference_rows(root, media, patch)
        print(f"[tensorfold] references: {len(geometry)} in all, "
              f"{0 if condition is None else condition.shape[0]} video rows, "
              f"{0 if audio_condition is None else audio_condition.shape[0]} audio rows held", flush=True)
    else:
        text, tags = encode_text(root, prompt, references if references is not None else image, args.text_encoder,
                                 stats)
    if references is not None:
        from tensorfold.families.h3 import sampler

        condition, shapes = encode_references(root, references, patch)
        sampler.layout = reference_layout(shapes)  # the sampler packs the references ahead of the clip
        print(f"[tensorfold] references: {len(shapes)} pictures at "
              f"{', '.join(f'{w * 16}x{h * 16}' for h, w in shapes)}, {condition.shape[0]} rows", flush=True)
    if image is not None:
        condition = encode_first_frame(root, image, args.width, args.height, patch)
    text_seconds = time.perf_counter() - started
    print(f"[tensorfold] text: {text.shape[1]} rows in {text_seconds:.1f}s", flush=True)

    started = time.perf_counter()
    fast = None
    if args.fasth3:
        from tensorfold.families.h3 import fasth3

        if args.fasth3_q8:
            import fasth3_q8

            dit, gates, fast = fasth3_q8.load(args.fasth3)
            # the blocks' weights are MLX quantized layers now; the int8 tensor-unit kernels do not apply to them
            args.int8_mlp = args.int8_qkv = args.int8_out = False
            mx.set_cache_limit(8 * 2**30)  # room for reuse without the default tens of GiB; 4 GiB cost a third in speed
            print("[tensorfold] FastH3 with 8-bit weights (MLX quantized layers)", flush=True)
        else:
            dit, gates, fast = fasth3.load_fasth3(args.fasth3)
        print(f"[tensorfold] FastH3: {fast.forwards} forwards, video shift {fast.video_shift}, sparsity "
              f"{fast.sparsity}, tile {fast.tile}, task {fast.task}", flush=True)
    else:
        dit = load_dit(args.model_dir)
    for spec in args.lora:
        path, _, strength = spec.partition(":")
        print(f"[tensorfold] {merge(dit, path, float(strength or 1.0))}", flush=True)
    if args.int8_mlp:
        print(f"[tensorfold] int8 MLP in {int8_mlp(dit)} blocks", flush=True)
    if args.int8_qkv or args.int8_out:
        changed = int8_attention(dit, qkv=args.int8_qkv, out=args.int8_out, fused=not args.unfused_qkv)
        print(f"[tensorfold] int8 attention projections: {changed}", flush=True)
    rounded = settle(dit)
    if rounded and args.lora:
        print(f"[tensorfold] {rounded} adapted projections rounded to bfloat16 (not on an int8 kernel)", flush=True)
    load_seconds = time.perf_counter() - started
    if args.parity:
        parity(root, dit, args, text, tags)

    points, subset = args.points, None
    if args.subset:
        points, subset = parse_subset(args.subset)
    schedule = {}
    if fast is not None:
        nodes = fast.nodes
        if args.fasth3_steps and args.fasth3_steps != len(nodes):
            nodes = tuple(nodes[0] * (1 - i / args.fasth3_steps) for i in range(args.fasth3_steps))
            print(f"[tensorfold] FastH3 resampled to {len(nodes)} forwards (trained for {fast.forwards})", flush=True)
        schedule = {"nodes": nodes, "video_shift": fast.video_shift}
        if args.audio_shift is None:
            args.audio_shift = fast.audio_shift
        if fast.sparsity > 0 and not args.dense:
            schedule["prepare"] = fasth3.route(dit, gates, fast.sparsity, fast.tile, args.vsa_impl)
    # reference videos and audio go through the family's own reference layout and its held audio rows
    held = {} if geometry is None else {"references": geometry, "audio_condition": audio_condition}
    held_audio = {} if audio_condition is None else {"audio_condition": audio_condition}
    started = time.perf_counter()
    latents = denoise(dit, text, tags, args.width, args.height, args.frames, points, args.seed, subset,
                      release=not args.keep_adaln, condition=condition,
                      keyframes=tuple(anchors),
                      on_step=lambda i, n, s: print(f"[tensorfold] step {i}/{n} {s:.2f}s", flush=True),
                      audio_shift=args.audio_shift, step_cache=args.step_cache,
                      attention_every=args.attention_every,
                      gates=tuple(int(v) for v in args.fast_gates.split(",")) if args.fast_gates else None,
                      **held, **schedule)
    denoise_seconds = time.perf_counter() - started
    revoice_seconds = 0.0
    if args.revoice:
        from tensorfold.families.h3.sampler import revoice

        started = time.perf_counter()
        del dit
        gc.collect()
        mx.clear_cache()
        dit = load_dit(args.model_dir)  # no adapter: the audio comes from the released weights
        if args.int8_mlp:
            int8_mlp(dit)
        if args.int8_qkv or args.int8_out:
            int8_attention(dit, qkv=args.int8_qkv, out=args.int8_out, fused=not args.unfused_qkv)
        latents.audio_rows = revoice(dit, text, latents, args.revoice + 1, args.seed, condition,
                                     exact=args.revoice_exact, release=not args.keep_adaln, **held_audio,
                                     on_step=lambda i, n, s: print(f"[tensorfold] voice {i}/{n} {s:.2f}s", flush=True))
        revoice_seconds = time.perf_counter() - started
        print(f"[tensorfold] re-voiced in {revoice_seconds:.1f}s", flush=True)
    del dit
    gc.collect()
    mx.clear_cache()

    if args.sheet:
        started = time.perf_counter()
        outputs = write_sheet(args.model_dir, latents, h3.DiTConfig.from_checkpoint(args.model_dir), args.sheet,
                              int8=not args.float_vae)
        print("[tensorfold] " + json.dumps({
            "outputs": outputs, "action": "turnaround", "size": args.width, "rows": latents.packed.rows,
            "forwards": len(latents.step_seconds), "text_s": round(text_seconds, 1), "load_s": round(load_seconds, 1),
            "denoise_s": round(denoise_seconds, 1), "decode_s": round(time.perf_counter() - started, 1),
            "per_forward_s": round(float(np.median(latents.step_seconds)), 2),
            "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}), flush=True)
        return

    started = time.perf_counter()
    frames, wave, rate = decode(args.model_dir, root, latents, h3.DiTConfig.from_checkpoint(args.model_dir),
                                int8=not args.float_vae, upscale_decoder=args.upscale_vae,
                                hard_clip=args.audio_hard_clip)
    if args.save_frames:
        np.save(args.save_frames, frames[::8])
    if args.crop:
        crop_w, crop_h = (int(v) for v in args.crop.lower().split("x"))
        full_h, full_w = frames.shape[1:3]
        if crop_w > full_w or crop_h > full_h or (full_w - crop_w) % 2 or (full_h - crop_h) % 2:
            raise SystemExit(f"cannot centre-crop {full_w}x{full_h} frames to {crop_w}x{crop_h}")
        top, left = (full_h - crop_h) // 2, (full_w - crop_w) // 2
        frames = np.ascontiguousarray(frames[:, top : top + crop_h, left : left + crop_w])
    from minimax_h3_mlx.media import save_mp4

    mux_started = time.perf_counter()
    save_mp4(args.output, frames, h3.FPS, audio=wave, sample_rate=rate)
    print(f"[tensorfold] mux {time.perf_counter() - mux_started:.2f}s", flush=True)
    decode_seconds = time.perf_counter() - started
    report = {"output": args.output, "rows": latents.packed.rows, "forwards": len(latents.step_seconds),
              "attention_reused": latents.reused_steps, "skipped": latents.skipped_steps,
              "text_s": round(text_seconds, 1), "load_s": round(load_seconds, 1),
              "denoise_s": round(denoise_seconds, 1), "revoice_s": round(revoice_seconds, 1), "decode_s": round(decode_seconds, 1),
              "per_forward_s": round(float(np.median(latents.step_seconds)), 2),
              "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}
    print("[tensorfold] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
