"""TensorFold Studio on a DGX Spark (CUDA): FastH3 video with sound and Qwen-Image-2.1 pictures through ComfyUI."""
# Credit: FastVideo (Hao AI Lab) for FastH3, MiniMax for H3, the Qwen team for Qwen-Image-2.1-Turbo, ComfyUI.
import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memory_guard  # noqa: E402  (what a clip needs and whether this machine has it, beside this file)
import spark_comfy  # noqa: E402
from spark_comfy import (MODELS, PREFIX, STATE, UPLOADED, URL, Saved, call, ensure_server, fetch,  # noqa: E402,F401
                         forget_uploads, fresh_server, installed, render, run, stopped, up, upload)
from spark_finish import (HEADROOM_DB, PEAK_DBFS, SOUND_BITRATE, crop_file, crop_frames, finish_sound,  # noqa: E402,F401
                          save_sound)
from spark_media import (ASPECT_LIMIT, AUDIO_LIMIT, AUDIO_RATE, CLIP_SECONDS, CLIP_TOLERANCE, FPS,  # noqa: E402,F401
                         MODEL_MAX_PIXELS, MODEL_SHORT_EDGE, PICTURE_LIMIT, REFERENCE_MULTIPLE, TOTAL_LIMIT,
                         TOTAL_SECONDS, VIDEO_LIMIT, _tool, check_clip_lengths, prepared_sound, prepared_video,
                         probe_media, reference_video_size)
ENGINE_LIB = PREFIX / "tf-h3" / "libtf_h3.so"

FASTH3 = {"bf16": "fastvideo_fasth3_8step_v2_pruned_bf16.safetensors",
          "int8": "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"}
# Ref2VA: references hold a subject; its Turbo adapter merges at run time, so only on bf16 (8-bit is rotated)
REF2VA = {"bf16": "minimax_h3_ref2va_pruned_bf16.safetensors",
          "int8": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"}
REF2VA_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
# MiniMax H3's own first-and-last-frame transformer (optional): for LoRAs trained on it and --base h3
H3_FL2VA = "minimax_h3_fl2va_pruned_bf16.safetensors"
H3_TEXT = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
# ComfyUI's 8-bit video decoder: half the fp16 one's decode time, same picture; SPARK_VAE picks another
H3_VAE = os.environ.get("SPARK_VAE", "minimax_h3_video_vae_int8_convrot.safetensors")
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
QWEN = "qwen_image_2.1_turbo_bf16.safetensors"   # Qwen-Image-2.1-Turbo, the 8-step checkpoint
QWEN_BASE = "qwen_image_2.1_bf16.safetensors"    # the 40-step base model, only when it has been downloaded
QWEN_TEXT = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
# the schedule saved with the Turbo checkpoint (model_index.json, sample_sigmas), then zero
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"


def fits_or_exit(args, lora: bool = False, references: bool = False, base: bool = False, pictures: int = 0,
                 reference_frames: int = 0) -> None:
    """The memory check before anything loads (memory_guard.py): refuse a clip over the floor, saying what fits."""

    found = memory_guard.check(args.width, args.height, args.frames, lora, references, base, args.engine == "tensorfold",
                               pictures, reference_frames)
    if found.get("route") is None:
        return
    print(f"[tensorfold] memory: about {found['need_mb'] / 1000:.0f} GB at the peak (give or take {found['error_mb'] / 1000:g}) of "
          f"{found['available_mb'] / 1000:.0f} GB, floor {found['floor_mb'] / 1000:.0f} GB", flush=True)
    if not found["fits"]:
        sys.exit("refused before anything was loaded: " + found["reason"])


def hub_loras(specs) -> list[tuple[str, float]]:
    """--lora PATH[:STRENGTH] as (name ComfyUI lists, strength); merged at run time into bf16 weights, never 8-bit."""

    loras = []
    for spec in specs:
        path, _, strength = spec.rpartition(":")
        try:
            strength = float(strength)
        except ValueError:
            path, strength = spec, 1.0
        if not path:
            path, strength = spec, 1.0
        source = Path(path).resolve()
        if not source.is_file():
            sys.exit(f"no LoRA at {path}")
        link = MODELS / "loras" / f"hub_{source.name}"
        if not link.exists() or link.resolve() != source:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(source)
        loras.append((link.name, strength))
    return loras


def reference(args) -> None:
    """A clip from references (Ref2VA): `<Picture i>`, `<Video i>`, `<Audio i>` are --ref, --ref-video, --ref-audio."""

    pictures, videos, audios = args.ref, args.ref_video, args.ref_audio
    for kind, limit, files in (("pictures", PICTURE_LIMIT, pictures), ("videos", VIDEO_LIMIT, videos), ("audio clips", AUDIO_LIMIT, audios)):
        if len(files) > limit:
            sys.exit(f"at most {limit} reference {kind}, got {len(files)}")
    if len(pictures) + len(videos) + len(audios) > TOTAL_LIMIT:
        sys.exit(f"at most {TOTAL_LIMIT} references in all, got {len(pictures) + len(videos) + len(audios)}")
    if audios and not (pictures or videos):
        sys.exit("a reference audio clip needs a reference picture or video beside it: the model does not take sound alone")
    if args.first_frame or args.last_frame:
        sys.exit("references and --first-frame / --last-frame do not combine: the reference model has no keyframe input")
    for kind, files in (("picture", pictures), ("video", videos), ("audio clip", audios)):
        for path in files:
            if not Path(path).is_file() or not Path(path).stat().st_size:
                sys.exit(f"no reference {kind} at {path}")
    # each video and each audio clip is 2 to 15 seconds, and 15 seconds in all per kind: measured before any model loads
    probes = [(path, probe_media(path)) for path in videos]
    for path, probe in probes:
        if probe["video"] is None:
            sys.exit(f"{path} has no picture: a reference video is a video file")
    check_clip_lengths("video", [(path, probe["seconds"]) for path, probe in probes])
    sounds = [(path, probe_media(path)) for path in audios]
    for path, probe in sounds:
        if not probe["audio"]:
            sys.exit(f"{path} has no sound: a reference audio clip is an audio file (or a video with sound)")
    check_clip_lengths("audio clip", [(path, probe["audio_seconds"]) for path, probe in sounds])
    sizes = [reference_video_size(*probe["video"], args.width, args.height, args.ref_video_size) for _, probe in probes]
    loras = hub_loras(args.lora)
    last = STATE.read_text().strip() if STATE.exists() else ""
    if up() and (last.startswith("video") or loras or "lora" in last):
        # a LoRA render, the one after it, or one after FastH3 (the engine keeps its weights) start from an empty server
        fresh_server("a reference render starts from a server with nothing loaded")
    fits_or_exit(args, bool(loras) or args.ref_mode == "turbo", True, pictures=len(pictures),
                 reference_frames=sum(min(args.frames, int(round((probe["seconds"] or 0) * FPS))) for _, probe in probes))
    ensure_server("ref2va")
    mode = args.ref_mode
    # a LoRA (Turbo adapter or hub) needs the bf16 file: merged into the rotated 8-bit weights it corrupts silently
    name = REF2VA["bf16"] if mode == "turbo" or loras or not installed("unet", REF2VA["int8"]) else REF2VA["int8"]
    missing = [f for kind, f in (("unet", name),) + ((("lora", REF2VA_TURBO),) if mode == "turbo" else ()) if not installed(kind, f)]
    if missing:
        sys.exit(f"not installed: {', '.join(missing)} (bash oneshot-setup-spark.sh --ref2va)")
    # the bf16 and the 8-bit transformer never stay loaded together
    ensure_server("ref2va-" + name + ("+lora" if loras else ""))
    steps = args.steps if args.steps is not None else (4 if mode == "turbo" else 20)
    if mode == "turbo" and steps != 4:
        print(f"[tensorfold] the Ref2VA Turbo adapter was trained for 4 passes; running {steps} as asked", flush=True)
    prompt = prompt_of(args)
    if not args.ref_plain and not any(tag in prompt for tag in ("<Picture", "<Video")):
        named = [f"<Picture {i + 1}>" for i in range(len(pictures))] or [f"<Video {i + 1}>" for i in range(len(videos))]
        prompt = "Use " + " and ".join(named) + " as the subject. " + prompt
    model = ["1", 0]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_TEXT, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": prompt, "width": args.width,
                         "height": args.height, "length": args.frames, "ref_image_size": args.ref_size}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        # the adapter was trained on plain Euler steps; the 20-step model on the multistep sampler ComfyUI ships it with
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler" if mode == "turbo" else "res_multistep"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"studio/ref_{uuid.uuid4().hex[:8]}",
                                                     "format": "auto", "codec": "auto"}},
    }
    for i, picture in enumerate(pictures):
        w[str(30 + i)] = {"class_type": "LoadImage", "inputs": {"image": upload(picture)}}
        w["6"]["inputs"][f"ref_images.ref_image_{i}"] = [str(30 + i), 0]
    seconds, said = args.frames / FPS, []
    with tempfile.TemporaryDirectory(prefix="studio-ref-") as scratch:
        # numbered as on the Mac: pictures, videos (a video's own sound is the <Audio j> before it), audio clips
        for i, ((path, probe), size) in enumerate(zip(probes, sizes)):
            clip, frames = prepared_video(path, size, args.frames, Path(scratch))
            w[str(40 + i)] = {"class_type": "LoadVideo", "inputs": {"file": upload(str(clip))}}
            w[str(50 + i)] = {"class_type": "GetVideoComponents", "inputs": {"video": [str(40 + i), 0]}}
            w["6"]["inputs"][f"ref_videos.ref_video_{i}"] = [str(50 + i), 0]
            used = frames - (frames - 5) % 17  # the node cuts the count down to 17n + 5
            said.append(f"<Video {i + 1}> {size[0]}x{size[1]}, {used} frames ({used / FPS:.1f} s)")
            if args.ref_video_sound:
                if not probe["audio"]:
                    sys.exit(f"{path} has no sound: leave --ref-video-sound out, or give videos that have sound")
                sound = prepared_sound(path, min(seconds, frames / FPS), Path(scratch))
                w[str(60 + i)] = {"class_type": "LoadAudio", "inputs": {"audio": upload(str(sound))}}
                w["6"]["inputs"][f"ref_video_audios.ref_video_audio_{i}"] = [str(60 + i), 0]
        for i, (path, probe) in enumerate(sounds):
            sound = prepared_sound(path, seconds, Path(scratch))
            w[str(70 + i)] = {"class_type": "LoadAudio", "inputs": {"audio": upload(str(sound))}}
            w["6"]["inputs"][f"ref_audios.ref_audio_{i}"] = [str(70 + i), 0]
    if mode == "turbo":
        w["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": REF2VA_TURBO, "strength_model": 1.0}}
        model = ["2", 0]
    for index, (lora, strength) in enumerate(loras):
        if not installed("lora", lora):
            sys.exit(f"ComfyUI does not list the LoRA {lora}")
        w[str(80 + index)] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": lora, "strength_model": strength}}
        model = [str(80 + index), 0]
        print(f"[tensorfold] LoRA {lora} at {strength:g} on the bf16 Ref2VA weights", flush=True)
    w["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    w["11"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}}
    guidance = float(os.environ.get("SPARK_REF_CFG") or 1.0)
    if guidance > 1.0:
        # an experiment: the checkpoint is guidance-distilled; CFG runs every pass twice (prompt, and an empty one)
        w["7"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}}
        w["11"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["6", 0], "negative": ["7", 0], "cfg": guidance}}
        print(f"[tensorfold] guidance {guidance:g} (experimental: two forwards a pass)", flush=True)
    w["12"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["6", 1]}}
    save_sound(w, ["14", 0], w["16"]["inputs"]["filename_prefix"] + "_sound")
    cropped = crop_frames(w, args.crop, args.width, args.height) if args.crop else True
    kind = "Ref2VA with the Turbo adapter" if mode == "turbo" else "Ref2VA"
    count = (f"{len(pictures)} reference picture(s)" + (f", {len(videos)} video(s)" if videos else "")
             + (f", {len(audios)} audio clip(s)" if audios else "") + (", the videos' own sound" if videos and args.ref_video_sound else ""))
    print(f"[tensorfold] engine: cuda (ComfyUI, MiniMax H3 {kind}, {name.split('pruned_')[1].split('.')[0]}), "
          f"{count}, {steps} passes, generating {args.width}x{args.height}, {args.frames} frames", flush=True)
    for line in said:
        print(f"[tensorfold] reference {line}", flush=True)
    try:
        item, seconds_taken = render(w, "12")
    finally:
        forget_uploads()
    target = Path(args.output)
    fetch(item, target)
    finish_sound(item, target, args.frames)
    if not cropped:
        crop_file(target, args.crop)
    print("[tensorfold] " + json.dumps({"output": str(target), "engine": "cuda", "model": "ref2va", "mode": mode,
                                        "forwards": steps, "total_s": round(seconds_taken, 1), "frames": args.frames,
                                        "size": [args.width, args.height], "references": len(pictures),
                                        "reference_videos": len(videos), "reference_audios": len(audios),
                                        "loras": [[lora, strength] for lora, strength in loras]}), flush=True)


def prompt_of(args) -> str:
    text = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    if not text or not text.strip():
        sys.exit("give --prompt or --prompt-file")
    return text.strip()


def video(args) -> None:
    for name, value, step in (("width", args.width, 32), ("height", args.height, 32)):
        if value % step:
            sys.exit(f"--{name} must be a multiple of {step}, got {value}")
    if (args.frames - 5) % 17:
        sys.exit(f"--frames must be 17n + 5 (22, 39, 56, ... 362), got {args.frames}")
    if args.ref or args.ref_video or args.ref_audio:
        return reference(args)
    base = args.base == "h3"
    if args.steps is None:
        args.steps = 25 if base else 8
    loras = hub_loras(args.lora)
    if loras:
        # the engine merges a LoRA into its int8 weights; ComfyUI merges the rest into bf16, never the rotated 8-bit
        if args.engine == "tensorfold" and not memory_guard.engine_takes_lora(ENGINE_LIB):
            print("[tensorfold] this engine build cannot take a LoRA (spark/build-engine.sh updates it): "
                  "ComfyUI's own blocks with the bf16 weights", flush=True)
            args.engine = "comfy"
        if args.weights != "bf16":
            print("[tensorfold] a LoRA renders with the bf16 weights", flush=True)
        args.weights = "bf16"
    if base:
        # the undistilled model: ComfyUI's own blocks, dense attention, its own schedule (video shift 12, audio 3)
        args.engine, args.weights, args.sparsity = "comfy", "bf16", 0.0
    sparsity = args.sparsity
    if sparsity is None and args.engine == "tensorfold":
        sparsity = 0.8    # what FastH3 was trained with; the engine's sparse kernel is the faster one at every size
    if sparsity is None:
        # sparse attention pays off once video rows dominate (on a GB10 from about 1344x768)
        sparsity = 0.8 if args.width * args.height >= 700_000 else 0.0
    if args.engine == "tensorfold" and not ENGINE_LIB.exists():
        sys.exit(f"no TensorFold engine at {ENGINE_LIB}: run spark/build-engine.sh, or use --engine comfy")
    last = STATE.read_text().strip() if STATE.exists() else ""
    on_comfy = bool(loras) and args.engine != "tensorfold"
    if up() and (on_comfy or base or "lora" in last or "h3base" in last):
        # the GPU shares memory with the system: bf16 weights plus a reloaded text encoder do not fit twice; start empty
        fresh_server("a LoRA render starts from a server with nothing loaded")
    fits_or_exit(args, bool(loras), False, base)
    # a LoRA on the engine needs no server of its own: the node merges it and puts the weights back after
    ensure_server("video" + ("+h3base" if base else "") + ("+lora" if on_comfy else ""))
    if base and not installed("unet", H3_FL2VA):
        sys.exit(f"MiniMax H3's own transformer ({H3_FL2VA}) is not installed: oneshot-setup-spark.sh --h3-base adds it "
                 "(40 GB), or render on FastH3 without --base h3")
    model = ["2", 0]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_FL2VA if base else FASTH3[args.weights], "weight_dtype": "default"}},
        # FastH3's schedule: video shift 10, audio shift 3 (fastvideo_inference.json); MiniMax H3's own: 12 and 3
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 12.0 if base else 10.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_TEXT, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": prompt_of(args), "width": args.width,
                         "height": args.height, "length": args.frames}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep" if base else "euler"}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"studio/clip_{uuid.uuid4().hex[:8]}",
                                                     "format": "auto", "codec": "auto"}},
    }
    if args.first_frame:
        w["20"] = {"class_type": "LoadImage", "inputs": {"image": upload(args.first_frame)}}
        w["6"]["inputs"]["first_frame"] = ["20", 0]
    if args.last_frame:
        # the end keyframe (first-and-last-frame mode): held at the last latent frame, centre-cropped beside a first
        w["23"] = {"class_type": "LoadImage", "inputs": {"image": upload(args.last_frame)}}
        w["6"]["inputs"]["last_frame"] = ["23", 0]
    for index, (name, strength) in enumerate(loras):
        if not installed("lora", name):
            sys.exit(f"ComfyUI does not list the LoRA {name}")
        node = str(30 + index)
        w[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": w["2"]["inputs"]["model"], "lora_name": name, "strength_model": strength}}
        w["2"]["inputs"]["model"] = [node, 0]
    if args.engine == "tensorfold":
        # every pass's 50 blocks on TensorFold's CUDA int8 engine with FastH3's sparse attention; ComfyUI does the rest
        w["22"] = {"class_type": "TensorFoldH3Blocks",
                   "inputs": {"model": model, "unet_name": FASTH3["bf16"], "sparsity": sparsity}}
        model = ["22", 0]
    elif sparsity > 0:
        # ComfyUI's block-sparse kernel with FastVideo's selection and FastH3's gates; text and audio rows stay exact
        w["21"] = {"class_type": "BlockSparseAttention",
                   "inputs": {"model": model, "selection": "vsa", "selection.keep_percent": round(100 * (1 - sparsity), 1),
                              "start_percent": 0.0, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 0,
                              "extra_tokens": 0, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["21", 0]
    w["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": args.steps,
                                                          "denoise": 1.0}}
    w["11"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["6", 0], "negative": ["7", 0],
                                                     "cfg": 1.0}}
    save_sound(w, ["14", 0], w["16"]["inputs"]["filename_prefix"] + "_sound")
    cropped = crop_frames(w, args.crop, args.width, args.height) if args.crop else True
    attention = f"sparse attention {sparsity}" if sparsity > 0 else "dense attention"
    runs = ("TensorFold int8 blocks in ComfyUI" + (", the LoRA merged into them" if loras else "")) if args.engine == "tensorfold" else "ComfyUI, MiniMax H3 bf16" if base else f"ComfyUI, FastH3 {args.weights}"
    print(f"[tensorfold] engine: cuda ({runs}, {attention}), generating "
          f"{args.width}x{args.height}, {args.frames} frames", flush=True)
    item, seconds = render(w, "12")
    target = Path(args.output)
    fetch(item, target)
    finish_sound(item, target, args.frames)
    if not cropped:
        crop_file(target, args.crop)
    print("[tensorfold] " + json.dumps({"output": str(target), "engine": "cuda", "forwards": args.steps,
                                        "total_s": round(seconds, 1), "frames": args.frames,
                                        "size": [args.width, args.height], "sparsity": sparsity,
                                        **({"cropped_to": args.crop} if args.crop else {})}), flush=True)


def image(args) -> None:
    for name, value in (("width", args.width), ("height", args.height)):
        if value % 16:
            sys.exit(f"--{name} must be a multiple of 16, got {value}")
    ensure_server("image")
    text = prompt_of(args)
    target = Path(args.output)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    for seed in seeds:
        w = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_BASE if args.base else QWEN,
                                                         "weight_dtype": "default"}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TEXT, "type": "qwen_image", "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
            "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "prompt": text,
                                                                    "negative_prompt": args.negative, "resolution": 1024}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
            "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/image_{uuid.uuid4().hex[:8]}"}},
        }
        if args.base:
            w["12"] = {"class_type": "KSampler",
                       "inputs": {"model": ["1", 0], "seed": seed, "steps": args.steps or 40, "cfg": args.cfg,
                                  "sampler_name": "euler", "scheduler": "simple", "positive": ["5", 0],
                                  "negative": ["5", 1], "latent_image": ["6", 0], "denoise": 1.0}}
        else:
            w["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
            w["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
            w["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}}
            w["11"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0],
                                                             "negative": ["5", 1], "cfg": 1.0}}
            w["12"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                                  "latent_image": ["6", 0]}}
        item, seconds = render(w, "12")
        out = target.with_name(f"{target.stem}_s{seed}{target.suffix}") if args.seeds else target
        fetch(item, out)
        print("[tensorfold] " + json.dumps({"output": str(out), "engine": "cuda", "width": args.width,
                                            "height": args.height, "seed": seed, "turbo": not args.base,
                                            "total_s": round(seconds, 1)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    kinds = parser.add_subparsers(dest="kind", required=True)
    v = kinds.add_parser("video")
    v.add_argument("--prompt", default=""); v.add_argument("--prompt-file", default="")
    v.add_argument("-o", "--output", required=True)
    v.add_argument("--width", type=int, default=864); v.add_argument("--height", type=int, default=480)
    v.add_argument("--frames", type=int, default=124); v.add_argument("--seed", type=int, default=0)
    v.add_argument("--steps", type=int, default=None, help="passes: 8 is what FastH3 was trained for (the default); "
                   "with --ref, 4 with the Turbo adapter and 20 without")
    v.add_argument("--ref", action="append", default=[], help="a reference picture (repeat for up to 9): the clip "
                   "comes from MiniMax H3 Ref2VA and `<Picture 1>` in the prompt is the first one")
    v.add_argument("--ref-video", action="append", default=[], help="a reference video (repeat for up to 3, each 2 to "
                   "15 s, 15 s in all): `<Video 1>`, ... in the prompt; a subject and how it moves")
    v.add_argument("--ref-audio", action="append", default=[], help="a reference audio clip (repeat for up to 3, each 2 "
                   "to 15 s, 15 s in all): `<Audio 1>`, ...; only beside a picture or a video")
    v.add_argument("--ref-video-sound", action="store_true", help="each reference video's own sound is a reference "
                   "too, numbered `<Audio 1>`, ... ahead of the --ref-audio clips")
    v.add_argument("--ref-video-size", choices=("match",), default="match",
                   help="a reference video is brought down to the clip's pixel area. The model's own 768-pixel canvas "
                        "(REF_VIDEO_SIZE=full on the Mac) is several times the rows and the memory and is not offered "
                        "on a Spark")
    v.add_argument("--ref-mode", choices=("turbo", "full"), default="full",
                   help="full: 20 steps; turbo: the bf16 model with LightX2V's 4-pass adapter")
    v.add_argument("--ref-size", choices=("match", "max"), default="match",
                   help="match: references scaled down to the clip's pixel area (what the adapter was trained with); "
                        "max: up to 2048 pixels on the short edge, several times slower")
    v.add_argument("--first-frame", default="", help="an image the clip starts from")
    v.add_argument("--last-frame", default="", help="an image the clip ends on (with or without --first-frame)")
    v.add_argument("--text-encoder", default=os.environ.get("TEXT_ENCODER") or "regular",
                   help="regular, or the file name of another MiniMax H3 text encoder in ComfyUI's text_encoders folder")
    v.add_argument("--crop", default="", help="WxH centre crop of the finished clip")
    v.add_argument("--weights", choices=sorted(FASTH3), default=os.environ.get("SPARK_WEIGHTS", "bf16"))
    v.add_argument("--engine", choices=("tensorfold", "comfy"),
                   default=os.environ.get("SPARK_ENGINE") or ("tensorfold" if ENGINE_LIB.exists() else "comfy"),
                   help="what runs the transformer blocks: TensorFold's CUDA family (the default once "
                        "spark/build-engine.sh has built it) or ComfyUI's own")
    v.add_argument("--base", choices=("fasth3", "h3"), default=os.environ.get("SPARK_BASE") or "fasth3",
                   help="the transformer of a clip: FastH3 (the default), or MiniMax H3's own first-and-last-frame "
                        "model when it is installed (25 passes by default; what keyframe LoRAs were trained on)")
    v.add_argument("--lora", action="append", default=[], help="a style LoRA, PATH or PATH:STRENGTH (repeat to stack)")
    v.add_argument("--ref-plain", action="store_true",
                   help="with references: the prompt as written, without `Use <Video 1> as the subject.` put ahead "
                        "of a prompt that names no reference (a LoRA trained on plain captions, such as line art)")
    v.add_argument("--sparsity", type=float, default=None, help="sparse attention (0 is dense); default by size")
    v.set_defaults(run=video)
    i = kinds.add_parser("image")
    i.add_argument("--prompt", default=""); i.add_argument("--prompt-file", default="")
    i.add_argument("-o", "--output", required=True)
    i.add_argument("--width", type=int, default=1344); i.add_argument("--height", type=int, default=768)
    i.add_argument("--seed", type=int, default=0); i.add_argument("--seeds", default="", help="several pictures: 1,2,3")
    i.add_argument("--model", choices=("qwen",), default=os.environ.get("IMAGE_MODEL") or "qwen", help="qwen: Qwen-Image-2.1-Turbo")
    i.add_argument("--base", action="store_true", help="the base model without the turbo adapter")
    i.add_argument("--steps", type=int, default=None, help="Qwen's base model (40)")
    i.add_argument("--cfg", type=float, default=4.0, help="base model only")
    i.add_argument("--negative", default="")
    i.set_defaults(run=image)
    args = parser.parse_args()
    if getattr(args, "text_encoder", "regular") != "regular":
        global H3_TEXT
        H3_TEXT, spark_comfy.TEXT_TAG = args.text_encoder, "+" + args.text_encoder
        ensure_server("video")
        if H3_TEXT not in call("/object_info/CLIPLoader")["CLIPLoader"]["input"]["required"]["clip_name"][0]:
            sys.exit(f"ComfyUI does not list the text encoder {H3_TEXT}")
    try:
        args.run(args)
    except Exception as error:  # noqa: BLE001 - the server went away under the render: say why, not a traceback
        if not isinstance(error, (OSError, ConnectionError)) and type(error).__module__.split(".")[0] != "aiohttp":
            raise
        sys.exit(stopped() or f"ComfyUI stopped answering during the render: {error} (log: {PREFIX}/comfy.log)")


if __name__ == "__main__":
    main()
