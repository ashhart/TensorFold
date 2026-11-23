"""dots.mocr's image references: the model card's example oracle, its tower taps, and the stage compare."""
import importlib.metadata
import json
import os
import shutil
import time

import numpy as np

from dots_mocr_reference import (REVISION, ORACLE_VERSIONS, Scores, SdpaProbe, example_prompt, flash_or_stand_in,
                                 load_model, pinned, require, sdpa_backend, selected_pci, sha, tower_config)

IMAGE_VERSIONS = {"pillow": "12.2.0", "qwen-vl-utils": "0.0.10"}  # the demo fixture's pixels; Pillow 13 changes them
VISION_ROPE_POSITIONS = [0, 1, 239, 3393]  # the tower rotary contract's positions
# The tower stages a native dump is compared on; vision-patch-bias is the patch embed with its bias, before its norm.
TAPS = ["vision-patch-bias", "vision-patch", "vision-block0", "vision-norm", "vision-merged"]


def vision_rope(torch, model, output):
    """Dump the tower's rotary inv_freq as HF builds it on the CPU, and its FP32 angles, cos and sin on the device."""
    rotary = model.vision_tower.rotary_pos_emb
    cpu = type(rotary)(2 * rotary.inv_freq.numel()).inv_freq
    if cpu.device.type != "cpu" or not torch.equal(cpu, rotary.inv_freq.cpu()):
        raise RuntimeError("the tower's device inv_freq differs from the CPU table HF builds")
    bits = cpu.numpy().astype("<f4").view("<u4")
    (output / "vision_inv_freq.txt").write_text(",".join(f"0x{v:08x}" for v in bits) + "\n")
    with torch.inference_mode():
        angles = rotary(max(VISION_ROPE_POSITIONS) + 1)[VISION_ROPE_POSITIONS]
        for name, tensor in {"inv_freq": cpu, "angles": angles, "cos": angles.cos(), "sin": angles.sin()}.items():
            np.save(output / f"vision-rope-{name}.npy", tensor.float().cpu().numpy())
    np.save(output / "vision-rope-positions.npy", np.array(VISION_ROPE_POSITIONS, dtype=np.int64))
    n = np.arange(0, 2 * len(bits), 2, dtype=np.float32)
    numpy32 = (np.float32(1) / np.float32(10000) ** (n / np.float32(2 * len(bits)))).astype("<f4")
    return {"inv_freq_hex": [f"0x{v:08x}" for v in bits], "inv_freq": bits.view("<f4").tolist(),
            "differs_from_numpy_float32_at": np.flatnonzero(bits != numpy32.view("<u4")).tolist()}


def patch_bias_rounding(module, x, tap):
    """Whether HF's patch embed rounds its bias once, bf16(conv + b), or twice, bf16(bf16(conv) + b), on this GPU."""
    plain = module._conv_forward(x, module.weight, None).flatten(1)
    twice = (plain.float() + module.bias.float()).bfloat16()
    # FP64 sum then FP64 -> FP32 -> BF16: the FP32 step can only matter at an exact BF16 tie.
    once = (module._conv_forward(x.double(), module.weight.double(), None).flatten(1) + module.bias.double()).float().bfloat16()
    tap, differ = tap.flatten(1), twice != once
    counts = {"elements": tap.numel(), "equal_twice": int((tap == twice).sum()), "equal_once": int((tap == once).sum()),
              "hypotheses_differ": int(differ.sum()), "twice_where_differ": int((tap == twice)[differ].sum()),
              "once_where_differ": int((tap == once)[differ].sum())}
    split, full = counts["hypotheses_differ"], counts["elements"]
    verdict = ("undecided" if not split else "twice" if counts["equal_twice"] == full else "once" if counts["equal_once"] == full
               else "once where they split" if counts["once_where_differ"] == split else "neither")
    return {"verdict": verdict, **counts}


def oracle(args):
    """Run the HF example, its tower on --attention (SDPA without flash-attn); keep comparison inputs and BF16 taps."""
    require(ORACLE_VERSIONS | IMAGE_VERSIONS)
    source_sha256 = pinned(args.checkpoint)
    flash = flash_or_stand_in()
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from transformers.generation import LogitsProcessorList

    pci = selected_pci(torch)
    if pci != args.pci.lower():
        raise RuntimeError(f"selected GPU {pci} differs from requested GPU {args.pci}")
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else example_prompt(args.checkpoint)
    image_paths = [args.image, *args.extra_image]
    messages = [{"role": "user", "content": [*({"type": "image", "image": str(p)} for p in image_paths),
                                               {"type": "text", "text": prompt}]}]
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    prepared_images = []
    for i, image in enumerate(image_inputs):
        path = args.output / f"prepared-image-{i}.png"
        image.save(path)
        prepared_images.append({"size": list(image.size), "sha256": sha(path)})
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    for key, tensor in inputs.items():
        np.save(args.output / f"input-{key}.npy", tensor.cpu().numpy())
    config = tower_config(args.checkpoint, args.attention)
    patch_rows = inputs.pixel_values.shape[0]
    image_arrays = (config.vision_config.num_hidden_layers + 4 if args.vision_only else 5)
    disk_required = image_arrays * patch_rows * config.vision_config.embed_dim * 2 + (2 << 30)
    if not args.vision_only:
        disk_required += args.max_new_tokens * config.vocab_size * 4
    if shutil.disk_usage(args.output).free < disk_required:
        raise RuntimeError(f"oracle needs {disk_required} free disk bytes including headroom")
    # Scores go directly to disk. Reserve weights, maximum KV, tower scratch, and 2 GiB device headroom.
    kv_bytes = 0 if args.vision_only else 2 * config.num_hidden_layers * 2 * config.num_key_value_heads * (
        config.hidden_size // config.num_attention_heads) * (inputs.input_ids.shape[1] + args.max_new_tokens)
    required = 6_078_358_528 + (1 << 30) + (2 << 30) + kv_bytes
    free, _ = torch.cuda.mem_get_info()
    if free < required:
        raise RuntimeError(f"oracle needs {required} free bytes including headroom, found {free}")
    model = load_model(torch, args.checkpoint, config, args.attention)
    sdpa_backend(torch, args.sdpa_backend)
    rope = vision_rope(torch, model, args.output)
    captured, bias_rounding = set(), {}

    def capture(name):
        def hook(module, module_args, output):
            if name in captured:
                return
            captured.add(name)
            tensor = (output[0] if isinstance(output, tuple) else output).detach()
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} is {tensor.dtype}, not BF16")
            tensor = tensor.flatten(1) if tensor.dim() == 4 else tensor  # the patch conv's [n, embed, 1, 1]
            np.save(args.output / f"{name}.npy", tensor.contiguous().view(torch.int16).cpu().numpy().view("<u2"))
            if name == "vision-patch-bias" and args.vision_only:
                bias_rounding.update(patch_bias_rounding(module, module_args[0], tensor))
        return hook

    hooks = [model.vision_tower.patch_embed.patchifier.proj.register_forward_hook(capture("vision-patch-bias")),
             model.vision_tower.patch_embed.register_forward_hook(capture("vision-patch")),
             model.vision_tower.blocks[0].register_forward_hook(capture("vision-block0")),
             model.vision_tower.post_trunk_norm.register_forward_hook(capture("vision-norm")),
             model.vision_tower.register_forward_hook(capture("vision-merged")),
             model.model.layers[0].register_forward_hook(capture("text-block0")),
             model.model.norm.register_forward_hook(capture("text-norm"))]
    hooks.extend(block.register_forward_hook(capture(f"vision-block{i}"))
                 for i, block in enumerate(model.vision_tower.blocks) if i != 0 and args.vision_only)
    recorder = None if args.vision_only else Scores(args.output / "generation-scores.npy", config.vocab_size)
    try:
        inputs = inputs.to("cuda:0")
        started = time.monotonic()
        with torch.inference_mode(), SdpaProbe(torch) as probe:
            if args.vision_only:
                model.vision_tower(inputs.pixel_values, inputs.image_grid_thw)
            else:
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           return_dict_in_generate=True, output_scores=False, do_sample=False,
                                           logits_processor=LogitsProcessorList([recorder]))
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
    finally:
        if recorder is not None:
            recorder.close()
        for hook in hooks:
            hook.remove()
    ids = [] if args.vision_only else generated.sequences[0, inputs.input_ids.shape[1]:].cpu().tolist()
    ends = model.generation_config.eos_token_id
    end_ids = ends if isinstance(ends, list) else [ends]
    if recorder is not None and recorder.count != len(ids):
        raise RuntimeError("score row count differs from the generated token count")
    output_text = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result = {
        "checkpoint_revision": REVISION, "source_sha256": source_sha256,
        "image_sha256": sha(args.image), "images_sha256": [sha(p) for p in image_paths],
        "prepared_images": prepared_images, "prompt": prompt, "rendered_text": text,
        "prompt_ids": inputs.input_ids[0].cpu().tolist(), "output_ids": ids, "text": output_text,
        "output_ids_sha256_u32_le": sha(np.array(ids, dtype="<u4").tobytes()),
        "pixel_sha256": sha(inputs.pixel_values.float().cpu().numpy().astype("<f4").tobytes()),
        "image_grid_thw": inputs.image_grid_thw.cpu().tolist(),
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "attn_implementation": args.attention,
        "vision_attn_implementation": config.vision_config.attn_implementation, "sdpa_backend": args.sdpa_backend,
        "sdpa": probe.summary(),
        "vision_rope": rope, "patch_bias_rounding": bias_rounding or None, "vision_only": args.vision_only,
        "eos_reached": bool(ids and ids[-1] in end_ids),
        "seconds": seconds, "peak_device_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(), "pci": pci, "hip_version": torch.version.hip, "cuda_version": torch.version.cuda,
        "flash_triton_amd": os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE"), "flash_attn": flash,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "qwen-vl-utils", "numpy", "pillow"]},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_tokens": len(ids), "seconds": seconds, "text": output_text}, ensure_ascii=False), flush=True)


def reference_images(torch, reference_dir, prompt, config):
    """A reference's pixel_values and image_grid_thw on cuda:0 for score, checked against its prompt's image rows."""
    images, image_rows = {}, prompt.count(config.image_token_id)
    if (reference_dir / "input-pixel_values.npy").exists():
        pixels = np.load(reference_dir / "input-pixel_values.npy", allow_pickle=False)
        grid = np.load(reference_dir / "input-image_grid_thw.npy", allow_pickle=False)
        if (grid.ndim != 2 or grid.shape[1] != 3 or pixels.ndim != 2 or pixels.shape[0] != int(grid.prod(axis=1).sum())
                or image_rows * config.vision_config.spatial_merge_size ** 2 != pixels.shape[0]):
            raise ValueError("the reference's pixel rows, image grid and prompt image rows disagree")
        images = {"pixel_values": torch.from_numpy(pixels).to("cuda:0"), "image_grid_thw": torch.from_numpy(grid).to("cuda:0")}
    elif image_rows:
        raise ValueError("the prompt holds image rows but the reference has no input-pixel_values.npy")
    return images


def bf16_bits(path):
    """A tap's BF16 bit patterns, stored as <u2."""
    if (array := np.load(path, allow_pickle=False)).dtype != np.dtype("<u2"):
        raise ValueError(f"{path} does not hold BF16 bit patterns (<u2)")
    return array


def stages(args):
    """Compare native tower taps with an `oracle --vision-only` reference: max abs and a BF16-ulp histogram per tap."""
    if missing := [name for name in TAPS if not (args.native_dir / f"{name}.npy").exists()]:
        raise SystemExit(f"native taps missing: {', '.join(missing)}")
    blocks = sorted((p.stem for p in args.native_dir.glob("vision-block*.npy") if p.stem != "vision-block0"),
                    key=lambda name: int(name.removeprefix("vision-block")))
    report, labels = {}, ["0", "1", "2", "3-4", "5-8", "9-16", "17-64", ">64"]
    for name in TAPS + blocks:
        ref, native = bf16_bits(args.reference_dir / f"{name}.npy"), bf16_bits(args.native_dir / f"{name}.npy")
        x, y = ((a.astype(np.uint32) << 16).view(np.float32) for a in (ref, native))
        if ref.shape != native.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{name}: reference shape {ref.shape}, native shape {native.shape}, or nonfinite values")
        # Sign-magnitude bits to integers that count BF16 steps along the real line, with -0 equal to +0.
        steps = [np.where(a & 0x8000, 0x8000 - a.astype(np.int32), a.astype(np.int32)) for a in (ref, native)]
        ulps, difference = np.abs(steps[0] - steps[1]), np.abs(x - y)
        histogram, _ = np.histogram(ulps, bins=[0, 1, 2, 3, 5, 9, 17, 65, 1 << 16])
        report[name] = {"shape": list(ref.shape), "max_abs": float(difference.max()), "max_ulps": int(ulps.max()),
                        "max_abs_at": [int(i) for i in np.unravel_index(int(difference.argmax()), ref.shape)],
                        "mean_abs": float(difference.mean(dtype=np.float64)), "ulps": dict(zip(labels, map(int, histogram)))}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
