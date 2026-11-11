import argparse
import collections
import contextlib
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((root / "sam3").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def flatten(torch, values, prefix, output, channels_last=False):
    if isinstance(values, torch.Tensor):
        if values.is_floating_point() and not torch.isfinite(values).all():
            raise ValueError(f"Non-finite reference tensor: {prefix}")
        if values.ndim == 4 and not channels_last:
            values = values.permute(0, 2, 3, 1)
        if prefix.endswith("obj_ptr"):
            values = values.reshape(-1, 256)
        elif prefix.endswith("object_score_logits"):
            values = values.reshape(-1)
        output[prefix] = values.detach().to("cpu").contiguous().clone()
    elif isinstance(values, dict):
        for key, value in values.items():
            flatten(torch, value, f"{prefix}.{key}", output, channels_last)
    elif isinstance(values, (list, tuple)):
        for index, value in enumerate(values):
            flatten(torch, value, f"{prefix}.{index}", output, channels_last)
    elif hasattr(values, "tensors"):
        flatten(torch, values.tensors, prefix, output, channels_last)


def write_tensors(path, tensors):
    from safetensors.torch import save

    with path.open("xb") as file:
        file.write(save(tensors))


def capture(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source.resolve()))
    os.environ["SAM3_DEVICE"] = args.device
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    from sam3 import build_sam3_predictor
    from sam3.model_builder import build_sam3_video_model
    from sam3.model.data_misc import NestedTensor
    from sam3.model.vl_combiner import TriHeadVisionOnly

    source_identity = source_hash(args.source)
    capture_identity = file_hash(__file__)

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no CPU fallback is permitted")
    if args.precision == "bf16" and args.device == "cpu":
        raise ValueError("Use FP32 for the CPU reference")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    frames = load_file(args.frames, device="cpu")["frames"]
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected U8 frames [N,H,W,3]")
    count, height, width = frames.shape[:3]
    direction = -1 if args.reverse else 1
    indices = [args.start + direction * step for step in range(args.count)]
    if not indices or min(indices) < 0 or max(indices) >= count:
        raise ValueError("Frame range is outside the input video")
    reference = load_file(args.seed_reference, device="cpu")["masks"]
    if reference.dtype != torch.uint8 or tuple(reference.shape) != (count, height, width):
        raise ValueError("Expected U8 seed masks [N,H,W]")
    seed_bytes = reference[args.start]
    if not torch.all((seed_bytes == 0) | (seed_bytes == 255)):
        raise ValueError("Expected binary seed mask (0 or 255)")
    seed = seed_bytes.float()[None, None] / 255
    selected = {int(frame) for frame in args.trace_frames.split(",") if frame}
    if selected - set(indices):
        raise ValueError("A requested trace frame is outside the captured range")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = file_hash(args.checkpoint)
    if args.checkpoint.suffix == ".safetensors":
        weights = load_file(args.checkpoint, device="cpu")
        torch_checkpoint = args.torch_checkpoint or args.output / "checkpoint.pt"
        if args.torch_checkpoint:
            container = torch.load(torch_checkpoint, map_location="cpu", weights_only=True)
            container = container.get("model", container)
            if container.keys() != weights.keys() or any(
                container[key].dtype != value.dtype or container[key].shape != value.shape or
                not torch.equal(container[key].contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))
                for key, value in weights.items()
            ):
                raise ValueError("Torch container does not contain the exact source checkpoint tensors")
            del container
        else:
            with torch_checkpoint.open("xb") as file:
                torch.save({"model": weights}, file)
    else:
        torch_checkpoint = args.checkpoint
        weights = torch.load(torch_checkpoint, map_location="cpu", weights_only=True)
        if "model" in weights:
            weights = weights["model"]
    weight_dtypes = dict(collections.Counter(str(tensor.dtype) for tensor in weights.values()))
    keys = set(weights)
    with (args.output / "build.log").open("x") as log, contextlib.redirect_stdout(log):
        if args.version == "sam3.1":
            predictor = build_sam3_predictor(
                version=args.version, checkpoint_path=str(torch_checkpoint),
                max_num_objects=1, multiplex_count=16, use_fa3=False, use_rope_real=True,
                compile=False, warm_up=False, async_loading_frames=False,
            )
            model = predictor.model
        else:
            model = build_sam3_video_model(
                checkpoint_path=str(torch_checkpoint), load_from_HF=False,
                device=args.device, compile=False, strict_state_dict_loading=False,
            )
    model = model.eval().float().to(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    allowed_unused = {"detector.backbone.language_backbone.encoder.text_projection"}
    missing = set(dict(model.named_parameters())) - keys
    if missing - allowed_unused:
        raise ValueError(f"Unloaded learned parameters: {sorted(missing - allowed_unused)}")
    for name, value in model.named_parameters():
        if name in allowed_unused:
            continue
        actual = value.detach().to("cpu").contiguous()
        expected = weights[name].float().contiguous()
        if actual.dtype != torch.float32 or actual.shape != expected.shape or not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            raise ValueError(f"Learned parameter differs from source checkpoint: {name}")
    del weights, actual, expected
    multiplex = args.version == "sam3.1"
    tracker = model.tracker.model if multiplex else model.tracker
    tracker.backbone = TriHeadVisionOnly(
        visual=model.detector.backbone.vision_backbone, n_features=256, scalp=0,
    ).eval() if multiplex else model.detector.backbone
    state = tracker.multiplex_controller.get_state(
        num_valid_entries=1, device=torch.device(args.device), dtype=torch.float32,
        random=False, object_ids=[args.object_id],
    ) if multiplex else None
    current_trace = {}

    def record_trunk(module, inputs, output):
        if current_trace.get("enabled"):
            flatten(torch, output, "trunk", current_trace["tensors"])

    visual = model.detector.backbone.vision_backbone
    hooks = [visual.trunk.register_forward_hook(record_trunk)]

    def record_pyramid(prefix, channels_last=False):
        def record(module, inputs, output):
            if current_trace.get("enabled"):
                flatten(torch, output, prefix, current_trace["tensors"], channels_last)
        return record

    for branch, kind in (("propagation_convs", "propagation"), ("interactive_convs", "interactive")) if multiplex else (("sam2_convs", "propagation"),):
        for index, neck in enumerate(getattr(visual, branch)):
            hooks.append(neck.register_forward_hook(record_pyramid(f"pyramid.{kind}.{index}")))
    if args.trace_blocks:
        for name in ("patch_embed", "ln_pre"):
            hooks.append(getattr(visual.trunk, name).register_forward_hook(record_pyramid(f"trunk.{name}", True)))
        for index, block in enumerate(visual.trunk.blocks):
            hooks.append(block.register_forward_hook(record_pyramid(f"trunk.blocks.{index}", True)))
    history = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    masks = torch.empty((args.count, height, width), dtype=torch.uint8)
    actual_dtypes = {}
    input_hashes = {}
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    with torch.inference_mode(), torch.autocast(args.device, enabled=args.precision != "fp32", dtype=dtype):
        for step, index in enumerate(indices):
            current_trace = {"enabled": index in selected, "tensors": {}}
            if args.prepared_input:
                prepared = load_file(args.prepared_input / f"frame-{index:06d}.safetensors", device="cpu")["preprocessed_nhwc"]
                if prepared.dtype != torch.float32 or tuple(prepared.shape) != (1, 1008, 1008, 3):
                    raise ValueError("Prepared image must be F32 [1,1008,1008,3]")
                if not torch.isfinite(prepared).all():
                    raise ValueError("Non-finite prepared image")
                input_hashes[str(index)] = hashlib.sha256(prepared.numpy()).hexdigest()
                image = prepared.permute(0, 3, 1, 2).to(args.device)
            else:
                image = frames[index].to(args.device, dtype=torch.float32).permute(2, 0, 1)[None] / 127.5 - 1
                image = functional.interpolate(image, (1008, 1008), mode="bilinear", align_corners=False)
            mask_input = functional.interpolate(seed.to(args.device), (1008, 1008), mode="bilinear", align_corners=False) if step == 0 else None
            common = dict(
                frame_idx=index, is_init_cond_frame=step == 0, image=image,
                point_inputs=None, mask_inputs=mask_input, output_dict=history,
                num_frames=count, track_in_reverse=args.reverse, run_mem_encoder=True,
            )
            if multiplex:
                backbone = tracker.forward_image(
                    NestedTensor(tensors=image, mask=None), need_sam3_out=False,
                    need_interactive_out=step == 0, need_propagation_out=True,
                )
                features = tracker._prepare_backbone_features(backbone)
                result = tracker.track_step(
                    **common, backbone_features_interactive=features.get("interactive"),
                    backbone_features_propagation=features["sam2_backbone_out"],
                    gt_masks=None, frames_to_add_correction_pt=[], multiplex_state=state,
                )
            else:
                backbone = tracker.forward_image(image)
                _, vision, positions, sizes = tracker._prepare_backbone_features(backbone)
                result = tracker.track_step(
                    **common, current_vision_feats=vision, current_vision_pos_embeds=positions,
                    feat_sizes=sizes,
                )
            history["cond_frame_outputs" if step == 0 else "non_cond_frame_outputs"][index] = result
            history["non_cond_frame_outputs"].pop(index - direction * 16, None)
            logits = functional.interpolate(result["pred_masks"].float(), (height, width), mode="bilinear", align_corners=False)
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite reference mask logits at frame {index}")
            masks[step] = (logits[0, 0] > 0).to(device="cpu", dtype=torch.uint8) * 255
            if current_trace["enabled"]:
                tensors = current_trace["tensors"]
                flatten(torch, image, "preprocessed_nhwc", tensors)
                flatten(torch, backbone, "backbone", tensors)
                for name in ("pred_masks", "pred_masks_high_res", "obj_ptr", "object_score_logits", "maskmem_features", "maskmem_pos_enc"):
                    if name in result:
                        flatten(torch, result[name], f"result.{name}", tensors)
                flatten(torch, logits, "output_logits_nhwc", tensors)
                if args.fp64_patch:
                    patch = visual.trunk.patch_embed.proj
                    reference64 = functional.conv2d(
                        image.detach().cpu().double(),
                        patch.weight.detach().cpu().double(),
                        None if patch.bias is None else patch.bias.detach().cpu().double(),
                        stride=patch.stride, padding=patch.padding,
                        dilation=patch.dilation, groups=patch.groups,
                    )
                    flatten(torch, reference64, "fp64.trunk.patch_embed", tensors)
                actual_dtypes[str(index)] = {name: str(value.dtype) for name, value in tensors.items()}
                if args.precision == "fp32" and any(value.dtype in (torch.bfloat16, torch.float16) for value in tensors.values()):
                    raise ValueError("FP32 reference produced reduced-precision stage tensors")
                write_tensors(args.output / f"frame-{index:06d}.safetensors", tensors)
            print(f"Reference {args.version} {args.precision}: frame {index}", flush=True)
    for hook in hooks:
        hook.remove()
    if source_hash(args.source) != source_identity or file_hash(__file__) != capture_identity:
        raise ValueError("Reference source changed during capture")
    write_tensors(args.output / "masks.safetensors", {
        "masks": masks, "object_ids": torch.tensor([args.object_id], dtype=torch.uint64),
        "frame_indices": torch.tensor(indices, dtype=torch.uint64),
        "active": torch.ones((args.count, 1), dtype=torch.uint8),
        "complete": torch.ones(1, dtype=torch.uint8),
    })
    report = {
        "schema": 1, "scope": "publisher core tracker, one fixed mask-seeded object",
        "version": args.version, "precision": args.precision, "device": args.device,
        "checkpoint_sha256": checkpoint_hash, "checkpoint_dtypes": weight_dtypes,
        "torch_container_sha256": file_hash(torch_checkpoint),
        "frames_file_sha256": file_hash(args.frames),
        "frames_tensor_sha256": hashlib.sha256(frames.numpy()).hexdigest(),
        "seed_reference_sha256": file_hash(args.seed_reference),
        "seed_mask_sha256": hashlib.sha256(seed_bytes.numpy()).hexdigest(),
        "source_tree_sha256": source_identity, "capture_tool_sha256": capture_identity,
        "masks_sha256": file_hash(args.output / "masks.safetensors"),
        "torch": torch.__version__, "python": platform.python_version(), "os": platform.platform(),
        "cpu_fallback": False, "compile": False, "torch_threads": args.threads,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_fp64_patch_reference": args.fp64_patch,
        "input_shape": list(frames.shape), "frame_indices": indices, "object_ids": [args.object_id],
        "history": "conditioning retained; non-conditioning expires at distance 16",
        "preprocessing": "float32 normalize /127.5-1, bilinear 1008x1008, align_corners=False",
        "provided_preprocessed_inputs": bool(args.prepared_input), "provided_input_hashes": input_hashes,
        "trace_layout": "rank-4 tensors use BHWC; pointers use object rows; scores use one dimension",
        "missing_unused_parameters": sorted(missing), "trace_dtypes": actual_dtypes,
        "used_learned_parameters_equal_source": True,
    }
    with (args.output / "manifest.json").open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Capture independent masks and stage tensors from the local Meta SAM source.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--torch-checkpoint", type=Path, help="Reuse a Torch container after exact tensor validation")
    parser.add_argument("--version", choices=("sam3", "sam3.1"), required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--prepared-input", type=Path, help="Use captured F32 image tensors to isolate model arithmetic")
    parser.add_argument("--seed-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--object-id", type=int, default=42)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--trace-frames", default="0,1")
    parser.add_argument("--trace-blocks", action="store_true")
    parser.add_argument("--fp64-patch", action="store_true", help="Also capture an independent CPU FP64 patch convolution")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.count <= 0 or not 0 <= args.object_id < 2**64:
        parser.error("positive --count and U64 --object-id are required")
    capture(args)


if __name__ == "__main__":
    main()
