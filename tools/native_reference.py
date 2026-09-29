"""Numerical oracle for the native Qwen port (no native process launching).

Generate reference logits with this checkout's actual lane forward, or compare
already generated .npy files. Artifacts live under build/native-checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def vision_fixture(model_dir, output, height, width, image_fixture=False, image_format="PNG", image_alpha=False, image_orientation=1, image_only=False, image_mode=None):
    from native_runtime import require_mlx
    require_mlx()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.config import VisionConfig
    from mlx_vlm.models.qwen3_5.vision import VisionModel
    from tensorfold.vision.qwen_checkpoint import load_vision_weights, quantization_predicate, vision_tensors
    config = json.loads((Path(model_dir) / "config.json").read_text())
    out = Path(output)
    (out / "python").mkdir(parents=True, exist_ok=True)
    if image_fixture:
        from PIL import Image
        from tensorfold.vision.qwen_processing import QwenImageProcessor
        import base64
        from tensorfold.vision.images import ImageSource, load_images
        image = Image.fromarray(np.random.default_rng(314159).integers(0, 256, (height, width, 4 if image_alpha else 3), dtype=np.uint8))
        if image_mode:
            image = image.convert(image_mode)
        exif = Image.Exif()
        exif[274] = image_orientation
        image_path = out / ("image." + image_format.lower())
        image.save(image_path, format=image_format, exif=exif)
        image = load_images([ImageSource("data:image/" + image_format.lower() + ";base64," + base64.b64encode(image_path.read_bytes()).decode())])[0].to_pil()
        processor = QwenImageProcessor.from_directory(model_dir).processor
        processed = processor(images=[image], max_pixels=4096 * 1024, min_pixels=65536)
        pixels = np.asarray(processed["pixel_values"], dtype=np.float32)
        _, height, width = map(int, processed["image_grid_thw"][0])
    else:
        pixels = np.random.default_rng(314159).uniform(-1, 1, (height * width, 1536)).astype(np.float32)
    np.save(out / "pixels.npy", pixels)
    (out / "grid.json").write_text(json.dumps(dict(height=height, width=width)))
    if image_only:
        print(f"Saved upstream image preprocessing in {out}")
        return
    tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
    weights = tower.sanitize(load_vision_weights(vision_tensors(Path(model_dir)), mx))
    nn.quantize(tower, class_predicate=quantization_predicate(config, weights))
    tower.load_weights(list(weights.items()), strict=True)
    tower.eval()
    def save(name, x):
        mx.eval(x)
        np.save(out / "python" / f"{name}.npy", np.array(x.astype(mx.float32)))
    grid = mx.array([[1, height, width]], dtype=mx.int32)
    h = tower.patch_embed(mx.array(pixels).astype(tower.patch_embed.proj.weight.dtype))
    save("patch", h)
    position = tower.fast_pos_embed_interpolate(grid)
    save("position", position)
    h = h + position
    freq = tower.rot_pos_emb(grid)
    save("frequencies", mx.concatenate([freq, freq], -1).reshape(1, height * width, 1, 72))
    cu = mx.array([0, height * width], dtype=mx.int32)
    for i, block in enumerate(tower.blocks):
        h = block(h, cu, freq)
        save(f"block-{i}", h)
    save("embeddings", tower.merger(h))
    print(f"Saved upstream vision stages in {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/models/Qwen3.8-27B-MLX-4bit")
    parser.add_argument("--tokens", help="Exact prompt IDs, also honored during generation")
    parser.add_argument("--dump-logits", type=Path, help="Save the final prefill block before generation")
    parser.add_argument("--metal-sampling", action="store_true")
    parser.add_argument("--simd", action="store_true", help="Use the production SIMD decoder and its MLX attention arithmetic")
    parser.add_argument("--production-kernels", action="store_true", help="Diagnostic: use production loading/fusion with lane-tree prefill (not the serving engine's prefill)")
    parser.add_argument("--disable-fusion", action="store_true", help="Diagnostic: turn off stacked consumers after the production load")
    parser.add_argument("--output", default="build/native-checks/reference.npy")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--vision-fixture", nargs=2, type=int, metavar=("GRID_HEIGHT", "GRID_WIDTH"))
    parser.add_argument("--image-fixture", action="store_true", help="Vision fixture dimensions describe a generated RGB PNG before preprocessing")
    parser.add_argument("--image-format", choices=("PNG", "JPEG", "WEBP"), default="PNG")
    parser.add_argument("--image-alpha", action="store_true")
    parser.add_argument("--image-only", action="store_true", help="Generate preprocessing oracle without loading the vision tower")
    parser.add_argument("--image-mode", choices=("RGB", "RGBA", "L", "CMYK"))
    parser.add_argument("--image-orientation", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--compare-vision", type=Path)
    parser.add_argument("--compare-arrays", nargs=2, type=Path)
    parser.add_argument("--compare-reports", nargs="+")
    parser.add_argument("--require-rounds", action="store_true", help="Reject trivial EOS-before-decode parity runs")
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.vision_fixture:
        return vision_fixture(args.model, args.output, *args.vision_fixture, args.image_fixture, args.image_format, args.image_alpha, args.image_orientation, args.image_only, args.image_mode)
    if args.compare_vision or args.compare_arrays:
        reference, actual = args.compare_arrays or (args.compare_vision / "python", args.compare_vision / "native")
        files = sorted(reference.glob("*.npy"))
        assert files, "No oracle arrays"
        failed = []
        for path in files:
            a = np.load(path)
            b = np.load(actual / path.name)
            equal = a.shape == b.shape and np.array_equal(a, b)
            if not equal or len(files) <= 64:
                difference = np.max(np.abs(a - b)) if a.shape == b.shape else "shape mismatch"
                print(f"{path.stem}: {'PASS' if equal else 'FAIL'}, max difference {difference}")
            if not equal:
                failed.append(path.stem)
        assert not failed, failed
        print(f"PASS: {len(files)} arrays bit-exact")
        return
    if args.compare_reports:
        reports = [json.loads(Path(p).read_text()) for p in args.compare_reports]
        if args.require_rounds:
            for report in reports:
                assert report["rounds"] > 0 and len(report["tokens"]) > 1, "No decode rounds exercised"
        for report in reports[1:]:
            assert report["prompt_tokens"] == reports[0]["prompt_tokens"], "Tokenizer mismatch"
            assert report["tokens"] == reports[0]["tokens"], "Output token mismatch"
        print(f"PASS: {len(reports)} reports have identical prompts and {len(reports[0]['tokens'])} output token IDs")
        return
    if args.compare:
        a, b = [np.load(p) for p in args.compare]
        if b.dtype.kind == "V" and b.dtype.itemsize == 2:
            b = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        print(f"shapes: {a.shape} / {b.shape}")
        print(f"equal elements: {np.count_nonzero(a == b)}/{a.size}")
        print(f"max absolute difference: {np.max(np.abs(a - b))}")
        if a.size // a.shape[-1] <= 32:
            print(f"argmax reference: {a.argmax(axis=-1).tolist()}")
            print(f"argmax native: {b.argmax(axis=-1).tolist()}")
        if not np.array_equal(a, b):
            raise SystemExit(1)
        return
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from tensorfold.families.qwen3_5 import load_lane_model
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_tree

    bonsai = json.loads((Path(args.model) / "config.json").read_text()).get("model_type") == "prism_hadamard_qwen35"
    if bonsai:
        from tensorfold.families.bonsai import pack
        from mlx_lm.utils import load_tokenizer
        model = pack.build(Path(args.model), form="packed" if args.simd else "lanes")
        tokenizer = load_tokenizer(Path(args.model))
        if args.simd:
            from tensorfold.kernels.qwen.dense.v1 import row_forward
            from tensorfold.families.qwen3_5 import install_row_decoder
            if not install_row_decoder(model):
                raise ValueError("Bonsai row decoder rejected this checkpoint")
            tree_forward, commit_tree = row_forward.forward, row_forward.commit
    elif args.simd:
        if args.production_kernels or args.disable_fusion:
            parser.error("--simd uses the production decoder without fusion overrides")
        from tensorfold.families.qwen3_5 import load
        from tensorfold.kernels.qwen.dense.v1 import row_forward
        family, tokenizer = load(Path(args.model), lane_kernels="off")
        model = family.inner
        tree_forward, commit_tree = row_forward.forward, row_forward.commit
    elif args.production_kernels:
        from tensorfold.families.qwen3_5 import load
        family, tokenizer = load(Path(args.model))
        model = family.inner
        if args.disable_fusion:
            from tensorfold.kernels.qwen.dense.v1 import lane_fuse
            lane_fuse.enabled = False
    else:
        if args.disable_fusion:
            parser.error("--disable-fusion requires --production-kernels")
        model, tokenizer = load_lane_model(Path(args.model))
    if not args.simd:
        tree_forward, commit_tree = lane_tree.tree_forward, lane_tree.commit_tree
        if not args.production_kernels:
            lane_qmm.install(model, rows=128, tile=True, wide=True)
    core = model.language_model.model
    head = model.language_model.lm_head
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    if not tokens:
        raise ValueError("Empty prompt")
    cache = make_prompt_cache(model)
    # Same 128-row grid as the native CLI; commit the whole prompt before decoding.
    for start in range(0, len(tokens), 128):
        block = tokens[start:start + 128]
        if args.simd:
            # Production SIMD verification uses windows of at most 16 rows;
            # wider calls switch to regular prompt attention arithmetic.
            parts = []
            for offset in range(0, len(block), 16):
                rows = block[offset:offset+16]
                part, record = tree_forward(core, head, rows, list(range(-1, len(rows)-1)), cache, start + offset)
                mx.eval(part)
                commit_tree(cache, record, list(range(len(rows))), len(rows), start + offset)
                parts.append(part)
            logits = mx.concatenate(parts, axis=1)
        else:
            logits, record = tree_forward(core, head, block, list(range(-1, len(block)-1)), cache, start)
            mx.eval(logits)
            commit_tree(cache, record, list(range(len(block))), len(block), start)
        if start % 1024 == 0:
            print(f"Prefill {start + len(block)}/{len(tokens)}", flush=True)
    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_logits, np.array(logits.astype(mx.float32)))
    if args.generate:
        from tensorfold.engine.exact_sampling import Sampling, sample_rows
        settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        def select(logits, position):
            last = logits[0, -1:]
            if args.metal_sampling:
                from tensorfold.engine.gpu_sampling import sample
                return int(sample(last, settings if args.temperature else None, [position]).item())
            return sample_rows(last, [position], settings)[0] if args.temperature else int(mx.argmax(last).item())
        position = len(tokens)
        pending = select(logits, position)
        generated = [pending]
        while len(generated) < args.generate and pending not in (248044, 248046):
            logits, record = tree_forward(core, head, [pending], [-1], cache, position)
            commit_tree(cache, record, [0], 1, position)
            position += 1
            pending = select(logits, position)
            generated.append(pending)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(np.array(generated, dtype="<u4").tobytes()).hexdigest()
        out.write_text(json.dumps(dict(prompt_tokens=tokens, tokens=generated, token_sha256=sha,
                                      peak_mlx_bytes=mx.get_peak_memory(), active_mlx_bytes=mx.get_active_memory())))
        print(f"Saved {out}: {len(generated)} tokens, SHA-256 {sha}")
        return
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(logits.astype(mx.float32)))
    print(f"Saved {out}: shape={logits.shape}, argmax={mx.argmax(logits, axis=-1).tolist()}")


if __name__ == "__main__":
    main()
