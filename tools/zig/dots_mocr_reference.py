"""dots.mocr references: the model card's Hugging Face example oracle, processor fixtures, native trace compare."""
import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

REVISION = "e539fbb52280393adc081b289ec597430a0f9031"
ORACLE_VERSIONS = {"transformers": "4.57.6"}
PROCESSOR_VERSIONS = {"transformers": "4.51.0", "pillow": "11.2.1", "numpy": "2.2.5"}
# The revision's non-weight sources: code, configuration, tokenizer, template, processor and the model card.
SOURCES = {
    "README.md": "381645b2bd63c2adff9d0c266f79e86b25657394c48c35e68551f1da73178259",
    "chat_template.json": "c8d629ba2b3133376f1e64b980f82438945741fba5334db768afb281af93c4a9",
    "config.json": "41d5c9aad72c730bb97062cab7a59cdb1d427eeec0b7a6e2f95e1ee72ad3b498",
    "configuration_dots.py": "aa85add6cdaedfd947d25b850410de4506fc299dc55140d0cc5352c9737151fe",
    "generation_config.json": "d9ca5bd51bbf3b3b2b418f2b2d296cb4286ef2f02dc76dd7361d82ffd9d4e7ca",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors.index.json": "0bb4faedeb3cc8b43c71fe1db6a761ea6557d4ab8ee3f54519e641078ded1c8a",
    "modeling_dots_ocr.py": "274b5bfa35f624f15001da70414e733b89f5cb44b5453d80023bff26a68d0135",
    "modeling_dots_vision.py": "f22d6251a05a03c7510d329ba1be9e959438cd54b32204eb51ccb1b56392a96f",
    "preprocessor_config.json": "fec8b3187ab0ae340f6d0b1f6f892a53c63b73b328836c83dd7bff1d5bc4c79a",
    "special_tokens_map.json": "92b060f39417514eed2f7652f454a3d5afb0e4a57b9f4b51dd4d1dfd8fdb9736",
    "tokenizer.json": "386545eb05f08c51352cde2fcc2c867f1592bb330f305efd1c6a57a93b1244cd",
    "tokenizer_config.json": "4f09a26dd6b20c82e9678d96b7363a58a778a948a21261d3315a949d47ed4cf4",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def require(versions):
    for package, want in versions.items():
        if (got := importlib.metadata.version(package)) != want:
            raise RuntimeError(f"{package}: expected {want}, found {got}")


def sha(data):
    return hashlib.sha256(data.read_bytes() if isinstance(data, Path) else data).hexdigest()


def pinned(checkpoint):
    """Return the checkpoint's source hashes, or refuse a checkpoint whose sources differ from REVISION."""
    found = {name: sha(checkpoint / name) if (checkpoint / name).is_file() else None for name in SOURCES}
    if wrong := sorted(name for name, digest in found.items() if digest != SOURCES[name]):
        raise RuntimeError(f"checkpoint sources differ from {REVISION}: {', '.join(wrong)}")
    return found


def sources(args):
    """Check a checkpoint's sources against the pinned revision and print their hashes; no model is loaded."""
    print(json.dumps({"checkpoint_revision": REVISION, "source_sha256": pinned(args.checkpoint)}, indent=2))


def example_prompt(checkpoint):
    text = (checkpoint / "README.md").read_text()
    section = text.split("### Hugginface inference\n", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    for node in ast.parse(section).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("the pinned model card has no example prompt")


def selected_pci(torch):
    if torch.version.hip is None or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the oracle requires exactly one visible HIP GPU")
    lib = ctypes.CDLL("libamdhip64.so")
    function = lib.hipDeviceGetPCIBusId
    function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    text = ctypes.create_string_buffer(32)
    if function(text, len(text), 0) != 0:
        raise RuntimeError("cannot identify the selected HIP device")
    return text.value.decode().lower()


def oracle(args):
    """Run the checkpoint's Hugging Face example and retain native-comparison inputs."""
    require(ORACLE_VERSIONS)
    source_sha256 = pinned(args.checkpoint)
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    from transformers.generation import LogitsProcessor, LogitsProcessorList

    class Scores(LogitsProcessor):
        """Retain the scores used for greedy selection without a growing device tensor history."""

        def __init__(self, path, vocab):
            self.file = path.open("w+b")
            self.vocab, self.count = vocab, 0
            self.header()

        def header(self):
            text = repr({"descr": "<f4", "fortran_order": False, "shape": (self.count, 1, self.vocab)})
            payload = text.encode() + b" " * (243 - len(text)) + b"\n"
            if len(payload) != 244:
                raise RuntimeError("score header exceeds its reserved NPY space")
            self.file.seek(0)
            self.file.write(b"\x93NUMPY\x02\x00" + (244).to_bytes(4, "little") + payload)
            self.file.seek(0, 2)
            self.file.flush()

        def __call__(self, input_ids, scores):
            array = scores.detach().float().cpu().numpy().astype("<f4", copy=False)
            if array.shape != (1, self.vocab) or not np.isfinite(array).all():
                raise RuntimeError("unexpected or nonfinite generation scores")
            self.file.write(array.tobytes())
            self.count += 1
            if self.count % 256 == 0:
                self.header()
                print(json.dumps({"recorded_score_rows": self.count}), flush=True)
            return scores

        def close(self):
            self.header()
            self.file.close()

    pci = selected_pci(torch)
    if pci != args.pci.lower():
        raise RuntimeError(f"selected GPU {pci} differs from requested GPU {args.pci}")
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = example_prompt(args.checkpoint)
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
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    config.vision_config.attn_implementation = args.attention
    patch_rows = inputs.pixel_values.shape[0]
    image_arrays = (config.vision_config.num_hidden_layers + 2 if args.vision_only else 3)
    disk_required = image_arrays * patch_rows * config.hidden_size * 4 + (2 << 30)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, config=config, attn_implementation=args.attention, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True, local_files_only=True,
    ).eval()
    captured = set()

    def capture(name):
        def hook(_module, _args, output):
            if name in captured:
                return
            captured.add(name)
            tensor = output[0] if isinstance(output, tuple) else output
            np.save(args.output / f"{name}.npy", tensor.detach().float().cpu().numpy())
        return hook

    hooks = [model.vision_tower.patch_embed.register_forward_hook(capture("vision-patch")),
             model.vision_tower.blocks[0].register_forward_hook(capture("vision-block0")),
             model.vision_tower.post_trunk_norm.register_forward_hook(capture("vision-norm")),
             model.vision_tower.register_forward_hook(capture("vision-merged")),
             model.model.layers[0].register_forward_hook(capture("text-block0")),
             model.model.norm.register_forward_hook(capture("text-norm"))]
    if args.vision_only:
        hooks.extend(block.register_forward_hook(capture(f"vision-block{i}"))
                     for i, block in enumerate(model.vision_tower.blocks) if i != 0)
    recorder = None if args.vision_only else Scores(args.output / "generation-scores.npy", config.vocab_size)
    try:
        inputs = inputs.to("cuda:0")
        started = time.monotonic()
        with torch.inference_mode():
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
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "attn_implementation": args.attention,
        "vision_only": args.vision_only,
        "eos_reached": bool(ids and ids[-1] in end_ids),
        "seconds": seconds, "peak_device_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(), "pci": pci, "hip_version": torch.version.hip,
        "flash_triton_amd": os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE"),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "transformers", "flash-attn", "qwen-vl-utils", "numpy", "pillow"]},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_tokens": len(ids), "seconds": seconds, "text": output_text}, ensure_ascii=False), flush=True)


def processor(args):
    """Record small processor fixtures with the checkpoint's Transformers version; no model weights are loaded."""
    require(PROCESSOR_VERSIONS)
    pinned(args.checkpoint)
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    image_processor = Qwen2VLImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    fixtures = []
    for h, w, seed in [(56, 84, 0), (217, 300, 1), (513, 511, 2), (56, 1120, 3)]:
        y, x, ch = np.indices((h, w, 3), dtype=np.uint32)
        rgb = ((7 * x + 13 * y + 50 * ch + (x * y) % 31 + seed) % 256).astype(np.uint8)
        result = image_processor(images=[Image.fromarray(rgb)], return_tensors="np")
        pixels = result["pixel_values"].astype("<f4")
        fixtures.append({"height": h, "width": w, "seed": seed,
                         "grid": result["image_grid_thw"][0].tolist(), "sha256": sha(pixels.tobytes())})
    args.output.write_text(json.dumps({"checkpoint_revision": REVISION, "versions": PROCESSOR_VERSIONS,
                                      "cases": fixtures}, indent=2) + "\n")


def compare(args):
    """Compare a native greedy trace with the oracle on the same complete token history."""
    reference_dir, native_prefix = args.reference_dir, str(args.native_prefix)
    reference = json.loads((reference_dir / "result.json").read_text())
    ids = json.loads(Path(native_prefix + ".ids.json").read_text())
    expected = reference["output_ids"]
    if ids != expected:
        first = next((i for i, (a, b) in enumerate(zip(ids, expected)) if a != b), min(len(ids), len(expected)))
        raise ValueError(f"token history differs at {first}: native {len(ids)} rows, reference {len(expected)} rows")
    native = np.load(native_prefix + ".scores.npy", mmap_mode="r", allow_pickle=False)
    scores = np.load(reference_dir / "generation-scores.npy", mmap_mode="r", allow_pickle=False)
    if native.dtype != np.dtype("<u2") or native.ndim != 2 or native.shape[0] != len(ids):
        raise ValueError("native scores must be BF16 bits with one row per generated token")
    if scores.dtype != np.dtype("<f4") or scores.shape != (len(ids), 1, native.shape[1]):
        raise ValueError("reference scores must be FP32 with matching token and vocabulary dimensions")
    max_abs = sum_squared = selected_max_abs = 0.0
    min_gap = float("inf")
    ties = separated = 0
    for start in range(0, len(ids), 16):
        x = (native[start:start + 16].astype(np.uint32) << 16).view(np.float32)
        y = np.asarray(scores[start:start + 16, 0])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"nonfinite score in rows starting at {start}")
        chosen = np.asarray(ids[start:start + len(x)])
        if not np.array_equal(np.argmax(x, axis=-1), chosen) or not np.array_equal(np.argmax(y, axis=-1), chosen):
            raise ValueError(f"saved scores disagree with selected tokens in rows starting at {start}")
        difference = np.abs(x - y)
        row_error = difference.max(axis=-1)
        max_abs = max(max_abs, float(row_error.max()))
        sum_squared += float(np.sum(difference.astype(np.float64) ** 2))
        selected_max_abs = max(selected_max_abs, float(difference[np.arange(len(x)), chosen].max()))
        top = np.partition(y, -2, axis=-1)[:, -2:]
        gap = top[:, 1] - top[:, 0]
        min_gap = min(min_gap, float(gap.min()))
        ties += int(np.count_nonzero(gap == 0))
        separated += int(np.count_nonzero(gap > 2 * row_error))
    print(json.dumps({
        "tokens": len(ids), "token_ids_identical": True, "scores_finite": True,
        "reference_eos_reached": reference.get("eos_reached"),
        "logit_max_abs": max_abs, "logit_rms_difference": (sum_squared / native.size) ** 0.5,
        "selected_logit_max_abs": selected_max_abs, "reference_min_top_gap": min_gap,
        "reference_top_ties": ties, "rows_gap_above_twice_observed_max_error": separated,
    }, indent=2))


def positive(text):
    if (value := int(text)) <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    mode = modes.add_parser("oracle", help=oracle.__doc__, description=oracle.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.add_argument("image", type=Path)
    mode.add_argument("output", type=Path)
    mode.add_argument("--max-new-tokens", type=positive, default=24000)
    mode.add_argument("--attention", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    mode.add_argument("--pci", required=True, help="the selected GPU's PCI address, checked before model allocation")
    mode.add_argument("--vision-only", action="store_true", help="capture every vision block without decoder generation")
    mode.add_argument("--extra-image", type=Path, action="append", default=[], help="append another image to test independent image spans")
    mode.set_defaults(run=oracle)
    mode = modes.add_parser("sources", help=sources.__doc__, description=sources.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.set_defaults(run=sources)
    mode = modes.add_parser("processor", help=processor.__doc__, description=processor.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.add_argument("output", type=Path)
    mode.set_defaults(run=processor)
    mode = modes.add_parser("compare", help=compare.__doc__, description=compare.__doc__)
    mode.add_argument("reference_dir", type=Path)
    mode.add_argument("native_prefix", type=Path)
    mode.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
