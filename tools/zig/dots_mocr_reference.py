"""dots.mocr references: the Hugging Face oracle, processor and OCR fixtures, our test pages, native trace compare."""
import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REVISION = "e539fbb52280393adc081b289ec597430a0f9031"
ORACLE_VERSIONS = {"transformers": "4.57.6"}
ROPE_POSITIONS = [0, 1, 4096, 131071]  # the rotary values the native kernel's contract compares, short and long
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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the oracle requires exactly one visible GPU")
    if torch.version.hip is None:
        p = torch.cuda.get_device_properties(0)
        return f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    lib = ctypes.CDLL("libamdhip64.so")
    function = lib.hipDeviceGetPCIBusId
    function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    text = ctypes.create_string_buffer(32)
    if function(text, len(text), 0) != 0:
        raise RuntimeError("cannot identify the selected HIP device")
    return text.value.decode().lower()


def tower_config(checkpoint):
    """The checkpoint's config with the tower on SDPA: the pinned tower config defaults to flash-attn."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    config.vision_config.attn_implementation = "sdpa"
    return config


def load_model(torch, checkpoint, config, attention):
    """Load the pinned model on cuda:0 and refuse a tower block whose attention is not VisionSdpaAttention."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, attn_implementation=attention, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True, local_files_only=True).eval()
    sdpa = sys.modules[type(model.vision_tower).__module__].VisionSdpaAttention
    if wrong := [i for i, block in enumerate(model.vision_tower.blocks) if type(block.attn) is not sdpa]:
        raise RuntimeError(f"tower blocks {wrong} do not use VisionSdpaAttention")
    return model


class SdpaProbe:
    """Count the backend Torch's dispatcher picks for each SDPA call: the tower's are 3-D, the decoder's 4-D."""

    def __init__(self, torch):
        self.torch, self.calls = torch, {}
        self.original = torch.nn.functional.scaled_dot_product_attention
        self.names = {int(v): k for k, v in torch.nn.attention.SDPBackend.__members__.items()}

    def __call__(self, *args, **kwargs):
        q, mask = args[0], args[3] if len(args) > 3 else kwargs.get("attn_mask")
        choice = self.names.get(int(self.torch._fused_sdp_choice(*args, **kwargs)), "unknown")
        entry = self.calls.setdefault("tower" if q.dim() == 3 else "decoder", {"first_q": [list(q.shape), str(q.dtype)], "backends": {},
                                      "first_mask": None if mask is None else [list(mask.shape), str(mask.dtype)]})
        entry["backends"][choice] = entry["backends"].get(choice, 0) + 1
        return self.original(*args, **kwargs)

    def __enter__(self):
        self.torch.nn.functional.scaled_dot_product_attention = self
        return self

    def __exit__(self, *_exc):
        self.torch.nn.functional.scaled_dot_product_attention = self.original

    def summary(self):
        cuda, names = self.torch.backends.cuda, ["flash", "mem_efficient", "math", "cudnn"]
        flags = {f"{name}_sdp_enabled": getattr(cuda, f"{name}_sdp_enabled")() for name in names}
        return {"calls": self.calls, "flags": {**flags, "bf16_reduction_math": cuda.fp16_bf16_reduction_math_sdp_allowed()}}


class Scores:
    """Retain the scores used for greedy selection without a growing device tensor history (a LogitsProcessor)."""

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


def pages(args):
    """Draw the OCR test pages, our own text in Liberation fonts: page.png and one-line.png."""
    from PIL import Image, ImageDraw, ImageFont

    def font(name, size):
        return ImageFont.truetype(str(args.fonts / f"Liberation{name}.ttf"), size)

    args.output.mkdir(parents=True, exist_ok=True)
    page = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(page)
    d.text((80, 60), "TENSORFOLD READER NOTES", font=font("Sans-Regular", 22), fill="black")
    d.text((1000, 60), "March 2026", font=font("Sans-Regular", 22), fill="black")
    d.line((80, 95, 1160, 95), fill="black", width=2)
    d.text((80, 150), "Reading a Page Twice", font=font("Sans-Bold", 48), fill="black")
    d.text((80, 260), "1. Why the second read matters", font=font("Sans-Bold", 32), fill="black")
    for i, line in enumerate(["A reader that sees the same page twice must give the same blocks twice.",
                              "We feed one scanned page to two readers and count the tokens each",
                              "one writes. The table below lists the counts for a page and a line."]):
        d.text((80, 330 + 44 * i), line, font=font("Serif-Regular", 30), fill="black")
    d.text((80, 520), "Table 1: Tokens written per input", font=font("Sans-Regular", 26), fill="black")
    x, y = [80, 420, 760, 1100], [570, 630, 690, 750, 810]
    for row in y:
        d.line((x[0], row, x[3], row), fill="black", width=2)
    for column, top in [(x[0], y[0]), (x[1], y[0]), (x[3], y[0]), (x[2], y[1])]:
        d.line((column, top, column, y[4]), fill="black", width=2)
    d.rectangle((x[0] + 2, y[1] - 1, x[1] - 1, y[1] + 1), fill="white")  # "Reader" spans both header rows
    for at, cell in [((100, 615), "Reader"), ((520, 585), "Tokens written"), ((560, 645), "Page"), ((900, 645), "Line")]:
        d.text(at, cell, font=font("Sans-Bold", 28), fill="black")
    for r, row in enumerate([("First reader", "612", "41"), ("Second reader", "598", "39")]):
        for c, cell in enumerate(row):
            d.text((x[c] + 20, y[2 + r] + 15), cell, font=font("Sans-Regular", 28), fill="black")
    d.text((80, 880), "2. What we check", font=font("Sans-Bold", 32), fill="black")
    for i, item in enumerate(["The blocks come back in reading order.", "Each box stays inside the page.",
                              "A table keeps its spanned header cell."]):
        d.text((100, 950 + 50 * i), "\u2022  " + item, font=font("Serif-Regular", 30), fill="black")
    d.text((612, 1680), "3", font=font("Sans-Regular", 24), fill="black")
    page.save(args.output / "page.png")
    line = Image.new("RGB", (960, 96), "white")
    ImageDraw.Draw(line).text((24, 24), "TensorFold reads one line of text.", font=font("Sans-Regular", 36), fill="black")
    line.save(args.output / "one-line.png")
    print(json.dumps({name: sha(args.output / name) for name in ["page.png", "one-line.png"]}, indent=2))


def record(args):
    """Write an OCR fixture from an oracle run: its input, frame and hashes, without the model's text."""
    result = json.loads((args.reference_dir / "result.json").read_text())
    if len(result["prepared_images"]) != 1:
        raise ValueError("an OCR fixture records a run on one image")
    from PIL import Image
    with Image.open(args.image) as image:
        original = list(image.size)
    if sha(args.image) != result["image_sha256"]:
        raise ValueError(f"{args.image} is not the image the oracle read")
    ids = np.asarray(result["output_ids"], dtype="<u4").tobytes()
    args.output.write_text(json.dumps({
        "source": args.image.name, "image_sha256": result["image_sha256"], "original": original,
        "frame": result["prepared_images"][0]["size"], "prompt_sha256": sha(result["prompt"].encode()),
        "max_new_tokens": result["max_new_tokens"], "eos_reached": result["eos_reached"],
        "output_tokens": len(result["output_ids"]), "output_ids_sha256": sha(ids),
        "text_sha256": sha(result["text"].encode()),
    }, indent=2) + "\n")


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


def text(args):
    """Greedy ids and scores for a text-only prompt: the decoder's own reference, on a HIP or CUDA GPU."""
    require(ORACLE_VERSIONS)
    source_sha256 = pinned(args.checkpoint)
    flash = flash_or_stand_in()
    import torch
    from transformers import AutoProcessor
    from transformers.generation import LogitsProcessorList

    pci = selected_pci(torch)
    if pci != args.pci.lower():
        raise RuntimeError(f"selected GPU {pci} differs from requested GPU {args.pci}")
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False)
    rendered = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                                             tokenize=False, add_generation_prompt=True)
    inputs = processor.tokenizer([rendered], return_tensors="pt")
    np.save(args.output / "input-input_ids.npy", inputs.input_ids.cpu().numpy())
    model = load_model(torch, args.checkpoint, tower_config(args.checkpoint), args.attention)
    rotary = model.model.rotary_emb
    np.save(args.output / "rope-inv_freq.npy", rotary.inv_freq.detach().float().cpu().numpy())
    with torch.inference_mode():
        at = torch.tensor([ROPE_POSITIONS], device="cuda:0")
        cos, sin = rotary(torch.zeros(1, device="cuda:0", dtype=torch.bfloat16), at)
    np.save(args.output / "rope-positions.npy", np.array(ROPE_POSITIONS, dtype=np.int64))
    np.save(args.output / "rope-cos.npy", cos[0].float().cpu().numpy())
    np.save(args.output / "rope-sin.npy", sin[0].float().cpu().numpy())
    recorder = Scores(args.output / "generation-scores.npy", model.config.vocab_size)
    try:
        inputs = inputs.to("cuda:0")
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, return_dict_in_generate=True,
                                       output_scores=False, do_sample=False, logits_processor=LogitsProcessorList([recorder]))
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
    finally:
        recorder.close()
    ids = generated.sequences[0, inputs.input_ids.shape[1]:].cpu().tolist()
    if recorder.count != len(ids):
        raise RuntimeError("score row count differs from the generated token count")
    ends = model.generation_config.eos_token_id
    end_ids = ends if isinstance(ends, list) else [ends]
    result = {
        "checkpoint_revision": REVISION, "source_sha256": source_sha256, "prompt": prompt, "rendered_text": rendered,
        "prompt_ids": inputs.input_ids[0].cpu().tolist(), "output_ids": ids,
        "text": processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0],
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "attn_implementation": args.attention,
        "eos_reached": bool(ids and ids[-1] in end_ids), "seconds": seconds,
        "device": torch.cuda.get_device_name(), "pci": pci, "hip_version": torch.version.hip, "cuda_version": torch.version.cuda,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy"]}, "flash_attn": flash,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"prompt_tokens": len(result["prompt_ids"]), "output_tokens": len(ids), "seconds": seconds}), flush=True)


def flash_or_stand_in():
    """The installed flash-attn version, or a stand-in that refuses calls: the tower imports it, SDPA never calls it."""
    try:
        return importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        import types
        from importlib.machinery import ModuleSpec

        def refuse(*_args, **_kwargs):
            raise RuntimeError("flash_attn is not installed: only the SDPA tower can run")
        stand_in = types.ModuleType("flash_attn")
        stand_in.__spec__ = ModuleSpec("flash_attn", None)
        stand_in.flash_attn_varlen_func = refuse
        sys.modules["flash_attn"] = stand_in
        return "absent: a stand-in that refuses calls (SDPA tower)"


def score(args):
    """HF's teacher-forced logits for each native id after a reference's prompt and pixels: exact, one step, or fail."""
    require(ORACLE_VERSIONS)
    pinned(args.checkpoint)
    flash_or_stand_in()
    import torch
    reference_images = image_modes().reference_images
    reference = json.loads((args.reference_dir / "result.json").read_text())
    ids = json.loads(args.native_ids.read_text())
    config = tower_config(args.checkpoint)
    prompt = reference["prompt_ids"]
    saved = np.load(args.reference_dir / "input-input_ids.npy", allow_pickle=False)
    if saved.ndim != 2 or saved.shape[0] != 1 or saved[0].tolist() != prompt:
        raise ValueError("the reference's input-input_ids.npy and result.json prompt_ids differ")
    if not prompt or not ids or not all(isinstance(t, int) and 0 <= t < config.vocab_size for t in ids):
        raise ValueError("the reference prompt and the native ids must be non-empty lists of vocabulary ids")
    images = reference_images(torch, args.reference_dir, prompt, config)
    model = load_model(torch, args.checkpoint, config, args.attention)
    with torch.inference_mode(), SdpaProbe(torch) as probe:
        sequence = torch.tensor([prompt + ids], device="cuda:0")
        logits = model(input_ids=sequence, use_cache=False, **images).logits[0, len(prompt) - 1:-1].float().cpu().numpy()
    if logits.shape[0] != len(ids) or not np.isfinite(logits).all():
        raise ValueError("the teacher-forced logits do not line up with the native ids")
    counts, worst = {"exact": 0, "one step": 0, "fail": 0}, []
    for i, (row, token) in enumerate(zip(logits, ids)):
        top = float(row.max())
        gap = top - float(row[token])
        step = 2.0 ** (np.floor(np.log2(abs(top))) - 7) if top != 0 else 0.0
        kind = "exact" if gap == 0 else ("one step" if gap <= step else "fail")
        counts[kind] += 1
        if kind != "exact":
            worst.append({"at": i, "native": token, "gap": gap, "step": step, "kind": kind})
    count = int(images["image_grid_thw"].shape[0]) if images else 0
    args.output.write_text(json.dumps({"ids": len(ids), "images": count, **counts, "differences": worst,
                                       "sdpa": probe.summary()}, indent=2) + "\n")
    print(json.dumps({"ids": len(ids), "images": count, **counts, "first_differences": worst[:16],
                      "all_in": str(args.output)}, indent=2))
    if counts["fail"]:
        raise SystemExit(f"{counts['fail']} native ids score more than one BF16 step below HF's top")


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


def image_modes():
    """The sibling dots_mocr_images.py by path, since python -I puts no script directory on sys.path."""
    if (loaded := sys.modules.get("dots_mocr_images")) is not None:
        return loaded
    sys.modules.setdefault("dots_mocr_reference", sys.modules[__name__])  # its helpers: this module, not a second copy
    spec = importlib.util.spec_from_file_location("dots_mocr_images", Path(__file__).with_name("dots_mocr_images.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["dots_mocr_images"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["dots_mocr_images"]
        raise
    return module


def positive(text):
    if (value := int(text)) <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    images = image_modes()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    mode = modes.add_parser("oracle", help=images.oracle.__doc__, description=images.oracle.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.add_argument("image", type=Path)
    mode.add_argument("output", type=Path)
    mode.add_argument("--max-new-tokens", type=positive, default=24000)
    mode.add_argument("--prompt", type=Path, help="a UTF-8 prompt file in place of the model card's example prompt")
    mode.add_argument("--attention", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    mode.add_argument("--pci", required=True, help="the selected GPU's PCI address, checked before model allocation")
    mode.add_argument("--vision-only", action="store_true", help="capture every vision block and the patch bias rounding, without decoder generation")
    mode.add_argument("--extra-image", type=Path, action="append", default=[], help="append another image to test independent image spans")
    mode.set_defaults(run=images.oracle)
    mode = modes.add_parser("sources", help=sources.__doc__, description=sources.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.set_defaults(run=sources)
    mode = modes.add_parser("pages", help=pages.__doc__, description=pages.__doc__)
    mode.add_argument("output", type=Path)
    mode.add_argument("--fonts", type=Path, default=Path("/usr/share/fonts/liberation"))
    mode.set_defaults(run=pages)
    mode = modes.add_parser("record", help=record.__doc__, description=record.__doc__)
    mode.add_argument("reference_dir", type=Path)
    mode.add_argument("image", type=Path)
    mode.add_argument("output", type=Path)
    mode.set_defaults(run=record)
    mode = modes.add_parser("processor", help=processor.__doc__, description=processor.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.add_argument("output", type=Path)
    mode.set_defaults(run=processor)
    mode = modes.add_parser("text", help=text.__doc__, description=text.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.add_argument("output", type=Path)
    prompts = mode.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    mode.add_argument("--max-new-tokens", type=positive, default=2048)
    mode.add_argument("--attention", choices=["flash_attention_2", "sdpa"], default="sdpa")
    mode.add_argument("--pci", required=True, help="the selected GPU's PCI address, checked before model allocation")
    mode.set_defaults(run=text)
    mode = modes.add_parser("score", help=score.__doc__, description=score.__doc__)
    mode.add_argument("checkpoint", type=Path)
    mode.add_argument("reference_dir", type=Path)
    mode.add_argument("native_ids", type=Path, help="a JSON list of the native reply ids")
    mode.add_argument("output", type=Path, help="new JSON file for every difference")
    mode.add_argument("--attention", choices=["flash_attention_2", "sdpa"], default="sdpa")
    mode.set_defaults(run=score)
    mode = modes.add_parser("compare", help=compare.__doc__, description=compare.__doc__)
    mode.add_argument("reference_dir", type=Path)
    mode.add_argument("native_prefix", type=Path)
    mode.set_defaults(run=compare)
    mode = modes.add_parser("stages", help=images.stages.__doc__, description=images.stages.__doc__)
    mode.add_argument("reference_dir", type=Path, help="an oracle --vision-only output directory")
    mode.add_argument("native_dir", type=Path, help="native taps under the same names, as <u2 BF16 bit patterns")
    mode.add_argument("--output", type=Path, help="also write the report to this JSON file")
    mode.set_defaults(run=images.stages)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
