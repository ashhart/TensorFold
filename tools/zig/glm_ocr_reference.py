"""Pinned offline Transformers GLM-OCR references: the model-card capture and the fixtures native Zig tests load."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import time
from pathlib import Path

MODEL_REVISION = "2e85a62840ccac27daa451df36c736c4636b8628"
IMAGE_REVISION = "9dd8e4710f6b2ddf5184d308639e0359bc7b55be"
VERSION = "5.5.4"
# The revision's non-weight sources: configuration, tokenizer, template, processor and the model card.
SOURCES = {
    "README.md": "172334449c3df98650bc9a42781c254c64c9d466b440e2ca8ce31f99829212b2",
    "chat_template.jinja": "062e7ee4cc8defa88a5938b5d456dc60366ffd80647f918946ee747bf09ddc7c",
    "config.json": "ad537817b2becde827cbb23569a1dd83a352ee64cc34ce751a4ef000ecce4b5f",
    "generation_config.json": "4e2a3412f65b2a21d315d928986c55723d3e60dfb92c3982a5b9fd56835b0aa5",
    "preprocessor_config.json": "02cc50c36240882ae35e8cd4077a25a379664108185d728d261cb785aefeccff",
    "processor_config.json": "a56fbd559186fefd6c22a3acc6a393a18a9fe52c264692a122a143d24042112b",
    "tokenizer.json": "aa0fd058c73a5718bb191f6672dc16d122ee0147b20c123d1726514298f9968a",
    "tokenizer_config.json": "95abd49942be9cdc6e2653528aed7b4c1cb9e180828421d3488ab38da3057016",
}
# The model and processor code ships in Transformers 5.5.4, not in the checkpoint; paths are package-relative.
LIBRARY = {
    "models/glm_ocr/configuration_glm_ocr.py": "7015c0aa18811ab058e228a9c343bc07bf266abb2753a66a2c371720b6612c53",
    "models/glm_ocr/modeling_glm_ocr.py": "f667126388eacde9ca93983c1fc34488f4c169717a6ff5900c1479d2c6bb7db4",
    "models/glm46v/image_processing_glm46v.py": "41fa88e258a4b291f167d77085eee989dbebff1776c336121e2f4185b6e0eef1",
    "models/glm46v/image_processing_pil_glm46v.py": "3506c3f76284c90d3ed9d4e44e5091cb09ceb4fcd8f382cb2a90aa18923c7598",
    "models/glm46v/processing_glm46v.py": "91ba14d315a2059e4f0f811618394f201401a9858648c1f66c4637f25fe628db",
}
SCHEMA = {
    "id_number": "", "last_name": "", "first_name": "", "date_of_birth": "",
    "address": {"street": "", "city": "", "state": "", "zip_code": ""},
    "dates": {"issue_date": "", "expiration_date": ""}, "sex": "",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_transformers(what):
    if importlib.metadata.version("transformers") != VERSION:
        raise RuntimeError(f"{what} requires official Transformers {VERSION}")


def hashed(root, pins, what):
    """Return the hashes of the pinned files under root, or refuse when any differs from its pin."""
    found = {name: sha(root / name) if (root / name).is_file() else None for name in pins}
    if wrong := sorted(name for name, digest in found.items() if digest != pins[name]):
        raise RuntimeError(f"{what} differs from its pin: {', '.join(wrong)}")
    return found


def pinned(checkpoint):
    """Return the checkpoint and Transformers source hashes, or refuse sources that differ from the pins."""
    pinned_transformers("the source pins")
    package = Path(importlib.util.find_spec("transformers").origin).parent
    return dict(checkpoint=hashed(Path(checkpoint), SOURCES, f"checkpoint {MODEL_REVISION}"),
                transformers=hashed(package, LIBRARY, f"Transformers {VERSION}"))


def sources(args):
    """Check the checkpoint and Transformers sources against the pins and print their hashes; no model loads."""
    print(json.dumps(dict(model_revision=MODEL_REVISION, transformers=VERSION, source_sha256=pinned(args.checkpoint)),
                     indent=2))


def device_pci(torch):
    if torch.version.hip is None or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one visible HIP GPU before loading the reference")
    lib = ctypes.CDLL("libamdhip64.so")
    get = lib.hipDeviceGetPCIBusId
    get.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    get.restype = ctypes.c_int
    text = ctypes.create_string_buffer(32)
    if get(text, len(text), 0) != 0:
        raise RuntimeError("cannot identify the selected HIP GPU")
    return text.value.decode().lower()


def runtime_versions():
    miopen = ctypes.CDLL("libMIOpen.so")
    values = [ctypes.c_size_t() for _ in range(3)]
    if miopen.miopenGetVersion(*[ctypes.byref(v) for v in values]) != 0:
        raise RuntimeError("cannot identify the MIOpen runtime")
    runtime = ctypes.c_int()
    if ctypes.CDLL("libamdhip64.so").hipRuntimeGetVersion(ctypes.byref(runtime)) != 0:
        raise RuntimeError("cannot identify the HIP runtime")
    return dict(miopen=[v.value for v in values], hip_runtime=runtime.value)


def capture(args):
    import numpy as np
    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    source_sha256 = pinned(args.checkpoint)
    pci = device_pci(torch)
    if pci != args.pci.lower():
        raise RuntimeError(f"selected GPU {pci} differs from requested GPU {args.pci}")
    free, total = torch.cuda.mem_get_info()
    if free < 6 << 30:
        raise RuntimeError("reference requires 6 GiB free VRAM; existing jobs are left unchanged")
    manifest = json.loads(args.examples.read_text())
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    image = args.examples.parent / case["image"]
    if sha(image) != case["image_sha256"]:
        raise RuntimeError("example image hash does not match the pinned manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "image", "url": str(image)},
                                              {"type": "text", "text": case["prompt"]}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    for key, value in inputs.items():
        np.save(args.output / f"input-{key}.npy", value.numpy())
    patches = int(inputs.image_grid_thw.prod(dim=1).max())
    required = (12 if patches > 8192 else 6) << 30
    if args.attention == "eager":
        required = max(required, (3 << 30) + 4 * 16 * patches * patches)
    free, total = torch.cuda.mem_get_info()
    if free < required:
        raise RuntimeError(f"{args.attention} reference needs at least {required} free bytes for this unchanged input")
    model = GlmOcrForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
        attn_implementation=args.attention,
    ).eval()
    captured = set()
    text_step = -1

    def save(name, value):
        if name.startswith("text-") and text_step > 0:
            if text_step > 1:
                return
            name = f"decode-{text_step}-" + name
        if name in captured:
            return
        captured.add(name)
        np.save(args.output / f"{name}.npy", value.detach().float().cpu().numpy())
        if value.dtype == torch.bfloat16:
            np.save(args.output / f"{name}-bf16.npy", value.detach().contiguous().view(torch.uint16).cpu().numpy())

    def tap(name):
        return lambda _module, _args, output: save(name, output[0] if isinstance(output, tuple) else output)

    v, t = model.model.visual, model.model.language_model
    taps = {"vision-patch": v.patch_embed, "vision-block0": v.blocks[0], "vision-last": v.blocks[-1],
            "vision-norm": v.post_layernorm, "vision-downsample": v.downsample, "vision-connector": v.merger,
            "text-block0": t.layers[0], "text-last": t.layers[-1], "text-norm": t.norm}
    hooks = [module.register_forward_hook(tap(name)) for name, module in taps.items()]
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    if args.text_details:
        block = t.layers[0]
        details = {"norm1": block.input_layernorm, "q": block.self_attn.q_proj, "k": block.self_attn.k_proj,
                   "v": block.self_attn.v_proj, "proj": block.self_attn.o_proj,
                   "attn-norm": block.post_self_attn_layernorm, "norm2": block.post_attention_layernorm,
                   "gate-up": block.mlp.gate_up_proj, "down": block.mlp.down_proj,
                   "mlp-norm": block.post_mlp_layernorm}
        hooks.extend(module.register_forward_hook(tap("text-block0-" + name)) for name, module in details.items())

        def text_input(_module, _positional, kwargs):
            nonlocal text_step
            text_step += 1
            save("text-input", kwargs["inputs_embeds"])
            name = (f"decode-{text_step}-" if text_step else "") + "text-positions"
            if text_step <= 1 and name not in captured:
                captured.add(name)
                np.save(args.output / f"{name}.npy", kwargs["position_ids"].detach().cpu().numpy())

        def rotary(_module, _positional, output):
            save("text-rope-cos", output[0])
            save("text-rope-sin", output[1])

        def attention(q, k, v, *positional, **kwargs):
            output = original_sdpa(q, k, v, *positional, **kwargs)
            if q.shape[-1] == 128:
                save("text-block0-qrope", q.transpose(1, 2).contiguous())
                save("text-block0-krope", k[:, :, -q.shape[2]:].transpose(1, 2).contiguous())
                if text_step == 1 and not (args.output / "decode-1-attention.json").exists():
                    (args.output / "decode-1-attention.json").write_text(json.dumps({
                        "q_shape": list(q.shape), "k_shape": list(k.shape), "q_stride": list(q.stride()),
                        "k_stride": list(k.stride()), "is_causal": kwargs.get("is_causal", False),
                        "mask": kwargs.get("attn_mask") is not None,
                    }, indent=2) + "\n")
                save("text-block0-attended", output.transpose(1, 2).contiguous())
            return output

        hooks.append(t.register_forward_pre_hook(text_input, with_kwargs=True))
        hooks.append(t.rotary_emb.register_forward_hook(rotary))
        save("text-inverse", t.rotary_emb.inv_freq)
        torch.nn.functional.scaled_dot_product_attention = attention
    try:
        inputs = inputs.to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=case["max_new_tokens"], do_sample=False,
                                       return_dict_in_generate=True, output_scores=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        ids = generated.sequences[0, inputs.input_ids.shape[-1]:].tolist()
        top = []
        for score in generated.scores:
            values, indices = score[0].float().topk(5)
            top.append(dict(ids=indices.tolist(), logits=values.tolist(), margin=(values[0] - values[1]).item()))
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].detach()
        np.save(args.output / "prefill-logits.npy", logits.float().cpu().numpy())
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for hook in hooks:
            hook.remove()
    if args.text_details:
        for step, score in enumerate(generated.scores):
            np.save(args.output / f"step-{step}-logits.npy", score.detach().float().cpu().numpy())
    receipt = dict(case=case, model_revision=manifest["model_revision"], transformers=VERSION,
                   torch=torch.__version__, hip=torch.version.hip, pci=pci, attention=args.attention, dtype="bfloat16",
                   gpu=torch.cuda.get_device_name(), free_before=free, total_memory=total,
                   peak_allocated=torch.cuda.max_memory_allocated(), seconds=elapsed, output_ids=ids,
                   text=processor.decode(ids, skip_special_tokens=True), top_tokens=top,
                   source_sha256=source_sha256, runtime=runtime_versions())
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("pci", "seconds", "text", "peak_allocated")}, ensure_ascii=False), flush=True)


def processor_fixtures(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, GlmOcrModel
    from transformers.models.glm46v.image_processing_glm46v import smart_resize

    pinned(args.checkpoint)
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "preprocessor_config.json"):
        (args.output / name).write_bytes((args.checkpoint / name).read_bytes())
    cases = []
    for height, width in [(84, 112), (85, 119), (42, 70), (17, 65), (147, 231), (643, 491)]:
        y, x, ch = np.indices((height, width, 3))
        rgb = ((y * 17 + x * 29 + ch * 83) % 256).astype(np.uint8)
        prepared = processor.image_processor(images=Image.fromarray(rgb), return_tensors="pt")
        values = prepared.pixel_values.numpy().astype("<f4")
        grid = prepared.image_grid_thw.tolist()[0]
        cases.append(dict(height=height, width=width, grid=grid,
                          tokens=grid[1] * grid[2] // 4,
                          sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    geometry = []
    for height, width in [(3000, 4000), (100, 8000), (28, 5600), (42, 70), (70, 98), (1, 1)]:
        h, w = smart_resize(2, height, width, min_pixels=12544, max_pixels=9633792)
        geometry.append(dict(height=height, width=width, h=h, w=w))
    with torch.device("meta"):
        model = GlmOcrModel(cfg)
    ids = [10, 59256] + [59280] * 6 + [59257, 20, 59256] + [59280] * 2 + [59257, 30]
    types = [int(i == 59280) for i in ids]
    grids = [[1, 4, 6], [1, 2, 4]]
    positions, delta = model.get_rope_index(torch.tensor([ids]), torch.tensor([types]), torch.tensor(grids))
    sources = {Path(inspect.getfile(obj)).name: sha(inspect.getfile(obj))
               for obj in (GlmOcrModel, type(processor.image_processor))}
    fixture = dict(revision=MODEL_REVISION, transformers=VERSION, torch=torch.__version__,
                   source_sha256=sources, cases=cases, geometry=geometry,
                   positions=dict(ids=ids, grids=grids, rows=positions[:, 0].T.tolist(), delta=delta.item()))
    (args.output / "processor.json").write_text(json.dumps(fixture, indent=2) + "\n")


def attention_fixtures(args):
    import numpy as np
    import torch
    from torch.nn import functional as F

    if torch.version.hip is None or torch.cuda.device_count() != 1:
        raise RuntimeError("select one HIP GPU before fixture generation")
    args.output.mkdir(exist_ok=False, parents=True)
    cases = []
    for name, heads, kv_heads, rows, keys, dim, causal in [
        ("vision", 16, 16, 67, 67, 64, False),
        ("decoder", 16, 8, 17, 17, 128, True),
        ("decode", 16, 8, 1, 19, 128, False),
    ]:
        tensors = []
        for i, (length, nh) in enumerate([(rows, heads), (keys, kv_heads), (keys, kv_heads)]):
            values = torch.arange(length * nh * dim, device="cuda", dtype=torch.float32)
            tensor = ((values * (i + 3)) % 31 / 16 - 1).reshape(1, length, nh, dim).to(torch.bfloat16)
            tensors.append(tensor.transpose(1, 2))
            np.save(args.output / f"{name}-{'qkv'[i]}.npy", tensor.view(torch.uint16).cpu().numpy())
        with torch.inference_mode():
            out = F.scaled_dot_product_attention(*tensors, is_causal=causal, enable_gqa=heads != kv_heads)
        np.save(args.output / f"{name}-out.npy", out.transpose(1, 2).contiguous().view(torch.uint16).cpu().numpy())
        cases.append(dict(name=name, heads=heads, kv_heads=kv_heads, rows=rows, keys=keys, dim=dim, causal=causal))
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def examples(args):
    from PIL import Image, ImageDraw, ImageFont

    args.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(str(args.font), 32)
    cases = []
    for name, file, limit in [("hf-car", "transformers/tasks/car.jpg", 512), ("hf-bee", "bee.jpg", 128)]:
        source = args.hf_images / file
        target = args.output / f"{name}.jpg"
        target.write_bytes(source.read_bytes())
        cases.append(dict(name=name, image=target.name, prompt="Text Recognition:", max_new_tokens=limit,
                          source=f"https://huggingface.co/datasets/huggingface/documentation-images/resolve/{IMAGE_REVISION}/{file}",
                          license="CC-BY-NC-SA-4.0"))
    data = [
        ("text", ["TensorFold OCR test", "Invoice 0042", "Total: 123.45 USD"], "Text Recognition:",
         {"contains": ["TensorFold", "0042", "123.45"]}),
        ("formula", ["a² + b² = c²"], "Formula Recognition:", {"formula": "a^2+b^2=c^2"}),
        ("table", [], "Table Recognition:", {"cells": [["Item", "Quantity"], ["Apples", "3"], ["Pears", "5"]]}),
        ("extraction", ["FICTIONAL ID - TEST DATA", "ID Number: ZX000042", "Last Name: EXAMPLE",
                        "First Name: ALEX", "Date of Birth: 1990-01-02", "Street: 42 Example Road",
                        "City: Sampletown", "State: CA", "ZIP Code: 00000", "Issue Date: 2025-01-02",
                        "Expiration Date: 2030-01-02", "Sex: X"],
         "请按下列JSON格式输出图中信息:\n" + json.dumps(SCHEMA, ensure_ascii=False, indent=4),
         {"json": {"id_number": "ZX000042", "last_name": "EXAMPLE", "first_name": "ALEX",
                   "date_of_birth": "1990-01-02", "address": {"street": "42 Example Road", "city": "Sampletown",
                   "state": "CA", "zip_code": "00000"}, "dates": {"issue_date": "2025-01-02",
                   "expiration_date": "2030-01-02"}, "sex": "X"}}),
    ]
    for name, lines, prompt, expected in data:
        image = Image.new("RGB", (840, max(196, 64 + 52 * len(lines))), "white")
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            draw.text((32, 26 + 52 * i), line, font=font, fill="black")
        if name == "table":
            image = Image.new("RGB", (560, 252), "white")
            draw = ImageDraw.Draw(image)
            for row, cells in enumerate(expected["cells"]):
                for col, text in enumerate(cells):
                    x, y = 12 + col * 268, 12 + row * 76
                    draw.rectangle((x, y, x + 268, y + 76), outline="black", width=2)
                    draw.text((x + 12, y + 16), text, font=font, fill="black")
        path = args.output / f"{name}.png"
        image.save(path)
        cases.append(dict(name=name, image=path.name, prompt=prompt, max_new_tokens=512,
                          expected=expected, source="model-card prompt with generated fictional fixture"))
    for case in cases:
        case["image_sha256"] = sha(args.output / case["image"])
    manifest = dict(model_revision=MODEL_REVISION, documentation_images_revision=IMAGE_REVISION,
                    task_source="https://huggingface.co/zai-org/GLM-OCR",
                    example_source="https://github.com/huggingface/transformers/blob/v5.5.4/docs/source/en/model_doc/glm_ocr.md",
                    font_sha256=sha(args.font), cases=cases)
    (args.output / "examples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pins = commands.add_parser("sources", help="check the checkpoint and Transformers sources against the pins")
    pins.add_argument("checkpoint", type=Path)
    pins.set_defaults(func=sources)
    run = commands.add_parser("capture", help="run the HIP reference for one examples.json case")
    run.add_argument("checkpoint", type=Path)
    run.add_argument("examples", type=Path)
    run.add_argument("case")
    run.add_argument("output", type=Path)
    run.add_argument("--pci", required=True)
    run.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    run.add_argument("--text-details", action="store_true")
    run.set_defaults(func=capture)
    proc = commands.add_parser("processor", help="write processor.json, config.json and preprocessor_config.json")
    proc.add_argument("checkpoint", type=Path)
    proc.add_argument("output", type=Path)
    proc.set_defaults(func=processor_fixtures)
    attn = commands.add_parser("attention", help="write cases.json and <case>-q/k/v/out.npy on one HIP GPU")
    attn.add_argument("output", type=Path)
    attn.set_defaults(func=attention_fixtures)
    ex = commands.add_parser("examples", help="write examples.json with its images")
    ex.add_argument("hf_images", type=Path)
    ex.add_argument("output", type=Path)
    ex.add_argument("--font", type=Path, required=True)
    ex.set_defaults(func=examples)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
