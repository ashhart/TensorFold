# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: OpenMDW-1.1

"""Offline Nemotron Parse references for native tests: HF example, positions, processor fingerprints, compare."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Adapted from run_transformers_inference.py in nvidia/Nemotron-3.5-Parse at the revision below.

REVISION = '4035ffb1d8f8bafcdb867dfaf6f7514f101464f9'
PROMPT = '</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>'
COMMANDS = ('reference', 'sources', 'positions', 'processor', 'compare')
# The revision's non-weight sources: model, processor and example code, configuration, tokenizer and the model card.
SOURCES = {
    'README.md': 'a97942afdd486e6e413afd66b5f7d955a995ad78b8eaf4a73cd010ac829813ef',
    'chat_template.jinja': '86e90f392a2779c96c96b5848248981a05750e7a11e0380b092a8c35e2c3e32f',
    'config.json': 'bff3460ae530573ad2e276627be86cab1b6859754b76e8b196410f17e7f0b7d5',
    'generation_config.json': '0ec0669b5d3df6c4f2fcbf41334c3d64f44fdee1d1e5f158a33b112edf1646c3',
    'hf_logits_processor.py': '37bb9612db8840eeda67f3bc9a2d8837aeebc3b206315229671d6d56c570704c',
    'hf_nemotron_parse_config.py': '5f4d2530457e41dc100df1fa614485aec8c94715f25cd69ba6e6ebdaf4611899',
    'hf_nemotron_parse_modeling.py': 'ec43d337be2797e2ad137326a68578af67694763eac304fd85b604b6775b93cc',
    'hf_nemotron_parse_processor.py': '983fa8aeb0d872da01296035b30d60c84fd857879a37c939c5c7203bad3c8b81',
    'hf_prefix_geometry.py': '41406f33399fd39bd0601fd2acd320f87fec5f9865f13fbcb4e4577d94b97048',
    'postprocessing.py': 'cef5a7379720d479f7fd6fe47fb0e83097ff47bc053f751e747134c882d5a3fe',
    'preprocessor_config.json': '16d54d73fc265b31f3771caf39770dc85213cae7594823ba7a6ac3f085315ada',
    'processor_config.json': 'b3e2ff2cc94f3d9412f179acf06d8bc5543305ec27966617977b3bd47fee9e5e',
    'run_transformers_inference.py': 'fc3568bb3e9cf48dbcb8ce0cd4e25ee7360cec9fa9da331141497ecee5dcea41',
    'special_tokens_map.json': '7073bb56328d2b66724495a7cf53841fccdcc9efcafb1e1b7cc916baa80e3c8c',
    'tokenizer.json': '34f566e728995769d971fc2707596bfb92c83298d357bc6ec29a7a1d3d12a74a',
    'tokenizer_config.json': '701dad9bcc5f964a6fa020e7d5018c71efc93aad79d471cfca541b9bb4f6d012',
}
VERSIONS = {'torch': '2.11.0', 'torchvision': '0.26.0', 'transformers': '5.14.1', 'timm': '1.0.22'}


def pinned(model_dir):
    """Return the checkpoint's source hashes, or refuse sources that differ from REVISION."""
    found = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() if (model_dir / name).is_file() else None
             for name in SOURCES}
    if wrong := sorted(name for name, digest in found.items() if digest != SOURCES[name]):
        raise RuntimeError(f"checkpoint sources differ from {REVISION}: {', '.join(wrong)}")
    return found


def versions():
    """Refuse a reference environment whose packages differ from the pinned versions."""
    import importlib.metadata

    if wrong := [f'{name} {importlib.metadata.version(name)}' for name, want in VERSIONS.items()
                 if importlib.metadata.version(name).split('+')[0] != want]:
        raise RuntimeError(f"reference packages differ from {VERSIONS}: {', '.join(wrong)}")


def sources(args):
    """Check a checkpoint's sources against the pins and print their hashes; no model or package is loaded."""
    print(json.dumps(dict(reference_revision=REVISION, versions=VERSIONS, source_sha256=pinned(args.model.resolve())),
                     indent=2))


def reference(args):
    model_dir = args.model.resolve()
    sources_sha256 = pinned(model_dir)
    versions()
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "input.image").write_bytes(args.image.read_bytes())
    source = model_dir / 'hf_nemotron_parse_modeling.py'
    manifest = dict(reference_revision=REVISION, device=args.device, dtype=args.dtype,
                    torch_version=torch.__version__, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(), sources_sha256=sources_sha256,
                    versions=VERSIONS, tensors={})

    def save(name, value):
        value = value.detach().cpu().contiguous()
        payload = value.view(torch.uint8).numpy().tobytes()
        filename = name + '.bin'
        (args.output / filename).write_bytes(payload)
        manifest['tensors'][name] = dict(file=filename, shape=list(value.shape), dtype=str(value.dtype),
                                         sha256=hashlib.sha256(payload).hexdigest())

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True,
                                      dtype=getattr(torch, args.dtype)).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    image = Image.open(args.image).convert('RGB')
    inputs = processor(images=[image], text=args.prompt, return_tensors='pt',
                       add_special_tokens=False).to(args.device)
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            save('input.' + name, value)

    handles = []
    capture = True

    def tap(name):
        def record(_module, _inputs, output):
            if not capture:
                return
            if isinstance(output, torch.Tensor):
                save(name, output)
            elif isinstance(output, tuple):
                for i, value in enumerate(output):
                    if isinstance(value, torch.Tensor):
                        save(f'{name}.{i}', value)
        return record

    handles.append(model.encoder.register_forward_hook(tap('encoder.neck')))
    blocks = model.encoder.model_encoder.radio_model.model.blocks
    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(tap(f'encoder.block.{i}')))
    handles.append(blocks[0].register_forward_pre_hook(lambda _module, values: save('encoder.positioned', values[0]) if capture else None))
    for i, layer in enumerate(model.decoder.layers):
        handles.append(layer.register_forward_hook(tap(f'decoder.layer.{i}')))
    handles.append(model.decoder.layernorm_embedding.register_forward_hook(tap('decoder.embedding_norm')))
    handles.append(model.decoder.layer_norm.register_forward_hook(tap('decoder.final_norm')))
    config = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    config.num_beams = 1
    config.repetition_penalty = 1.1
    generation_step = 0

    def save_logits(_module, _inputs, output):
        nonlocal generation_step, capture
        if generation_step == 0:
            save('prefill.logits', output)
            capture = False
        save(f'generation.logits.{generation_step}', output[:, -1:, :])
        generation_step += 1

    generation_hook = model.lm_head.register_forward_hook(save_logits)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=config)
    generation_hook.remove()
    for handle in handles:
        handle.remove()
    manifest['prompt_ids'] = inputs['input_ids'].tolist()[0]
    manifest['output_ids'] = output_ids.tolist()[0]
    manifest['new_ids'] = manifest['output_ids'][len(manifest['prompt_ids']):]
    manifest['text'] = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    manifest['settings'] = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                repetition_penalty=1.1)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest['text'])


def positions(args):
    """Content-normalized RADIO positions from the checkpoint's hf_prefix_geometry.py, one-patch axes included."""
    pinned(args.model.resolve())
    import importlib.util

    import torch

    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('parse_geometry', args.model / 'hf_prefix_geometry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    hidden, prefix = 8, 10
    table = (((torch.arange(128 * 128 * hidden) * 17) % 193).float() - 96).div(97).to(torch.bfloat16)
    cls = torch.arange(prefix * hidden).float().div(83).to(torch.bfloat16)

    def save(name, value):
        (args.output / name).write_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())

    save('table.bin', table)
    save('cls.bin', cls)
    cells = []
    for i, (height, width) in enumerate([(1, 1), (8, 73), (17, 65), (113, 97), (512, 256), (2047, 1663)]):
        gh, gw = (height + 15) // 16, ((width + 63) // 64) * 4
        patch = (((torch.arange(gh * gw * hidden) * 13) % 101).float() - 50).div(257).to(torch.bfloat16)
        with torch.inference_mode():
            positioned = module._normalized_2d_position_pages(
                torch, patches=patch.reshape(1, gh * gw, hidden), pos_embed=table.reshape(1, 128 * 128, hidden),
                source_rows=128, source_cols=128, image_sizes=[(gh * 16, gw * 16)], patch_size=16,
                content_sizes=[(height, width)], content_normalized=True)
        save(f'patches-{i}.bin', patch)
        save(f'expected-{i}.bin', torch.cat([cls.reshape(prefix, hidden), positioned[0]], dim=0))
        cells.append(dict(height=height, width=width, gh=gh, gw=gw))
    (args.output / 'manifest.json').write_text(json.dumps(dict(hidden=hidden, prefix=prefix, cells=cells), indent=2))


# Formulas of nvidia/Nemotron-3.5-Parse@4035ffb1d8f8bafcdb867dfaf6f7514f101464f9 processor; no native code.
PROCESSOR_CASES = [(1, 1), (17, 65), (31, 129), (113, 97), (2049, 101), (99, 2001), (2201, 1777), (2048, 1664)]


def processor_case(np, Image, height, width):
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    y, x, ch = np.indices((height, width, 3), dtype=np.uint32)
    rgb = ((y * 17 + x * 29 + ch * 83) % 256).astype(np.uint8)
    ratio = width / height
    h, w = height, width
    if h > 2048:
        h = 2048
        w = int(h * ratio)
    if w > 1664:
        w = 1664
        h = int(w / ratio)
    image = Image.fromarray(rgb)
    if (h, w) != (height, width):
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    content = np.array(image)
    padded = np.pad(content, ((0, -h % 16), (0, -w % 64), (0, 0)), constant_values=255)
    # ToTensor divides float32 by 255; Normalize uses two separate float32 operations.
    normalized = (padded.astype(np.float32) / np.float32(255.0) - mean) / std
    ph, pw = padded.shape[:2]
    patches = normalized.reshape(ph // 16, 16, pw // 16, 16, 3).transpose(0, 2, 4, 1, 3)
    data = patches.astype('<f4').tobytes()
    return dict(height=height, width=width, content_h=h, content_w=w, padded_h=ph, padded_w=pw,
                tokens=(ph // 16) * (pw // 64) + 1, sha256=hashlib.sha256(data).hexdigest())


def processor(_args):
    import numpy as np
    from PIL import Image

    print(json.dumps(dict(source_revision=REVISION,
                          cases=[processor_case(np, Image, h, w) for h, w in PROCESSOR_CASES]), indent=2))


def compare(args):
    import torch
    from transformers import RepetitionPenaltyLogitsProcessor

    manifest = json.loads((args.reference / 'manifest.json').read_text())
    ids = list(manifest['prompt_ids'])
    penalty = RepetitionPenaltyLogitsProcessor(manifest['settings']['repetition_penalty'])
    mismatches, choices = [], []
    for step, want in enumerate(manifest['new_ids']):
        raw = args.native / f'generation.logits.{step}.bin'
        values = torch.frombuffer(bytearray(raw.read_bytes()), dtype=torch.bfloat16).float().reshape(1, -1)
        if values.shape[1] != 72512 or not values.isfinite().all():
            raise ValueError(f'invalid native logits at step {step}')
        scores = penalty(torch.tensor([ids], dtype=torch.long), values)
        # GenerationConfig forces EOS only at its maximum length; this matches the shipped example.
        if step + 1 == manifest['settings']['max_new_tokens']:
            got = 2
        else:
            got = int(scores.argmax(dim=-1).item())
        choices.append(got)
        if got != want:
            mismatches.append(dict(step=step, native=got, reference=want,
                                   native_score=float(scores[0, got]), reference_score=float(scores[0, want])))
        ids.append(want)
    result = dict(compared=len(choices), matched=len(choices) - len(mismatches),
                  native_choices=choices, reference_ids=manifest['new_ids'], mismatches=mismatches,
                  method='teacher-forced native model; original HF RepetitionPenaltyLogitsProcessor on FP32 scores')
    (args.native / 'draw-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if mismatches else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('reference', help='save the HF example workflow (default command)')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--prompt', default=PROMPT)
    p.set_defaults(run=reference)
    p = commands.add_parser('sources', help='check the checkpoint sources against the pins')
    p.add_argument('--model', type=Path, required=True)
    p.set_defaults(run=sources)
    p = commands.add_parser('positions', help='build HIP position tests for TF_PARSE_POSITION_DIR')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(run=positions)
    p = commands.add_parser('processor', help='print zig/src/families/nemotron_parse/fixtures/processor.json')
    p.set_defaults(run=processor)
    p = commands.add_parser('compare', help='check native teacher-forced logits from the probe')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.set_defaults(run=compare)
    argv = sys.argv[1:]
    if not argv or argv[0] not in COMMANDS + ('-h', '--help'):
        argv = ['reference'] + argv
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    main()
