#!/usr/bin/env python3
"""Capture Transformers residuals for the finite public CUDA layer diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from qwen35_reference_forward import library_forward, load_model, manual_forward, model_load_report

    torch.set_grad_enabled(False)
    ids_bytes = Path(args.ids).read_bytes()
    ids = json.loads(ids_bytes)
    if not isinstance(ids, list) or not 2 <= len(ids) <= 4096:
        raise ValueError('expected one finite fixture of token IDs')
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('invalid token ID')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise ValueError('reference output must be empty')
    model, loading = load_model(args.model)
    inputs = torch.tensor([ids], dtype=torch.long)
    layers = model.model.language_model.config.num_hidden_layers
    hidden = model.model.language_model.config.hidden_size
    reports = {}
    for arm in ('bf16', 'fp32'):
        dest = out / arm
        dest.mkdir()
        seen = []

        def capture(index, value):
            if index != len(seen) or list(value.shape) != [1, len(ids), hidden]:
                raise ValueError('incomplete or inconsistent reference boundary')
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite reference')
            data = value.detach().float().cpu().contiguous().numpy().astype('<f4').tobytes()
            (dest / ('%d.f32' % index)).write_bytes(data)
            seen.append(index)

        checks = []
        final, stats = manual_forward(model, inputs, upcast=arm == 'fp32', capture=capture,
                                      verify_layers={0, 3, layers - 1}, verify_report=checks,
                                      track_params=True)
        if seen != list(range(layers + 2)):
            raise ValueError('missing reference boundaries')
        if arm == 'bf16':
            library_hidden, _ = library_forward(model, inputs, logits_to_keep=1)
            equal = torch.equal(final, library_hidden)
            if not equal:
                raise ValueError('captured BF16 forward differs from public library entry point')
            reports[arm] = {'library_final_hidden_equal': equal, 'stats': stats}
            del library_hidden
        else:
            if not checks or not all(c['exact_equal'] for c in checks):
                raise ValueError('FP32 layer upcast differs from independently cloned FP32 layer')
            reports[arm] = {'independent_cloned_layer_checks': checks, 'stats': stats}
        del final
    metadata = {'rows': len(ids), 'hidden': hidden, 'layers': layers, 'boundaries': layers + 2,
                'ids_sha256': hashlib.sha256(ids_bytes).hexdigest(), 'arms': reports,
                'python': platform.python_version(), 'torch': torch.__version__,
                'transformers': transformers.__version__, 'device': 'cpu',
                'model': model_load_report(model, loading)}
    (out / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
