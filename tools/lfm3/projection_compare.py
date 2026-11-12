#!/usr/bin/env python3
"""Compare saved native projection rows and MLX against independently dequantized FP64 checkpoint products."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fp64_reference import Oracle, bf
from reference import Reference


def read_rows(path, width):
    if path.stat().st_size % 2:
        raise ValueError('BF16 input contains a partial value')
    words = np.fromfile(path, dtype='<u2')
    if len(words) % width:
        raise ValueError('BF16 input does not contain whole rows')
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(-1, width).astype(np.float64)


def metrics(values, expected):
    if values.shape != expected.shape or not np.isfinite(values).all() or not np.isfinite(expected).all():
        raise ValueError('Projection comparison requires matching finite arrays')
    error = values - expected
    per_column = np.sqrt(np.mean(error * error, axis=0))
    return {
        'rms': float(np.sqrt(np.mean(error * error))),
        'max_abs': float(np.max(np.abs(error))),
        'worst_column_rms': float(np.max(per_column)),
        'worst_column': int(np.argmax(per_column)),
        'correctly_rounded': int(np.count_nonzero(values == bf(expected))),
        'outputs': int(values.size),
    }, per_column


def compare(case, model, reference, inputs, candidate, output, mx):
    x = read_rows(inputs / case['input_file'], case['k'])
    one_input = read_rows(inputs / case['one_row_file'], case['k'])
    if one_input.shape != (1, case['k']) or not np.array_equal(one_input, x[:1]):
        raise ValueError('Decode input must equal the first prompt input row')
    if len(x) != case['rows']:
        raise ValueError('Fixture row count differs from its manifest')
    native = read_rows(candidate / (case['name'] + '.bf16'), case['n'])
    one = read_rows(candidate / (case['name'] + '-one.bf16'), case['n'])
    if native.shape != (len(x), case['n']) or one.shape != (1, case['n']):
        raise ValueError('Native output geometry differs from its manifest')
    expected = np.empty(native.shape, dtype=np.float64)
    for start in range(0, case['n'], 512):
        end = min(start + 512, case['n'])
        expected[:, start:end] = x @ model.weights(case['module'], slice(start, end)).T
    wide_mx = reference.linear(mx.array(x.astype(np.float32), dtype=mx.bfloat16), case['module'])
    one_mx = reference.linear(mx.array(x[:1].astype(np.float32), dtype=mx.bfloat16), case['module'])
    mx.eval(wide_mx, one_mx)
    mlx = np.array(wide_mx.astype(mx.float32)).astype(np.float64)
    mlx_one = np.array(one_mx.astype(mx.float32)).astype(np.float64)
    records = []
    for label, actual, second, target in [('prompt', native, mlx, expected), ('decode', one, mlx_one, expected[:1])]:
        native_metrics, nc = metrics(actual, target)
        mlx_metrics, mc = metrics(second, target)
        increase = nc - mc
        worst = np.argsort(increase)[-min(20, len(increase)):][::-1]
        records.append({
            'mode': label, 'native': native_metrics, 'mlx': mlx_metrics,
            'columns_native_worse': int(np.count_nonzero(nc > mc)),
            'columns_native_better': int(np.count_nonzero(nc < mc)),
            'worst_column_increases': [{'column': int(i), 'native_rms': float(nc[i]),
                                        'mlx_rms': float(mc[i]), 'increase': float(increase[i])} for i in worst],
        })
        np.savez_compressed(output / (case['name'] + '-' + label + '-columns.npz'), native=nc, mlx=mc)
    return {
        'name': case['name'], 'model': case['model_directory'], 'module': case['module'],
        'shape': [len(x), case['k'], case['n']], 'quantization': list(model.quant(case['module'])),
        'input_sha256': hashlib.sha256((inputs / case['input_file']).read_bytes()).hexdigest(),
        'native_prompt_sha256': hashlib.sha256((candidate / (case['name'] + '.bf16')).read_bytes()).hexdigest(),
        'native_decode_sha256': hashlib.sha256((candidate / (case['name'] + '-one.bf16')).read_bytes()).hexdigest(),
        'native_prompt_decode_equal': bool(np.array_equal(native[:1], one)),
        'mlx_prompt_decode_equal': bool(np.array_equal(mlx[:1], mlx_one)), 'comparisons': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.inputs / 'manifest.json').read_text())
    records, current = [], None
    for case in cases:
        if current != case['model_directory']:
            if current is not None:
                del reference, model
                mx.clear_cache()
            current = case['model_directory']
            model = Oracle(args.models / current)
            reference = Reference(args.models / current, mx)
        record = compare(case, model, reference, args.inputs, args.candidate, args.output, mx)
        records.append(record)
        (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    summary = {
        'cells': len(records), 'projection_outputs': sum(c['shape'][0] * c['shape'][2] for c in records),
        'native_prompt_decode_equal_cells': sum(c['native_prompt_decode_equal'] for c in records),
        'mlx_prompt_decode_equal_cells': sum(c['mlx_prompt_decode_equal'] for c in records),
        'acceptance_threshold': None,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
