#!/usr/bin/env python3
"""Compare native residuals with the Transformers BF16-to-FP32 distance; failure exits 1."""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def read_values(path, dtype, count):
    data = path.read_bytes()
    if len(data) != count * (2 if dtype == 'bf16' else 4):
        raise ValueError('tensor byte count mismatch: ' + path.name)
    if dtype == 'bf16':
        words = array.array('H', data)
        if sys.byteorder != 'little':
            words.byteswap()
        expanded = array.array('I', (x << 16 for x in words))
        values = array.array('f', expanded.tobytes())
    else:
        values = array.array('f', data)
        if sys.byteorder != 'little':
            values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite capture: ' + path.name)
    return values


def distance(native, bf16, fp32, hidden):
    if not native or len(native) != len(bf16) or len(native) != len(fp32):
        raise ValueError('empty or mismatched comparison')
    a = [abs(n - f) for n, f in zip(native, fp32)]
    b = [abs(t - f) for t, f in zip(bf16, fp32)]
    maximum = max(a)
    own = max(b)
    i = a.index(maximum)
    return {'native_fp32_max_abs': maximum, 'bf16_fp32_max_abs': own,
            'native_fp32_median_abs': statistics.median(a),
            'bf16_fp32_median_abs': statistics.median(b),
            'native_fp32_rms': math.sqrt(sum(x*x for x in a) / len(a)),
            'bf16_fp32_rms': math.sqrt(sum(x*x for x in b) / len(b)),
            'native_bf16_max_abs': max(abs(n-t) for n, t in zip(native, bf16)),
            'elements_exceeding_own_distance': sum(x > y for x, y in zip(a, b)),
            'worst_native_row': i // hidden, 'worst_native_column': i % hidden,
            'native_at_worst': native[i], 'fp32_at_worst': fp32[i], 'bf16_at_worst': bf16[i],
            'within_bf16_max_distance': maximum <= own}


def compare(native_dir, reference_dir):
    native_dir, reference_dir = Path(native_dir), Path(reference_dir)
    nm = json.loads((native_dir / 'meta.json').read_text())
    tm = json.loads((reference_dir / 'meta.json').read_text())
    for key in ('rows', 'hidden', 'layers', 'boundaries'):
        if type(nm[key]) is not int or nm[key] <= 0 or nm[key] != tm[key]:
            raise ValueError('inconsistent capture metadata: ' + key)
    if nm['boundaries'] != nm['layers'] + 2 or not nm.get('trace_plain_logits_equal'):
        raise ValueError('incomplete or perturbing native trace')
    count = nm['rows'] * nm['hidden']
    result, all_rows, tensors = [], [], []
    scored_rows = nm["rows"] - 1
    if scored_rows < 1:
        raise ValueError("fixture needs a scored row")
    for i in range(nm['boundaries']):
        paths = [(native_dir / ('%d.bf16' % i), 'bf16'),
                 (reference_dir / 'bf16' / ('%d.f32' % i), 'f32'),
                 (reference_dir / 'fp32' / ('%d.f32' % i), 'f32')]
        values = []
        for path, dtype in paths:
            values.append(read_values(path, dtype, count))
            tensors.append({'boundary': i, 'arm': ('native' if dtype == 'bf16' else path.parent.name),
                            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        full = distance(*values, nm['hidden'])
        row = distance(*(v[:scored_rows * nm['hidden']] for v in values), nm['hidden'])
        row.update(boundary=i, layer=i-1 if 0 < i <= nm['layers'] else None,
                   kind='embedding' if i == 0 else 'final_norm' if i > nm['layers'] else 'residual')
        full.update(boundary=i, layer=row["layer"], kind=row["kind"])
        all_rows.append(full)
        result.append(row)
    failures = [r for r in result if not r['within_bf16_max_distance']]
    first_layer = next((r['layer'] for r in failures if r['kind'] == 'residual'), None)
    return {'rule': 'per boundary, max |native-FP32| <= max |Transformers BF16-FP32|; no added tolerance',
            'status': 'failed' if failures else 'passed', 'first_exceeding_layer_zero_based': first_layer,
            'shape': {k: nm[k] for k in ('rows', 'hidden', 'layers', 'boundaries')},
            'compared_rows': scored_rows, 'row_scope': 'hidden rows 0..N-2 predict the served prompt IDs 1..N-1',
            'all_input_rows_boundaries': all_rows,
            'boundaries': result, 'tensors': tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.native, args.reference)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'first_exceeding_layer_zero_based')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
