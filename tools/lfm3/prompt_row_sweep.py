#!/usr/bin/env python3
"""Per-prompt-length boundary-row error versus independent FP32 for several native routes and official BF16.

A single prompt-boundary row is a very noisy statistic: the official BF16 row can land well below its own
long-run error by chance. This tool scores many boundary rows of one causal sequence instead. For prompt length
N the native capture for `tokens[:N]` with `prompt:N` has one row (position N-1); FP32 and official BF16 are
produced once for the whole sequence with `full_model_reference.py --rows ...` and indexed by absolute position.
It reports facts only and applies no pass/fail rule.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VOCAB = 128000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bf16(path):
    raw = np.fromfile(path, dtype='<u2')
    if raw.size != VOCAB:
        raise ValueError(f'{path}: expected one full-vocabulary row, got {raw.size} values')
    return (raw.astype(np.uint32) << 16).view(np.float32)


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32', type=Path, required=True, help='FP32 rows NPY from full_model_reference.py')
    p.add_argument('--reference-bf16', type=Path, required=True, help='official BF16 rows NPY, same --rows')
    p.add_argument('--positions', required=True, help='comma separated absolute positions used for --rows')
    p.add_argument('--route', action='append', required=True, help='NAME=DIR/PREFIX; file is DIR/PREFIX_<N>.bf16')
    p.add_argument('--lengths', required=True, help='comma separated prompt lengths N (row position N-1)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    positions = [int(x) for x in a.positions.split(',')]
    lengths = [int(x) for x in a.lengths.split(',')]
    fp32 = np.load(a.fp32, allow_pickle=False)
    ref = np.load(a.reference_bf16, allow_pickle=False)
    if fp32.shape != (len(positions), VOCAB) or ref.shape != fp32.shape:
        raise ValueError('reference geometry does not match --positions')
    routes = dict(r.split('=', 1) for r in a.route)
    rows, hashes = [], {str(a.fp32): sha(a.fp32), str(a.reference_bf16): sha(a.reference_bf16)}
    for n in lengths:
        f, b = fp32[positions.index(n - 1)], ref[positions.index(n - 1)]
        top = np.argsort(-f, kind='stable')[:20]
        pf = softmax(f)[top]
        def margin(x):
            two = np.sort(np.partition(x.astype(np.float64), -2)[-2:])
            return float(two[1] - two[0])

        entry = {'prompt_length': n, 'fp32_top1': int(f.argmax()), 'fp32_top1_margin': margin(f)}

        def score(x):
            d = x.astype(np.float64) - f
            return {
                'top1': int(x.argmax()),
                'top1_margin': margin(x),
                'fp32_margin_of_this_top1': float(f[x.argmax()] - f.max()),
                'rms': float(np.sqrt((d**2).mean())),
                'max': float(np.abs(d).max()),
                'top20_probability_max': float(np.abs(softmax(x)[top] - pf).max()),
                'top1_matches_fp32': bool(x.argmax() == f.argmax()),
            }

        entry['official_bf16'] = score(b)
        for name, spec in routes.items():
            path = Path(f'{spec}_{n}.bf16')
            hashes[str(path)] = sha(path)
            entry[name] = score(bf16(path))
        rows.append(entry)
    names = ['official_bf16', *routes]
    summary = {
        name: {
            key: float(np.mean([r[name][key] for r in rows]))
            for key in ['rms', 'max', 'top20_probability_max']
        }
        | {'top1_misses': sum(not r[name]['top1_matches_fp32'] for r in rows)}
        for name in names
    }
    summary['rows_where_route_rms_exceeds_official_bf16'] = {
        name: sum(r[name]['rms'] > r['official_bf16']['rms'] for r in rows) for name in routes
    }
    out = {
        'scope': 'Per-row boundary error versus independent FP32; descriptive statistics, no pass/fail rule',
        'rows': len(rows),
        'summary': summary,
        'per_row': rows,
        'files_sha256': hashes,
    }
    a.output.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
