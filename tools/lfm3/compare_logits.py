#!/usr/bin/env python3
"""Report every teacher-forced top-token difference without applying an acceptance tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(native, reference, row_offset=0):
    native, reference = np.asarray(native), np.asarray(reference)
    if native.shape != reference.shape or native.ndim != 2 or native.shape[1] < 2 or not len(native):
        raise ValueError('Expected equally shaped, nonempty row-by-vocabulary arrays')
    if not np.isfinite(native).all() or not np.isfinite(reference).all():
        raise ValueError('Logits must be finite')
    differences, worst = [], []
    sum_squared, reference_squared, maximum = 0.0, 0.0, 0.0
    column_squared = np.zeros(native.shape[1], dtype=np.float64)
    column_maximum = np.zeros(native.shape[1], dtype=np.float64)
    for index, (actual, expected) in enumerate(zip(native, reference)):
        actual, expected = actual.astype(np.float64), expected.astype(np.float64)
        error = actual - expected
        square = float(error @ error)
        denominator = float(expected @ expected)
        column_squared += error * error
        column_maximum = np.maximum(column_maximum, np.abs(error))
        sum_squared += square
        reference_squared += denominator
        maximum = max(maximum, float(np.abs(error).max()))
        worst.append({'row': index + row_offset, 'rms_error': float(np.sqrt(square / len(error))),
                      'maximum_absolute_error': float(np.abs(error).max())})
        a, b = int(actual.argmax()), int(expected.argmax())
        if a == b:
            continue
        aset = np.flatnonzero(actual == actual[a]).tolist()
        bset = np.flatnonzero(expected == expected[b]).tolist()
        atop = np.partition(actual, -2)[-2:]
        btop = np.partition(expected, -2)[-2:]
        differences.append({'row': index + row_offset, 'native_top': a, 'reference_top': b,
                            'native_maximum_set': aset, 'reference_maximum_set': bset,
                            'maximum_sets_overlap': bool(set(aset) & set(bset)),
                            'native_margin': float(atop[-1] - atop[-2]),
                            'reference_margin': float(btop[-1] - btop[-2]),
                            'native_reference_contender_gap': float(actual[a] - actual[b]),
                            'reference_native_contender_gap': float(expected[b] - expected[a])})
    columns = np.argsort(column_squared)[-20:][::-1]
    worst_columns = [{'column': int(c), 'rms_error': float(np.sqrt(column_squared[c] / len(native))),
                      'maximum_absolute_error': float(column_maximum[c])} for c in columns]
    return {'rows': len(native), 'vocabulary': native.shape[1], 'top1_differences': len(differences),
            'disjoint_maximum_sets': sum(not item['maximum_sets_overlap'] for item in differences),
            'relative_rms': float(np.sqrt(sum_squared / reference_squared)) if reference_squared else None,
            'maximum_absolute_error': maximum, 'differences': differences,
            'worst_rows': sorted(worst, key=lambda item: item['rms_error'], reverse=True)[:20],
            'worst_columns': worst_columns, 'accepted_threshold': None, 'fidelity_pass': None}


def read_bf16(path, vocabulary):
    if path.stat().st_size % 2:
        raise ValueError('BF16 file contains an incomplete two-byte value')
    values = np.fromfile(path, dtype='<u2')
    if not values.size or values.size % vocabulary:
        raise ValueError('BF16 byte count must contain complete vocabulary rows')
    return (values.astype(np.uint32) << 16).view(np.float32).reshape(-1, vocabulary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True, help='Identical teacher-forced token fixture')
    parser.add_argument('--vocabulary', type=int, required=True)
    parser.add_argument('--phase', choices=('prompt', 'decode'), required=True)
    parser.add_argument('--row-offset', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.vocabulary < 2 or args.row_offset < 0:
        parser.error('Vocabulary must be at least two and row offset nonnegative')
    native, reference = (read_bf16(path, args.vocabulary) for path in (args.native, args.reference))
    if args.phase == 'decode' and len(native) != 1:
        parser.error('Decode receipts must contain exactly one measured output row')
    tokens = np.load(args.tokens, allow_pickle=False)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or not tokens.size:
        parser.error('Expected a nonempty one-dimensional teacher-forced integer token fixture')
    if args.row_offset + len(native) > len(tokens):
        parser.error('Measured rows exceed teacher-forced fixture')
    result = compare(native, reference, args.row_offset)
    result.update({'phase': args.phase,
                   'token_fixture_sha256': hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
                   'native_logits_sha256': hashlib.sha256(args.native.read_bytes()).hexdigest(),
                   'reference_logits_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                   'input_identity': 'Caller declares fixture; matching run commands establish execution identity'})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
