#!/usr/bin/env python3
"""Compare an FP32/native/actual-reference-BF16 triplet without adding numerical slack."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def input_pins(metadata):
    entries = metadata['inputs_sha256']
    pins = {Path(key).name: value for key, value in entries.items()}
    if len(pins) != len(entries):
        raise ValueError('Reference input basenames must be unique')
    return pins


def metrics(x, f):
    e = np.asarray(x, dtype=np.float64) - np.asarray(f, dtype=np.float64)
    r = e.reshape(-1, e.shape[-1])
    den = float(np.linalg.norm(np.asarray(f, dtype=np.float64).ravel()))
    return {
        'mean_absolute_error': float(np.mean(np.abs(e))),
        'rms_error': float(np.sqrt(np.mean(e * e))),
        'relative_rms': float(np.linalg.norm(e.ravel()) / den) if den else None,
        'maximum_absolute_error': float(np.max(np.abs(e))),
        'maximum_coordinate': list(np.unravel_index(np.argmax(np.abs(e)), e.shape)),
    }, r


def compare(n, b, f, space='operator'):
    if n.shape != b.shape or n.shape != f.shape or n.ndim < 2 or n.size == 0:
        raise ValueError('Same nonempty shape required')
    if not all(np.isfinite(x).all() for x in [n, b, f]):
        raise ValueError('Nonfinite triplet is not a finite envelope pass')
    nm, ne = metrics(n, f)
    bm, be = metrics(b, f)
    rows_n = np.sqrt(np.mean(ne * ne, axis=-1))
    rows_b = np.sqrt(np.mean(be * be, axis=-1))
    cols_n = np.sqrt(np.mean(ne * ne, axis=0))
    cols_b = np.sqrt(np.mean(be * be, axis=0))
    excess = np.abs(ne) - np.abs(be)
    result = {
        'shape': list(f.shape),
        'native': nm,
        'reference_bf16': bm,
        'zero_added_slack': True,
        'proposed_metric_results': {k: nm[k] <= bm[k] for k in ['rms_error', 'maximum_absolute_error']},
        'pointwise_native_worse_count': int(np.count_nonzero(excess > 0)),
        'maximum_pointwise_excess': float(np.max(excess)),
        'pointwise_is_diagnostic_not_invented_gate': True,
        'all_row_errors': [
            {
                'flat_row': i,
                'native_rms': float(rows_n[i]),
                'reference_rms': float(rows_b[i]),
                'native_max': float(np.abs(ne[i]).max()),
                'reference_max': float(np.abs(be[i]).max()),
            }
            for i in range(len(ne))
        ],
        'worst_columns': [
            {
                'column': int(i),
                'native_rms': float(cols_n[i]),
                'reference_rms': float(cols_b[i]),
                'native_max': float(np.abs(ne[:, i]).max()),
                'reference_max': float(np.abs(be[:, i]).max()),
            }
            for i in np.argsort(cols_n)[-20:][::-1]
        ],
    }
    result['worst_rows'] = sorted(result['all_row_errors'], key=lambda x: x['native_rms'], reverse=True)[:20]
    if space == 'logits':
        if n.ndim != 2:
            raise ValueError('Logits must be rows,vocabulary')

        def prob(x):
            x = x.astype(np.float64)
            p = np.exp(x - x.max(axis=1, keepdims=True))
            return p / p.sum(axis=1, keepdims=True)

        pn, pb, pf = map(prob, [n, b, f])
        top = np.argsort(-f, axis=1, kind='stable')[:, : min(20, f.shape[1])]
        take = lambda x: np.take_along_axis(x, top, axis=1)
        pe_n = np.abs(take(pn) - take(pf))
        pe_b = np.abs(take(pb) - take(pf))
        result['probability_scoring'] = (
            'Shared stable NumPy binary64 scoring of captured logits only, not FP64 model forward; fixed top20 selected by F, ties use column order'
        )
        result['top20_probability'] = {
            'native_mean': float(pe_n.mean()),
            'reference_mean': float(pe_b.mean()),
            'native_max': float(pe_n.max()),
            'reference_max': float(pe_b.max()),
        }
        result['proposed_metric_results'].update(
            top20_probability_mean=pe_n.mean() <= pe_b.mean(), top20_probability_max=pe_n.max() <= pe_b.max()
        )
        misses = []
        for row, (x, y, z) in enumerate(zip(n, b, f)):
            ni, bi, fi = map(lambda a: int(a.argmax()), [x, y, z])
            if ni == bi == fi:
                continue
            sets = [np.flatnonzero(a == a.max()).tolist() for a in [x, y, z]]
            fs = np.sort(z.astype(np.float64))
            native_sorted = np.sort(x.astype(np.float64))
            reference_sorted = np.sort(y.astype(np.float64))
            misses.append(
                {
                    'row': row,
                    'native_top': ni,
                    'reference_bf16_top': bi,
                    'fp32_top': fi,
                    'native_maximum_set': sets[0],
                    'reference_maximum_set': sets[1],
                    'fp32_maximum_set': sets[2],
                    'native_margin': float(native_sorted[-1] - native_sorted[-2]),
                    'reference_bf16_margin': float(reference_sorted[-1] - reference_sorted[-2]),
                    'fp32_margin': float(fs[-1] - fs[-2]),
                    'fp32_native_contender_gap': float(z[fi]) - float(z[ni]),
                    'fp32_reference_contender_gap': float(z[fi]) - float(z[bi]),
                    'native_only_miss': ni != fi and bi == fi,
                }
            )
        result['top_pick_differences'] = misses
        result['native_fp32_top1_misses'] = int(np.count_nonzero(n.argmax(-1) != f.argmax(-1)))
        result['reference_fp32_top1_misses'] = int(np.count_nonzero(b.argmax(-1) != f.argmax(-1)))
        result['top_pick_acceptance'] = (
            'No invented tie waiver or automatic pick acceptance; root gate must specify rule'
        )
    result['proposed_metrics_no_worse'] = all(bool(v) for v in result['proposed_metric_results'].values())
    result['proposed_metric_results'] = {k: bool(v) for k, v in result['proposed_metric_results'].items()}
    result['full_model_fidelity_pass'] = None
    result['scope'] = (
        'Finite held-input operator'
        if space == 'operator'
        else 'Teacher-forced captured logits; scope and top-pick acceptance must be supplied by cell manifest'
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ['fp32', 'reference-bf16', 'native', 'output']:
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--native-format', choices=['bf16', 'f32'], default='bf16')
    p.add_argument('--space', choices=['operator', 'logits'], default='operator')
    a = p.parse_args()
    f = np.load(a.fp32, allow_pickle=False)
    b = np.load(a.reference_bf16, allow_pickle=False)
    if f.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError('FP32 files required')
    if np.any(b.view(np.uint32) & 0xFFFF):
        raise ValueError('Reference file is not actual BF16 values promoted to FP32')
    if a.native_format == 'bf16':
        n = (np.fromfile(a.native, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    else:
        n = np.fromfile(a.native, dtype='<f4')
    if n.size != f.size:
        raise ValueError('Native byte geometry differs')
    n = n.reshape(f.shape)
    fm = json.loads(a.fp32.with_suffix('.json').read_text())
    bm = json.loads(a.reference_bf16.with_suffix('.json').read_text())
    if fm['backend'] != 'fp32' or bm['backend'] != 'mlx':
        raise ValueError('Require independent FP32 and actual official MLX BF16')
    if input_pins(fm) != input_pins(bm):
        raise ValueError('Reference input identity mismatch')
    for k in ['operator', 'rows', 'history', 'window', 'scale_bits', 'n', 'k', 'bits', 'group', 'shape']:
        if fm.get(k) != bm.get(k):
            raise ValueError(('Reference geometry mismatch', k))
    if sha(a.fp32) != fm['output_sha256'] or sha(a.reference_bf16) != bm['output_sha256']:
        raise ValueError('Reference output hash mismatch')
    out = compare(n, b, f, a.space)
    out.update(
        native_format=a.native_format,
        native_fp32_is_diagnostic=a.native_format == 'f32',
        input_pins_verified=True,
        files_sha256={str(p): sha(p) for p in [a.fp32, a.reference_bf16, a.native]},
        comparator_sha256=sha(__file__),
    )

    def encoder(x):
        return x.item() if isinstance(x, np.generic) else x

    a.output.write_text(json.dumps(out, indent=2, default=encoder, allow_nan=False) + '\n')
    print(
        json.dumps(
            {
                'proposed_metrics_no_worse': out['proposed_metrics_no_worse'],
                'native': out['native'],
                'reference': out['reference_bf16'],
            },
            default=encoder,
        )
    )


if __name__ == '__main__':
    main()
