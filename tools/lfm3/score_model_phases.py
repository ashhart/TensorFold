#!/usr/bin/env python3
"""Score prompt boundary and propagated teacher-forced continuation separately; never pool away a failing phase."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from fp32_bf16_envelope import compare

p = argparse.ArgumentParser()
for key in ['fp32', 'reference-bf16', 'native', 'output']:
    p.add_argument('--' + key, type=Path, required=True)
a = p.parse_args()
aggregate = a.output.with_suffix('.aggregate.json')
inputs = {path.resolve() for path in [a.fp32, a.reference_bf16, a.native]}
inputs |= {path.with_suffix('.json').resolve() for path in [a.fp32, a.reference_bf16]}
if a.output.resolve() in inputs or aggregate.resolve() in inputs:
    raise ValueError('Output must not overwrite an input or reference metadata')
aggregate.unlink(missing_ok=True)
a.output.unlink(missing_ok=True)
r = subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name('full_model_envelope.py')),
        '--fp32',
        str(a.fp32),
        '--reference-bf16',
        str(a.reference_bf16),
        '--native',
        str(a.native),
        '--output',
        str(aggregate),
    ]
)
if not aggregate.exists():
    raise SystemExit('Reference identity validation failed')
d = json.loads(aggregate.read_text())
f = np.load(a.fp32, allow_pickle=False)
b = np.load(a.reference_bf16, allow_pickle=False)
n = (np.fromfile(a.native, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(f.shape)
positions = np.asarray(d['selected_token_positions'])
prompt = d['prompt_len']
phases = {}
for label, mask in [
    ('prompt_boundary', positions < prompt),
    ('teacher_forced_continuation', positions >= prompt),
]:
    if not mask.any():
        continue
    result = compare(n[mask], b[mask], f[mask], space='logits')
    result['absolute_token_positions'] = positions[mask].tolist()
    result['row_max_exceedances'] = [
        x for x in result['all_row_errors'] if x['native_max'] > x['reference_max']
    ]
    result['row_rms_exceedances'] = [
        x for x in result['all_row_errors'] if x['native_rms'] > x['reference_rms']
    ]
    phases[label] = result
out = {
    'scope': 'Independent propagated full-vocabulary prompt-boundary and fixed teacher-forced continuation comparison. Not greedy decode acceptance; no top-pick tie waiver.',
    'files_sha256': d['files_sha256'],
    'aggregate_metrics_pass': d['proposed_metrics_no_worse'],
    'phases': phases,
    'all_phase_metrics_pass': all(x['proposed_metrics_no_worse'] for x in phases.values()),
    'top_pick_acceptance': None,
    'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
a.output.write_text(
    json.dumps(out, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n'
)
print(
    json.dumps(
        {
            'aggregate': out['aggregate_metrics_pass'],
            'all_phases': out['all_phase_metrics_pass'],
            'phases': {
                k: {
                    'pass': v['proposed_metrics_no_worse'],
                    'native': v['native'],
                    'reference': v['reference_bf16'],
                    'native_top1_misses': v['native_fp32_top1_misses'],
                    'reference_top1_misses': v['reference_fp32_top1_misses'],
                }
                for k, v in phases.items()
            },
        },
        default=lambda x: x.item() if isinstance(x, np.generic) else x,
    )
)
if r.returncode or not out['all_phase_metrics_pass']:
    raise SystemExit(1)
