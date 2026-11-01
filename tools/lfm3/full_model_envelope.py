#!/usr/bin/env python3
"""Validate propagated full-logit reference identities, then apply the unchanged canonical scorer."""
import argparse, json, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--fp32', type=Path, required=True)
p.add_argument('--reference-bf16', type=Path, required=True)
p.add_argument('--native', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
f = json.loads(a.fp32.with_suffix('.json').read_text())
b = json.loads(a.reference_bf16.with_suffix('.json').read_text())
for k in [
    'operator',
    'selected_token_positions',
    'prompt_len',
    'teacher_forced_continuation',
    'reference_chunk',
    'image_start',
    'shape',
]:
    if f[k] != b[k]:
        raise ValueError(('Full-model reference mismatch', k))
if f['operator'] != 'full_model_logits' or not b['official_sources_sha256']:
    raise ValueError('Need actual official propagated full-model reference')
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name('fp32_bf16_envelope.py')),
        '--fp32',
        str(a.fp32),
        '--reference-bf16',
        str(a.reference_bf16),
        '--native',
        str(a.native),
        '--output',
        str(a.output),
        '--space',
        'logits',
    ],
    check=True,
)
d = json.loads(a.output.read_text())
d['selected_token_positions'] = f['selected_token_positions']
d['prompt_len'] = f['prompt_len']
d['teacher_forced_continuation'] = f['teacher_forced_continuation']
d['scope'] = (
    'Propagated decoder selected full-vocabulary logits; no standalone operator substitution. Aggregate whole-cell RMS/max and fixed-F-top20 mean/max gate; row/column and pick differences explicit.'
)
d['row_max_exceedances'] = [r for r in d['all_row_errors'] if r['native_max'] > r['reference_max']]
d['row_rms_exceedances'] = [r for r in d['all_row_errors'] if r['native_rms'] > r['reference_rms']]
a.output.write_text(json.dumps(d, indent=2) + '\n')
if not d['proposed_metrics_no_worse']:
    raise SystemExit('FullModelMetricEnvelopeFailure')
