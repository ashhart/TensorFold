"""Failure controls for the finite FP32/reference-BF16 comparison proposal."""

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fp32_bf16_envelope import compare, input_pins


class EnvelopeTests(unittest.TestCase):
    def test_actual_reference_is_its_own_envelope(self):
        f = np.array([[1.001, -0.501, 0.1], [0.25, 0.125, -0.75]], dtype=np.float32)
        b = np.array([[1, -0.5, 0.1], [0.25, 0.125, -0.75]], dtype=np.float32)
        result = compare(b, b, f, 'logits')
        self.assertTrue(result['proposed_metrics_no_worse'])
        self.assertEqual(result['top_pick_differences'], [])

    def test_better_mean_does_not_hide_worse_maximum(self):
        f = np.zeros((2, 8), dtype=np.float32)
        b = np.full_like(f, 0.125)
        n = f.copy()
        n[0, 0] = 0.25
        result = compare(n, b, f)
        self.assertTrue(result['proposed_metric_results']['rms_error'])
        self.assertFalse(result['proposed_metric_results']['maximum_absolute_error'])
        self.assertFalse(result['proposed_metrics_no_worse'])

    def test_duplicate_input_basenames_are_refused(self):
        with self.assertRaises(ValueError):
            input_pins({'inputs_sha256': {'a/tokens.npy': 'first', 'b/tokens.npy': 'second'}})

    def test_nonfinite_is_not_a_pass(self):
        f = np.ones((1, 4), dtype=np.float32)
        n = f.copy()
        n[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare(n, f, f)

    def test_top_difference_is_retained_with_margin(self):
        f = np.array([[1, 0.5, 0]], dtype=np.float32)
        n = np.array([[0.5, 1, 0]], dtype=np.float32)
        result = compare(n, f, f, 'logits')
        self.assertEqual(result['native_fp32_top1_misses'], 1)
        self.assertEqual(result['top_pick_differences'][0]['fp32_margin'], 0.5)
        self.assertFalse(result['proposed_metrics_no_worse'])

    def test_reference_identity_failure_cannot_reuse_stale_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            f = np.ones((2, 4), dtype=np.float32)
            for name, backend in [('f', 'fp32'), ('b', 'mlx')]:
                path = root / (name + '.npy')
                np.save(path, f)
                metadata = dict(
                    operator='full_model_logits',
                    selected_token_positions=[0, 1],
                    prompt_len=1,
                    teacher_forced_continuation=1,
                    reference_chunk=128,
                    image_start=None,
                    shape=[2, 4],
                    backend=backend,
                    inputs_sha256={'tokens.npy': 'matching-input'},
                    output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    official_sources_sha256={'model.py': 'pinned-source'},
                )
                if name == 'b':
                    metadata['prompt_len'] = 2
                path.with_suffix('.json').write_text(json.dumps(metadata))
            native = root / 'native.bf16'
            (f.view(np.uint32) >> 16).astype('<u2').tofile(native)
            output = root / 'result.json'
            aggregate = output.with_suffix('.aggregate.json')
            aggregate.write_text('{"stale": true}')
            output.write_text('{"stale": true}')
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name('score_model_phases.py')),
                    '--fp32',
                    str(root / 'f.npy'),
                    '--reference-bf16',
                    str(root / 'b.npy'),
                    '--native',
                    str(native),
                    '--output',
                    str(output),
                ],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(aggregate.exists())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
