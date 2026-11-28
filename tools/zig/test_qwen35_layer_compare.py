#!/usr/bin/env python3
"""Negative controls for finite layer captures; no optional ML dependencies."""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from qwen35_layer_compare import compare, distance, read_values


class LayerComparisonTests(unittest.TestCase):
    def test_first_drift_and_nonzero_process(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            native, reference = root / 'native', root / 'reference'
            native.mkdir()
            for arm in ('bf16', 'fp32'):
                (reference / arm).mkdir(parents=True)
            meta = dict(rows=2, hidden=1, layers=2, boundaries=4, trace_plain_logits_equal=True)
            for dest in (native, reference):
                (dest / 'meta.json').write_text(json.dumps(meta))
            for i in range(4):
                (native / ('%d.bf16' % i)).write_bytes(struct.pack('<HH', 0x4000 if i == 2 else 0x3f80, 0x3f80))
                (reference / 'bf16' / ('%d.f32' % i)).write_bytes(struct.pack('<ff', 1.25, 1.25))
                (reference / 'fp32' / ('%d.f32' % i)).write_bytes(struct.pack('<ff', 1.0, 1.0))
            result = compare(native, reference)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['first_exceeding_layer_zero_based'], 1)
            script = Path(__file__).with_name('qwen35_layer_compare.py')
            proc = subprocess.run([sys.executable, str(script), '--native', str(native),
                                   '--reference', str(reference), '--output', str(root / 'out.json')],
                                  capture_output=True, timeout=10)
            self.assertEqual(proc.returncode, 1)
            (native / '2.bf16').write_bytes(struct.pack('<HH', 0x3f80, 0x3f80))
            self.assertEqual(compare(native, reference)['status'], 'passed')
            (native / '2.bf16').write_bytes(struct.pack('<HH', 0x3f80, 0x4100))
            served = compare(native, reference)
            self.assertEqual(served['status'], 'passed')
            self.assertEqual(served['compared_rows'], 1)
            self.assertFalse(served['all_input_rows_boundaries'][2]['within_bf16_max_distance'])
            (native / '3.bf16').unlink()
            with self.assertRaises(FileNotFoundError):
                compare(native, reference)

    def test_zero_distance_and_finite_inputs(self):
        self.assertTrue(distance([1.0], [1.0], [1.0], 1)['within_bf16_max_distance'])
        self.assertFalse(distance([1.0001], [1.0], [1.0], 1)['within_bf16_max_distance'])
        with self.assertRaises(ValueError):
            distance([], [], [], 1)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'bad.f32'
            path.write_bytes(struct.pack('<f', float('nan')))
            with self.assertRaises(ValueError):
                read_values(path, 'f32', 1)
            with self.assertRaises(ValueError):
                read_values(path, 'f32', 2)


if __name__ == '__main__':
    unittest.main()
