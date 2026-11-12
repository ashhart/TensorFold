"""Host checks for bounded checkpoint metadata inspection."""

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inspect_checkpoint as inspector


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


class InspectorTests(unittest.TestCase):
    def test_refuses_full_weight_response_before_reading(self):
        response = Response(b'weight data', 200, {})
        with patch.object(inspector.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(inspector.InspectionError, 'refused'):
                inspector.Remote.fetch('https://example.org/model', 8, (0, 7))
        self.assertEqual(response.read_calls, 0)

    def test_rejects_wrong_range(self):
        response = Response(b'12345678', 206, {'Content-Range': 'bytes 8-15/100'})
        with patch.object(inspector.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(inspector.InspectionError, 'unexpected'):
                inspector.Remote.fetch('https://example.org/model', 8, (0, 7))

    def test_rejects_truncated_range(self):
        response = Response(b'123', 206, {'Content-Range': 'bytes 0-7/100'})
        with patch.object(inspector.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(inspector.InspectionError, 'Truncated'):
                inspector.Remote.fetch('https://example.org/model', 8, (0, 7))

    def test_accepts_exact_range(self):
        response = Response(b'12345678', 206, {'Content-Range': 'bytes 0-7/100', 'Content-Length': '8'})
        with patch.object(inspector.urllib.request, 'urlopen', return_value=response):
            self.assertEqual(inspector.Remote.fetch('https://example.org/model', 8, (0, 7)), b'12345678')

    def test_caps_non_range_metadata(self):
        response = Response(b'123456789', 200, {})
        with patch.object(inspector.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(inspector.InspectionError, 'body limit'):
                inspector.Remote.fetch('https://example.org/config', 8)

    def test_rejects_oversize_header_and_unsafe_names(self):
        with self.assertRaises(inspector.InspectionError):
            inspector.header_size(struct.pack('<Q', inspector.MAX_HEADER + 1))
        for name in ('../model', '/model', 'sub/../../model', 'sub\\model'):
            with self.subTest(name=name), self.assertRaises(inspector.InspectionError):
                inspector.safe_name(name)

    def test_truncated_local_tensor_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            header = json.dumps({'weight': {
                'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4],
            }}).encode()
            (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'12')
            with self.assertRaisesRegex(inspector.InspectionError, 'bounds'):
                inspector.inspect(inspector.Local(root))

    def remote_with_header(self, header, payload_length, second_total_delta=0):
        data = json.dumps(header).encode()
        total = 8 + len(data) + payload_length
        responses = [
            Response(struct.pack('<Q', len(data)), 206, {'Content-Range': f'bytes 0-7/{total}'}),
            Response(data, 206, {'Content-Range': f'bytes 8-{7 + len(data)}/{total + second_total_delta}'}),
        ]
        remote = inspector.Remote.__new__(inspector.Remote)
        remote.base = 'https://example.org/'
        remote.repository = 'owner/model'
        remote.revision = 'a' * 40
        remote.files = {'model.safetensors'}
        remote.json = lambda name: {}
        return remote, responses

    def test_remote_tensor_offsets_exceed_real_file(self):
        remote, responses = self.remote_with_header({
            'weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
        }, 2)
        with patch.object(inspector.urllib.request, 'urlopen', side_effect=responses):
            with self.assertRaisesRegex(inspector.InspectionError, 'bounds'):
                inspector.inspect(remote)

    def test_remote_total_must_stay_constant(self):
        remote, responses = self.remote_with_header({}, 4, second_total_delta=1)
        with patch.object(inspector.urllib.request, 'urlopen', side_effect=responses):
            with self.assertRaisesRegex(inspector.InspectionError, 'size changed'):
                remote.header('model.safetensors')

    def test_validates_descriptor_shapes_and_offsets(self):
        bad = [
            {'dtype': 'BF16', 'shape': [-1], 'data_offsets': [0, 2]},
            {'dtype': 'BF16', 'shape': [True], 'data_offsets': [0, 2]},
            {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0]},
            {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 2]},
            {'dtype': 'unsupported', 'shape': [2], 'data_offsets': [0, 4]},
        ]
        for descriptor in bad:
            with self.subTest(descriptor=descriptor), self.assertRaises(inspector.InspectionError):
                inspector.validate_tensor('weight', descriptor, 4)

    def test_rejects_overlapping_tensor_data(self):
        tensor = {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}
        remote, responses = self.remote_with_header({'a': tensor, 'b': tensor}, 4)
        with patch.object(inspector.urllib.request, 'urlopen', side_effect=responses):
            with self.assertRaisesRegex(inspector.InspectionError, 'Overlapping'):
                inspector.inspect(remote)

    def test_local_checkpoint_and_index_cross_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{"model_type":"lfm3"}')
            tensor = {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}
            header = json.dumps({'layer.weight': tensor}).encode()
            (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
            result = inspector.inspect(inspector.Local(root))
            self.assertEqual(result['tensor_count'], 1)
            self.assertEqual(result['shard_data_lengths'], {'model.safetensors': 4})
            self.assertEqual(result['dtype_counts'], {'BF16': 1})
            (root / 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'wrong.weight': 'model.safetensors'},
            }))
            with self.assertRaisesRegex(inspector.InspectionError, 'disagree'):
                inspector.inspect(inspector.Local(root))


if __name__ == '__main__':
    unittest.main()
