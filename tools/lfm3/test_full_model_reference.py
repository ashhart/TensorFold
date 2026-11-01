import unittest, tempfile, json, struct
from pathlib import Path
import numpy as np
from full_model_reference import Checkpoint, FP32, inject_check
from fp32_graph_reference import Graph


class Tests(unittest.TestCase):
    def checkpoint(self, prefix, index):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / 'config.json').write_text(json.dumps({'model_type': 'lfm3', 'tie_word_embeddings': True}))
            name = prefix + 'embed_tokens.weight'
            h = {name: {'dtype': 'U32', 'shape': [1, 2], 'data_offsets': [0, 8]}}
            raw = json.dumps(h).encode()
            (d / 'model.safetensors').write_bytes(
                struct.pack('<Q', len(raw)) + raw + np.array([3, 5], dtype='<u4').tobytes()
            )
            if index:
                (d / 'model.safetensors.index.json').write_text(
                    json.dumps({'weight_map': {name: 'model.safetensors'}})
                )
            c = Checkpoint(d)
            self.assertEqual(c.prefix, prefix)
            np.testing.assert_array_equal(c.tensor(name), [[3, 5]])

    def test_text_single(self):
        self.checkpoint('model.', False)

    def test_vl_index(self):
        self.checkpoint('language_model.model.', True)

    def test_query_tiling(self):
        r = np.random.default_rng(1502)
        q = r.normal(size=(2, 17, 8)).astype(np.float32)
        k = r.normal(size=q.shape).astype(np.float32)
        v = r.normal(size=q.shape).astype(np.float32)
        f = object.__new__(FP32)
        f.query_tile = 3
        f.boundaries = 'fp32'
        for window in [0, 1, 7]:
            np.testing.assert_allclose(
                f.attention_value(q, k, v, window),
                Graph.attention_value(f, q, k, v, window),
                rtol=1e-6,
                atol=2e-7,
            )

    def test_image_span(self):
        inject_check(np.array([1, 125030, 125030, 2]), np.zeros((2, 8), np.float32), 1, 8)
        with self.assertRaises(ValueError):
            inject_check(np.array([125030, 1, 125030, 2]), np.zeros((2, 8), np.float32), 1, 8)


if __name__ == '__main__':
    unittest.main()
