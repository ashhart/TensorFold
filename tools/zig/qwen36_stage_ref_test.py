"""The float64 reference's pieces against mlx's own ops, and the staged comparer's refusals and planted errors."""
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_ops

from qwen36_f64 import Checkpoint, Model, delta_rule, rms
from qwen36_stage_ref import Dump, bf16, measure, stage_names, step


def tiny_checkpoint(bits, out_rows=8, width=128, group=64):
    """A Checkpoint whose one tensor `w` is mlx-quantized from random weights, without a snapshot on disk."""
    ck = Checkpoint.__new__(Checkpoint)
    ck.bits, ck.group, ck.widths = bits, group, {}
    w = mx.random.normal((out_rows, width), key=mx.random.key(bits)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=group, bits=bits)
    ck.shards = {"w.weight": q, "w.scales": s, "w.biases": b}
    return ck, mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.float32)


class ReferenceOpsTest(unittest.TestCase):
    def test_dequantize_matches_mlx_for_every_width(self):
        for bits in (4, 8):
            ck, want = tiny_checkpoint(bits)
            got = ck.dense("w")
            # mlx dequantizes in bf16; ours is exact, so they agree to one bf16 step of each value.
            np.testing.assert_allclose(got, np.array(want), rtol=2 ** -8, atol=1e-6)
            np.testing.assert_array_equal(ck.dense("w", rows=mx.array([3, 1])), got[[3, 1]])

    def test_rope_matches_mlx_fast_rope(self):
        m = Model.__new__(Model)
        m.rotary, m.theta = 64, 1e7
        x = np.random.default_rng(0).standard_normal((5, 2, 256))
        got = m.rope(x, np.arange(7, 12))
        want = mx.fast.rope(mx.array(x.transpose(1, 0, 2), mx.float32)[None], 64, traditional=False, base=1e7,
                            scale=1.0, offset=7)
        np.testing.assert_allclose(got, np.array(want)[0].transpose(1, 0, 2), atol=2e-5)

    def test_delta_rule_matches_mlx_ops(self):
        rng = np.random.default_rng(1)
        t, hk, hv, d = 6, 2, 4, 16
        q, k = rng.standard_normal((t, hk, d)), rng.standard_normal((t, hk, d))
        v = rng.standard_normal((t, hv, d))
        g, beta = rng.uniform(0.5, 1.0, (t, hv)), rng.uniform(0.0, 1.0, (t, hv))
        f = lambda a: mx.array(a, mx.float32)[None]
        want, _ = gated_delta_ops(f(q), f(k), f(v), f(g), f(beta))
        np.testing.assert_allclose(delta_rule(q, k, v, g, beta), np.array(want)[0], atol=1e-4)

    def test_route_breaks_ties_to_the_lower_id_and_renormalizes(self):
        m = Model.__new__(Model)
        m.top_k = 2
        picks, weights, gap = m.route(np.array([[1.0, 3.0, 3.0, 3.0]]))
        np.testing.assert_array_equal(picks, [[1, 2]])
        np.testing.assert_allclose(weights, [[0.5, 0.5]])
        self.assertEqual(gap[0], 0.0)

    def test_rms(self):
        x = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(rms(x, np.array([2.0, 1.0]), 0.0), [[3 / np.sqrt(12.5) * 2, 4 / np.sqrt(12.5)]])


def write_dump(path: Path, layers, hidden, vocab, rows, logits, positions=None, names=None, shapes=None,
               logit_positions=None, version=1):
    """Logits row i holds bf16(i) everywhere, so a selection of rows reads back; `shapes` overrides a stage's shape."""
    path.mkdir(parents=True, exist_ok=True)
    stages = []
    for n in names or stage_names(layers):
        count, width = (shapes or {}).get(n, (logits, vocab) if n == "logits" else (rows, hidden))
        data = np.zeros((count, width), np.uint16)
        if n == "logits":
            data[:] = (np.arange(count, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)[:, None]
        data.tofile(path / f"{n}.bf16")
        stages.append({"name": n, "width": width, "rows": count})
    index = {"version": version, "dtype": "bf16", "layers": layers, "hidden": hidden, "vocab": vocab,
             "stages": stages, "positions": positions if positions is not None else list(range(rows)),
             "logit_positions": logit_positions if logit_positions is not None else list(range(rows - logits, rows))}
    (path / "index.json").write_text(json.dumps(index))


class ComparerTest(unittest.TestCase):
    def test_dump_refuses_other_shapes_positions_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good"
            write_dump(good, 2, 4, 6, 5, 2)
            args = (2, 4, 6, list(range(5)), [3, 4])
            self.assertEqual(Dump("e", good, *args).get("l01.block").shape, (5, 4))
            # Each case's files agree with its own index, so only the check against the model can refuse it.
            cases = {
                "layers": dict(layers=3), "positions": dict(positions=[0, 1, 2, 4, 5]),
                "names": dict(names=stage_names(2)[:-1]), "repeated": dict(names=stage_names(2) + ["final"]),
                "version": dict(version=2), "repeated logits": dict(logit_positions=[3, 3]),
            }
            for n in ("embed", "l00.mixer", "l01.block", "final", "logits"):
                rows, width = (2, 6) if n == "logits" else (5, 4)
                for bad in ((rows, 1), (rows + 1, width), (rows - 1, width), (rows, width * 2)):
                    cases[f"{n} {bad}"] = dict(shapes={n: bad})
            for label, change in cases.items():
                path = Path(tmp) / label
                kw = dict(layers=2, hidden=4, vocab=6, rows=5, logits=2) | change
                write_dump(path, **kw)
                with self.assertRaises(SystemExit, msg=label):
                    Dump("e", path, *args)
            (good / "final.bf16").write_bytes(b"\0" * 6)
            with self.assertRaises(SystemExit):
                Dump("e", good, *args).get("final")

    def test_every_row_logits_give_the_teacher_rows_and_a_missing_row_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            every = Path(tmp) / "every"
            write_dump(every, 1, 4, 6, 5, 5)  # one-row steps: logits at positions 0..4
            got = Dump("e", every, 1, 4, 6, list(range(5)), [3, 4]).get("logits")
            np.testing.assert_array_equal(got[:, 0], [3.0, 4.0])
            with self.assertRaises(SystemExit):
                Dump("e", Path(tmp) / "every", 1, 4, 6, list(range(5)), [4, 5])

    def test_step_is_bf16_spacing_at_every_exponent(self):
        for e in range(-100, 100):
            for mant in (1.0, 1.5, 1.9921875):
                v = mant * 2.0 ** e
                self.assertEqual(step(np.array([v]))[0], 2.0 ** (e - 7))
                self.assertEqual(step(np.array([-v]))[0], 2.0 ** (e - 7))

    def test_a_planted_step_is_found_on_every_row(self):
        rng = np.random.default_rng(2)
        ref = bf16(rng.integers(0x3f00, 0x4100, (6, 8), dtype=np.uint16))
        for row in range(6):
            for col in range(8):
                for k in (1, 3):
                    engine = ref.copy()
                    engine[row, col] += k * step(np.abs(ref[row]).max(keepdims=True))[0]
                    m = measure(engine, ref, {"r": row})
                    self.assertEqual(m["worst_row"], row)
                    self.assertAlmostEqual(m["max_steps"], k)
                    self.assertAlmostEqual(m["rows"]["r"]["steps"], k)
        self.assertEqual(measure(ref, ref, {})["max_steps"], 0.0)


if __name__ == "__main__":
    unittest.main()
