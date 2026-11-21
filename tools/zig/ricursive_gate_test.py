"""The Ripple gate tools on synthetic files: KL, top-1, worst rows, identical dumps, replies, an exact-length chat."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ricursive_gate_compare as cmp
import ricursive_gate_ids as ids_tool

VOCAB, ROWS = 50, 12


def bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cmp.main(list(argv))
    return code, out.getvalue()


class CompareTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        rng = np.random.default_rng(1)
        self.ref = bf16(rng.normal(0, 3, (ROWS, VOCAB)))
        self.eng = self.ref.copy()
        self.eng[3] = bf16(rng.normal(0, 3, VOCAB))  # row 3 is another distribution
        self.ids = [int(t) for t in rng.integers(0, VOCAB, ROWS)]
        (self.dir / "ids.json").write_text(json.dumps(self.ids))

    def raw(self, name: str, bits: np.ndarray) -> str:
        path = self.dir / name
        bits.astype("<u2").tofile(path)
        return str(path)

    def kl_dump(self, name: str, bits: np.ndarray, fed: list[int]) -> str:
        path = self.dir / name
        with open(path, "wb") as f:
            f.write(np.array([bits.shape[0], bits.shape[1]], dtype="<u4").tobytes())
            for r in range(bits.shape[0]):
                f.write(np.array([fed[r]], dtype="<u4").tobytes() + bits[r].astype("<u2").tobytes())
        return str(path)

    def expected_kl(self, r: int) -> float:
        p, q = f32(self.ref[r]).astype(np.float64), f32(self.eng[r]).astype(np.float64)
        p, q = p - np.log(np.exp(p).sum()), q - np.log(np.exp(q).sum())
        return float((np.exp(p) * (p - q)).sum())

    def test_kl_matches_the_formula_for_both_formats(self):
        fed = self.ids[1:] + [0]
        for eng in (self.raw("e.bin", self.eng), self.kl_dump("e.kl", self.eng, fed)):
            code, text = run("kl", self.raw("r.bin", self.ref), eng, "--ids", str(self.dir / "ids.json"),
                             "--vocab", str(VOCAB), "--out", str(self.dir / "rep.json"))
            rep = json.loads((self.dir / "rep.json").read_text())
            self.assertEqual(code, 0)
            self.assertAlmostEqual(rep["kl_mean"], self.expected_kl(3) / ROWS, places=9)
            self.assertEqual(rep["worst"][0]["row"], 3)
            self.assertAlmostEqual(rep["worst"][0]["kl"], self.expected_kl(3), places=9)
            self.assertEqual(rep["worst"][0]["next_token"], self.ids[4])
            self.assertIn("worst rows", text)
            same = float(np.mean(f32(self.ref).argmax(-1) == f32(self.eng).argmax(-1)))
            self.assertAlmostEqual(rep["top1_agreement"], same)

    def test_pass_rule_flags_kl_and_nonfinite_logits(self):
        args = ("--ids", str(self.dir / "ids.json"), "--vocab", str(VOCAB))
        ref = self.raw("r.bin", self.ref)
        eng = self.raw("e.bin", self.eng)
        self.assertEqual(run("kl", ref, eng, *args, "--max-kl", "10")[0], 0)
        code, text = run("kl", ref, eng, *args, "--max-kl", "0.0001")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", text)
        bad = self.eng.copy()
        bad[5, 2] = 0x7FC0  # a bf16 NaN
        self.assertEqual(run("kl", ref, self.raw("n.bin", bad), *args)[0], 1)

    def test_a_dump_that_fed_other_tokens_is_refused(self):
        dump = self.kl_dump("e.kl", self.eng, [t + 1 for t in self.ids])
        with self.assertRaises(SystemExit):
            run("kl", self.raw("r.bin", self.ref), dump, "--ids", str(self.dir / "ids.json"), "--vocab", str(VOCAB))

    def test_same_finds_the_first_differing_row(self):
        a = self.raw("a.bin", self.ref)
        self.assertEqual(run("same", a, self.raw("b.bin", self.ref), "--vocab", str(VOCAB))[0], 0)
        code, text = run("same", a, self.raw("c.bin", self.eng), "--vocab", str(VOCAB))
        self.assertEqual(code, 1)
        self.assertIn("1 of 12 rows differ", text)
        self.assertIn("first row 3", text)

    def test_reply_equal_and_not(self):
        prompt = 5
        ref = self.ref
        reply = [int(f32(ref[prompt - 1 + i]).argmax()) for i in range(4)]
        path = self.dir / "reply.json"
        path.write_text(json.dumps({"tokens": reply}))  # the run command's report
        self.assertEqual(run("reply", self.raw("r.bin", ref), str(path), "--prompt-len", str(prompt), "--vocab", str(VOCAB))[0], 0)
        reply[2] = (reply[2] + 1) % VOCAB
        path.write_text(json.dumps(reply))
        code, text = run("reply", self.raw("r.bin", ref), str(path), "--prompt-len", str(prompt), "--vocab", str(VOCAB))
        self.assertEqual(code, 1)
        self.assertIn("3 equal", text)
        self.assertIn("reply token 2", text)

    def test_a_longer_reference_is_read_for_its_first_rows(self):
        longer = np.concatenate([self.ref, self.ref[:3]])
        code, _ = run("kl", self.raw("r.bin", longer), self.raw("e.bin", self.eng), "--ids", str(self.dir / "ids.json"),
                      "--vocab", str(VOCAB), "--out", str(self.dir / "rep.json"))
        self.assertEqual(json.loads((self.dir / "rep.json").read_text())["rows"], ROWS)
        self.assertEqual(code, 0)

    def test_finite_reports_nan_rows_and_next_token_hits(self):
        good = self.raw("g.bin", self.ref)
        code, text = run("finite", good, "--vocab", str(VOCAB), "--ids", str(self.dir / "ids.json"))
        self.assertEqual(code, 0)
        self.assertIn("0 with a non-finite logit", text)
        hits = 100 * np.mean(f32(self.ref).argmax(-1)[:-1] == np.array(self.ids[1:]))
        self.assertIn(f"{hits:.1f}%", text)
        bad = self.ref.copy()
        bad[4, 0] = 0x7F80  # bf16 +inf
        code, text = run("finite", self.raw("b.bin", bad), "--vocab", str(VOCAB))
        self.assertEqual(code, 1)
        self.assertIn("1 with a non-finite logit", text)

    def test_split_writes_the_prompt_and_prints_the_rest(self):
        code, text = run("split", str(self.dir / "ids.json"), str(self.dir / "p.json"), "--prompt", "2")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.dir / "p.json").read_text()), self.ids[:2])
        self.assertEqual(text.strip(), ",".join(map(str, self.ids[2:])))

    def test_join(self):
        (self.dir / "reply.json").write_text(json.dumps({"tokens": [7, 8]}))
        run("join", str(self.dir / "ids.json"), str(self.dir / "reply.json"), str(self.dir / "all.json"))
        self.assertEqual(json.loads((self.dir / "all.json").read_text()), self.ids + [7, 8])


class IdsTests(unittest.TestCase):
    def count(self, chat):  # a stand-in tokenizer: a token a word, two for a long word, and four a turn
        return sum(sum(2 if len(w) > 3 else 1 for w in m["content"].split()) + 4 for m in chat) + 3

    def test_fit_is_exact_and_deterministic(self):
        text = " ".join(f"w{i}" for i in range(2000))
        for length in (1500, 1501, 1502, 1503):
            chat = ids_tool.fit(self.count, text, length)
            self.assertEqual(self.count(chat), length)
            self.assertEqual(chat, ids_tool.fit(self.count, text, length))

    def test_fit_refuses_a_short_text_and_long_fixed_turns(self):
        with self.assertRaises(SystemExit):
            ids_tool.fit(self.count, " ".join(["w"] * 400), 2586)
        with self.assertRaises(SystemExit):
            ids_tool.fit(self.count, " ".join(["w"] * 2000), 100)

    def test_the_release_template_renders_the_chat(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2 is not installed")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        (Path(folder.name) / "chat_template.jinja").write_text(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        text = ids_tool.load_renderer(Path(folder.name))(ids_tool.messages("a b", "c d"))
        self.assertTrue(text.startswith("<|im_start|>system\n"))
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()
