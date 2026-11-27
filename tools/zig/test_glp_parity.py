"""The GLP parity comparator rejects mismatched values and layer identities."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glp_parity as parity

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class Compare(unittest.TestCase):
    """The comparison the script runs on the two readers' views of one file."""

    def setUp(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy is required")
        self.meta = {
            "glp.mode": "add",
            "glp.hook_point": "residual_stream_post_layer",
            "glp.alpha_default": "1.0",
            "glp.layer_ids_zero_based": "7",
        }
        self.tensors = {"direction.7": numpy.zeros(8, dtype=numpy.float32)}
        self.tensors["direction.7"][3] = 1.0
        self.ours = {
            "hook_point": "residual_stream_post_layer",
            "alpha": 1.0,
            "method": "",
            "contrast": "",
            "base_model": "",
            "base_revision": "",
            "sha256": "0" * 64,
            "layers": [7],
        }
        self.blob = self.tensors["direction.7"].astype("<f4").tobytes()

    def test_the_same_file_is_the_same_view(self):
        problems: list[str] = []
        parity.compare(self.meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual([], problems)

    def test_a_different_layer_is_reported(self):
        problems: list[str] = []
        ours = dict(self.ours, layers=[8])
        parity.compare(self.meta, self.tensors, ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("layer ids differ", problems[0])

    def test_a_different_number_is_reported(self):
        problems: list[str] = []
        blob = bytearray(self.blob)
        blob[0] = 0x3F  # a different first element
        parity.compare(self.meta, self.tensors, self.ours, bytes(blob), problems)
        self.assertEqual(1, len(problems))
        self.assertIn("direction.7 differs", problems[0])

    def test_a_layer_list_that_does_not_match_the_tensors_is_reported(self):
        problems: list[str] = []
        meta = dict(self.meta, **{"glp.layer_ids_zero_based": "9"})
        parity.compare(meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("names layers", problems[0])

    def test_several_errors_on_one_file_do_not_agree_or_go_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_file = root / "ref" / "plugins" / "vllm" / "weightless_steer" / "container.py"
            ref_file.parent.mkdir(parents=True)
            vec_ok = root / "ok.gguf"
            vec_bad = root / "bad.gguf"
            vec_ok.write_bytes(b"OK")
            vec_bad.write_bytes(b"BAD")
            ok_sha = hashlib.sha256(b"OK").hexdigest()
            bad_sha = hashlib.sha256(b"BAD").hexdigest()
            ref_file.write_text(
                "import numpy as np\n"
                "def read_gguf_cvec(path):\n"
                "    t = {'direction.7': np.array([1.0, 2.0], dtype=np.float32)}\n"
                "    if path.endswith('bad.gguf'):\n"
                "        return ({'glp.mode': 'add', 'glp.hook_point': 'residual_stream_post_layer', "
                "'glp.alpha_default': '1.0', 'glp.layer_ids_zero_based': '9'}, t)\n"
                "    return ({'glp.mode': 'add', 'glp.hook_point': 'residual_stream_post_layer', "
                "'glp.alpha_default': '1.0', 'glp.layer_ids_zero_based': '7'}, t)\n"
            )
            exe = root / "fake_exe.py"
            exe.write_text(
                f"#!{sys.executable}\n"
                "import json, struct, sys\n"
                "from pathlib import Path\n"
                "cmd, vec = sys.argv[1], sys.argv[2]\n"
                "if cmd == 'read':\n"
                "    is_bad = vec.endswith('bad.gguf')\n"
                f"    fsha = {bad_sha!r} if is_bad else {ok_sha!r}\n"
                "    Path(sys.argv[3]).write_text(json.dumps({\n"
                "        'hook_point': 'wrong_hook' if is_bad else 'residual_stream_post_layer',\n"
                "        'alpha': 2.0 if is_bad else 1.0,\n"
                "        'method': '', 'contrast': '', 'base_model': '', 'base_revision': '',\n"
                "        'sha256': '0' * 64, 'file_sha256': fsha, 'width': 2, 'layers': [7],\n"
                "    }))\n"
                "    vals = (9.0, 2.0) if is_bad else (1.0, 2.0)\n"
                "    Path(sys.argv[4]).write_bytes(struct.pack('<2f', *vals))\n"
                "elif cmd == 'rewrite':\n"
                "    Path(sys.argv[3]).write_bytes(Path(vec).read_bytes())\n"
            )
            exe.chmod(0o755)
            out_single = io.StringIO()
            with contextlib.redirect_stdout(out_single):
                rc_single = parity.main(["--reference", str(root / "ref"), "--vector", str(vec_bad), "--exe", str(exe), "--work", str(root / "w1")])
            self.assertEqual(1, rc_single)
            text_single = out_single.getvalue()
            self.assertNotIn("reference and Zig agree", text_single)
            self.assertIn("glp parity: 0/1 files agree", text_single)
            self.assertGreaterEqual(text_single.count("\ndiffers:"), 3)

            out_two = io.StringIO()
            with contextlib.redirect_stdout(out_two):
                rc_two = parity.main(["--reference", str(root / "ref"), "--vector", str(vec_ok), "--vector", str(vec_bad), "--exe", str(exe), "--work", str(root / "w2")])
            self.assertEqual(1, rc_two)
            text_two = out_two.getvalue()
            self.assertIn("ok.gguf:", text_two)
            self.assertIn("reference and Zig agree", text_two)
            self.assertNotIn("bad.gguf: 2 layers", text_two.split("reference and Zig agree")[0].split("\n")[-1])
            self.assertIn("glp parity: 1/2 files agree", text_two)


if __name__ == "__main__":
    unittest.main()
