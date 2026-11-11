"""The guidance check's verdict and voice requirement, without a server or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import drama_guidance_check as check


def replies(names):
    return {name: {"sha256": name, "bytes": 4, "first_audio_s": 0.1, "total_s": 0.2} for name in names}


def matching():
    names = [s[0] for step in check.plan("vesper") for s in (step if isinstance(step, list) else [step])]
    guided = replies(names)
    modes = {"plain": replies(n for n in names if n.endswith("-g1")), "separate": guided, "paired": copy.deepcopy(guided)}
    return {rows: copy.deepcopy(modes) for rows in check.ROWS}


class VerdictTests(unittest.TestCase):
    def test_equal_pcm_passes(self):
        self.assertEqual(check.verdict(matching()), [])

    def test_paired_that_differs_from_separate_fails(self):
        runs = matching()
        runs[1024]["paired"]["lane-b"]["sha256"] = "other"
        self.assertEqual(check.verdict(runs), ["rows=1024 lane-b: paired PCM differs from separate"])

    def test_gamma_one_that_differs_from_plain_fails(self):
        runs = matching()
        runs[1]["plain"]["t2-g1"]["sha256"] = "other"
        self.assertEqual(check.verdict(runs), ["rows=1 t2-g1: separate gamma 1 differs from the server without CFG",
                                               "rows=1 t2-g1: paired gamma 1 differs from the server without CFG"])

    def test_replies_without_pcm_fail_even_when_hashes_agree(self):
        runs = matching()
        for modes in runs.values():
            for replies_ in modes.values():
                for reply in replies_.values():
                    reply.update(bytes=0, sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        problems = check.verdict(runs)
        self.assertIn("rows=1 t0-g1: plain returned no PCM", problems)
        self.assertIn("rows=1024 lane-a: paired returned no PCM", problems)


class VoiceTests(unittest.TestCase):
    def test_only_a_reference_backed_catalog_voice_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            voices = Path(tmp)
            (voices / "vesper.json").write_text(json.dumps({"references": [{"text": "A cloud.", "codes": "x"}]}))
            (voices / "bare.json").write_text(json.dumps({"references": []}))
            check.require_reference(voices, "vesper")
            for voice in ("default", "Default", "bare", "missing"):
                with self.assertRaises(SystemExit, msg=voice):
                    check.require_reference(voices, voice)


if __name__ == "__main__":
    unittest.main()
