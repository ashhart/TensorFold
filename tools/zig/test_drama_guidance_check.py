"""The guidance check's verdict, without a server or GPU."""
import copy
import unittest

import drama_guidance_check as check


def replies(names):
    return {name: {"sha256": name, "bytes": 4, "first_audio_s": 0.1, "total_s": 0.2} for name in names}


def matching():
    names = [s[0] for step in check.plan("vesper") for s in (step if isinstance(step, list) else [step])]
    guided = replies(names)
    return {"plain": replies(n for n in names if n.endswith("-g1")), "separate": guided, "paired": copy.deepcopy(guided)}


class VerdictTests(unittest.TestCase):
    def test_equal_pcm_passes(self):
        self.assertEqual(check.verdict(matching()), [])

    def test_paired_that_differs_from_separate_fails(self):
        modes = matching()
        modes["paired"]["lane-b"]["sha256"] = "other"
        self.assertEqual(check.verdict(modes), ["lane-b: paired PCM differs from separate"])

    def test_gamma_one_that_differs_from_plain_fails(self):
        modes = matching()
        modes["plain"]["t2-g1"]["sha256"] = "other"
        self.assertEqual(check.verdict(modes), ["t2-g1: separate gamma 1 differs from the server without CFG",
                                                "t2-g1: paired gamma 1 differs from the server without CFG"])


if __name__ == "__main__":
    unittest.main()
