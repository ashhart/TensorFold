"""The checkpoints check's verdict and peak-memory source, without a server or GPU."""
import copy
import unittest

import drama3_checkpoints_check as check


def matching():
    replies = {f"{mode}-{other}": {"sha256": mode, "bytes": 4, "first_audio_s": 0.1, "total_s": 0.2}
               for mode, _, _ in check.SOLO for other in ("first", "repeat", "lane")}
    return {name: {"held": held, "replies": copy.deepcopy(replies)} for name, held in check.HELD.items()}


class VerdictTests(unittest.TestCase):
    def test_equal_pcm_and_expected_experts_pass(self):
        self.assertEqual(check.verdict(matching()), [])

    def test_a_checkpoint_holding_other_experts_fails(self):
        runs = matching()
        runs["pruned"]["held"] = 10240
        runs["pruned-5900"]["held"] = None
        self.assertEqual(check.verdict(runs), ["pruned: routed experts held 10240, expected 7369",
                                               "pruned-5900: routed experts held None, expected 5900"])

    def test_a_repeat_that_differs_fails(self):
        runs = matching()
        runs["fp8"]["replies"]["whole-repeat"]["sha256"] = "other"
        self.assertEqual(check.verdict(runs), ["fp8 whole: the repeat reply's PCM differs from the first"])

    def test_a_concurrent_lane_that_differs_from_solo_fails(self):
        runs = matching()
        runs["pruned-5900"]["replies"]["stream-lane"]["sha256"] = "other"
        self.assertEqual(check.verdict(runs), ["pruned-5900 stream: the lane reply's PCM differs from the first"])

    def test_empty_replies_fail_even_when_hashes_agree(self):
        runs = matching()
        for reply in runs["pruned"]["replies"].values():
            reply["bytes"] = 0
        problems = check.verdict(runs)
        self.assertEqual(len(problems), 6)
        self.assertIn("pruned stream-lane: no PCM", problems)


class PeakTests(unittest.TestCase):
    def test_device_peak_is_preferred(self):
        memory = {"device": {"peak_bytes": 3}, "process": {"lifetime_peak_physical_footprint_bytes": 9}}
        self.assertEqual(check.peak_bytes(memory), (3, "device.peak_bytes"))

    def test_without_a_device_peak_the_process_peak_is_used(self):
        memory = {"process": {"lifetime_peak_physical_footprint_bytes": 9}}
        self.assertEqual(check.peak_bytes(memory), (9, "process.lifetime_peak_physical_footprint_bytes"))


if __name__ == "__main__":
    unittest.main()
