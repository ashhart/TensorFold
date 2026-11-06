"""The reference cache check's verdict, without a server or GPU."""
import copy
import unittest

import drama_reference_cache_check as check


def reply(sha):
    return {"sha256": sha, "bytes": 4, "first_audio_s": 0.1, "total_s": 0.2}


def matching():
    names = ("fill", "default", "hit", "repeat", "lane-a", "lane-b")
    run = {"replies": {name: reply(name if name != "repeat" else "fill") for name in names}, "cache_builds": 1}
    return {1: {"off": {**copy.deepcopy(run), "cache_hits": 0}, "on": {**copy.deepcopy(run), "cache_hits": 3}}}


class VerdictTests(unittest.TestCase):
    def test_equal_pcm_with_hits_passes(self):
        self.assertEqual(check.verdict(matching()), [])

    def test_a_reply_that_changes_with_the_cache_fails(self):
        passes = matching()
        passes[1]["on"]["replies"]["lane-b"]["sha256"] = "other"
        self.assertEqual(check.verdict(passes), ["rows=1 lane-b: PCM differs with the cache on"])

    def test_a_repeat_that_differs_from_its_first_request_fails(self):
        passes = matching()
        passes[1]["off"]["replies"]["repeat"]["sha256"] = "other"
        self.assertIn("rows=1 cache=off: the repeated request's PCM differs from the first", check.verdict(passes))

    def test_no_hit_with_the_cache_on_or_a_hit_with_it_off_fails(self):
        passes = matching()
        passes[1]["on"]["cache_hits"], passes[1]["off"]["cache_hits"] = 0, 1
        self.assertEqual(sorted(check.verdict(passes)), ["rows=1: no reference cache hit with the cache on",
                                                         "rows=1: the cache hit with TF_DRAMA_REFERENCE_CACHE=0"])

    def test_the_plan_sends_voice_default_voice_then_the_repeat_and_two_lanes_together(self):
        steps = check.plan("vesper")
        self.assertEqual([s[0] for s in steps[:4]], ["fill", "default", "hit", "repeat"])
        self.assertEqual(steps[3][2], steps[0][2])
        self.assertEqual([s[1] for s in steps[4]], ["vesper", "vesper"])


if __name__ == "__main__":
    unittest.main()
