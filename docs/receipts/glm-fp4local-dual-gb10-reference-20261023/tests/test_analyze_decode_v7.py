"""Synthetic receipts: no GPU, service, model or external dependencies."""
import io
import json
import unittest

from analyze_decode_v7 import Analyzer, compact, context_bucket, validate_route


def route(call=1, rank=0, layer=3, rows=2, lo=8192, hi=None, hist=None):
    hist = hist if hist is not None else [0, 8, 4]
    return {"call": call, "label": f"verify {rows} rows", "rank": rank, "layer": layer,
            "rows": rows, "streams": 1, "context_min": lo, "context_max": lo if hi is None else hi,
            "unique": sum(hist), "pairs": sum(i * n for i, n in enumerate(hist)), "pair_hist": hist}


def coarse(call=1, rank=0, rows=2, lo=8192, hi=None, gpu=40, calls=1, label=None):
    label = label or f"verify {rows} rows"
    hi = lo if hi is None else hi
    return (f"info: glm prof {label} ({calls} calls): rank {rank}, seq {call}, rows {rows}, "
            f"context {lo}..{hi}; wall {gpu+1} ms, gpu {gpu} ms a call; kda 12 routed 18 shared 3\n")


def rline(**kwargs):
    return "info: glm route " + json.dumps(route(**kwargs)) + "\n"


def round_line(lo=8192, hi=8441, initial=False, calls=10, kept=20):
    return (f"info: glm round host: streams=1 rows=3 context={lo}..{hi} "
            f"initial={str(initial).lower()} calls={calls} kept={kept} next_drafts=70 "
            "wall_ms=60 min_ms=58 max_ms=65 verify_call_ms=52 draw_nested_ms=46 "
            "draft_call_ms=5 keep_call_ms=1 commit_nested_ms=0.5 other_ms=2\n")


class AnalyzerTests(unittest.TestCase):
    def analyze(self, text):
        a = Analyzer()
        a.read(io.StringIO(text), "test-rank0.log")
        result = a.result()
        json.dumps(result, allow_nan=False)
        return result

    def test_context_boundaries_and_mixed_exclusion(self):
        for n, expected in ((0, None), (8191, None), (8192, "8K"), (32767, "8K"),
                            (32768, "32K"), (131071, "32K"), (131072, "128K")):
            self.assertEqual(context_bucket(n, n)[0], expected)
        self.assertEqual(context_bucket(8191, 8192), (None, "mixed_context"))
        self.assertEqual(context_bucket(32767, 32768), (None, "mixed_context"))
        self.assertEqual(context_bucket(131071, 131072), (None, "mixed_context"))
        with self.assertRaises(ValueError):
            context_bucket(10, 9)

    def test_weighted_route_reuse_and_detail_association(self):
        text = rline() + rline(layer=4, hist=[0, 16, 0])
        text += "info: glm detail verify 2 rows (1 calls, nested ms/call): kda_project 3.5 kda_copy 0.5\n"
        text += coarse()
        result = self.analyze(text)
        g = result["groups"][0]
        self.assertEqual(g["routes"]["unique_experts_mean"], 14)
        self.assertEqual(g["routes"]["pairs_mean"], 16)
        self.assertAlmostEqual(g["routes"]["pairs_per_unique"], 32 / 28)
        self.assertEqual(g["routes"]["pair_hist"], {"1": 24, "2": 4})
        self.assertEqual(g["detail_nested"]["ms"]["kda_copy"]["mean"], .5)
        self.assertEqual(g["coarse"]["ms"]["mla"]["mean"], 0)
        self.assertEqual(result["counts"]["route_records_matched_coarse"], 2)
        self.assertEqual(result["route_call_layer_count_hist"], {2: 1})
        self.assertEqual([r["layer"] for r in g["route_layers"]], [3, 4])
        self.assertIn("pairs/unique=1.143", compact(result))

    def test_call_weighting_and_rank_width_separation(self):
        text = coarse(gpu=30, calls=2) + coarse(call=3, gpu=60)
        text += coarse(rank=1, gpu=40) + coarse(rows=1, gpu=20) + coarse(lo=32768, gpu=50)
        result = self.analyze(text)
        self.assertEqual(len(result["groups"]), 4)
        row = next(g for g in result["groups"] if (g["bucket"], g["rank"], g["rows"]) == ("8K", 0, 2))
        self.assertEqual(row["coarse"]["calls"], 3)
        self.assertEqual(row["coarse"]["ms"]["gpu"]["mean"], 40)
        self.assertEqual(row["coarse"]["ms"]["gpu"]["min_report_mean"], 30)
        self.assertEqual(result["counts"]["coarse_aggregated_reports"], 1)

    def test_startup_prefill_and_cross_bucket_ignored(self):
        result = self.analyze(coarse(lo=100) + coarse(lo=32760, hi=32770) +
                              coarse(label="prompt chunk 64", lo=8192))
        self.assertFalse(result["groups"])
        self.assertEqual(result["excluded"]["coarse_startup"], 1)
        self.assertEqual(result["excluded"]["coarse_mixed_context"], 1)
        self.assertEqual(result["excluded"]["coarse_non_decode"], 1)

    def test_malformed_histogram_is_rejected(self):
        r = route()
        r["pairs"] += 1
        with self.assertRaises(ValueError):
            validate_route(r)
        result = self.analyze("glm route " + json.dumps(r) + "\n")
        self.assertEqual(result["counts"]["malformed"], 1)
        self.assertFalse(result["groups"])
        r = route(hist=[0, 0, 0, 2])
        with self.assertRaises(ValueError):
            validate_route(r)

    def test_duplicate_route_layer_not_counted_twice(self):
        result = self.analyze(rline() + rline() + coarse())
        self.assertEqual(result["counts"]["malformed"], 1)
        self.assertEqual(result["groups"][0]["routes"]["layer_records"], 1)

    def test_detail_mismatch_orphan_and_legacy_reports(self):
        result = self.analyze("glm detail verify 2 rows (2 calls, nested ms/call): kda_project 1\n" +
                              coarse() + "glm detail verify 1 row (1 calls, nested ms/call): kda_project 2\n" +
                              "glm prof verify 2 rows (64 calls): wall 45 ms, gpu 44 ms a call; routed 18\n")
        self.assertEqual(result["excluded"]["detail_call_mismatch"], 1)
        self.assertEqual(result["excluded"]["detail_orphan"], 1)
        self.assertEqual(result["excluded"]["coarse_legacy_or_unparsed"], 1)

    def test_host_round_nested_not_added_and_initial_separate(self):
        result = self.analyze(round_line() + round_line(initial=True) +
                              round_line(lo=32760, hi=32780) + round_line(lo=100, hi=110) +
                              "glm round host end: buckets=4 dropped=1 aborted=2; host calls include async waits\n")
        self.assertEqual(len(result["host_rounds"]), 2)
        row = result["host_rounds"][0]
        self.assertFalse(row["initial"])
        self.assertEqual(row["ms"]["wall_ms"]["mean"], 60)
        self.assertEqual(row["kept_per_round"], 2)
        self.assertAlmostEqual(row["kept_per_host_second"], 1000 / 30)
        self.assertEqual(row["verified_draft_rows_per_round"], 2)
        self.assertEqual(result["excluded"]["round_mixed_context"], 1)
        self.assertEqual(result["excluded"]["round_startup"], 1)
        self.assertEqual(result["counts"]["round_dropped"], 1)
        self.assertEqual(result["counts"]["round_aborted"], 2)


if __name__ == "__main__":
    unittest.main()
