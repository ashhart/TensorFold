#!/usr/bin/env python3
"""Unit checks for MTP receipt coverage and primary metrics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_openai_ordered as ordered_openai
import prompt_ordered
import resume_exactness
import run_receipt


class ReceiptTests(unittest.TestCase):
    def test_openai_receipt_requires_all_cells_and_three_valid_hashes(self):
        data = [{"prompt": prompt, "temperature": temp,
                 "evidence": [{"output_sha256": "a" * 64} for _ in range(3)]}
                for prompt in ("fibonacci-raw", "gpu-chat-no-think") for temp in (0.0, 1.0)]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "receipt.json"
            report.write_text(json.dumps(data))
            _, hashes = ordered_openai.read_hashes(report)
            self.assertEqual(set(hashes), ordered_openai.EXPECTED_KEYS)
            self.assertEqual({len(row) for row in hashes.values()}, {3})
            report.write_text(json.dumps(data[:-1]))
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                ordered_openai.read_hashes(report)
            malformed = copy.deepcopy(data)
            malformed[0]["evidence"][0]["output_sha256"] = "bad"
            report.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "malformed SHA-256"):
                ordered_openai.read_hashes(report)

    def test_decode_comparison_rejects_missing_expected_cells(self):
        token = "b" * 64

        def fixture():
            cells = []
            for prompt in run_receipt.PROMPTS:
                for temperature in (0.0, 1.0):
                    for streams in (1, 2, 4):
                        comparisons = [{role: {"complete": True, "token_sha256": token}
                                        for role in ("concurrent", "alone", "serial")}
                                       for _ in range(streams)]
                        cells.append({"prompt": prompt["name"], "temperature": temperature,
                                      "streams": streams,
                                      "reps": [{"comparisons": copy.deepcopy(comparisons)} for _ in range(2)]})
            return {"cells": cells}

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{index}.json" for index in range(3)]
            for path in paths:
                path.write_text(json.dumps(fixture()))
            self.assertTrue(run_receipt.cross_compare(*paths)["all_equal"])
            missing = fixture()
            missing["cells"].pop()
            paths[2].write_text(json.dumps(missing))
            result = run_receipt.cross_compare(*paths)
            self.assertFalse(result["all_equal"])
            self.assertTrue(any(check.get("key") == "coverage" for check in result["checks"]))

    def test_prompt_gate_uses_first_request_not_secondary_rows(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True)]
        result = {}
        for name, seconds in (("main", 10.0), ("candidate-no-drafts", 10.0),
                              ("candidate-mtp-after", 9.0)):
            result[f"rep-00-{name}"] = {
                "qualified": True,
                "first_request_cold": True,
                "primary_first_request": {"ttft_s": seconds, "token_sha256": "c" * 64},
                "secondary_requests": [{"ttft_s": 0.1}],
            }
        case = {"results": result, "supported": True}
        prompt_ordered.summarize_case(case, specs, 1)
        self.assertEqual(case["summary"]["main"]["median_first_ttft_s"], 10.0)
        self.assertTrue(case["comparisons_vs_main"]["candidate-mtp-after"]["not_slower"])
        self.assertTrue(case["qualified"])

    def test_prompt_gate_allows_regression_within_point_one_percent_of_main(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True)]
        results = {}
        for name, seconds in (("main", 1.0), ("candidate-no-drafts", 1.0005),
                              ("candidate-mtp-after", 1.0009)):
            results[f"rep-00-{name}"] = {
                "qualified": True,
                "first_request_cold": True,
                "primary_first_request": {"ttft_s": seconds, "token_sha256": "c" * 64},
            }
        case = {"results": results, "supported": True}
        prompt_ordered.summarize_case(case, specs, 1)
        self.assertTrue(case["comparisons_vs_main"]["candidate-mtp-after"]["within_regression_tolerance"])
        self.assertTrue(case["qualified"])

    def test_dflash2_prompt_arm_is_reported_as_diagnostic_not_speed_gate(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True),
                 ("candidate-dflash2", Path("candidate"), False)]
        results = {}
        for name, seconds in (("main", 1.0), ("candidate-no-drafts", 1.0),
                              ("candidate-mtp-after", 1.0), ("candidate-dflash2", 1.2)):
            results[f"rep-00-{name}"] = {
                "qualified": True,
                "first_request_cold": True,
                "primary_first_request": {"ttft_s": seconds, "token_sha256": "f" * 64},
            }
        case = {"results": results, "supported": True}
        prompt_ordered.summarize_case(case, specs, 1)
        diagnostic = case["comparisons_vs_main"]["candidate-dflash2-diagnostic-vs-main"]
        self.assertAlmostEqual(diagnostic["delta_percent_vs_main"], 20.0)
        self.assertTrue(diagnostic["token_hashes_match_main"])
        self.assertNotIn("not_slower", diagnostic)
        self.assertTrue(case["dflash2_diagnostic_valid"])
        self.assertTrue(case["qualified"])

    def test_prompt_runner_clears_inherited_mtp_overrides(self):
        with mock.patch.dict("os.environ", {"TF_MTP_DRAFTS": "10", "TF_MTP_TREES": "1"}):
            clean = prompt_ordered.clean_env()
            self.assertNotIn("TF_MTP_DRAFTS", clean)
            self.assertNotIn("TF_MTP_TREES", clean)
            self.assertEqual(clean["TF_QWEN_NVFP4_TC"], "1")

    def test_benchmark_env_overrides_and_records_inherited_settings(self):
        for solo_chunk_rows, inherited_chunk_rows in ((4096, 8192), (8192, 4096)):
            inherited = {
                "TF_MTP_BATCH": "0",
                "TF_QWEN35_CHUNK_SOLO": str(inherited_chunk_rows),
                "TF_QWEN_NVFP4_TC": "0",
            }
            with self.subTest(solo_chunk_rows=solo_chunk_rows), mock.patch.dict("os.environ", inherited):
                env, provenance = run_receipt.benchmark_env(solo_chunk_rows)
            self.assertEqual(env["TF_MTP_BATCH"], "1")
            self.assertEqual(env["TF_QWEN35_CHUNK_SOLO"], str(solo_chunk_rows))
            self.assertEqual(env["TF_QWEN_NVFP4_TC"], "1")
            self.assertEqual(provenance["inherited_values_overridden"], inherited)
            self.assertEqual(provenance["effective"], {
                "TF_QWEN_NVFP4_TC": "1",
                "TF_MTP_BATCH": "1",
                "TF_QWEN35_CHUNK_SOLO": str(solo_chunk_rows),
            })

    def test_openai_receipt_requests_one_stream_and_opts_into_trees_only_for_tree_variant(self):
        env = {"TF_MTP_TREES": "1", "KEEP_ME": "yes"}
        with mock.patch.object(ordered_openai.rr, "start_server", return_value="started") as start:
            self.assertEqual(ordered_openai.start_variant("mtp-chain", Path("engine"), True, False, env),
                             "started")
            args, kwargs = start.call_args
            self.assertEqual(args[4], ordered_openai.PARALLEL)
            self.assertEqual(args[5]["KEEP_ME"], "yes")
            self.assertNotIn("TF_MTP_TREES", args[5])
            self.assertEqual(kwargs, {})

            ordered_openai.start_variant("mtp-tree", Path("engine"), True, True, env)
            args, _ = start.call_args
            self.assertEqual(args[4], 1)
            self.assertEqual(args[5]["TF_MTP_TREES"], "1")

            with self.assertRaisesRegex(ValueError, "requires the MTP drafter"):
                ordered_openai.start_variant("main-tree", Path("engine"), False, True, env)
            self.assertEqual(start.call_count, 2)

    def test_prompt_runner_parses_speculative_and_decode_server_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "[tensorfold] done id prompt=2048 tokens=64 finish=length rounds=12 "
                "accepted=24/36 ttft=1.2 decode=0.08 elapsed=1.28 tok/s=300.0\n"
            )
            completion, = prompt_ordered.parse_server_completions(log)
        self.assertEqual(completion["rounds"], 12)
        self.assertEqual(completion["accepted"], 24)
        self.assertEqual(completion["drafted"], 36)
        self.assertAlmostEqual(completion["acceptance_rate"], 2 / 3)
        self.assertEqual(completion["tok_s"], 300.0)

    def test_mtp_overhead_vs_candidate_target_only_is_diagnostic(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True)]
        results = {}
        for name, seconds in (("main", 1.00), ("candidate-no-drafts", 0.90),
                              ("candidate-mtp-after", 0.95)):
            results[f"rep-00-{name}"] = {
                "qualified": True,
                "first_request_cold": True,
                "primary_first_request": {"ttft_s": seconds, "token_sha256": "d" * 64},
            }
        case = {"results": results, "supported": True}
        prompt_ordered.summarize_case(case, specs, 1)
        own = case["comparisons_vs_main"]["candidate-mtp-after-vs-candidate-no-drafts"]
        self.assertFalse(own["not_slower"])
        self.assertFalse(own["within_regression_tolerance"])
        self.assertTrue(case["qualified"])
        self.assertEqual(case["status"], "passed")

    def test_dflash2_capacity_diagnostic_does_not_fail_main_gate(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True),
                 ("candidate-dflash2", Path("candidate"), False)]
        results = {}
        for name in ("main", "candidate-no-drafts", "candidate-mtp-after"):
            results[f"rep-00-{name}"] = {
                "qualified": True,
                "first_request_cold": True,
                "primary_first_request": {"ttft_s": 1.0, "token_sha256": "d" * 64},
            }
        results["rep-00-candidate-dflash2"] = {
            "qualified": False,
            "error": "RuntimeError: server exited during startup with 1",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original_out = prompt_ordered.OUT
            try:
                prompt_ordered.OUT = output
                (output / "prompt-8192-rep-01-candidate-dflash2.server.log").write_text(
                    "the CUDA memory budget fits no stream"
                )
                case = {"results": results, "supported": True, "prompt_tokens_expected": 8192}
                prompt_ordered.summarize_case(case, specs, 1)
            finally:
                prompt_ordered.OUT = original_out
        self.assertTrue(case["qualified"])
        self.assertEqual(case["dflash2_diagnostic_status"], "capacity_limited")

    def test_resume_server_cache_wrapper_accepts_drafter_path(self):
        captured = {}

        def fake_start(label, binary, mtp, context, parallel, env):
            captured["args"] = resume_exactness.rr.server_args(binary, 1, context, parallel, mtp, None)
            return captured["args"]

        with mock.patch.object(resume_exactness.rr, "TARGET", Path("/target")), \
             mock.patch.object(resume_exactness.rr, "start_server", side_effect=fake_start):
            result = resume_exactness.start_server("resume", Path("binary"), False, 2)
        self.assertEqual(result[captured["args"].index("--prompt-cache-gib") + 1], "2")

    def test_unsupported_required_context_is_skipped_not_passed(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True)]
        case = {"results": {}, "supported": False}
        prompt_ordered.summarize_case(case, specs, 1)
        self.assertEqual(case["status"], "skipped_unsupported")
        self.assertIsNone(case["qualified"])
        overall = prompt_ordered.summarize_overall(
            {"2048": {"status": "passed", "qualified": True},
             "8192": case}, [2048, 8192])
        self.assertFalse(overall["qualified"])
        self.assertEqual(overall["skipped_required_contexts"], [8192])

    def test_unsupported_context_with_failed_startup_skips_before_metric_reads(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True)]
        case = {"results": {"rep-00-main": {"qualified": False, "error": "memory refusal"}},
                "supported": False}
        prompt_ordered.summarize_case(case, specs, 1)
        self.assertEqual(case["summary"], {})
        self.assertEqual(case["status"], "skipped_unsupported")
        self.assertIsNone(case["qualified"])

    def test_failed_candidate_sample_is_reported_as_failed_without_crashing(self):
        specs = [("main", Path("main"), False),
                 ("candidate-no-drafts", Path("candidate"), False),
                 ("candidate-mtp-after", Path("candidate"), True)]
        results = {}
        for name, seconds in (("main", 1.0), ("candidate-no-drafts", 1.0)):
            results[f"rep-00-{name}"] = {
                "qualified": True,
                "first_request_cold": True,
                "primary_first_request": {"ttft_s": seconds, "token_sha256": "e" * 64},
            }
        results["rep-00-candidate-mtp-after"] = {
            "qualified": False,
            "error": "RuntimeError: CUDA memory budget refusal",
        }
        case = {"results": results, "supported": True}
        prompt_ordered.summarize_case(case, specs, 1)
        self.assertEqual(case["status"], "failed")
        self.assertFalse(case["qualified"])
        self.assertEqual(case["summary"]["candidate-mtp-after"]["samples"], [None])
        self.assertEqual(case["results"]["rep-00-candidate-mtp-after"]["error"],
                         "RuntimeError: CUDA memory budget refusal")

    def test_only_baseline_capacity_refusal_skips_long_context(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("the CUDA memory budget fits no stream")
            refusal = {"qualified": False, "error": "server exited during startup"}
            self.assertTrue(prompt_ordered.baseline_capacity_refusal("main", refusal, log))
            self.assertFalse(prompt_ordered.baseline_capacity_refusal("candidate-mtp-after", refusal, log))

    def test_prompt_gate_cannot_pass_when_required_contexts_are_omitted(self):
        overall = prompt_ordered.summarize_overall(
            {"2048": {"status": "passed", "qualified": True}}, [2048])
        self.assertFalse(overall["qualified"])
        self.assertEqual(overall["missing_required_contexts"], [8192])

    def test_server_can_attach_dflash2_separately_from_mtp(self):
        target = Path("/target")
        dflash = Path("/dflash2")
        with mock.patch.object(run_receipt, "TARGET", target):
            mtp_args = run_receipt.server_args(Path("tensorfold"), 8080, 4096, 1, True)
            dflash_args = run_receipt.server_args(Path("tensorfold"), 8080, 4096, 1, False, dflash)
            plain_args = run_receipt.server_args(Path("tensorfold"), 8080, 4096, 1, False)
        self.assertIn(str(target), mtp_args)
        self.assertIn("--mtp-drafts", mtp_args)
        self.assertIn(str(dflash), dflash_args)
        self.assertIn("--drafter", dflash_args)
        self.assertNotIn("--no-drafts", dflash_args)
        self.assertIn("--no-drafts", plain_args)

    def test_main_receipt_marks_pr_gate_unassessed_until_prompt_speed_is_supplied(self):
        fields = run_receipt.pr_qualification_fields()
        self.assertEqual(fields["pr_qualification_status"], "not_assessed")
        self.assertIn("cold 2048/8192 prompt-speed A/B", fields["pr_qualification_missing_gates"])
        self.assertNotIn("qualified", fields)
        self.assertNotIn("pr_qualified", fields)

    def test_stream_capacity_and_memory_refusal_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("CUDA sm_120 device 0: 4 streams at once, 0.4 GiB each")
            self.assertEqual(run_receipt.stream_capacity(log), 4)
            self.assertFalse(run_receipt.memory_capacity_refusal(log))
            log.write_text("--parallel 4 needs 2.0 GiB; lower --parallel or --context")
            self.assertTrue(run_receipt.memory_capacity_refusal(log))
            self.assertIsNone(run_receipt.suggested_parallel_capacity(log))
            log.write_text(
                "tensorfold: --parallel 4 needs 2.68 GiB for its streams at a 1024-token window, "
                "and the CUDA memory budget leaves 2.31 GiB: serve --parallel 3, or lower --context")
            self.assertTrue(run_receipt.memory_capacity_refusal(log))
            self.assertEqual(run_receipt.suggested_parallel_capacity(log), 3)
            log.write_text("tensorfold: CUDA model load failed: invalid checkpoint file")
            self.assertFalse(run_receipt.memory_capacity_refusal(log))

    def test_server_capacity_advice_is_retried_and_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tried = []

            def fake_start(label, _binary, _mtp, _context, parallel, _env):
                tried.append(parallel)
                log = output / f"{label}.server.log"
                if parallel == 4:
                    log.write_text(
                        "tensorfold: --parallel 4 needs 2.68 GiB for its streams at a 1024-token window, "
                        "and the CUDA memory budget leaves 2.31 GiB: serve --parallel 3, or lower --context")
                    raise RuntimeError("capacity refusal")
                log.write_text("CUDA sm_120 device 0: 3 streams at once, 0.4 GiB each")
                return object(), None, None, "http://127.0.0.1:1", 0.01, log

            with (mock.patch.object(run_receipt, "OUT", output),
                  mock.patch.object(run_receipt, "start_server", side_effect=fake_start),
                  mock.patch.object(run_receipt, "telemetry", return_value={"reported_mtp_enabled": False}),
                  mock.patch.object(run_receipt, "bench"),
                  mock.patch.object(run_receipt, "stop_server", return_value={"gpu_peak_used_mib": 1})):
                result = run_receipt.run_variant(
                    "capacity-test", Path("unused-binary"), False, "decode", 1024, 4, {})

        self.assertEqual(tried, [4, 3])
        self.assertEqual(result["parallel_admitted"], 3)
        self.assertEqual(result["supported_stream_levels"], [1, 2, 3])

    def test_readiness_timeout_uses_monotonic_clock(self):
        with self.assertRaises(TimeoutError):
            run_receipt.readiness('http://127.0.0.1', None, timeout=0)

    def test_only_base_context_capacity_errors_mark_long_context_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("the CUDA memory budget fits no stream")
            self.assertTrue(prompt_ordered.context_is_unsupported(log))
            log.write_text("--context 65536 exceeds this model's 32768-token window")
            self.assertTrue(prompt_ordered.context_is_unsupported(log))
            log.write_text("failed to load model: corrupt tensor")
            self.assertFalse(prompt_ordered.context_is_unsupported(log))

    def test_resume_requires_at_least_4096_actual_cached_tokens(self):
        fresh = {"prompt_tokens": 4345, "cached_tokens": 0}
        self.assertFalse(resume_exactness.cache_reuse_gate(
            {"prompt_tokens": 4345, "cached_tokens": 4095}, fresh))
        self.assertTrue(resume_exactness.cache_reuse_gate(
            {"prompt_tokens": 4345, "cached_tokens": 4096}, fresh))
        self.assertFalse(resume_exactness.cache_reuse_gate(
            {"prompt_tokens": 4344, "cached_tokens": 4096}, fresh))


if __name__ == "__main__":
    unittest.main()
