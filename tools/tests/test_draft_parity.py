"""Host fixtures for exact device comparison, setup refusals and owned native-process orchestration."""
import contextlib
import argparse
import copy
import io
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import draft_parity as parity
import draft_parity_native as native


def response(body, drafted):
    return {"model": body["model"], "choices": [{"text": "same text", "finish_reason": "length"}],
            "tensorfold": {"drafts": drafted, "sampling": "exact" if body["temperature"] else "greedy",
                           "token_ids": list(range(10, 10 + body["max_tokens"]))},
            "usage": {"completion_tokens": body["max_tokens"], "prompt_tokens": 23},
            "speculative": {"rounds": 4, "drafted": 7 if drafted else 0, "accepted": 3 if drafted else 0}}


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "native"
        self.binary.write_text("fixture binary, never executed")
        self.binary.chmod(0o755)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"nemotron_h"}')
        self.prompts = self.root / "prompts.json"
        self.prompts.write_text(json.dumps({"text": "one two three", "ids": [1, 2, 3]}))
        self.calls, self.servers = [], []
        self.active = 0
        self.transform = lambda body, drafted, reply: reply
        self.labels = {True: "MTP + copies", False: "off"}
        self.backends = {True: "fixture", False: "fixture"}

    def options(self, extra=()):
        return parity.parse_args([str(self.model), "--binary", str(self.binary), "--output", str(self.root / "out"),
                                  "--prompts", str(self.prompts), "--tokens", "8", *extra])

    def factory(self, command, name, port, directory, environ, startup_timeout, request_timeout):
        test = self
        drafted = "--no-drafts" not in command

        class Fake:
            health = {"model": name, "backend": test.backends[drafted], "version": "fixture", "device": "fixture"}
            returncode = None
            killed = False

            def __enter__(self):
                test.assertEqual(test.active, 0)
                test.active += 1
                test.servers.append(command)
                return self

            def request(self, body):
                test.calls.append((drafted, copy.deepcopy(body)))
                return test.transform(body, drafted, response(body, drafted))

            def __exit__(self, kind, value, traceback):
                test.active -= 1
                self.returncode = -15

            def draft_label(self):
                return test.labels[drafted]
        return Fake()

    def run_check(self, extra=()):
        with patch.object(parity, "unused_port", return_value=12345):
            return parity.execute(self.options(extra), self.factory)

    def test_exact_ids_and_all_sampling_cells_use_same_requests_and_binary(self):
        report = self.run_check()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 8)
        self.assertEqual(len(self.servers), 2)
        self.assertEqual(self.servers[0], self.servers[1][:-1])
        self.assertEqual(self.servers[1][-1], "--no-drafts")
        self.assertNotIn("--no-drafts", self.servers[0])
        self.assertEqual(self.servers[0][0], str(self.binary.resolve()))
        self.assertIn("--no-update-check", self.servers[0])
        self.assertEqual(self.active, 0)
        first, second = self.calls[:8], self.calls[8:]
        self.assertTrue(all(mode for mode, _ in first))
        self.assertTrue(all(not mode for mode, _ in second))
        self.assertEqual([body for _, body in first], [body for _, body in second])
        cells = {(b["temperature"], b["top_p"], b["seed"]) for _, b in first}
        self.assertEqual(cells, {(0, 1, 0), (1, .95, 0), (1, .95, 1), (1, .95, 1234)})
        self.assertEqual(sum("messages" in body for _, body in first), 4)
        self.assertEqual(sum("prompt" in body for _, body in first), 4)
        for _, body in self.calls:
            self.assertNotIn("draft", body)
            self.assertIs(body["return_token_ids"], True)
            self.assertIs(body["ignore_eos"], True)
            self.assertEqual((body["top_k"], body["min_p"], body["n"]), (0, 0, 1))
        for index in range(8):
            a = self.root / "out/default" / f"{index:03d}.request.json"
            b = self.root / "out/plain" / f"{index:03d}.request.json"
            self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(report["binary"]["sha256"], native.sha256(self.binary))
        self.assertEqual(native.read_json((self.root / "out/report.json").read_text()), report)

    def test_sampled_id_mismatch_fails_even_when_text_is_identical(self):
        def change(body, drafted, reply):
            if drafted and body["temperature"] == 1 and body["seed"] == 1:
                reply["tensorfold"]["token_ids"][3] = 999
            return reply
        self.transform = change
        report = self.run_check()
        self.assertEqual(report["status"], "divergent")
        self.assertFalse(report["passed"])
        bad = [case for case in report["cases"] if case["status"] == "divergent"]
        self.assertEqual(bad[0]["first_difference"], {"index": 3, "plain_id": 13, "drafted_id": 999})

    def test_greedy_id_mismatch_also_fails(self):
        def change(body, drafted, reply):
            if drafted and body["temperature"] == 0:
                reply["tensorfold"]["token_ids"][0] = 998
            return reply
        self.transform = change
        self.assertEqual(self.run_check()["status"], "divergent")

    def test_proposed_but_no_accepted_drafts_is_setup_failure(self):
        def change(body, drafted, reply):
            if drafted:
                reply["speculative"]["accepted"] = 0
            return reply
        self.transform = change
        report = self.run_check()
        self.assertEqual(report["status"], "setup_error")
        self.assertTrue(all(case["default"]["drafted"] == 7 for case in report["cases"]))
        self.assertTrue(all("no accepted drafts" in case["setup_error"] for case in report["cases"]))

    def test_one_inactive_case_cannot_hide_behind_other_accepted_drafts(self):
        def change(body, drafted, reply):
            if drafted and body["seed"] == 1234:
                reply["speculative"]["accepted"] = 0
            return reply
        self.transform = change
        report = self.run_check()
        self.assertEqual(report["status"], "setup_error")
        self.assertTrue(any(case["status"] == "pass" for case in report["cases"]))
        self.assertFalse(report["passed"])

    def test_none_source_never_passes_even_with_inconsistent_positive_counts(self):
        self.labels[True] = "none"
        report = self.run_check()
        self.assertFalse(report["passed"])
        self.assertIn("drafts: none", report["errors"][0])

    def test_require_mtp_detects_checkpoint_with_copies_only(self):
        self.labels[True] = "copies only"
        self.assertFalse(self.run_check(["--require-mtp"])["passed"])

    def test_require_mtp_accepts_the_actual_mtp_label(self):
        self.assertTrue(self.run_check(["--require-mtp"])["passed"])

    def test_missing_startup_label_cannot_pass(self):
        self.labels[True] = None
        self.assertFalse(self.run_check()["passed"])

    def test_backend_identity_change_cannot_pass(self):
        self.backends[False] = "other"
        report = self.run_check()
        self.assertFalse(report["passed"])
        self.assertIn("native health backend changed between runs", report["errors"])

    def test_plain_label_must_be_off(self):
        self.labels[False] = "MTP + copies"
        self.assertFalse(self.run_check()["passed"])

    def test_request_error_does_not_skip_cleanup_or_become_a_pass(self):
        def fail(body, drafted, reply):
            raise OSError("fixture request refused")
        self.transform = fail
        report = self.run_check()
        self.assertFalse(report["passed"])
        self.assertEqual(self.active, 0)
        self.assertEqual(len(self.servers), 2)
        self.assertEqual(report["cases"][0]["default"]["error"], "fixture request refused")

    def test_keyboard_interrupt_closes_owned_process(self):
        def stop(body, drafted, reply):
            raise KeyboardInterrupt
        self.transform = stop
        report = self.run_check()
        self.assertEqual(report["status"], "interrupted")
        self.assertFalse(report["passed"])
        self.assertEqual(self.active, 0)

    def test_binary_change_cannot_be_compared(self):
        def change(body, drafted, reply):
            if drafted:
                self.binary.write_text("changed binary")
            return reply
        self.transform = change
        report = self.run_check()
        self.assertFalse(report["passed"])
        self.assertIn("binary changed", report["errors"][0])

    def test_checkpoint_change_cannot_be_compared(self):
        def change(body, drafted, reply):
            if drafted:
                (self.model / "config.json").write_text("changed config")
            return reply
        self.transform = change
        self.assertIn("checkpoint changed", self.run_check()["errors"][0])

    def test_existing_output_is_refused_without_overwriting(self):
        options = self.options()
        options.output.mkdir()
        marker = options.output / "report.json"
        marker.write_text("old evidence")
        with self.assertRaises(FileExistsError):
            parity.execute(options, self.factory)
        self.assertEqual(marker.read_text(), "old evidence")
        self.assertEqual(self.servers, [])

    def test_cli_success_and_divergence_exit_codes(self):
        for expected in (0, 1):
            def change(body, drafted, reply):
                if expected and drafted:
                    reply["tensorfold"]["token_ids"][0] = 999
                return reply
            self.transform = change
            target = self.root / f"cli-{expected}"
            args = [str(self.model), "--binary", str(self.binary), "--output", str(target), "--tokens", "8"]
            with patch.object(parity, "unused_port", return_value=12345):
                with patch.object(parity, "NativeServer", self.factory), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(parity.main(args), expected)
            report = native.read_json((target / "report.json").read_text())
            self.assertEqual(report["status"], "pass" if expected == 0 else "divergent")

    def test_cli_exit_codes_and_zero_count_output(self):
        def change(body, drafted, reply):
            if drafted:
                reply["speculative"]["accepted"] = 0
            return reply
        self.transform = change
        output = io.StringIO()
        args = [str(self.model), "--binary", str(self.binary), "--output", str(self.root / "cli"), "--tokens", "8"]
        with patch.object(parity, "unused_port", return_value=12345), patch.object(parity, "NativeServer", self.factory):
            with contextlib.redirect_stdout(output):
                code = parity.main(args)
        self.assertEqual(code, 2)
        self.assertIn('"accepted": 0', output.getvalue())
        self.assertIn("no accepted drafts", output.getvalue())
        self.assertIn('"status": "setup_error"', output.getvalue())
        self.assertIn('"startup_drafts":', output.getvalue())

    def test_terminal_hides_prompt_names_and_drafter_paths(self):
        prompt_secret = "/private/box/prompt-secret"
        drafter_secret = "/private/box/drafter-secret"
        self.prompts.write_text(json.dumps({prompt_secret: "one two three"}))
        self.labels[True] = f"local ({drafter_secret})"
        output = io.StringIO()
        args = [str(self.model), "--binary", str(self.binary), "--output", str(self.root / "safe"),
                "--prompts", str(self.prompts), "--tokens", "8"]
        with patch.object(parity, "unused_port", return_value=12345), patch.object(parity, "NativeServer", self.factory):
            with contextlib.redirect_stdout(output):
                self.assertEqual(parity.main(args), 0)
        text = output.getvalue()
        self.assertNotIn(prompt_secret, text)
        self.assertNotIn(drafter_secret, text)
        self.assertIn('"index": 0', text)
        self.assertIn('"startup_drafts": "local"', text)

    def test_setup_errors_print_only_exception_classes(self):
        missing = self.root / "private-missing-native"
        output, errors = io.StringIO(), io.StringIO()
        args = [str(self.model), "--binary", str(missing), "--output", str(self.root / "early"), "--tokens", "8"]
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(parity.main(args), 2)
        text = output.getvalue() + errors.getvalue()
        self.assertIn("FileNotFoundError", text)
        self.assertNotIn(str(missing), text)
        existing = self.root / "existing"
        existing.mkdir()
        args[args.index(str(self.root / "early"))] = str(existing)
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(parity.main(args), 2)
        text = output.getvalue() + errors.getvalue()
        self.assertIn("FileExistsError", text)
        self.assertNotIn(str(existing), text)

    def test_startup_timeout_fails_cleans_owned_process_and_uses_safe_terminal_code(self):
        process = Mock()
        process.poll.side_effect = [None, -15]
        process.wait.return_value = -15
        output = io.StringIO()
        args = [str(self.model), "--binary", str(self.binary), "--output", str(self.root / "startup"),
                "--tokens", "8", "--startup-timeout", "1"]
        with patch.object(parity, "unused_port", return_value=12345), \
                patch.object(native.subprocess, "Popen", return_value=process), \
                patch.object(native.time, "monotonic", side_effect=[0.0, 2.0]), contextlib.redirect_stdout(output):
            self.assertEqual(parity.main(args), 2)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=20)
        report = native.read_json((self.root / "startup/report.json").read_text())
        self.assertEqual((report["status"], report["passed"]), ("setup_error", False))
        self.assertEqual(report["errors"], ["native server startup timed out"])
        self.assertIn("TimeoutError", output.getvalue())
        self.assertNotIn("native server startup timed out", output.getvalue())

    def test_request_timeout_accounts_for_every_case_cleans_up_and_uses_safe_terminal_code(self):
        secret = "/private/box/request-secret"
        def stop(body, drafted, reply):
            raise TimeoutError(secret)
        self.transform = stop
        output = io.StringIO()
        args = [str(self.model), "--binary", str(self.binary), "--output", str(self.root / "requests"), "--tokens", "8"]
        with patch.object(parity, "unused_port", return_value=12345), patch.object(parity, "NativeServer", self.factory):
            with contextlib.redirect_stdout(output):
                self.assertEqual(parity.main(args), 2)
        report = native.read_json((self.root / "requests/report.json").read_text())
        self.assertEqual((report["status"], report["passed"]), ("setup_error", False))
        self.assertEqual((len(report["cases"]), len(self.calls), len(self.servers), self.active), (8, 16, 2, 0))
        self.assertEqual([len(run["observations"]) for run in report["modes"].values()], [8, 8])
        for case in report["cases"]:
            self.assertEqual(case["default"]["error_code"], "TimeoutError")
            self.assertEqual(case["plain"]["error_code"], "TimeoutError")
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("TimeoutError", output.getvalue())


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.body = parity.cases_for({"p": [1, 2]}, [0, 1, 2], 8, "owned")[1]["request"]

    def test_accepted_counter_may_include_the_final_budget_clipped_rows(self):
        reply = response(self.body, True)
        reply["speculative"].update(drafted=12, accepted=10)
        observed = parity.observe(reply, self.body, True)
        self.assertEqual((len(observed["tokens"]), observed["drafted"], observed["accepted"]), (8, 12, 10))

    def test_rejects_missing_or_invalid_evidence(self):
        changes = [
            lambda r: r.pop("speculative"),
            lambda r: r["speculative"].pop("accepted"),
            lambda r: r["speculative"].update(accepted=-1),
            lambda r: r["speculative"].update(accepted=True),
            lambda r: r["speculative"].update(accepted=8),
            lambda r: r["speculative"].update(rounds=0),
            lambda r: r["tensorfold"].pop("token_ids"),
            lambda r: r["tensorfold"]["token_ids"].__setitem__(0, True),
            lambda r: r["tensorfold"]["token_ids"].__setitem__(0, -1),
            lambda r: r["tensorfold"].update(token_ids=[1]),
            lambda r: r["tensorfold"].update(drafts=False),
            lambda r: r["tensorfold"].update(sampling="greedy"),
            lambda r: r["tensorfold"].update(seed=999),
            lambda r: r["choices"][0].update(finish_reason="stop"),
            lambda r: r["usage"].update(completion_tokens=7),
            lambda r: r["usage"].update(prompt_tokens=0),
            lambda r: r.update(model="foreign"),
            lambda r: r.update(error={"message": "refused"}),
        ]
        for change in changes:
            reply = response(self.body, True)
            change(reply)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parity.observe(reply, self.body, True)

    def test_plain_cannot_report_proposals_or_accepted_drafts(self):
        reply = response(self.body, False)
        reply["speculative"]["drafted"] = 1
        with self.assertRaisesRegex(ValueError, "performed draft work"):
            parity.observe(reply, self.body, False)

    def test_lengths_and_prompt_tokenization_cannot_pass(self):
        plain = parity.observe(response(self.body, False), self.body, False)
        drafted = parity.observe(response(self.body, True), self.body, True)
        case = {"name": "p", "request": self.body}
        drafted["tokens"].append(99)
        result = parity.compare_case(case, drafted, plain)
        self.assertEqual(result["status"], "divergent")
        self.assertEqual(result["first_difference"], {"index": 8, "plain_id": None, "drafted_id": 99})
        drafted["tokens"].pop()
        drafted["prompt_tokens"] += 1
        self.assertEqual(parity.compare_case(case, drafted, plain)["status"], "setup_error")

    def test_text_is_not_used_as_the_equality_oracle(self):
        left, right = response(self.body, True), response(self.body, False)
        left["choices"][0]["text"] = "different text"
        self.assertEqual(parity.observe(left, self.body, True)["tokens"],
                         parity.observe(right, self.body, False)["tokens"])

    def test_prompt_and_seed_validation(self):
        for prompts in ({}, {"p": []}, {"p": [True]}, {"p": [2**32]}, {"p": " "}):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                parity.prompts_from(prompts)
        for seeds in ("0,1", "0,1,1", "-1,2,3", "x,1,2", "0,1,9223372036854775808"):
            with self.subTest(seeds=seeds), self.assertRaises(argparse.ArgumentTypeError):
                parity.seed_list(seeds)

    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        for text in ('{"accepted":0,"accepted":2}', '{"n":NaN}', '{"n":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                native.read_json(text)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def server(self):
        return native.NativeServer(["fixture"], "owned", 12345, self.root, {}, 1, 1)

    def test_foreign_health_is_refused_and_only_owned_process_is_terminated(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = -15
        server = self.server()
        with patch.object(native.subprocess, "Popen", return_value=process) as spawn, patch.object(native, "fetch") as fetch:
            fetch.return_value = {"model": "foreign", "status": "ok", "warming": False}
            with self.assertRaisesRegex(ValueError, "newly started server"):
                server.__enter__()
        self.assertEqual(spawn.call_args.kwargs["cwd"], Path.cwd())
        process.terminate.assert_called_once()
        self.assertTrue(server.log.closed)

    def test_start_failure_closes_the_log(self):
        server = self.server()
        with patch.object(native.subprocess, "Popen", side_effect=OSError("fixture start failure")):
            with self.assertRaises(OSError):
                server.__enter__()
        self.assertTrue(server.log.closed)

    def test_request_checks_process_before_and_after_response(self):
        server = self.server()
        server.process = Mock()
        server.process.poll.side_effect = [None, 9]
        with patch.object(native, "fetch", return_value={"fixture": True}):
            with self.assertRaisesRegex(RuntimeError, "exited 9"):
                server.request({})

    def test_text_uses_chat_template_and_raw_ids_use_completions(self):
        server = self.server()
        server.process = Mock()
        server.process.poll.return_value = None
        with patch.object(native, "fetch", return_value={}) as fetch:
            server.request({"messages": [{"role": "user", "content": "text"}]})
            self.assertEqual(fetch.call_args.args[1], "http://127.0.0.1:12345/v1/chat/completions")
            server.request({"prompt": [1, 2]})
            self.assertEqual(fetch.call_args.args[1], "http://127.0.0.1:12345/v1/completions")


    def test_request_passes_its_bounded_timeout_to_fetch(self):
        server = self.server()
        server.process = Mock()
        server.process.poll.return_value = None
        with patch.object(native, "fetch", side_effect=TimeoutError("request timed out")) as fetch:
            with self.assertRaises(TimeoutError):
                server.request({"prompt": [1, 2]})
        self.assertEqual(fetch.call_args.args[3], 1)
        server.close()
        server.process.terminate.assert_called_once()

    def test_cleanup_escalates_only_its_own_stuck_process(self):
        server = self.server()
        process = Mock()
        process.poll.side_effect = [None, -9]
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 20), -9]
        server.process = process
        server.close()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertTrue(server.killed)
        self.assertEqual(server.returncode, -9)

    def test_truncated_http_response_is_a_reportable_setup_error(self):
        reply = Mock()
        reply.read.side_effect = http.client.IncompleteRead(b"partial", 9)
        opener = Mock()
        opener.open.return_value = contextlib.nullcontext(reply)
        with self.assertRaisesRegex(OSError, "incomplete or invalid"):
            native.fetch(opener, "http://127.0.0.1:12345/health", None, 1)

    def test_redirect_is_refused(self):
        with self.assertRaisesRegex(ValueError, "redirect refused"):
            native.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "http://foreign")

    def test_startup_label_uses_the_serving_line_and_stops_at_context(self):
        log = self.root / "server.log"
        log.write_text("serving m at http://127.0.0.1:12345/v1 (sampling: greedy; drafts: MTP + copies; context: 8192)\n"
                       "some request says drafts: none\n")
        self.assertEqual(native.startup_drafts(log), "MTP + copies")


if __name__ == "__main__":
    unittest.main()
