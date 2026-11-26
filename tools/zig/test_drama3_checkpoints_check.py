"""The checkpoints check's verdict, peak-memory source and reply timing, without a model or GPU."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
import unittest.mock

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


class SlowStream(BaseHTTPRequestHandler):
    """A streamed reply whose first 4 bytes come at once and the rest a second later."""
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for delay, data in ((0, b"\0" * 4), (1.0, b"\1" * 8)):
            time.sleep(delay)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *_):
        pass


class SpeakTests(unittest.TestCase):
    def test_first_audio_is_the_first_bytes_not_a_full_read(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), SlowStream) as server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            reply = check.speak(server.server_address[1], "text", True, 10)
            server.shutdown()
        self.assertEqual(reply["bytes"], 12)
        self.assertLess(reply["first_audio_s"], 0.5)
        self.assertGreaterEqual(reply["total_s"], 1.0)


class RunTests(unittest.TestCase):
    """run_checkpoint rebinds and retries once when the port it probed is taken before the server binds."""

    class FakeServer:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"process": {"lifetime_peak_physical_footprint_bytes": 7}}'

    def run_with(self, ready_side_effects, log, writes=((), ())):
        args = unittest.mock.Mock(bin="bin", voices="voices", voice="v", ready_timeout=1, timeout=1)
        reply = {"sha256": "x", "bytes": 4, "first_audio_s": 0.1, "total_s": 0.2}

        def popen(*popen_args, **kwargs):
            if writes:
                for line in writes[min(popen.call_count, len(writes) - 1)]:
                    kwargs["stdout"].write(line + "\n")
                    kwargs["stdout"].flush()
                popen.call_count += 1
            return self.FakeServer()
        popen.call_count = 0
        with unittest.mock.patch.object(check.subprocess, "Popen", side_effect=popen), \
                unittest.mock.patch.object(check, "ready", side_effect=ready_side_effects), \
                unittest.mock.patch.object(check, "speak", return_value=reply), \
                unittest.mock.patch.object(check.urllib.request, "urlopen", return_value=self.FakeResponse()):
            return check.run_checkpoint(args, "ckpt", log)

    def test_a_taken_port_is_retried_once(self):
        log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
        log.close()
        run = self.run_with([RuntimeError("port in use"), None], Path(log.name))
        self.assertEqual(run["held"], None)

    def test_a_failed_attempts_held_count_does_not_pass_for_the_successful_one(self):
        log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
        log.close()
        run = self.run_with([RuntimeError("port in use"), None], Path(log.name),
                            writes=[("routed experts held 7369 of 10240",), ()])
        problems = check.verdict({"pruned": run})
        self.assertIn("pruned: routed experts held None, expected 7369", problems)

    def test_a_server_that_keeps_dying_still_fails(self):
        log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
        log.close()
        with self.assertRaises(RuntimeError):
            self.run_with([RuntimeError("port in use"), RuntimeError("port in use")], Path(log.name))


class TableTests(unittest.TestCase):
    def test_an_empty_reply_prints_its_missing_values_not_a_crash(self):
        runs = matching()
        for reply in runs["pruned"]["replies"].values():
            reply.update(bytes=0, first_audio_s=None)
        for other in runs.values():
            other.update(load_s=1, peak_bytes=1, total_s=1, first_audio_s=1, audio_s=1, rtf=1)
        runs["pruned"].update(load_s=0, peak_bytes=0, total_s=0, first_audio_s=None, audio_s=0, rtf=None, held=None)
        line = [l for l in check.table(runs).splitlines() if l.startswith("| pruned |")][0]
        self.assertIn("no reply", line)
        self.assertNotIn("None", line)


if __name__ == "__main__":
    unittest.main()
