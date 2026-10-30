"""Model-free route/auth integration: pass fake_serve and its fixture directory as the two arguments."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request


def spawn_server(argv, **kwargs):
    # Keep unread lines in the pipe so selectors sees the PORT line after a warning.
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, **kwargs)


def listener_port(proc, timeout=10):
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if not selector.select(max(0, deadline - time.monotonic())):
                break
            line = proc.stdout.readline()
            if line.startswith(b"PORT "):
                return int(line.split()[1])
            if not line:
                break
    raise RuntimeError("fake server did not announce its listener")


class ListenerStartup(unittest.TestCase):
    def child(self, output, wait=True):
        code = f"import os,sys; os.write(1, {output!r}); "
        if wait:
            code += "sys.stdin.buffer.read()"
        proc = spawn_server([sys.executable, "-c", code], stdin=subprocess.PIPE)
        self.addCleanup(self.stop_child, proc)
        return proc

    @staticmethod
    def stop_child(proc):
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)

    def test_warning_and_port_in_one_write(self):
        proc = self.child(b"warning: fixture\nPORT 12345\n")
        self.assertEqual(listener_port(proc, timeout=1), 12345)

    def test_eof_without_port_is_refused(self):
        proc = self.child(b"warning: fixture\n", wait=False)
        with self.assertRaisesRegex(RuntimeError, "did not announce"):
            listener_port(proc, timeout=1)

    def test_silent_child_times_out(self):
        proc = self.child(b"")
        with self.assertRaisesRegex(RuntimeError, "did not announce"):
            listener_port(proc, timeout=0.1)


class MemoryHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = spawn_server([str(BINARY), "serve", str(FIXTURES), "--name", "fixture", "--port", "0",
                                     "--host", "127.0.0.1", "--api-key", "public-test-key", "--metrics-open",
                                     "--keep-warm", "0"],
                                    env={**os.environ, "TF_FAKE_CONTEXT": "8192", "TENSORFOLD_NO_LIVE": "1"})
        cls.addClassCleanup(cls.stop)
        cls.base = "http://127.0.0.1:" + str(listener_port(cls.proc))

    @classmethod
    def stop(cls):
        cls.proc.terminate()
        try:
            cls.proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
            cls.proc.communicate(timeout=5)

    def get(self, path, authorized=False):
        headers = {"Authorization": "Bearer public-test-key"} if authorized else {}
        req = urllib.request.Request(self.base + path, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as reply:
                return reply.status, reply.read()
        except urllib.error.HTTPError as reply:
            with reply:
                return reply.code, reply.read()

    def test_both_routes_are_gated_even_with_metrics_open(self):
        for path in ["/memory", "/v1/memory", "/memory/?reset_peak=1"]:
            self.assertEqual(self.get(path)[0], 401)
            code, body = self.get(path, authorized=True)
            self.assertEqual(code, 200)
            value = json.loads(body)
            self.assertEqual(value["schema_version"], 1)
            self.assertEqual(value["context_window"], 8192)
            self.assertNotIn("prompt_cache_plan", value)
            self.assertNotIn("model", value)
            self.assertNotIn("uuid", body.decode().lower())

    def test_health_redaction_is_preserved(self):
        code, body = self.get("/health")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})

    def test_process_peak_scope_and_metrics(self):
        _, first = self.get("/memory", authorized=True)
        self.get("/health?reset_peak=1", authorized=True)
        code, second = self.get("/memory?reset_peak=1", authorized=True)
        self.assertEqual(code, 200)
        a, b = json.loads(first), json.loads(second)
        code, metrics = self.get("/metrics")
        self.assertEqual(code, 200)
        if sys.platform == "darwin":
            current = b["process"]["physical_footprint_bytes"]
            peak = b["process"]["lifetime_peak_physical_footprint_bytes"]
            self.assertGreater(current, 0)
            self.assertGreaterEqual(peak, current)
            self.assertGreaterEqual(peak, a["process"]["lifetime_peak_physical_footprint_bytes"])
            self.assertIn(b"tensorfold:process_footprint_peak_bytes ", metrics)
        else:
            self.assertNotIn("process", b)
            self.assertNotIn(b"tensorfold:process_footprint_peak_bytes ", metrics)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: test_memory_http.py FAKE_SERVE FIXTURES")
    BINARY, FIXTURES = (Path(arg).resolve() for arg in sys.argv[1:])
    unittest.main(argv=[sys.argv[0]])
