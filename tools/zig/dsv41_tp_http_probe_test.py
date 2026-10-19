"""Synthetic controls for HTTP receipt validation, not model-performance evidence."""
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


class ProbeControls(unittest.TestCase):
    def exercise(self, mode):
        class Handler(http.server.BaseHTTPRequestHandler):
            calls = 0

            def log_message(self, *args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                text = 'fixture'
                if not data.get('stream'):
                    type(self).calls += 1
                    if mode == 'unstable-repeat' and self.calls > 3:
                        text = 'changed'
                    body = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
                    raw = json.dumps(body).encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    chunks = [
                        {'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]},
                        {'choices': [{'index': 0, 'delta': {'content': text}}]},
                        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
                    ]
                    for item in chunks:
                        self.wfile.write(('data: ' + json.dumps(item) + '\n\n').encode())
                    if mode != 'truncated-stream':
                        self.wfile.write(b'data: [DONE]\n\n')

        probe = Path(__file__).with_name('dsv41_tp_http_probe.py')
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / mode
            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                result = subprocess.run(
                    [sys.executable, str(probe), 'baseline',
                     'http://127.0.0.1:' + str(server.server_port), str(out)],
                    capture_output=True, timeout=30,
                )
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
            if mode == 'valid':
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(json.loads((out / 'result.json').read_text())['state'], 'passed')
                self.assertEqual(len(json.loads((out / 'benchmark.json').read_text())), 9)
                for repeat in range(3):
                    receipt = json.loads((out / f'repeat-{repeat}-stream.json').read_text())
                    self.assertGreaterEqual(receipt['first_delta_seconds'], 0)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((out / 'result.json').exists())

    def test_repeated_answers_and_complete_streams_pass(self):
        self.exercise('valid')

    def test_changed_repeated_answer_fails(self):
        self.exercise('unstable-repeat')

    def test_stream_without_done_marker_fails(self):
        self.exercise('truncated-stream')


if __name__ == '__main__':
    unittest.main()
