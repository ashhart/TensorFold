"""Offline control for Qwen's paged raw-prompt tail preparation."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1]))
import cuda_round_device as client


class PagedCacheServer(ThreadingHTTPServer):
    """Raw tails stay at 16-token page edges only with shared pages enabled."""

    def __init__(self):
        super().__init__(('127.0.0.1', 0), PagedCacheHandler)
        self.nonce = ''
        self.page_tokens = 0
        self.kept = []
        self.requests = []


class PagedCacheHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != '/health':
            self.send_error(404)
            return
        self.reply({'nonce': self.server.nonce})

    def do_POST(self):
        if self.path != '/v1/completions':
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ids = body['prompt']
        tokens = body['max_tokens']
        self.server.requests.append(ids)
        cached = max((len(prefix) for prefix in self.server.kept
                      if len(prefix) <= len(ids) and ids[:len(prefix)] == prefix), default=0)
        if self.server.page_tokens:
            at = len(ids) // self.server.page_tokens * self.server.page_tokens
            self.server.kept.append(ids[:at])
        chunk = {
            'id': 'cmpl-paged-cache',
            'choices': [{'index': 0, 'text': 'x' * tokens, 'finish_reason': 'length'}],
            'usage': {
                'prompt_tokens': len(ids),
                'completion_tokens': tokens,
                'total_tokens': len(ids) + tokens,
                'prompt_tokens_details': {'cached_tokens': cached},
            },
            'tensorfold': {'token_sha256': client.token_hash([7] * tokens)},
            'speculative': {'rounds': 1, 'drafted': 2, 'accepted': 1},
        }
        payload = f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Preparation(unittest.TestCase):
    def setUp(self):
        self.server = PagedCacheServer()
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.stop)
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.prompt = list(range(1, 33))

    def stop(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def restart(self):
        self.stop()
        self.server = PagedCacheServer()
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def test_raw_tail_control_and_paged_arm(self):
        client.completion(self.base, self.prompt[:-1], 1, 5)
        old = client.completion(self.base, self.prompt, 4, 5)
        with self.assertRaisesRegex(ValueError, 'prepared prefixes were not restored'):
            client.require(old['cached_tokens'] >= len(self.prompt) - 16,
                           'prepared prefixes were not restored; do not call this the prepared-context decode cell')
        self.assertEqual([self.prompt[:-1], self.prompt], self.server.requests)
        self.restart()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'native'
            binary.write_bytes(b'synthetic executable')
            args = argparse.Namespace(
                output=root, binary=binary, model_dir=root / 'model', drafter=root / 'draft',
                max_tokens=4, cache_gib=24, server_arg=[], conditions='fixture',
                start_timeout=5, stream_timeout=5, prompts_from=None, source='fixture')
            process = mock.Mock()
            process.poll.return_value = None
            spawned = {}

            def spawn(command, **options):
                env = options['env']
                spawned.update(env)
                self.server.nonce = env['TF_CUDA_ROUND_HEALTH_NONCE']
                if env.get('TF_QWEN_KV_PAGES') == '16' and int(env.get('TF_QWEN_KV_SHARE', '0')) > 0:
                    self.server.page_tokens = 16
                options['stdout'].write(
                    b'[tensorfold] serving fixture (drafts: DFlash2)\n'
                    b'TF_POLICY_CONFIG {"timed":true}\n'
                    b'TF_POLICY {"kind":"done","key":0,"mark":"cost","ms":0}\n'
                    b'TF_POLICY_END complete=true events=1\n')
                options['stdout'].flush()
                return process

            real_socket = client.socket.socket

            def socket_for(*args, **kwargs):
                if args:
                    return real_socket(*args, **kwargs)
                sock = mock.MagicMock()
                sock.__enter__.return_value.getsockname.return_value = (
                    '127.0.0.1', self.server.server_port)
                return sock

            with mock.patch.object(client.socket, 'socket', side_effect=socket_for), \
                    mock.patch.object(client.subprocess, 'Popen', side_effect=spawn), \
                    mock.patch.object(client, 'prompts', return_value=[self.prompt]):
                result = client.arm(args, 'code', 32, 1, 0, 0, calibration=True)
            self.assertGreaterEqual(result['replies'][0]['cached_tokens'], 16)
            self.assertEqual('16', spawned['TF_QWEN_KV_PAGES'])
            self.assertEqual('8', spawned['TF_QWEN_KV_SHARE'])
            self.assertEqual([self.prompt[:-1], self.prompt], self.server.requests)
            prepared = json.loads((root / 'code-n32-s1-r0-t0-cal' / 'prepared.json').read_text())
            self.assertEqual(1, len(prepared))
            process.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
