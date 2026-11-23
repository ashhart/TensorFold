"""replay_request.py on journal fixtures; every send goes to a fake server, and no socket is opened."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('replay_request', Path(__file__).with_name('replay_request.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

FP = 'ab' * 32
TOKENS = 'c0' * 32


def whole(content, reply_id='chatcmpl-1', created=1, tokens=TOKENS, seconds=0.5):
    return json.dumps({'id': reply_id, 'object': 'chat.completion', 'created': created, 'model': 'm',
                       'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
                       'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 0}},
                       'tensorfold': {'token_sha256': tokens, 'seconds': seconds}}).encode()


def streamed(pieces, reply_id='chatcmpl-1', keepalive=False):
    frames = [{'id': reply_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}]
    frames += [{'id': reply_id, 'choices': [{'index': 0, 'delta': {'content': p}, 'finish_reason': None}]} for p in pieces]
    frames.append({'id': reply_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'tensorfold': {'token_sha256': TOKENS}})
    out = b''.join(b'data: ' + json.dumps(f).encode() + b'\n\n' for f in frames)
    return (b': keepalive\n\n' if keepalive else b'') + out + b'data: [DONE]\n\n'


class FakeServer:
    def __init__(self, reply, status=200):
        self.reply, self.status, self.calls = reply, status, []

    def send(self, url, body):
        self.calls.append((url, body))
        return self.status, self.reply


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'requests.jsonl'

    def write(self, records, path=None):
        (path or self.path).write_text(''.join(json.dumps(dict(version=1, **x)) + '\n' for x in records))

    def journal(self, reply=whole('ok'), stream=False, body=None, seed=42, kept=True):
        start = {'kind': 'start', 'id': 'r', 'at': int(time.time()), 'path': '/v1/chat/completions', 'fingerprint': FP, 'seed': None,
                 'body': {'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 1} if body is None else body}
        finish = {'kind': 'finish', 'id': 'r', 'status': 200, 'stream': stream, 'outcome': 'completed', 'reply_bytes': len(reply),
                  'reply_sha256': hashlib.sha256(reply).hexdigest(), 'token_sha256': TOKENS}
        if kept:
            finish['reply_base64'] = base64.b64encode(reply).decode()
        self.write([start, {'kind': 'prepared', 'id': 'r', 'seed': seed}, finish])
        return r.read_journal(self.path)[0]['r']

    def test_the_drawn_seed_is_sent_and_equal_output_passes_despite_new_ids_and_timings(self):
        server = FakeServer(whole('ok', reply_id='chatcmpl-2', created=2, seconds=0.9))
        result, _ = r.replay(self.journal(), 'http://example.test', server.send)
        self.assertTrue(result['equal'])
        self.assertEqual(result['comparison'], 'output')
        self.assertEqual(server.calls[0][0], 'http://example.test/v1/chat/completions')
        self.assertEqual(server.calls[0][1]['seed'], 42)
        own = self.journal(body={'messages': [], 'seed': 7})
        server = FakeServer(whole('ok'))
        r.replay(own, 'http://example.test', server.send)
        self.assertEqual(server.calls[0][1]['seed'], 7)  # the client's own seed is never replaced

    def test_different_text_or_tokens_are_a_reported_mismatch(self):
        result, _ = r.replay(self.journal(), 'http://example.test', FakeServer(whole('oK')).send)
        self.assertFalse(result['equal'])
        self.assertIsNotNone(result['first_difference'])
        result, _ = r.replay(self.journal(), 'http://example.test', FakeServer(whole('ok', tokens='d1' * 32)).send)
        self.assertFalse(result['equal'])
        self.assertFalse(r.replay(self.journal(), 'http://example.test', FakeServer(whole('ok'), 500).send)[0]['equal'])

    def test_whole_body_compares_every_byte(self):
        entry = self.journal()
        self.assertTrue(r.replay(entry, 'http://example.test', FakeServer(whole('ok')).send, whole_body=True)[0]['equal'])
        result, _ = r.replay(entry, 'http://example.test', FakeServer(whole('ok', reply_id='chatcmpl-2')).send, whole_body=True)
        self.assertFalse(result['equal'])
        self.assertEqual(result['comparison'], 'whole_body')

    def test_a_stream_compares_its_output_whatever_its_frames_and_keepalives(self):
        entry = self.journal(streamed(['o', 'k']), stream=True)
        self.assertTrue(r.replay(entry, 'http://example.test', FakeServer(streamed(['ok'], 'chatcmpl-9', True)).send)[0]['equal'])
        self.assertFalse(r.replay(entry, 'http://example.test', FakeServer(streamed(['o', 'h'])).send)[0]['equal'])
        self.assertFalse(r.replay(entry, 'http://example.test', FakeServer(streamed(['ok'], keepalive=True)).send, whole_body=True)[0]['equal'])

    def test_a_reply_too_large_to_keep_is_compared_by_its_token_hash(self):
        entry = self.journal(kept=False)
        result, _ = r.replay(entry, 'http://example.test', FakeServer(whole('anything')).send)
        self.assertEqual(result['comparison'], 'tokens')
        self.assertTrue(result['equal'])
        self.assertFalse(r.replay(entry, 'http://example.test', FakeServer(whole('ok', tokens='d1' * 32)).send)[0]['equal'])

    def test_redacted_media_and_an_omitted_body_need_the_original_before_any_send(self):
        for body in ({'messages': [{'content': [{'type': 'image_url', 'image_url': {'url': '<redacted>'}}]}]}, None):
            server = FakeServer(whole('ok'))
            entry = self.journal(body=body if body else {})
            if body is None:
                entry['start'].pop('body')
            with self.assertRaises(r.ReplayError):
                r.replay(entry, 'http://example.test', server.send)
            self.assertEqual(server.calls, [])
            self.assertTrue(r.replay(entry, 'http://example.test', server.send, replacement={'messages': []})[0]['equal'])

    def test_a_crashed_request_is_sent_only_on_purpose_and_has_no_reference(self):
        crash = {'kind': 'quarantine', 'id': 'r', 'fingerprint': FP, 'at': int(time.time())}
        self.write([{'kind': 'start', 'id': 'r', 'path': '/v1/chat/completions', 'fingerprint': FP, 'body': {'messages': []}}, crash, crash])
        entry = r.read_journal(self.path)[0]['r']
        self.assertTrue(entry['refused'])
        server = FakeServer(b'{}')
        with self.assertRaises(r.ReplayError):
            r.replay(entry, 'http://example.test', server.send)
        self.assertEqual(server.calls, [])
        result, _ = r.replay(entry, 'http://example.test', server.send, allow_quarantined=True)
        self.assertIsNone(result['equal'])
        self.assertEqual(result['comparison'], 'no_reference')

    def test_rotated_files_are_read_oldest_first_and_torn_or_bad_lines_passed_over(self):
        self.write([{'kind': 'start', 'id': 'r', 'path': '/v1/completions', 'fingerprint': FP, 'body': {}}], Path(f'{self.path}.2'))
        self.path.write_bytes(b'bad\n' + json.dumps({'version': 1, 'kind': 'finish', 'id': 'r', 'status': 200}).encode() + b'\n{"kind":')
        entries, skipped = r.read_journal(self.path)
        self.assertEqual(skipped, 1)
        self.assertEqual(entries['r']['finish']['status'], 200)
        self.assertFalse(entries['r']['in_flight'])

    def test_two_recent_crashes_refuse_and_a_lapsed_one_or_a_clear_does_not(self):
        old = int(time.time()) - r.QUARANTINE_S - 1
        start = {'kind': 'start', 'id': 'r', 'path': '/v1/completions', 'fingerprint': FP, 'body': {}}
        lapsed, recent = ({'kind': 'quarantine', 'id': 'r', 'fingerprint': FP, 'at': at} for at in (old, int(time.time())))
        self.write([start, lapsed, recent])
        self.assertFalse(r.read_journal(self.path)[0]['r']['refused'])
        self.write([start, recent, recent])
        self.assertTrue(r.read_journal(self.path)[0]['r']['refused'])
        self.write([start, recent, recent, {'kind': 'clear', 'fingerprint': FP}])
        self.assertFalse(r.read_journal(self.path)[0]['r']['refused'])

    def test_route_and_origin_are_checked_before_any_send(self):
        entry, server = self.journal(), FakeServer(b'{}')
        for base in ('file:///tmp/a', 'http://name:secret@example.test', 'http://example.test/v1', 'http://example.test/?a=1'):
            with self.assertRaises(r.ReplayError):
                r.replay(entry, base, server.send)
        entry['start']['path'] = '/admin'
        with self.assertRaises(r.ReplayError):
            r.replay(entry, 'http://example.test', server.send)
        self.assertEqual(server.calls, [])

    @unittest.skipIf(os.name == 'nt', 'the lock is flock here; Windows locks a byte')
    def test_cli_list_never_sends_a_mismatch_exits_one_and_clear_waits_for_the_server_to_stop(self):
        import fcntl
        self.journal()
        quiet = contextlib.redirect_stdout(io.StringIO())
        with patch.object(r, 'http_send', side_effect=AssertionError('no send')), quiet:
            self.assertEqual(r.main([str(self.path), '--list']), 0)
        with patch.object(r, 'http_send', return_value=(200, whole('no'))), quiet:
            self.assertEqual(r.main([str(self.path), '--id', 'r', '--base', 'http://example.test']), 1)
        lock = os.open(f'{self.path}.lock', os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, lock)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # as a running server holds it
        with self.assertRaises(r.ReplayError):
            r.clear(self.path, 'r', r.read_journal(self.path)[0])
        fcntl.flock(lock, fcntl.LOCK_UN)
        with quiet:
            self.assertEqual(r.main([str(self.path), '--clear', 'r']), 0)
        last = json.loads(self.path.read_text().splitlines()[-1])
        self.assertEqual((last['kind'], last['fingerprint']), ('clear', FP))


if __name__ == '__main__':
    unittest.main()
