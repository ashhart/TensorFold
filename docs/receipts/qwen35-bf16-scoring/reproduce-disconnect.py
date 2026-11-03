"""Observe native cancellation and score recovery against a loopback server started with --dashboard."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import struct
import sys
import time
import urllib.parse
import urllib.request

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'tools/zig'))
from check_qwen35_prompt_scores import digest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base')
    parser.add_argument('model')
    parser.add_argument('fixtures')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = args.base.rstrip('/')
    origin = urllib.parse.urlsplit(base)
    if origin.scheme != 'http' or origin.hostname not in ('localhost', '127.0.0.1', '::1'):
        parser.error('use a loopback HTTP endpoint')
    fixtures = json.loads(Path(args.fixtures).read_text())
    report = {'schema': 1, 'client_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': [], 'passed': True}

    def record(name, passed, **fields):
        report['checks'].append({'check': name, 'passed': bool(passed), **fields})
        report['passed'] &= bool(passed)

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return json.load(response)

    def counter():
        with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
            text = response.read().decode()
        match = re.search(r'^tensorfold:client_disconnections_total (\d+)$', text, re.M)
        if match is None:
            raise RuntimeError('missing_cancellation_counter')
        return int(match.group(1))

    def wait(predicate, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            result = predicate()
            if result is not None:
                return result
            time.sleep(.1)
        raise TimeoutError('observation_deadline')

    def score():
        body = {'model': args.model, 'prompt': fixtures[1], 'max_tokens': 8, 'temperature': 0,
                'seed': 1234, 'draft': True, 'ignore_eos': True, 'return_token_ids': True, 'prompt_logprobs': 2}
        request = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.load(response)
        return {'token_ids': data['tensorfold']['token_ids'],
                'score_sha256': digest_rows(data['choices'][0]['prompt_logprobs']),
                'cached_tokens': data['usage']['prompt_tokens_details']['cached_tokens']}

    def acknowledged(before):
        value = counter()
        return value if value > before else None

    try:
        reference = score()
        record('reference_cache_off', reference['cached_tokens'] == 0, reference=reference)
        before = counter()
        body = {'model': args.model, 'prompt': fixtures[-1], 'max_tokens': 128, 'temperature': 0,
                'seed': 1234, 'ignore_eos': True, 'prompt_logprobs': 2}
        encoded = json.dumps(body).encode()
        target = origin.path.rstrip('/') + '/v1/completions'
        connection = socket.create_connection((origin.hostname, origin.port or 80), timeout=10)
        try:
            header = ('POST ' + target + ' HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                      'Content-Length: ' + str(len(encoded)) + '\r\n\r\n').encode()
            connection.sendall(header + encoded)

            def prefilling():
                status = get('/stats')
                return status if status.get('fill_prompt_tokens') == len(fixtures[-1]) and status.get('fill_tokens_left', 0) > 0 else None

            observed = wait(prefilling, 30)
            record('prefill_observed_before_reset', True,
                   status={k: observed.get(k) for k in ('running', 'fill_prompt_tokens', 'fill_tokens_left', 'generation_tokens_running')})
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        finally:
            connection.close()
        after = wait(lambda: acknowledged(before), 180)
        record('prefill_cancellation_acknowledged', after == before + 1, counter_before=before, counter_after=after)
        recovered = score()
        record('score_after_prefill_cancel', recovered == reference, observed=recovered)

        before = counter()
        body = {'model': args.model, 'prompt': fixtures[1], 'max_tokens': 128, 'temperature': 0,
                'seed': 1234, 'stream': True, 'ignore_eos': True}
        request = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                if raw.startswith(b'data: ') and raw.strip() != b'data: [DONE]':
                    data = json.loads(raw[6:])
                    if any(choice.get('text') for choice in data.get('choices', [])):
                        def decoding():
                            status = get('/stats')
                            generated = status.get('generation_tokens_running', 0)
                            return status if 0 < generated < 128 else None

                        observed = wait(decoding, 10)
                        record('decode_observed_before_close', True,
                               status={k: observed.get(k) for k in ('running', 'generation_tokens_running')})
                        break
            else:
                raise RuntimeError('no_streamed_token')
        after = wait(lambda: acknowledged(before), 30)
        record('decode_cancellation_acknowledged', after == before + 1, counter_before=before, counter_after=after)
        recovered = score()
        record('score_after_decode_cancel', recovered == reference, observed=recovered)
    except Exception as error:
        record('execution_error', False, error_class=type(error).__name__)
    encoded = json.dumps(report, indent=2) + '\n'
    Path(args.output).write_text(encoded)
    print(encoded, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
