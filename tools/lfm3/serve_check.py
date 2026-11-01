#!/usr/bin/env python3
"""Run owned loopback servers to check LFM3 served token equality, prompt reuse and image behavior."""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_evidence import Evidence, token_equal


def solid_png(red, green, blue):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = (b'\0' + bytes((red, green, blue)) * 256) * 256
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 256, 256, 8, 2, 0, 0, 0)) + chunk(
        b'IDAT', zlib.compress(rows),
    ) + chunk(b'IEND', b'')


def image_url(data):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}


class Server:
    def __init__(self, binary, directory, cache, log_directory):
        self.binary, self.directory, self.cache = binary, directory, cache
        self.log_directory = log_directory
        self.process = None

    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.base = f'http://127.0.0.1:{port}'
        self.log = tempfile.NamedTemporaryFile(mode='w+b', prefix='server-', suffix='.log',
                                               dir=self.log_directory, delete=False)
        command = [str(self.binary), 'serve', str(self.directory), '--host', '127.0.0.1', '--port', str(port),
                   '--name', 'lfm3-check', '--context', '2048', '--parallel', '4', '--temperature', '0',
                   '--prompt-cache-gib', str(self.cache), '--no-thinking', '--no-update-check']
        self.command = command
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Server exited {self.process.returncode}; see {self.log.name}')
                try:
                    with urllib.request.urlopen(self.base + '/v1/models', timeout=1) as response:
                        if response.status == 200:
                            return self
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f'Server startup timed out; see {self.log.name}')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()

    def request(self, item, draft=False, gate=None):
        body = {'model': 'lfm3-check', 'max_tokens': 24, 'temperature': item.get('temperature', 0),
                'seed': 1234, 'stream': True, 'stream_options': {'include_usage': True},
                'ignore_eos': True, 'draft': draft}
        if body['temperature']:
            body.update(top_k=20, top_p=0.95)
        chat = 'messages' in item
        if chat:
            body['messages'] = item['messages']
            body['chat_template_kwargs'] = {'enable_thinking': False}
        else:
            body['prompt'] = item['prompt']
        url = self.base + ('/v1/chat/completions' if chat else '/v1/completions')
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        if gate:
            gate.wait(timeout=30)
        evidence, usage, runtime, text = Evidence(), {}, {}, []
        complete = False
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if line == 'data: [DONE]':
                        complete = True
                        break
                    if not line.startswith('data:'):
                        continue
                    event = json.loads(line[5:])
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    usage = event.get('usage') or usage
                    runtime = event.get('tensorfold') or runtime
                    for choice in event.get('choices', []):
                        evidence.add(choice)
                        text.append(choice.get('text') or (choice.get('delta') or {}).get('content') or '')
            result = evidence.finish(usage, runtime, complete)
            result['text'] = ''.join(text)
            if not complete:
                result['error'] = 'incomplete_stream'
            return result
        except urllib.error.HTTPError as exc:
            return {'error': exc.read(8192).decode(errors='replace'), 'status': exc.code, 'complete': False}
        except Exception as exc:
            return {'error': f'{type(exc).__name__}: {exc}', 'complete': False}


    def disconnections(self):
        with urllib.request.urlopen(self.base + '/metrics', timeout=5) as response:
            text = response.read(1 << 20).decode()
        match = re.search(r'^tensorfold:client_disconnections_total (\d+)$', text, re.MULTILINE)
        if match is None:
            raise RuntimeError('Missing server client-disconnection counter')
        return int(match.group(1))

    def cancel_stream(self, item):
        before = self.disconnections()
        body = {'model': 'lfm3-check', 'max_tokens': 512, 'temperature': 0, 'seed': 1234,
                'stream': True, 'ignore_eos': True, 'draft': True}
        is_chat = 'messages' in item
        if is_chat:
            body['messages'] = item['messages']
            body['chat_template_kwargs'] = {'enable_thinking': False}
        else:
            body['prompt'] = item['prompt']
        endpoint = '/v1/chat/completions' if is_chat else '/v1/completions'
        request = urllib.request.Request(self.base + endpoint, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        first_content, request_id, already_finished = None, None, False
        with urllib.request.urlopen(request, timeout=30) as response:
            for raw in response:
                line = raw.decode().strip()
                if line == 'data: [DONE]':
                    already_finished = True
                    break
                if not line.startswith('data:'):
                    continue
                event = json.loads(line[5:])
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                request_id = event.get('id') or request_id
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    content = choice.get('text') or delta.get('content') or delta.get('reasoning_content')
                    already_finished = already_finished or choice.get('finish_reason') is not None
                    if content:
                        first_content = content
                        break
                if first_content:
                    break
        deadline = time.monotonic() + 20
        after = before
        while time.monotonic() < deadline and after <= before and first_content and not already_finished:
            after = self.disconnections()
            if after <= before:
                time.sleep(0.05)
        return {'request_id': request_id, 'first_content': first_content, 'requested_tokens': 512,
                'closed_before_done': bool(first_content) and not already_finished,
                'server_disconnections_before': before, 'server_disconnections_after': after,
                'server_confirmed_cancel': after > before}


def concurrent_requests(server, items):
    gate = threading.Barrier(len(items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
        futures = [executor.submit(server.request, item, True, gate) for item in items]
        return [future.result() for future in futures]


def chat(text, images=()):
    content = [image_url(image) for image in images] + [{'type': 'text', 'text': text}]
    return {'messages': [{'role': 'user', 'content': content if images else text}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-images', action='store_true', help='Diagnostic omission, recorded as skipped')
    parser.add_argument('--cancel', action='store_true', help='Also check real HTTP cancellation and recovery')
    parser.add_argument('--cancel-only', action='store_true', help='Run only fresh baselines and cancellation checks')
    parser.add_argument('--sampled', action='store_true', help='Also test seeded sampling')
    args = parser.parse_args()
    args.binary, args.model_dir = args.binary.resolve(), args.model_dir.resolve()
    config = json.loads((args.model_dir / 'config.json').read_text())
    vision = config.get('model_type') == 'lfm3_vl'
    receipt = {'model': args.model_dir.name, 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               'tests': [], 'logs': [], 'commands': [], 'speed_qualified': False}

    def check(name, passed, evidence=None, skip=False):
        status = 'skipped' if skip else 'passed' if passed is True else 'failed'
        receipt['tests'].append({'name': name, 'status': status, 'evidence': evidence})
        print(f'{status}: {name}', flush=True)

    code = {'prompt': 'Write a Python function that returns the larger of two integers.'}
    prose = chat('Explain why the sky is blue in two sentences.')
    story = ('The river flows past a quiet village. Children watch birds near the bridge. ' * 60)
    long = {'prompt': story + '\nSummarize the story in one sentence.'}
    edited = {'prompt': 'A mountain rises beside a busy town. ' + story[70:] + '\nSummarize in one sentence.'}
    items = {'code': code, 'chat': prose, 'long_prefix': long, 'edited_prefix': edited}
    if args.sampled:
        items['sampled_code'] = dict(code, temperature=0.7)
        items['sampled_chat'] = dict(prose, temperature=0.7)
    if vision and not args.skip_images:
        red, blue = solid_png(255, 0, 0), solid_png(0, 0, 255)
        question = 'What single solid color fills this image? Reply with only the color name.'
        items.update(red=chat(question, [red]), blue=chat(question, [blue]))
        order = 'Name the solid color of the first image, then the second image. Reply with two color names in order.'
        items.update(red_blue=chat(order, [red, blue]), blue_red=chat(order, [blue, red]))
    elif vision:
        check('images', None, 'Explicit --skip-images diagnostic run', skip=True)
    if args.cancel_only:
        items = {name: items[name] for name in ('code', 'chat', 'red') if name in items}
    baselines = {}
    log_directory = tempfile.mkdtemp(prefix='lfm3-serve-check-')
    try:
        with Server(args.binary, args.model_dir, 0, log_directory) as server:
            receipt['logs'].append(server.log.name)
            receipt['commands'].append(server.command)
            for name, item in items.items():
                result = server.request(item)
                baselines[name] = result
                check(name + '/fresh_plain', result.get('complete') and bool(result.get('token_sha256')), result)
            if 'long_prefix' in baselines:
                check('long_prefix/length_over_256', (baselines['long_prefix'].get('prompt_tokens') or 0) > 256,
                      baselines['long_prefix'].get('prompt_tokens'))
        with Server(args.binary, args.model_dir, 0.1, log_directory) as server:
            receipt['logs'].append(server.log.name)
            receipt['commands'].append(server.command)
            if not args.cancel_only:
                for name, item in items.items():
                    for draft in (False, True):
                        result = server.request(item, draft=draft)
                        if name == 'edited_prefix' and not draft:
                            check('edited_prefix/no_stale_prefix_reuse', result.get('cached_tokens') == 0, result)
                        check(name + ('/drafted_equals_fresh' if draft else '/plain_equals_fresh'),
                              token_equal(result, baselines[name]), result)
                    if name == 'long_prefix':
                        warm = server.request(item, draft=False)
                        check('long_prefix/warm_reuses_and_matches_fresh',
                              (warm.get('cached_tokens') or 0) > 0 and token_equal(warm, baselines[name]) is True, warm)
                    if name == 'edited_prefix':
                        result = server.request(item)
                        check('edited_prefix/matches_cache_off', token_equal(result, baselines[name]), result)
                for level in (1, 2, 4):
                    names = ['code', 'chat'] if not args.sampled else ['code', 'chat', 'sampled_code', 'sampled_chat']
                    for name in names:
                        results = concurrent_requests(server, [items[name]] * level)
                        check(f'{name}/concurrent_{level}_equals_solo',
                              all(token_equal(result, baselines[name]) is True for result in results), results)
                if vision and not args.skip_images:
                    for name, expected in [('red', ['red']), ('blue', ['blue']),
                                           ('red_blue', ['red', 'blue']), ('blue_red', ['blue', 'red'])]:
                        found = re.findall(r'\b(?:red|blue)\b', baselines[name].get('text', '').lower())
                        check(name + '/color_semantics', found[:len(expected)] == expected,
                              {'expected': expected, 'observed': found, 'text': baselines[name].get('text')})
                    image_items = [items['red'], items['blue'], items['red_blue'], items['blue_red']]
                    results = concurrent_requests(server, image_items)
                    check('images/concurrent_equals_solos', all(
                        token_equal(result, baselines[name]) is True
                        for result, name in zip(results, ('red', 'blue', 'red_blue', 'blue_red'))
                    ), results)
                    corrupt = server.request(chat('Describe this image.', [b'\x89PNG\r\n\x1a\ninvalid']))
                    check('images/corrupt_png_rejected', 400 <= corrupt.get('status', 0) < 500, corrupt)
            if args.cancel or args.cancel_only:
                for name in ('code', 'chat', 'red'):
                    if name not in items:
                        continue
                    cancelled = server.cancel_stream(items[name])
                    check(name + '/cancel_server_confirmed',
                          cancelled['closed_before_done'] and cancelled['server_confirmed_cancel'], cancelled)
                    recovered = server.request(items[name], draft=False)
                    check(name + '/cancel_next_plain_equals_fresh', token_equal(recovered, baselines[name]), recovered)
                    recovered = server.request(items[name], draft=True)
                    check(name + '/cancel_next_drafted_equals_fresh',
                          token_equal(recovered, baselines[name]), recovered)

    except Exception as exc:
        check('harness_execution', False, f'{type(exc).__name__}: {exc}')
    receipt['counts'] = {status: sum(test['status'] == status for test in receipt['tests'])
                         for status in ('passed', 'failed', 'skipped')}
    receipt['cancellation_checks'] = sum('/cancel_' in test['name'] for test in receipt['tests'])
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['counts']))
    return int(receipt['counts']['failed'] > 0)


if __name__ == '__main__':
    sys.exit(main())
