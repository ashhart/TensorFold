#!/usr/bin/env python3
"""Public, exact-length chat fixtures; native /tokenize and existing benchmark evidence rules."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'upstream' / 'tools'))
from bench_evidence import Evidence


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def post(base, path, body, timeout=1800):
    request = urllib.request.Request(base.rstrip('/') + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def body(model, prompt, effort, seed, tokens):
    return dict(model=model, messages=[dict(role='user', content=prompt)],
                temperature=1.0, top_p=0.95, seed=seed, reasoning_effort=effort,
                chat_template_kwargs={'enable_thinking': True}, max_tokens=tokens,
                ignore_eos=True)


TASKS = {
    'prose': '请用中文总结资料中备份、恢复演练、访问权限和日志审计的原则，解释它们的联系，至少600字。',
    'code': '请依据资料生成完整单文件HTML介绍页：深色主题、服务器配置表、备份状态及审计功能卡片。只输出代码。',
}


def prompt_text(cell, records, padding):
    prefix = f'这是公开合成性能测试资料，任务类别为{cell}。资料中的编号仅为条目编号。\n'
    lines = (f'记录{i:06d}：项目需要独立备份、恢复演练、最小访问权限及可追溯的操作日志。\n'
             for i in range(records))
    return prefix + ''.join(lines) + '\n填充标记：' + ' x' * padding + '\n' + TASKS[cell]


def make_fixture(args, cell, target):
    def tokenize(records, padding):
        request = body(args.model, prompt_text(cell, records, padding), args.effort, args.seed, args.tokens)
        encoded = post(args.base, '/tokenize', request)
        ids = encoded['tokens']
        if encoded['count'] != len(ids):
            raise ValueError('tokenizer count disagrees with IDs')
        return request, ids

    lo, hi = 0, max(1, target // 16)
    while len(tokenize(hi, 0)[1]) <= target:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(tokenize(mid, 0)[1]) <= target:
            lo = mid
        else:
            hi = mid
    records = lo
    request, ids = tokenize(records, 0)
    # With this tokenizer each spaced x normally adds one token; verify, never assume it.
    padding = max(0, target - len(ids))
    seen = set()
    for _ in range(12):
        request, ids = tokenize(records, padding)
        if len(ids) == target:
            return dict(cell=f'{cell}-{target}', prompt_tokens=target, request=request, prompt_ids=ids,
                        prompt_sha256=hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest())
        seen.add(padding)
        padding += target - len(ids)
        if padding < 0 or padding in seen:
            break
    raise ValueError(f'Cannot form exact {target}-token chat prompt; no generation was sent')


def streamed(args, request, cell, rep):
    request = dict(request, stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(args.base.rstrip('/') + '/chat/completions', json.dumps(request).encode(),
                                 {'Content-Type': 'application/json'})
    evidence = Evidence()
    chunks, runtime, usage, speculative = [], {}, None, {}
    start = time.monotonic()
    first = last = None
    complete, hook = False, None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == 'data: [DONE]':
                complete = True
                break
            if not line.startswith('data:'):
                continue
            chunk = json.loads(line[5:])
            chunks.append(chunk)
            if 'error' in chunk:
                raise RuntimeError(chunk['error'])
            runtime = chunk.get('tensorfold') or runtime
            usage = chunk.get('usage') or usage
            speculative = chunk.get('speculative') or speculative
            for choice in chunk.get('choices', []):
                evidence.add(choice)
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content') or choice.get('text'):
                    now = time.monotonic()
                    if first is None:
                        first = now
                        if args.capture_hook_json:
                            values = {'label': args.label, 'cell': cell, 'rep': str(rep)}
                            argv = json.loads(args.capture_hook_json)
                            for key, value in values.items():
                                argv = [part.replace('{' + key + '}', value) for part in argv]
                            env = dict(os.environ, **{'TF_CAPTURE_' + k.upper(): v for k, v in values.items()})
                            hook = subprocess.Popen(argv, env=env, stdout=sys.stderr, stderr=sys.stderr)
                    last = now
    wall = time.monotonic() - start
    observed = evidence.finish(usage, runtime, complete)
    if not complete:
        raise RuntimeError('SSE ended without DONE')
    if hook is not None:
        # Waiting happens after the measured request; never block SSE consumption for the capture hook.
        hook_status = hook.wait(timeout=90)
    else:
        hook_status = None
    return dict(usage=usage, tensorfold=runtime, speculative=speculative, chunks=chunks,
                evidence=observed, capture_hook_status=hook_status,
                client_ttft_s=None if first is None else first - start,
                client_decode_s=None if first is None else last - first), wall


def run(args, fixture, rep):
    request = fixture['request']
    start = time.monotonic()
    if args.stream:
        result, wall = streamed(args, request, fixture['cell'], rep)
    else:
        result = post(args.base, '/chat/completions', request)
        wall = time.monotonic() - start
    runtime = result.get('tensorfold', {})
    usage = result.get('usage') or {}
    errors = []
    if len(runtime.get('token_sha256', '')) != 64:
        errors.append('missing full output token hash')
    if runtime.get('enable_thinking') is not True:
        errors.append('thinking not enabled')
    if usage.get('prompt_tokens') != fixture['prompt_tokens']:
        errors.append('served prompt count differs from fixture')
    if usage.get('completion_tokens') != args.tokens:
        errors.append('output length differs from requested fixed length')
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
    if args.require_cold and cached != 0:
        errors.append('prompt cache state is not verified cold')
    if args.stream and result.get('capture_hook_status') not in (None, 0):
        errors.append('capture hook failed')
    emit(label=args.label, cell=fixture['cell'], rep=rep, wall=wall,
         request=request, prompt_sha256=fixture['prompt_sha256'],
         expected_prompt_tokens=fixture['prompt_tokens'], result=result, errors=errors)
    if errors:
        raise RuntimeError('; '.join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8890/v1')
    parser.add_argument('--model', default='glm-5.3-flash-fp4-local')
    parser.add_argument('--label', required=True)
    parser.add_argument('--lengths', default='8192,32768,131072')
    parser.add_argument('--cells', default='prose,code')
    parser.add_argument('--effort', choices=['low', 'high'], default='low')
    parser.add_argument('--seed', type=int, default=119)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--fixtures', type=Path, required=True, help='Create once, then reuse exact requests and token IDs')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--require-cold', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--capture-hook-json',
                        help='JSON argv on first delta; supports {label}/{cell}/{rep} and TF_CAPTURE_* environment')
    args = parser.parse_args()
    if args.capture_hook_json:
        argv = json.loads(args.capture_hook_json)
        if not args.stream or not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            parser.error('--capture-hook-json requires --stream and a nonempty JSON string argv array')
    if args.tokens < 1 or args.reps < 1:
        parser.error('tokens/reps must be positive')
    cells = args.cells.split(',')
    lengths = [int(x) for x in args.lengths.split(',')]
    if any(x not in TASKS for x in cells) or any(x < 256 for x in lengths):
        parser.error('unknown cell or context length under 256')
    if args.fixtures.exists():
        all_fixtures = json.loads(args.fixtures.read_text())
        fixtures = [f for f in all_fixtures if f['prompt_tokens'] in lengths and f['cell'].split('-')[0] in cells]
        if len(fixtures) != len(cells) * len(lengths):
            parser.error('fixture file does not contain all requested cells')
        for f in fixtures:
            expected = body(args.model, f['request']['messages'][0]['content'], args.effort, args.seed, args.tokens)
            if f['request'] != expected:
                parser.error('fixture settings differ; use the original settings or a separate fixture file')
            ids = post(args.base, '/tokenize', f['request'])['tokens']
            if ids != f['prompt_ids']:
                parser.error('tokenizer/rendered prompt changed; refusing incomparable run')
    else:
        fixtures = [make_fixture(args, cell, n) for n in lengths for cell in cells]
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        with args.fixtures.open('x') as output:
            json.dump(fixtures, output, ensure_ascii=False)
    emit(kind='fixtures', label=args.label, path=str(args.fixtures.resolve()),
         cells=[dict(cell=f['cell'], prompt_tokens=f['prompt_tokens'], prompt_sha256=f['prompt_sha256']) for f in fixtures])
    if args.prepare_only:
        return
    for rep in range(args.reps):
        for fixture in fixtures:
            run(args, fixture, rep)


if __name__ == '__main__':
    main()
