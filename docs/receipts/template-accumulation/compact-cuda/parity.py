"""Collect the fixed candidate-only 256-token drafted/plain parity gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

SHARED_SHA256 = 'e3e531b83e5099c223d984a9f5b7879df6e856e77ac7e6fc4274c7dc177bde92'
CANDIDATE = '8d00b490f5ddf45721546e9b9816a083a46e5fa2'
TOKENS = 256
DEADLINE = 900.0
FIXTURES = ('multiturn', 'prose', 'code')
KEYS = tuple((name, temp, seed) for name in FIXTURES
             for temp, seed in ((0.0, 1234), (1.0, 1234), (1.0, 1235)))
SCOPE = 'Candidate-only 256-token drafted/plain equality; no timing or cached-resume claim.'


def require(condition: bool, message: str) -> None:
    """Reject incomplete, malformed or unequal evidence."""
    if not condition:
        raise ValueError(message)


def load_shared(path: Path) -> Any:
    """Load the unchanged shared HTTP client after checking its frozen source hash."""
    require(hashlib.sha256(path.read_bytes()).hexdigest() == SHARED_SHA256, 'shared_client_hash')
    spec = importlib.util.spec_from_file_location('compact_parity_shared', path)
    require(spec is not None and spec.loader is not None, 'shared_client_import')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPLY_TOKENS = TOKENS
    module.REPETITIONS = 1
    module.LEVELS = (1,)
    module.DEADLINE_SECONDS = DEADLINE
    return module


def fingerprint(run: dict[str, Any]) -> tuple[Any, ...]:
    """Compare all token IDs, full hashes and the complete output strings."""
    return (tuple(run['token_ids']), run['token_sha256'], run['output_sha256'],
            run['content'], run['reasoning_content'])


def validate_response(client: Any, run: dict[str, Any], key: tuple, draft: bool) -> None:
    """Recheck identities, full hashes and raw streaming arithmetic independently."""
    require(not client.validate(run), 'shared_response_validation')
    require(run.get('valid') is True and run.get('validation_errors') == [], 'response_invalid')
    require((run.get('fixture'), run.get('temperature'), run.get('seed')) == key, 'response_key')
    require(type(run.get('seed')) is int and type(run.get('temperature')) in (int, float), 'key_types')
    require(run.get('draft') is draft, 'response_draft')
    ids = run.get('token_ids')
    require(isinstance(ids, list) and len(ids) == TOKENS, 'token_count')
    require(all(type(n) is int and 0 <= n <= 0xffffffff for n in ids), 'token_type')
    digest = hashlib.sha256(','.join(map(str, ids)).encode('ascii')).hexdigest()
    require(run.get('token_sha256') == digest and run.get('token_sha') == digest[:12], 'token_hash')
    content, reasoning = run.get('content'), run.get('reasoning_content')
    require(isinstance(content, str) and isinstance(reasoning, str), 'output_text')
    output = hashlib.sha256(hashlib.sha256(content.encode()).digest() +
                            hashlib.sha256(reasoning.encode()).digest()).hexdigest()
    require(run.get('output_sha256') == output, 'output_hash')
    for field in ('sent', 'first', 'last', 'completed'):
        value = run.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'timestamp_' + field)
    require(run['sent'] < run['first'] < run['last'] <= run['completed'], 'timestamp_order')
    for field, expected in (('ttft_s', run['first'] - run['sent']),
                            ('decode_tps', 255 / (run['last'] - run['first'])),
                            ('elapsed_s', run['completed'] - run['sent'])):
        value = run.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and
                math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-8), 'arithmetic_' + field)
    pieces = run.get('pieces')
    require(isinstance(pieces, list) and 4 <= len(pieces) <= TOKENS, 'stream_chunks')
    require(all(isinstance(p, list) and len(p) == 2 and type(p[0]) in (int, float) and
                math.isfinite(p[0]) and type(p[1]) is int and p[1] > 0 for p in pieces), 'chunk_shape')
    times = [p[0] for p in pieces]
    require(times == sorted(times) and times[0] == run['first'] and times[-1] == run['last'], 'chunk_order')
    offsets = run.get('chunk_offsets_s')
    require(isinstance(offsets, list) and len(offsets) == len(times), 'chunk_offsets')
    require(all(type(value) in (int, float) and math.isfinite(value) and
                math.isclose(value, at - run['sent'], rel_tol=1e-9, abs_tol=1e-8)
                for value, at in zip(offsets, times)), 'chunk_offset_arithmetic')


def collect(client: Any) -> None:
    """Collect nine serial pairs, counting the first cold request once."""
    previous = 0.0
    for index, key in enumerate(KEYS):
        name, temp, seed = key
        drafted = client.stream(name, temp, seed, True)
        if index == 0:
            client.report['first_cold'] = drafted
            client.save()
            require(drafted.get('valid') is True, 'first_cold_invalid')
            client.snapshots()
        pair = {'fixture': name, 'temperature': temp, 'seed': seed, 'drafted': drafted}
        client.report['pairs'].append(pair)
        client.save()
        validate_response(client, drafted, key, True)
        require(drafted['sent'] >= previous, 'pair_chronology')
        plain = client.stream(name, temp, seed, False)
        pair['plain'] = plain
        client.save()
        validate_response(client, plain, key, False)
        require(plain['sent'] >= drafted['completed'], 'draft_plain_chronology')
        pair['equal'] = fingerprint(drafted) == fingerprint(plain)
        client.save()
        require(pair['equal'], 'draft_plain_output_mismatch')
        previous = plain['completed']
    client.report['qualified'] = len(client.report['pairs']) == 9


def run_client(args: argparse.Namespace, shared: Any) -> int:
    """Checkpoint complete evidence and fail closed before any later experiment phase."""
    client = shared.Client(args)
    client.report.update({
        'label': 'P0', 'phase': 'candidate_drafted_plain_parity', 'block': 0, 'side': 'P',
        'candidate': CANDIDATE, 'scope': SCOPE, 'pairs': [], 'pair_order': [list(key) for key in KEYS],
        'client_deadline_seconds': DEADLINE, 'inference_responses_expected': 18,
        'first_cold_counted_in_pairs': True,
        'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_client_sha256': SHARED_SHA256,
    })
    try:
        if not client.model:
            models = client.json_request('/v1/models').get('data')
            require(isinstance(models, list) and len(models) == 1 and
                    isinstance(models[0].get('id'), str), 'expected_one_served_model')
            client.model = models[0]['id']
        collect(client)
    except Exception as error:
        client.report['errors'].append(type(error).__name__ + ': ' + str(error))
    client.report['elapsed_s'] = time.monotonic() - client.started
    client.report['inference_responses'] = sum(
        int('drafted' in pair) + int('plain' in pair) for pair in client.report['pairs'])
    client.save()
    print(json.dumps({field: client.report[field] for field in
                      ('qualified', 'inference_responses', 'elapsed_s', 'errors')}))
    return 0 if client.report['qualified'] else 1


def self_test(shared: Any) -> dict[str, Any]:
    """Exercise the complete eighteen-response schedule and malformed evidence offline."""
    import copy

    class Fake:
        def __init__(self) -> None:
            self.report: dict[str, Any] = {'pairs': [], 'qualified': False}
            self.clock = 10.0
            self.prompt_counts = {name: 100 for name in FIXTURES}
            self.calls: list[tuple] = []
            self.checker = shared.Client.__new__(shared.Client)
            self.checker.prompt_counts = self.prompt_counts

        def save(self) -> None:
            pass

        def snapshots(self) -> None:
            require(len(self.calls) == 1, 'tokenize_after_first_request')

        def validate(self, run: dict) -> list[str]:
            return self.checker.validate(run)

        def stream(self, name: str, temp: float, seed: int, draft: bool) -> dict:
            self.calls.append((name, temp, seed, draft))
            sent = self.clock
            self.clock += 4
            ids = list(range(TOKENS))
            content = name + str(temp) + str(seed)
            digest = hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()
            times = [sent + 1, sent + 1.5, sent + 2, sent + 3]
            return {'fixture': name, 'temperature': temp, 'seed': seed, 'draft': draft,
                    'complete': True, 'valid': True, 'error': None, 'validation_errors': [],
                    'tokens': TOKENS, 'cached_tokens': 0, 'cache_state': 'cold', 'prompt_tokens': 100,
                    'token_ids': ids, 'token_sha256': digest, 'token_sha': digest[:12],
                    'content': content, 'reasoning_content': '',
                    'output_sha256': hashlib.sha256(hashlib.sha256(content.encode()).digest() +
                                                     hashlib.sha256(b'').digest()).hexdigest(),
                    'sent': sent, 'first': sent + 1, 'last': sent + 3, 'completed': sent + 3.1,
                    'ttft_s': 1.0, 'decode_tps': 127.5, 'elapsed_s': sent + 3.1 - sent,
                    'pieces': [[at, 1] for at in times], 'chunk_offsets_s': [at - sent for at in times]}

    fake = Fake()
    collect(fake)
    require(fake.report['qualified'] is True and len(fake.calls) == 18, 'complete_schedule')
    require(fake.calls == [(*key, draft) for key in KEYS for draft in (True, False)], 'request_order')
    valid = fake.report['pairs'][0]['drafted']
    failures = ({'cached_tokens': 1}, {'tokens': 255}, {'token_sha256': 'f' * 64},
                {'output_sha256': 'f' * 64}, {'complete': False}, {'first': valid['last']},
                {'chunk_offsets_s': []}, {'draft': False}, {'token_ids': [True] * TOKENS})
    for changes in failures:
        try:
            validate_response(fake, {**copy.deepcopy(valid), **changes}, KEYS[0], True)
        except ValueError:
            continue
        raise AssertionError('malformed_response_accepted')
    original_stream = fake.stream

    def unequal(name: str, temp: float, seed: int, draft: bool) -> dict:
        run = original_stream(name, temp, seed, draft)
        if not draft:
            run['token_ids'][0] = 999
            digest = hashlib.sha256(','.join(map(str, run['token_ids'])).encode()).hexdigest()
            run.update(token_sha256=digest, token_sha=digest[:12])
        return run

    fake = Fake()
    original_stream = fake.stream
    fake.stream = unequal
    try:
        collect(fake)
    except ValueError as error:
        require(str(error) == 'draft_plain_output_mismatch' and len(fake.calls) == 2, 'mismatch_stop')
    else:
        raise AssertionError('mismatch_accepted')
    return {'status': 'passed', 'responses': 18, 'keys': 9, 'malformed_rejections': len(failures),
            'complete_pair_captured_before_mismatch_stop': True}


def main() -> int:
    """Run an offline self-test or the single explicitly identified parity phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--url')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tools', type=Path)
    parser.add_argument('--shared-client', type=Path, required=True)
    parser.add_argument('--block', type=int, choices=(0,))
    parser.add_argument('--side', choices=('P',))
    parser.add_argument('--model')
    parser.add_argument('--api-key-env')
    args = parser.parse_args()
    shared = load_shared(args.shared_client.resolve(strict=True))
    if args.self_test:
        print(json.dumps(self_test(shared), sort_keys=True))
        return 0
    if any(value is None for value in (args.url, args.output, args.tools, args.block, args.side)):
        parser.error('--url, --output, --tools, --block 0 and --side P are required')
    return run_client(args, shared)


if __name__ == '__main__':
    raise SystemExit(main())
