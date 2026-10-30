"""Run one bounded side of a paired decode-equivalence block."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import threading
import time
from typing import Any

TOKENS = 256
REPETITIONS = 2
SEED = 1234
DEADLINE = 900.0
ORDER = (('A', 'B'), ('B', 'A'), ('B', 'A'), ('A', 'B'), ('A', 'B'), ('B', 'A'))
FIXTURES = ('multiturn', 'prose', 'code')


def cells_for(block: int) -> list[tuple[str, float, int]]:
    """Return one counterbalanced schedule shared by both sides of a block."""
    if not 1 <= block <= 6:
        raise ValueError('block must be between 1 and 6')
    cells = [(name, temp, streams) for name in FIXTURES for temp in (0.0, 1.0) for streams in (1, 2)]
    offset = (block - 1) * 2
    cells = cells[offset:] + cells[:offset]
    return list(reversed(cells)) if block % 2 == 0 else cells


def matches(shared: Any, reference: dict[str, Any], run: dict[str, Any]) -> bool:
    """Require valid complete evidence and identical tokens and output."""
    return shared.equal(reference, run) and reference.get('content') == run.get('content') and (
        reference.get('reasoning_content') == run.get('reasoning_content')
    )


def self_test(shared: Any = None) -> None:
    """Exercise protocol counts and fail-closed equality without network access."""
    class Shared:
        @staticmethod
        def equal(left: dict, right: dict) -> bool:
            return bool(left.get('valid') and right.get('valid') and left['token_ids'] == right['token_ids']
                        and left['token_sha256'] == right['token_sha256']
                        and left['output_sha256'] == right['output_sha256'])

    for block in range(1, 7):
        cells = cells_for(block)
        assert len(cells) == len(set(cells)) == 12
        assert sum(streams * REPETITIONS for _, _, streams in cells) == 36
    assert sum(order == ('A', 'B') for order in ORDER) == 3
    assert sum(order == ('B', 'A') for order in ORDER) == 3
    run = {'valid': True, 'token_ids': [1], 'token_sha256': 'a', 'output_sha256': 'b',
           'content': 'x', 'reasoning_content': ''}
    assert matches(Shared, run, dict(run))
    for changes in ({'valid': False}, {'token_ids': [2]}, {'token_sha256': 'c'},
                    {'output_sha256': 'c'}, {'content': 'y'}, {'reasoning_content': 'y'}):
        assert not matches(Shared, run, {**run, **changes})
    if shared is not None:
        checker = shared.Client.__new__(shared.Client)
        checker.prompt_counts = {'prose': 10}
        ids = list(range(TOKENS))
        valid = {'complete': True, 'error': None, 'tokens': TOKENS, 'cached_tokens': 0,
                 'cache_state': 'cold', 'prompt_tokens': 10, 'fixture': 'prose', 'token_ids': ids,
                 'token_sha256': shared.token_hash(ids), 'output_sha256': 'a' * 64,
                 'ttft_s': 1.0, 'decode_tps': 1.0, 'elapsed_s': 1.0}
        assert not checker.validate(valid)
        for changes in ({'complete': False}, {'cached_tokens': 1}, {'tokens': TOKENS - 1},
                        {'token_ids': ids[:-1]}, {'token_sha256': 'b' * 64}, {'decode_tps': 0.0}):
            assert checker.validate({**valid, **changes})
    assert len(FIXTURES) == 3
    print('PROTOCOL SELF-TEST PASSED: 6 blocks, 12 cells, 36 measured requests, 3 solo references' +
          ', 1 warmup, 6 equality failures' +
          (' and 6 validation failures' if shared is not None else ''))


def load_shared(path: Path) -> Any:
    """Load the frozen model client and set this protocol's immutable constants."""
    spec = importlib.util.spec_from_file_location('template_model_client', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('shared client cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPLY_TOKENS = TOKENS
    module.REPETITIONS = REPETITIONS
    module.DEADLINE_SECONDS = DEADLINE
    return module


def run_client(args: argparse.Namespace, shared: Any) -> int:
    """Save each complete cell and stop immediately on invalid evidence or drift."""
    client = shared.Client(args)
    client.report.update({
        'scope': '256-token decode equivalence; prior 64-token plain gate remains separate',
        'block': args.block, 'side': args.side, 'block_order': list(ORDER[args.block - 1]),
        'cell_order': [list(cell) for cell in cells_for(args.block)],
        'seeds': [SEED, SEED + 1], 'client_deadline_seconds': DEADLINE,
        'shared_client_sha256': hashlib.sha256(args.shared_client.read_bytes()).hexdigest(),
        'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'warmup_excluded': True, 'solo_references_excluded': True, 'solo_references': [],
    })
    try:
        if not client.model:
            models = client.json_request('/v1/models').get('data')
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0].get('id'), str):
                raise ValueError('expected_one_served_model')
            client.model = models[0]['id']
        client.report['first_cold'] = client.stream('multiturn', 0.0, SEED)
        client.save()
        if not client.report['first_cold']['valid']:
            raise ValueError('warmup_invalid')
        client.snapshots()
        if not client.report['first_cold']['valid']:
            raise ValueError('warmup_tokenize_mismatch')
        client.save()
        references: dict[tuple[str, float, int], dict[str, Any]] = {
            ('multiturn', 0.0, SEED): client.report['first_cold'],
        }
        for name in FIXTURES:
            solo = client.stream(name, 1.0, SEED + 1)
            client.report['solo_references'].append({'fixture': name, 'temperature': 1.0,
                                                     'seed': SEED + 1, 'run': solo})
            client.save()
            if not solo.get('valid'):
                raise ValueError('solo_reference_invalid')
            references[(name, 1.0, SEED + 1)] = solo
        for name, temperature, streams in cells_for(args.block):
            cell: dict[str, Any] = {'fixture': name, 'temperature': temperature, 'streams': streams, 'reps': []}
            for repetition in range(REPETITIONS):
                seeds = [SEED + i if temperature else SEED for i in range(streams)]
                gate = threading.Barrier(streams)
                with ThreadPoolExecutor(max_workers=streams) as pool:
                    futures = [pool.submit(client.stream, name, temperature, seed, True, gate) for seed in seeds]
                    runs = [future.result() for future in futures]
                comparisons = []
                for seed, run in zip(seeds, runs):
                    key = (name, temperature, seed)
                    if not run.get('valid'):
                        comparisons.append(False)
                    elif key not in references:
                        references[key] = run
                        comparisons.append(True)
                    else:
                        comparisons.append(matches(shared, references[key], run))
                timing = client.timing.workload(runs)
                aggregate = None
                if all(run.get('valid') for run in runs):
                    span = max(run['last'] for run in runs) - min(run['first'] for run in runs)
                    if math.isfinite(span) and span > 0:
                        aggregate = streams * (TOKENS - 1) / span
                cell['reps'].append({'repetition': repetition, 'runs': runs, 'comparisons': comparisons,
                                     'timing': timing, 'aggregate_tps': aggregate})
                if not all(comparisons) or aggregate is None or not math.isfinite(aggregate):
                    cell['qualified'] = False
                    client.report['cells'].append(cell)
                    client.save()
                    raise ValueError('invalid_request_or_output_drift')
            cell['qualified'] = True
            cell['median_aggregate_tps'] = statistics.median(rep['aggregate_tps'] for rep in cell['reps'])
            cell['median_end_to_end_tps'] = statistics.median(rep['timing']['end_to_end_tps'] for rep in cell['reps'])
            cell['median_request_ttft_s'] = statistics.median(
                run['ttft_s'] for rep in cell['reps'] for run in rep['runs'])
            client.report['cells'].append(cell)
            client.save()
        client.report['qualified'] = len(client.report['cells']) == 12
    except Exception as error:
        client.report['errors'].append(type(error).__name__ + ': ' + str(error))
    client.report['elapsed_s'] = time.monotonic() - client.started
    client.save()
    print(json.dumps({'qualified': client.report['qualified'], 'cells': len(client.report['cells']),
                      'elapsed_s': client.report['elapsed_s'], 'errors': client.report['errors']}))
    return 0 if client.report['qualified'] else 1


def main() -> int:
    """Run the offline protocol checks or one explicitly identified server side."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--url')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tools', type=Path)
    parser.add_argument('--shared-client', type=Path)
    parser.add_argument('--block', type=int, choices=range(1, 7))
    parser.add_argument('--side', choices=('A', 'B'))
    parser.add_argument('--model')
    parser.add_argument('--api-key-env')
    args = parser.parse_args()
    if args.self_test:
        self_test(load_shared(args.shared_client.resolve(strict=True)) if args.shared_client else None)
        return 0
    if any(value is None for value in (args.url, args.output, args.tools, args.shared_client, args.block, args.side)):
        parser.error('--url, --output, --tools, --shared-client, --block and --side are required')
    return run_client(args, load_shared(args.shared_client.resolve(strict=True)))


if __name__ == '__main__':
    raise SystemExit(main())
