"""Validate the fixed six-pair decode-equivalence experiment without interim inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

FIXTURES = ("multiturn", "prose", "code")
TOKENS = 256
REPETITIONS = 2
ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
LABELS = tuple(side + str(block) for block, order in enumerate(ORDERS, 1) for side in order)
T_CRITICAL = 2.570581835636305
BOUNDS = (0.98, 1.02)
PROTOCOL_SHA256 = "f86dae04e9f80398d198204380048b5efeaebf828bf71866bf0c077d313ff181"
HASH = re.compile(r"[0-9a-f]{64}\Z")
CellKey = tuple[str, float, int]
RequestKey = tuple[str, float, int, bool]
HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    """Stop at the first invalid or unequal piece of evidence."""
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: float, tag: str) -> None:
    """Check finite unrounded arithmetic with small floating-point tolerance."""
    require(type(actual) in (int, float) and math.isfinite(actual), tag + ": invalid number")
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), tag + ": arithmetic differs")


def ids_hash(ids: list[int]) -> str:
    """Hash comma-separated decimal IDs, independently of the benchmark helpers."""
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def valid_ids(ids: Any, tag: str, length: int | None = None) -> None:
    """Require actual unsigned token integers and the expected nonempty length."""
    require(isinstance(ids, list) and bool(ids), tag + ": missing IDs")
    require(length is None or len(ids) == length, tag + ": ID count")
    require(all(type(n) is int and 0 <= n <= 0xffffffff for n in ids), tag + ": malformed IDs")


def fingerprint(run: dict[str, Any]) -> tuple[Any, ...]:
    """Compare full tokens and independent content/reasoning hashes."""
    return tuple(run["token_ids"]), run["token_sha256"], run["output_sha256"]


def validate_request(
    run: dict[str, Any], expected: RequestKey, prompt_count: int, tag: str,
) -> None:
    """Recompute hashes and observed request timing without trusting qualification flags."""
    name, temperature, seed, draft = expected
    require(type(run.get("temperature")) in (int, float), tag + ": temperature type")
    require(run["fixture"] == name and run["temperature"] == temperature, tag + ": request identity")
    require(type(run["seed"]) is int and run["seed"] == seed, tag + ": seed")
    require(run["draft"] is draft, tag + ": draft mode")
    require(run.get("complete") is True and run.get("error") is None, tag + ": incomplete/error")
    require(run.get("valid") is True and run.get("validation_errors") == [], tag + ": client invalid")
    require(type(run.get("tokens")) is int and run["tokens"] == TOKENS, tag + ": reply count")
    require(type(run.get("cached_tokens")) is int and run["cached_tokens"] == 0, tag + ": cache reused")
    require(run.get("cache_state") == "cold", tag + ": unverified cache")
    require(type(run.get("prompt_tokens")) is int and run["prompt_tokens"] == prompt_count,
            tag + ": tokenize/prompt count")
    require(0 < prompt_count <= 8192 - TOKENS, tag + ": context bound")
    valid_ids(run.get("token_ids"), tag, TOKENS)
    require(run.get("token_sha256") == ids_hash(run["token_ids"]), tag + ": token hash")
    for field in ("token_sha256", "output_sha256"):
        require(isinstance(run.get(field), str) and HASH.fullmatch(run[field]) is not None,
                tag + ": invalid " + field)
    require(run.get("token_sha") == run["token_sha256"][:12], tag + ": short token hash")
    content, reasoning = run.get("content"), run.get("reasoning_content")
    require(isinstance(content, str) and isinstance(reasoning, str), tag + ": missing output text")
    digest = hashlib.sha256(
        hashlib.sha256(content.encode()).digest() + hashlib.sha256(reasoning.encode()).digest()
    ).hexdigest()
    require(run["output_sha256"] == digest, tag + ": output hash")
    for field in ("sent", "first", "last", "completed"):
        value = run.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                tag + ": invalid timestamp " + field)
    require(run["sent"] < run["first"] < run["last"] <= run["completed"], tag + ": event order")
    close(run["ttft_s"], run["first"] - run["sent"], tag + " TTFT")
    close(run["decode_tps"], (TOKENS - 1) / (run["last"] - run["first"]), tag + " decode")
    close(run["elapsed_s"], run["completed"] - run["sent"], tag + " elapsed")
    pieces = run.get("pieces")
    require(isinstance(pieces, list) and 4 <= len(pieces) <= TOKENS, tag + ": missing/excess streamed chunks")
    require(all(isinstance(p, list) and len(p) == 2 for p in pieces), tag + ": chunk shape")
    require(all(type(p[0]) in (int, float) and math.isfinite(p[0]) for p in pieces),
            tag + ": chunk timestamps")
    require(all(type(p[1]) is int and p[1] > 0 for p in pieces), tag + ": chunk character counts")
    times = [p[0] for p in pieces]
    require(times == sorted(times), tag + ": chunk order")
    close(times[0], run["first"], tag + " first chunk")
    close(times[-1], run["last"], tag + " last chunk")
    offsets = run.get("chunk_offsets_s")
    require(isinstance(offsets, list) and len(offsets) == len(times), tag + ": chunk offsets")
    for actual, at in zip(offsets, times):
        close(actual, at - run["sent"], tag + " chunk offset")


def validate_metadata(data: dict[str, Any], label: str) -> dict[str, int]:
    """Recompute fixture and tokenization fingerprints and require the complete protocol."""
    require(data.get("schema_version") == 1 and data.get("label") == label, label + ": header")
    require(data.get("qualified") is True and data.get("errors") == [], label + ": not qualified")
    require(data.get("requested_reply_tokens") == TOKENS and data.get("repetitions") == REPETITIONS,
            label + ": protocol size")
    require(data.get("levels") == [1, 2] and data.get("temperatures") == [0.0, 1.0], label + ": grid")
    require(data.get("resume_test") == "not_run: prompt cache is required to be disabled",
            label + ": resume scope")
    require(set(data["fixtures"]) == set(FIXTURES), label + ": fixtures")
    require(set(data["tokenize"]) == set(FIXTURES), label + ": tokenization fixtures")
    helpers = data.get("helper_sha256")
    expected_helpers = {"bench_evidence.py", "bench_timing.py", "bench_metrics.py"}
    require(isinstance(helpers, dict) and set(helpers) == expected_helpers, label + ": helper hashes")
    require(all(isinstance(h, str) and HASH.fullmatch(h) for h in helpers.values()),
            label + ": helper hash shape")
    prompt_counts = {}
    for name in FIXTURES:
        record = data["fixtures"][name]
        messages = record.get("messages")
        require(isinstance(messages, list) and len(messages) == (34 if name == "multiturn" else 1),
                label + ": fixture message count " + name)
        require(record.get("message_count") == len(messages), label + ": fixture count metadata")
        raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
        require(record.get("sha256") == hashlib.sha256(raw).hexdigest(), label + ": fixture hash " + name)
        snapshots = data["tokenize"][name]
        require(set(snapshots) == {"generation", "history"}, label + ": snapshot phases")
        for phase, snapshot in snapshots.items():
            ids = snapshot.get("token_ids")
            valid_ids(ids, label + ": tokenize " + name + "/" + phase)
            require(type(snapshot.get("count")) is int and snapshot["count"] == len(ids),
                    label + ": tokenize count")
            require(len(ids) + TOKENS <= 8192, label + ": tokenizer context bound")
            require(snapshot.get("token_sha256") == ids_hash(ids), label + ": tokenizer hash")
        prompt_counts[name] = snapshots["generation"]["count"]
    return prompt_counts


def cells_for(block: int) -> list[CellKey]:
    """Independently reconstruct the fixed counterbalanced cell schedule."""
    cells = [(name, temp, n) for name in FIXTURES for temp in (0.0, 1.0) for n in (1, 2)]
    offset = (block - 1) * 2
    cells = cells[offset:] + cells[:offset]
    return list(reversed(cells)) if block % 2 == 0 else cells


def frozen_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    """Read the frozen public protocol and clients without any network activity."""
    require(hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest() == PROTOCOL_SHA256,
            'frozen protocol differs')
    path = HERE.parent / 'model/client.py'
    spec = importlib.util.spec_from_file_location('frozen_fixtures', path)
    require(spec is not None and spec.loader is not None, 'cannot load public fixtures')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixtures(), {
        'client_sha256': hashlib.sha256((HERE / 'client.py').read_bytes()).hexdigest(),
        'shared_client_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def validate_run(data: dict[str, Any], label: str) -> tuple[dict, dict, int]:
    """Recompute all forty responses and twelve medians, including independent solo equality."""
    require(label in LABELS, 'unexpected label ' + label)
    prompts = validate_metadata(data, label)
    block, side = int(label[1:]), label[0]
    require(type(data.get('block')) is int and data['block'] == block and data.get('side') == side,
            label + ': block/side identity')
    require(data.get('block_order') == list(ORDERS[block - 1]), label + ': block order')
    require(data.get('cell_order') == [list(key) for key in cells_for(block)], label + ': cell schedule')
    require(data.get('seeds') == [1234, 1235], label + ': seeds')
    require(data.get('warmup_excluded') is True and data.get('solo_references_excluded') is True,
            label + ': exclusions')
    require(data.get('references') == [], label + ': unexpected old/plain references')
    close(data.get('client_deadline_seconds'), 900.0, label + ' client deadline')
    require(data.get('scope') == '256-token decode equivalence; prior 64-token plain gate remains separate',
            label + ': scope')
    fixtures, hashes = frozen_inputs()
    for field, digest in hashes.items():
        require(data.get(field) == digest, label + ': frozen ' + field)
    for name in FIXTURES:
        require(data['fixtures'][name]['messages'] == fixtures[name], label + ': public fixture differs')
    seen: dict[RequestKey, tuple[Any, ...]] = {}
    solo: dict[RequestKey, tuple[Any, ...]] = {}
    responses: list[dict] = []

    def observe(run: dict, key: RequestKey, tag: str, is_solo: bool = False) -> None:
        """Validate raw fields and compare independently of client equality flags."""
        validate_request(run, key, prompts[key[0]], label + ': ' + tag)
        fp = fingerprint(run)
        require(key not in seen or seen[key] == fp, label + ': repeated output differs ' + str(key))
        seen[key] = fp
        if is_solo:
            solo[key] = fp
        responses.append(run)

    observe(data['first_cold'], ('multiturn', 0.0, 1234, True), 'excluded warmup')
    refs = data.get('solo_references')
    require(isinstance(refs, list) and len(refs) == 3, label + ': solo reference count')
    previous = data['first_cold']['completed']
    for name, ref in zip(FIXTURES, refs):
        require((ref['fixture'], ref['temperature'], ref['seed']) == (name, 1.0, 1235),
                label + ': solo reference identity')
        observe(ref['run'], (name, 1.0, 1235, True), 'excluded solo', True)
        require(ref['run']['sent'] >= previous, label + ': reference chronology')
        previous = ref['run']['completed']
    cells = data.get('cells')
    require(isinstance(cells, list) and len(cells) == 12, label + ': twelve cells required')
    values = {}
    concurrent: list[tuple[RequestKey, tuple]] = []
    for expected, cell in zip(cells_for(block), cells):
        name, temp, n = expected
        require((cell['fixture'], cell['temperature'], cell['streams']) == expected,
                label + ': measured cell schedule')
        require(type(cell['streams']) is int and type(cell['temperature']) in (int, float),
                label + ': cell numeric types')
        require(cell.get('qualified') is True, label + ': unqualified cell')
        reps = cell.get('reps')
        require(isinstance(reps, list) and len(reps) == 2, label + ': two repetitions required')
        aggregate, e2e, ttft = [], [], []
        for ri, rep in enumerate(reps):
            require(type(rep.get('repetition')) is int and rep['repetition'] == ri, label + ': rep index')
            runs = rep.get('runs')
            require(isinstance(runs, list) and len(runs) == n, label + ': stream count')
            require(rep.get('comparisons') == [True] * n, label + ': client comparisons')
            for si, run in enumerate(runs):
                key = (name, temp, 1234 + si if temp else 1234, True)
                observe(run, key, 'measured ' + str(expected), n == 1)
                if n == 2:
                    concurrent.append((key, fingerprint(run)))
                ttft.append(run['first'] - run['sent'])
            require(min(r['sent'] for r in runs) >= previous, label + ': repetition chronology')
            previous = max(r['completed'] for r in runs)
            makespan = previous - min(r['sent'] for r in runs)
            span = max(r['last'] for r in runs) - min(r['first'] for r in runs)
            timing = rep.get('timing')
            require(isinstance(timing, dict) and timing.get('timing_valid') is True, label + ': timing flag')
            require(timing.get('requested_requests') == timing.get('timed_requests') == n,
                    label + ': timing count')
            close(timing.get('makespan_s'), makespan, label + ' makespan')
            close(timing.get('end_to_end_tps'), n * TOKENS / makespan, label + ' E2E')
            close(rep.get('aggregate_tps'), n * (TOKENS - 1) / span, label + ' aggregate')
            aggregate.append(n * (TOKENS - 1) / span)
            e2e.append(n * TOKENS / makespan)
        metrics = {'aggregate_decode_tps': statistics.median(aggregate),
                   'end_to_end_tps': statistics.median(e2e), 'ttft_s': statistics.median(ttft)}
        for field, metric in (('median_aggregate_tps', 'aggregate_decode_tps'),
                              ('median_end_to_end_tps', 'end_to_end_tps'),
                              ('median_request_ttft_s', 'ttft_s')):
            close(cell.get(field), metrics[metric], label + ' ' + field)
        values[expected] = metrics
    expected_keys = {(name, temp, seed, True) for name in FIXTURES for temp in (0.0, 1.0)
                     for seed in ([1234] if temp == 0 else [1234, 1235])}
    require(set(solo) == set(seen) == expected_keys, label + ': complete solo reference grid required')
    require(all(solo.get(key) == fp for key, fp in concurrent), label + ': solo/concurrent differs')
    require(len(responses) == 40, label + ': forty responses required')
    elapsed = data.get('elapsed_s')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 < elapsed <= 905,
            label + ': client elapsed/deadline')
    require(previous - data['first_cold']['sent'] <= elapsed + 1e-8, label + ': elapsed contradicts timestamps')
    return values, seen, len(responses)


def interval(ratios: list[float]) -> dict[str, Any]:
    """Use six block log ratios as the only independent statistical units."""
    require(len(ratios) == 6 and all(math.isfinite(r) and r > 0 for r in ratios), 'six positive ratios required')
    logs = [math.log(r) for r in ratios]
    mean = statistics.mean(logs)
    sd = statistics.stdev(logs)
    margin = T_CRITICAL * sd / math.sqrt(6)
    lower, upper = math.exp(mean - margin), math.exp(mean + margin)
    status = ('equivalent' if BOUNDS[0] <= lower and upper <= BOUNDS[1] else
              'not_equivalent' if upper < BOUNDS[0] or lower > BOUNDS[1] else 'inconclusive')
    return {'paired_ratios': ratios, 'mean_log_ratio': mean, 'sample_sd_log_ratio': sd,
            'geometric_mean_ratio': math.exp(mean), 'ci95_ratio': [lower, upper], 'status': status}


def identity(data: dict[str, dict], checked: dict[str, tuple]) -> None:
    """Check identical fixtures, helpers, tokenization and outputs over all supplied runs."""
    first = next(iter(data))
    for label in data:
        for field in ('fixtures', 'helper_sha256', 'tokenize', 'client_sha256', 'shared_client_sha256'):
            require(data[label][field] == data[first][field], label + ': cross-run ' + field + ' differs')
        require(checked[label][1] == checked[first][1], label + ': cross-run output differs')


def verify(directory: Path, partial: bool = False) -> dict[str, Any]:
    """Require the exact fixed sample before calculating any confidence interval."""
    paths = {p.stem: p for p in directory.glob('[AB]*.json')}
    require(paths and set(paths) <= set(LABELS), 'unexpected receipt labels')
    if not partial:
        require(set(paths) == set(LABELS), 'all twelve fixed receipts required; no interim intervals')
    raw = {label: paths[label].read_bytes() for label in LABELS if label in paths}
    data = {label: json.loads(blob) for label, blob in raw.items()}
    checked = {label: validate_run(run, label) for label, run in data.items()}
    identity(data, checked)
    result: dict[str, Any] = {
        'evidence_valid': True, 'analysis_complete': not partial, 'protocol_sha256': PROTOCOL_SHA256,
        'responses_per_run': {label: value[2] for label, value in checked.items()},
        'total_inference_responses': sum(value[2] for value in checked.values()), 'cells_per_run': 12,
        'fixture_helper_tokenize_and_output_equality': True,
        'input_sha256': {label: hashlib.sha256(blob).hexdigest() for label, blob in raw.items()},
        'plain_256_test': 'not_run; previous 64-token plain gate remains separate',
    }
    if partial:
        result['status'] = 'validated_evidence_only_no_intervals'
        return result
    for earlier, later in zip(LABELS, LABELS[1:]):
        finished = max(run['completed'] for cell in data[earlier]['cells']
                       for rep in cell['reps'] for run in rep['runs'])
        require(finished <= data[later]['first_cold']['sent'], 'fixed run execution order differs')
    rows = []
    for key in sorted(checked['A1'][0]):
        per_run = {label: checked[label][0][key] for label in LABELS}
        ratios = [per_run['B' + str(i)]['aggregate_decode_tps'] /
                  per_run['A' + str(i)]['aggregate_decode_tps'] for i in range(1, 7)]
        rows.append({'fixture': key[0], 'temperature': key[1], 'streams': key[2],
                     'per_run': per_run, **interval(ratios)})
    statuses = {row['status'] for row in rows}
    status = ('equivalent' if statuses == {'equivalent'} else
              'not_equivalent' if 'not_equivalent' in statuses else 'inconclusive')
    result.update({'status': status, 'gate_pass': status == 'equivalent', 'cells': rows,
                   'first_cold_excluded_ttft_s': {label: run['first_cold']['ttft_s'] for label, run in data.items()},
                   'method': 'Six paired log ratios of per-run median aggregate decode; df5 t95 interval.',
                   'global_test': 'All twelve per-cell intervals must be inside [0.98,1.02]; intersection-union.',
                   'limitations': 'Independent approximately normal block log ratios assumed; TTFT descriptive only.'})
    return result


def synthetic_run(label: str, speed: float = 1.0) -> dict[str, Any]:
    """Construct a complete deterministic offline receipt for validator rejection tests."""
    fixtures, hashes = frozen_inputs()
    block = int(label[1:])
    data: dict[str, Any] = {
        'schema_version': 1, 'label': label, 'qualified': True, 'errors': [],
        'requested_reply_tokens': TOKENS, 'repetitions': REPETITIONS,
        'levels': [1, 2], 'temperatures': [0.0, 1.0],
        'resume_test': 'not_run: prompt cache is required to be disabled',
        'helper_sha256': {name: 'a' * 64 for name in
                          ('bench_evidence.py', 'bench_timing.py', 'bench_metrics.py')},
        'fixtures': {}, 'tokenize': {}, 'references': [], 'cells': [],
        'block': block, 'side': label[0], 'block_order': list(ORDERS[block - 1]),
        'cell_order': [list(key) for key in cells_for(block)], 'seeds': [1234, 1235],
        'client_deadline_seconds': 900.0, 'warmup_excluded': True,
        'solo_references_excluded': True, 'solo_references': [],
        'scope': '256-token decode equivalence; prior 64-token plain gate remains separate', **hashes,
    }
    for name, messages in fixtures.items():
        raw = json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()
        data['fixtures'][name] = {'messages': messages, 'message_count': len(messages),
                                  'sha256': hashlib.sha256(raw).hexdigest()}
        data['tokenize'][name] = {phase: {'count': 100, 'token_ids': list(range(100)),
                                        'token_sha256': ids_hash(list(range(100)))}
                                  for phase in ('generation', 'history')}
    origin = 1000.0 + LABELS.index(label) * 1000.0
    clock = origin

    def request(name: str, temp: float, seed: int, sent: float) -> dict[str, Any]:
        """Produce consistent output fingerprints and independently calculable timestamps."""
        content, reasoning = 'synthetic output', ''
        ids = list(range(256))
        first, last = sent + 0.2, sent + 0.2 + 2 / speed
        times = [first + (last - first) * i / 3 for i in range(4)]
        return {
            'fixture': name, 'temperature': temp, 'seed': seed, 'draft': True,
            'complete': True, 'valid': True, 'error': None, 'validation_errors': [],
            'tokens': TOKENS, 'cached_tokens': 0, 'cache_state': 'cold', 'prompt_tokens': 100,
            'token_ids': ids, 'token_sha256': ids_hash(ids), 'token_sha': ids_hash(ids)[:12],
            'content': content, 'reasoning_content': reasoning,
            'output_sha256': hashlib.sha256(hashlib.sha256(content.encode()).digest() +
                                             hashlib.sha256(reasoning.encode()).digest()).hexdigest(),
            'sent': sent, 'first': first, 'last': last, 'completed': last + 0.01,
            'ttft_s': first - sent, 'decode_tps': 255 / (last - first),
            'elapsed_s': last + 0.01 - sent, 'pieces': [[at, 1] for at in times],
            'chunk_offsets_s': [at - sent for at in times],
        }

    data['first_cold'] = request('multiturn', 0.0, 1234, clock)
    clock = data['first_cold']['completed'] + 0.1
    for name in FIXTURES:
        run = request(name, 1.0, 1235, clock)
        data['solo_references'].append({'fixture': name, 'temperature': 1.0, 'seed': 1235, 'run': run})
        clock = run['completed'] + 0.1
    for name, temp, n in cells_for(block):
        reps = []
        for ri in range(2):
            runs = [request(name, temp, 1234 + si if temp else 1234, clock) for si in range(n)]
            clock = max(run['completed'] for run in runs) + 0.1
            makespan = max(r['completed'] for r in runs) - min(r['sent'] for r in runs)
            span = max(r['last'] for r in runs) - min(r['first'] for r in runs)
            reps.append({'repetition': ri, 'runs': runs, 'comparisons': [True] * n,
                         'aggregate_tps': n * 255 / span,
                         'timing': {'requested_requests': n, 'timed_requests': n, 'timing_valid': True,
                                    'makespan_s': makespan, 'end_to_end_tps': n * 256 / makespan}})
        data['cells'].append({
            'fixture': name, 'temperature': temp, 'streams': n, 'qualified': True, 'reps': reps,
            'median_aggregate_tps': statistics.median(rep['aggregate_tps'] for rep in reps),
            'median_end_to_end_tps': statistics.median(rep['timing']['end_to_end_tps'] for rep in reps),
            'median_request_ttft_s': statistics.median(r['ttft_s'] for rep in reps for r in rep['runs']),
        })
    data['elapsed_s'] = clock - origin
    return data


def self_test() -> dict[str, Any]:
    """Exercise complete fixed-sample inference and malformed evidence without any server."""
    import tempfile

    narrow = [0.999, 1.001, 1.0, 0.9995, 1.0005, 1.0]
    require(interval(narrow)['status'] == 'equivalent', 'narrow interval test')
    require(interval([0.9, 1.1, 0.95, 1.05, 0.98, 1.02])['status'] == 'inconclusive', 'wide interval test')
    require(interval([0.95] * 6)['status'] == 'not_equivalent', 'outside interval test')
    require(interval([1.03] * 6)['status'] == 'not_equivalent', 'faster outside interval test')
    result = interval([math.exp(v) for v in (-0.01, 0.0, 0.01, -0.01, 0.0, 0.01)])
    expected_margin = T_CRITICAL * math.sqrt(0.0004 / 5) / math.sqrt(6)
    close(result['ci95_ratio'][0], math.exp(-expected_margin), 'known lower interval')
    close(result['ci95_ratio'][1], math.exp(expected_margin), 'known upper interval')
    original = synthetic_run('A1')
    require(validate_run(original, 'A1')[2] == 40, 'valid synthetic receipt')
    mutations = [
        lambda d: d['first_cold'].update(cached_tokens=1),
        lambda d: d['first_cold'].update(tokens=255),
        lambda d: d['first_cold'].update(token_sha256='f' * 64),
        lambda d: d['first_cold'].update(output_sha256='f' * 64),
        lambda d: d['first_cold'].update(first=d['first_cold']['last']),
        lambda d: d['first_cold'].update(complete=False),
        lambda d: d['first_cold']['token_ids'].pop(),
        lambda d: d['cells'].pop(),
        lambda d: d['cells'][0]['reps'].pop(),
        lambda d: d['cells'][0].update(median_aggregate_tps=1.0),
        lambda d: d['cells'][0]['reps'][0].update(comparisons=[False]),
        lambda d: d['solo_references'].pop(),
        lambda d: d.update(warmup_excluded=False),
        lambda d: d.update(client_sha256='f' * 64),
        lambda d: d['fixtures']['prose'].update(sha256='f' * 64),
        lambda d: d['tokenize']['prose']['generation'].update(count=99),
        lambda d: d['cells'][0]['reps'][0]['runs'][0].update(seed=1235),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(original)
        mutate(broken)
        try:
            validate_run(broken, 'A1')
        except (ValueError, KeyError, TypeError):
            continue
        raise ValueError('malformed receipt unexpectedly accepted')
    with tempfile.TemporaryDirectory(prefix='tf-decode-validator-') as temporary:
        directory = Path(temporary)
        for label in LABELS:
            speed = narrow[int(label[1:]) - 1] if label[0] == 'B' else 1.0
            (directory / (label + '.json')).write_text(json.dumps(synthetic_run(label, speed)))
        full = verify(directory)
        require(full['gate_pass'] and full['total_inference_responses'] == 480, 'full fixed-sample test')
        (directory / 'B6.json').unlink()
        partial = verify(directory, True)
        require('cells' not in partial and 'gate_pass' not in partial, 'partial intervals leaked')
        try:
            verify(directory)
        except ValueError:
            pass
        else:
            raise ValueError('missing fixed receipt accepted')
        altered = synthetic_run('B5')
        altered['helper_sha256']['bench_metrics.py'] = 'b' * 64
        (directory / 'B5.json').write_text(json.dumps(altered))
        try:
            verify(directory, True)
        except ValueError:
            pass
        else:
            raise ValueError('cross-run metadata drift accepted')
    return {'self_test': 'passed', 'full_sample_responses': 480, 'cells': 12,
            'statistical_cases': 5, 'bad_receipt_rejections': len(mutations),
            'missing_sample_rejected': True, 'partial_intervals_absent': True,
            'cross_run_drift_rejected': True}


def main() -> int:
    """Validate one run, compare partial identity only, or evaluate the complete fixed experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-run', type=Path)
    parser.add_argument('--partial', action='store_true',
                        help='Validate supplied complete runs; never compute intervals.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
        elif args.validate_run:
            data = json.loads(args.validate_run.read_bytes())
            cells, _, count = validate_run(data, args.validate_run.stem)
            result = {'evidence_valid': True, 'analysis_complete': False,
                      'status': 'validated_run_only_no_intervals', 'responses': count, 'cells_per_run': len(cells)}
        else:
            require(args.directory is not None, 'directory or --validate-run or --self-test is required')
            result = verify(args.directory, args.partial)
    except (ValueError, KeyError, TypeError, OSError, IndexError, OverflowError) as error:
        result = {'evidence_valid': False, 'analysis_complete': False, 'status': 'invalid',
                  'error': type(error).__name__ + ': ' + str(error)}
    raw = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(raw)
    print(raw, end='')
    if result.get('evidence_valid') is False:
        return 1
    return 0 if result.get('gate_pass', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
