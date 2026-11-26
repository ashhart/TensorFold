#!/usr/bin/env python3
"""Validate opt-in CUDA round JSONL and keep event spans separate from complete rounds."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

OPS = ['other', 'qmmf', 'attention', 'logits_sampling', 'h2d', 'd2h', 'd2d', 'memset']
PHASES = ['other', 'forward', 'land', 'proposal', 'proposal_hold', 'proposal_select',
          'candidate_frame', 'token_readback', 'logits_readback', 'keep_rollback', 'state_publication']
VERIFY = {'forward', 'land', 'token_readback', 'logits_readback'}
PROPOSAL = {'proposal', 'proposal_hold', 'proposal_select', 'candidate_frame'}


class InvalidTrace(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise InvalidTrace(why)


def integer(n, why, minimum=0):
    require(type(n) is int and minimum <= n < 2**64, why)
    return n


def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def loads(text):
    def invalid(value):
        raise InvalidTrace('non-finite JSON ' + value)
    return json.loads(text, parse_constant=invalid)


def union(spans):
    total = 0
    left = right = 0
    for first, last in sorted(spans):
        integer(first, 'invalid interval start')
        integer(last, 'invalid interval end')
        require(first <= last, 'backwards interval')
        if first > right:
            total += right - left
            left = first
        right = max(right, last)
    return total + right - left


def calculate(record):
    b, ds, hs = record['budget'], record['device'], record['host']
    wall = integer(b['complete_round_wall_ns'], 'missing complete-round wall', 1)
    functional = integer(b['functional_return_ns'], 'missing functional boundary')
    envelope = integer(b['device_envelope_ns'], 'missing device envelope')
    require(functional <= wall, 'functional return after complete round')
    for events, bound in ((ds, envelope), (hs, wall)):
        for d in events:
            require(integer(d['end_ns'], 'invalid interval end') <= bound, 'event outside round')
            require(integer(d['begin_ns'], 'invalid interval start') <= d['end_ns'], 'backwards interval')
    interval = lambda d: (d['begin_ns'], d['end_ns'])
    covered = union(map(interval, ds))
    return dict(complete_round_wall_ns=wall, functional_return_ns=functional,
                device_envelope_ns=envelope, device_event_span_union_ns=covered,
                kernel_event_span_sum_ns=sum(d['end_ns'] - d['begin_ns'] for d in ds if d['op'] in OPS[:4]),
                device_envelope_gap_ns=envelope - covered,
                target_verify_device_event_union_ns=union(interval(d) for d in ds if d['phase'] in VERIFY),
                proposal_device_event_union_ns=union(interval(d) for d in ds if d['phase'] in PROPOSAL),
                device_by_operation_union_ns=[union(interval(d) for d in ds if d['op'] == op) for op in OPS],
                device_by_phase_union_ns=[union(interval(d) for d in ds if d['phase'] == p) for p in PHASES],
                host_wait_union_ns=union(interval(h) for h in hs if h['kind'] == 'wait'),
                state_publication_union_ns=union(interval(h) for h in hs if h['kind'] == 'span' and h['phase'] == 'state_publication'),
                attribution_drain_union_ns=union(interval(h) for h in hs if h['kind'] == 'attribution_drain'))


def validate(r, header):
    require(r['schema'] == 'tensorfold.cuda-round.v1', 'unsupported schema')
    require(r['complete'] is True and r['problem'] is None, 'partial or failed receipt')
    require(r['requests'] and r['windows'] and r['frames'] and r['device'], 'empty round')
    require(header['operation_names'] == OPS and header['phase_names'] == PHASES, 'unknown budget indices')
    require(sha(header['binary_sha256']) and sha(header['checkpoint']['checkpoint_sha256']), 'incomplete binary/checkpoint identity')
    require(re.fullmatch('[0-9a-f]{40}:[0-9a-f]{40}', header['source']) is not None, 'invalid source attestation')
    require('quantization' in header and 'conditions' in header, 'incomplete run identity')
    requests = {q['stream']: q for q in r['requests']}
    require(len(requests) == len(r['requests']), 'ambiguous stream identity')
    for q in requests.values():
        require(sha(q['prompt_sha256']), 'missing actual prompt identity')
        sampling = loads(q['sampling_json'])
        require(set(sampling) == {'sampling', 'penalties', 'drafts'}, 'missing sampling settings')
        integer(q['cached'], 'invalid cached count')
    for f in r['frames']:
        require(0 < f['live_rows'] <= f['shape_bucket'] and 0 < f['windows'] <= f['live_rows'], 'invalid frame')
        require(type(f['mt2']) is bool, 'missing actual MT2 selection')
    require(sum(f['live_rows'] for f in r['frames']) == sum(w['offered_rows'] for w in r['windows']), 'frames do not cover all windows')
    require(sum(f['windows'] for f in r['frames']) == len(r['windows']), 'frame/window mismatch')
    for w in r['windows']:
        require(w['stream'] in requests, 'unbound committed stream')
        offered = integer(w['offered_rows'], 'invalid offered rows', 1)
        path = w['kept_path']
        require(path and path[0] == 0 and all(type(i) is int and 0 <= i < offered for i in path), 'invalid keep path')
        require(all(a < b for a, b in zip(path, path[1:])), 'invalid connected keep order')
        require(w['connected_accepted_depth'] == (0 if w['kind'] == 'forced' else len(path) - 1), 'depth is not connected accepted depth')
        require(w['proposal_nodes'] == (0 if w['kind'] == 'forced' else offered - 1), 'invalid proposal count')
        require(w['rolled_back_rows'] == offered - len(path), 'invalid rollback count')
        require(len(w['committed_ids']) <= len(path), 'commit exceeds keep path')
        for token in w['committed_ids']:
            require(integer(token, 'invalid committed ID') < 2**32, 'invalid token')
        integer(w['cached_after'], 'missing cache publication')
        if w['pending_after'] is not None:
            integer(w['pending_after'], 'invalid pending publication')
    hosts = {h['correlation']: h for h in r['host']}
    require(len(hosts) == len(r['host']), 'duplicate host correlation')
    seen = set()
    for d in r['device']:
        require(d['correlation'] not in seen, 'duplicate device correlation')
        seen.add(d['correlation'])
        require(d['op'] in OPS and d['phase'] in PHASES, 'unknown operation or phase')
        scope = d['parent_scope']
        require(scope in hosts and hosts[scope]['kind'] == 'span', 'unjoined device scope')
        require((d['graph_replay'] if d['graph_replay'] is not None else d['correlation']) in hosts, 'unjoined device launch')
        if d['op'] == 'qmmf':
            require(d['qmmf'], 'unresolved QMMF descriptor')
            for q in d['qmmf']:
                for key in ('m', 'n', 'k', 'sk'):
                    integer(q[key], 'invalid QMMF ' + key, 1)
                require(len(q['cluster']) == 3 and all(type(x) is int and x > 0 for x in q['cluster']), 'invalid cluster')
                require(type(q['reduce']) is bool and q['mode'] != 'unresolved', 'invalid QMMF reduction/mode')
    computed = calculate(r)
    require(computed == r['budget'], 'budget differs from independently recomputed intervals')
    return r


def read(lines):
    headers, rounds, seen = {}, [], set()
    for line in lines:
        if line.startswith('TF_ROUND_ERROR '):
            raise InvalidTrace(line.strip())
        if line.startswith('TF_ROUND_IDENTITY '):
            _, key, text = line.rstrip('\n').split(' ', 2)
            require(hashlib.sha256(text.encode()).hexdigest() == key, 'header digest mismatch')
            headers[key] = loads(text)
        elif line.startswith('TF_ROUND '):
            r = loads(line[len('TF_ROUND '):])
            key = r['identity']
            require(key in headers, 'unbound round identity')
            require(r['round'] not in seen, 'duplicate round')
            seen.add(integer(r['round'], 'invalid round ID', 1))
            rounds.append(validate(r, headers[key]))
    require(rounds, 'no complete round receipts')
    return headers, rounds


def summary(data):
    headers, rs = data
    device = sum(r['budget']['device_event_span_union_ns'] for r in rs)
    qmmf = sum(r['budget']['device_by_operation_union_ns'][OPS.index('qmmf')] for r in rs)
    return dict(rounds=len(rs), identity=list(headers),
                median_complete_round_ns=statistics.median(r['budget']['complete_round_wall_ns'] for r in rs),
                median_functional_return_ns=statistics.median(r['budget']['functional_return_ns'] for r in rs),
                median_device_envelope_ns=statistics.median(r['budget']['device_envelope_ns'] for r in rs),
                qmmf_event_span_share=(qmmf / device if device else None),
                actual_mt2_on_frames=sum(f['mt2'] for r in rs for f in r['frames']),
                committed_tokens=sum(len(w['committed_ids']) for r in rs for w in r['windows']),
                accepted_depth=sum(w['connected_accepted_depth'] for r in rs for w in r['windows']),
                qmmf_60_percent_gate='UNQUALIFIED: event brackets are not profiler kernel-busy intervals',
                exactness='UNQUALIFIED: compare trace-off IDs and continued device state separately')


def compare(on, off):
    a, b = summary(on), summary(off)
    ha, ra = on
    hb, rb = off
    require([r['round'] for r in ra] == [r['round'] for r in rb],
            'ordered round IDs differ; do not compare unmatched medians')
    for x, y in zip(ra, rb):
        ix, iy = ha[x['identity']], hb[y['identity']]
        for key in ('source', 'binary_sha256', 'checkpoint', 'drafter', 'quantization', 'kv8', 'geometry', 'conditions',
                    'driver', 'sm_major', 'sm_minor', 'checkpoint_config_sha256'):
            require(ix[key] == iy[key], 'A/B identity differs: ' + key)
        def normalized(r):
            order = {q['stream']: i for i, q in enumerate(r['requests'])}
            reqs = [{k: v for k, v in q.items() if k != 'stream'} for q in r['requests']]
            windows = [dict(w, stream=order[w['stream']]) for w in r['windows']]
            return reqs, windows
        require(normalized(x) == normalized(y), 'A/B token or published host state mismatch')
        def frames(r):
            return [dict({k: v for k, v in f.items() if k not in ('correlation', 'mt2', 'shape_key')},
                         shape_key=f['shape_key'] & ~(1 << 29)) for f in r['frames']]
        require(frames(x) == frames(y), 'A/B frame workload differs')
        require(all(f['mt2'] for f in x['frames']) and not any(f['mt2'] for f in y['frames']), 'A/B did not select actual MT2 on/off')
    gain = 1 - a['median_complete_round_ns'] / b['median_complete_round_ns']
    return dict(mt2_instrumented_round_gain=gain,
                experiment_decision='do not invest in fusion' if gain < 0.05 else 'candidate only; exact state and overhead proof still required',
                release_gate='UNQUALIFIED', on=a, off=b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--mt2-off', type=Path)
    args = p.parse_args()
    try:
        data = read(args.log.read_text().splitlines(keepends=True))
        result = compare(data, read(args.mt2_off.read_text().splitlines(keepends=True))) if args.mt2_off else summary(data)
    except (InvalidTrace, KeyError, TypeError, OSError, ValueError) as error:
        p.exit(2, 'REFUSED: ' + str(error) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
