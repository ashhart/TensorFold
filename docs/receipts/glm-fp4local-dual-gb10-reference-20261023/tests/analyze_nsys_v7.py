#!/usr/bin/env python3
"""Read-only, dependency-free summary of a bounded Nsight Systems SQLite export.

This reports traced activity union/gaps, NOT SM occupancy or an exact critical
path. API correlation supplies evidence about late submission; it cannot prove
CPU causation, recover network dependencies or align clocks across machines.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys


def columns(db, table):
    return {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}


def percentile(values, pct):
    if not values:
        return None
    v = sorted(values)
    return v[round((len(v) - 1) * pct)]


def interval_summary(events, low, high, min_gap):
    """Union across all streams; include leading/trailing gaps separately."""
    intervals = [(max(low, e['start']), min(high, e['end']), e)
                 for e in events if e['end'] > low and e['start'] < high]
    # Use explicit key because dictionaries are not ordered for equal times.
    intervals.sort(key=lambda q: (q[0], q[1]))
    busy, gaps, cursor = 0, [], low
    for begin, end, event in intervals:
        if begin > cursor:
            gaps.append({'start_ns': cursor, 'end_ns': begin,
                         'duration_ns': begin - cursor, 'next': event})
        if end > cursor:
            busy += end - max(begin, cursor)
            cursor = end
    trailing = max(0, high - cursor)
    return busy, [g for g in gaps if g['duration_ns'] >= min_gap], trailing


def load_events(db, max_rows):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(db.execute('SELECT id,value FROM StringIds')) if 'StringIds' in tables else {}
    events, apis, notes = [], [], []
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY',
                  'CUPTI_ACTIVITY_KIND_MEMSET'):
        if table not in tables:
            continue
        cols = columns(db, table)
        if not {'start', 'end', 'deviceId'} <= cols:
            notes.append(f'{table}: unexpected schema, skipped')
            continue
        for row in db.execute(f'SELECT * FROM "{table}"'):
            r = dict(row)
            if len(events) >= max_rows:
                raise ValueError(f'more than {max_rows} GPU events; use a shorter capture or explicitly raise --max-rows')
            if r['end'] < r['start']:
                continue
            kind = table.rsplit('_', 1)[-1].lower()
            sid = next((r[x] for x in ('demangledName', 'shortName', 'mangledName', 'nameId')
                        if x in r and r[x] is not None), None)
            name = strings.get(sid, str(sid)) if sid is not None else kind
            events.append({'kind': kind, 'name': name, 'start': r['start'], 'end': r['end'],
                           'device': r['deviceId'], 'stream': r.get('streamId'),
                           'pid': r.get('globalPid'), 'correlation': r.get('correlationId'),
                           'grid': [r.get(k) for k in ('gridX', 'gridY', 'gridZ')],
                           'block': [r.get(k) for k in ('blockX', 'blockY', 'blockZ')]})
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        cols = columns(db, table)
        if not {'start', 'end', 'nameId'} <= cols:
            continue
        for row in db.execute(f'SELECT * FROM "{table}"'):
            r = dict(row)
            if len(apis) >= max_rows * 4:
                raise ValueError('too many CUDA API events for bounded analysis')
            tid = r.get('globalTid')
            apis.append({'name': strings.get(r['nameId'], str(r['nameId'])),
                         'start': r['start'], 'end': r['end'], 'tid': tid,
                         'pid': (tid & 0xFFFFFFFFFF000000) if tid is not None else None,
                         'correlation': r.get('correlationId'), 'table': table})
    if not events:
        raise ValueError('no GPU activity found; capture injection may have failed')
    return events, apis, notes


def summarize(db, args):
    events, apis, notes = load_events(db, args.max_rows)
    first = min(e['start'] for e in events)
    last = max(e['end'] for e in events)
    low = first + round(args.start_ms * 1e6)
    high = min(last, first + round(args.end_ms * 1e6)) if args.end_ms is not None else last
    if high <= low:
        raise ValueError('empty time interval')
    # Never correlate across processes solely by correlationId.
    launch = defaultdict(list)
    for a in apis:
        if a['correlation'] is not None and re.search(r'Launch|Graph', a['name']):
            launch[(a['pid'], a['correlation'])].append(a)
    result = {'window': {'first_gpu_ns': first, 'start_ns': low, 'end_ns': high,
                         'duration_ms': (high - low) / 1e6}, 'devices': [], 'notes': notes,
              'limitations': [
                  'Kernel duration sums overlap across streams; activity union avoids double counting.',
                  'GPU activity gaps are gaps in this traced process, not proof that the whole GPU was idle.',
                  'Launch-late classification is evidence only, not an exact CPU/GPU/network critical path.',
                  'No host clock alignment across ranks is inferred. Use within-rank durations or explicit synchronization evidence.',
                  'Capture and CUDA-event instrumentation may change timing; compare an unprofiled control.',
                  'No GPU memory-bandwidth, SM occupancy or DRAM-byte attribution is possible from this trace alone.',
              ]}
    for device in sorted({e['device'] for e in events}):
        selected = [e for e in events if e['device'] == device and e['end'] > low and e['start'] < high]
        busy, gaps, trailing = interval_summary(selected, low, high, round(args.gap_us * 1000))
        kernels = defaultdict(list)
        launch_durations, queue_delays = [], []
        matched = 0
        for e in selected:
            if e['kind'] != 'kernel':
                continue
            kernels[(e['name'], tuple(e['grid']), tuple(e['block']))].append(min(high, e['end']) - max(low, e['start']))
            matches = launch.get((e['pid'], e['correlation']), [])
            if matches:
                a = min(matches, key=lambda q: abs(e['start'] - q['end']))
                matched += 1
                launch_durations.append(a['end'] - a['start'])
                queue_delays.append(max(0, e['start'] - a['end']))
        grouped = []
        for (name, grid, block), durations in kernels.items():
            grouped.append({'name': name, 'grid': grid, 'block': block, 'count': len(durations),
                            'sum_ms': sum(durations) / 1e6, 'mean_us': statistics.mean(durations) / 1000,
                            'p50_us': percentile(durations, .5) / 1000, 'p95_us': percentile(durations, .95) / 1000})
        grouped.sort(key=lambda q: q['sum_ms'], reverse=True)
        classified = defaultdict(lambda: {'count': 0, 'sum_ms': 0.0})
        for g in gaps:
            e = g.pop('next')
            g['next_name'], g['next_kind'], g['next_stream'] = e['name'], e['kind'], e['stream']
            matches = launch.get((e['pid'], e['correlation']), [])
            label = 'unmatched_or_non_kernel'
            if e['kind'] == 'kernel' and matches:
                a = min(matches, key=lambda q: abs(e['start'] - q['end']))
                g['api_name'], g['api_start_ns'], g['api_end_ns'] = a['name'], a['start'], a['end']
                if a['start'] >= g['start_ns']:
                    label = 'next_launch_started_during_gap'
                elif a['end'] <= g['start_ns']:
                    label = 'next_launch_already_returned_before_gap'
                else:
                    label = 'next_launch_overlaps_gap'
            g['classification'] = label
            classified[label]['count'] += 1
            classified[label]['sum_ms'] += g['duration_ns'] / 1e6
        result['devices'].append({'device': device, 'traced_busy_union_ms': busy / 1e6,
                                  'traced_gap_total_ms': (high - low - busy) / 1e6,
                                  'trailing_gap_ms': trailing / 1e6,
                                  'kernel_count': sum(k['count'] for k in grouped), 'api_matched_kernel_count': matched,
                                  'launch_api_p50_us': percentile(launch_durations, .5) / 1000 if launch_durations else None,
                                  'launch_api_p95_us': percentile(launch_durations, .95) / 1000 if launch_durations else None,
                                  'queue_delay_p50_us': percentile(queue_delays, .5) / 1000 if queue_delays else None,
                                  'top_kernels_by_duration_sum': grouped[:args.top],
                                  'gaps_at_least_us': args.gap_us, 'gap_classification': dict(classified),
                                  'largest_gaps': sorted(gaps, key=lambda q: q['duration_ns'], reverse=True)[:args.top]})
    api_totals = defaultdict(list)
    for a in apis:
        if a['end'] > low and a['start'] < high:
            api_totals[a['name']].append(min(high, a['end']) - max(low, a['start']))
    result['api_sums_not_exclusive_wall'] = sorted(
        ({'name': n, 'count': len(v), 'sum_ms': sum(v) / 1e6, 'p95_us': percentile(v, .95) / 1000}
         for n, v in api_totals.items()), key=lambda q: q['sum_ms'], reverse=True)[:args.top]
    return result


def self_test():
    ev = [dict(start=0, end=10), dict(start=5, end=12), dict(start=20, end=30)]
    busy, gaps, tail = interval_summary(ev, 0, 35, 1)
    assert (busy, tail) == (22, 5)
    assert [g['duration_ns'] for g in gaps] == [8]
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT);'
                     'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);'
                     'CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER,correlationId INTEGER);')
    db.executemany('INSERT INTO StringIds VALUES(?,?)', [(1, 'first'), (2, 'second'), (3, 'cuLaunchKernel')])
    gp = 17 << 24
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?)', [(0, 10000, 0, 7, gp, 1, 1), (30000, 40000, 0, 7, gp, 2, 2)])
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(?,?,?,?,?)', [(-1000, -100, 3, gp + 8, 1), (20000, 21000, 3, gp + 8, 2)])
    args = argparse.Namespace(max_rows=100, start_ms=0, end_ms=None, gap_us=1, top=20)
    s = summarize(db, args)['devices'][0]
    assert s['traced_busy_union_ms'] == .02 and s['traced_gap_total_ms'] == .02
    assert s['api_matched_kernel_count'] == 2
    assert s['largest_gaps'][0]['classification'] == 'next_launch_started_during_gap'
    # Same timestamp events must not compare dictionaries during sorting.
    interval_summary([dict(start=0, end=10), dict(start=0, end=10)], 0, 10, 1)
    print('self-test passed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite', nargs='?', type=Path)
    p.add_argument('--start-ms', type=float, default=0, help='relative to first traced GPU event')
    p.add_argument('--end-ms', type=float)
    p.add_argument('--gap-us', type=float, default=20)
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--max-rows', type=int, default=500000)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if args.sqlite is None:
        p.error('provide a SQLite export or --self-test')
    db = sqlite3.connect(args.sqlite.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    try:
        print(json.dumps(summarize(db, args), ensure_ascii=False, indent=2))
    finally:
        db.close()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, sqlite3.Error) as exc:
        sys.exit(str(exc))
