#!/usr/bin/env python3
"""Read-only, all-gap audit and conservative kernel-family grouping of v7 traces.

Durations are per-rank; no clock alignment, SM occupancy or bandwidth inference.
Family sums can overlap. KDA projection identification requires a source-matched
local sequence; generic linear names alone do not identify their model owner.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

from analyze_nsys_v7 import interval_summary, load_events


def family(name):
    if "tf_nvfp4_experts::nvfp4_expert_kernel" in name:
        return "routed_nvfp4_expert"
    if name.startswith("roce_") or name.startswith("nccl"):
        return "communication_named"
    if "tf_qmmf::" in name:
        return "packed_linear_qmmf_unattributed"
    if "tf_qmm_group::" in name:
        return "grouped_linear_qmm_unattributed"
    if name.startswith("glm_mm"):
        return "bf16_linear_glm_mm_unattributed"
    if name.startswith("tf_glm_kda_"):
        return "kda_state_named"
    if name.startswith("df_"):
        return "dflash_named_only_not_full_drafter"
    return "other_named"


def gap_band(ns):
    if ns < 1000:
        return "0-1us"
    if ns < 5000:
        return "1-5us"
    if ns < 20000:
        return "5-20us"
    if ns < 100000:
        return "20-100us"
    return ">=100us"


def kda_roles(kernels):
    """Recognize the current non-direct KDA projection sequence, not by shape alone."""
    roles, hits, failures = {}, 0, 0
    grouped = defaultdict(list)
    for e in kernels:
        grouped[(e['device'], e['stream'])].append(e)
    for events in grouped.values():
        events.sort(key=lambda e: e['start'])
        for i, e in enumerate(events):
            if e['name'] != 'tf_glm_kda_prep':
                continue
            # Optional reduction follows the small rest projection. Other paths
            # remain unattributed rather than being inferred from grid alone.
            start = None
            for n in (7, 6):
                if i < n:
                    continue
                seq = events[i - n:i]
                names = [x['name'] for x in seq]
                middle = ['glm_mm16_16', 'glm_mm_reduce'] if n == 7 else ['glm_mm16_16']
                if ('tf_qmmf::qmmf_kernel<(int)2,' in names[0]
                        and names[1] == 'tf_glm_copy_rows'
                        and names[2:-3] == middle
                        and names[-3:] == ['tf_glm_copy_rows', 'glm_mm16_16', 'glm_mm16_16']):
                    start = i - n
                    break
            if start is None:
                failures += 1
                continue
            hits += 1
            for item in events[start:i]:
                roles[id(item)] = 'kda_input_projection_sequence'
            if (i + 3 < len(events)
                    and events[i + 1]['name'] == 'tf_glm_kda_step'
                    and events[i + 2]['name'] == 'tf_glm_kda_out'
                    and 'tf_qmmf::qmmf_kernel<(int)2,' in events[i + 3]['name']):
                roles[id(events[i + 3])] = 'kda_output_projection_sequence'
    return roles, hits, failures


def summarize(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    try:
        events, apis, notes = load_events(db, 1000000)
    finally:
        db.close()
    launch = defaultdict(list)
    for api in apis:
        if api['correlation'] is not None and re.search('Launch|Graph', api['name']):
            launch[(api['pid'], api['correlation'])].append(api)
    out = {'source': str(path), 'devices': [], 'notes': notes,
           'limitations': ['GPU activity coverage is not SM occupancy.',
                           'All positive gaps are retained; family duration sums can overlap.',
                           'Per-rank time only; no cross-node clock alignment.',
                           'DFlash named kernels exclude shared linear/communication names.',
                           'KDA roles require the exact local kernel order in cuda_kda_block.zig.',
                           'Launch timing evidence does not prove the root cause of a GPU gap.']}
    for device in sorted({e['device'] for e in events}):
        selected = [e for e in events if e['device'] == device]
        lo = min(e['start'] for e in selected)
        hi = max(e['end'] for e in selected)
        duration = hi - lo
        busy, gaps, trailing = interval_summary(selected, lo, hi, 1)
        kernels = [e for e in selected if e['kind'] == 'kernel']
        kernel_busy, _, _ = interval_summary(kernels, lo, hi, 1)
        roles, kda_hits, kda_failures = kda_roles(kernels)
        bins, labels, combined = defaultdict(list), defaultdict(list), defaultdict(list)
        for g in gaps:
            ns, e = g['duration_ns'], g['next']
            matches = launch.get((e['pid'], e['correlation']), [])
            label = 'unmatched_or_non_kernel'
            if e['kind'] == 'kernel' and matches:
                api = min(matches, key=lambda a: abs(e['start'] - a['end']))
                if api['start'] >= g['start_ns']:
                    label = 'launch_started_after_gap_begin'
                elif api['end'] <= g['start_ns']:
                    label = 'launch_already_returned_before_gap'
                else:
                    label = 'launch_overlaps_gap'
            bins[gap_band(ns)].append(ns)
            labels[label].append(ns)
            combined[(gap_band(ns), label)].append(ns)
        families = defaultdict(list)
        for e in kernels:
            families[roles.get(id(e), family(e['name']))].append(e)
        def durations(values):
            return {'count': len(values), 'sum_ms': sum(values) / 1e6,
                    'window_pct': 100 * sum(values) / duration}
        grouped = []
        for name, members in families.items():
            total = sum(e['end'] - e['start'] for e in members)
            union, _, _ = interval_summary(members, lo, hi, 1)
            grouped.append({'family': name, 'count': len(members), 'sum_ms': total / 1e6,
                            'union_ms': union / 1e6, 'sum_window_pct': 100 * total / duration})
        out['devices'].append({
            'device': device, 'start_ns': lo, 'end_ns': hi, 'window_ms': duration / 1e6,
            'activity_union_ms': busy / 1e6, 'activity_union_pct': 100 * busy / duration,
            'kernel_union_ms': kernel_busy / 1e6, 'kernel_union_pct': 100 * kernel_busy / duration,
            'all_gap_ms': (duration - busy) / 1e6, 'all_gap_count': len(gaps),
            'trailing_ms': trailing / 1e6,
            'gap_bins': {k: durations(v) for k, v in bins.items()},
            'gap_launch_evidence': {k: durations(v) for k, v in labels.items()},
            'gap_bin_launch_evidence': [{'bin': k[0], 'class': k[1], **durations(v)}
                                        for k, v in sorted(combined.items())],
            'kernel_families': sorted(grouped, key=lambda x: x['sum_ms'], reverse=True),
            'kda_sequence_matches': kda_hits, 'kda_sequence_unmatched': kda_failures,
        })
    return out


def self_test():
    assert [gap_band(n) for n in (1, 1000, 4999, 5000, 19999, 20000, 100000)] == [
        '0-1us', '1-5us', '1-5us', '5-20us', '5-20us', '20-100us', '>=100us']
    names = ['void tf_qmmf::qmmf_kernel<(int)2,', 'tf_glm_copy_rows', 'glm_mm16_16',
             'glm_mm_reduce', 'tf_glm_copy_rows', 'glm_mm16_16', 'glm_mm16_16',
             'tf_glm_kda_prep', 'tf_glm_kda_step', 'tf_glm_kda_out',
             'void tf_qmmf::qmmf_kernel<(int)2,']
    events = [dict(name=n, start=i, device=0, stream=0) for i, n in enumerate(names)]
    roles, hits, failures = kda_roles(events)
    assert (len(roles), hits, failures) == (8, 1, 0)
    events[2]['name'] = 'not_a_projection'
    assert kda_roles(events)[1:] == (0, 1)
    print('self-test passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', nargs='*', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps([summarize(p) for p in args.sqlite], ensure_ascii=False, indent=2))
