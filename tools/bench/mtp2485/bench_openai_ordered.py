#!/usr/bin/env python3
"""Run the PR-receipt single-stream client main / MTP / main on the same GPU."""
import json
from pathlib import Path
import re
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_receipt as rr  # noqa: E402
OUT = rr.OUT

EXPECTED_KEYS = {(name, temp) for name in ('fibonacci-raw', 'gpu-chat-no-think') for temp in (0.0, 1.0)}
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
PARALLEL = 1


def start_variant(label, binary, mtp, tree_mode, env):
    """Use the same one-stream limit for every cell in this correctness receipt."""
    run_env = env.copy()
    run_env.pop('TF_MTP_TREES', None)
    if tree_mode:
        if not mtp:
            raise ValueError('MTP tree mode requires the MTP drafter')
        run_env['TF_MTP_TREES'] = '1'
    return rr.start_server(label, binary, mtp, 1024, PARALLEL, run_env)


def run_client(label, url):
    output = OUT / f'{label}.openai.json'
    log = OUT / f'{label}.openai.log'
    cmd = [sys.executable, str(rr.PROJECT / 'tools/bench_openai.py'), url, rr.MODEL,
           '--tokens', '128', '--reps', '3', '--temperatures', '0,1.0',
           '--label', label, '--output', str(output)]
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=rr.PROJECT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            print(line, end='', flush=True)
        rc = p.wait()
    if rc:
        raise RuntimeError(f'bench_openai exited {rc} for {label}; see {log}')
    return output


def read_hashes(path):
    data = json.loads(path.read_text())
    out = {}
    for row in data:
        key = (row['prompt'], float(row['temperature']))
        if key in out:
            raise ValueError(f'duplicate prompt/temperature evidence row: {key}')
        evidence = row.get('evidence')
        if not isinstance(evidence, list) or len(evidence) != 3:
            raise ValueError(f'{key}: expected exactly three output hashes, found {len(evidence) if isinstance(evidence, list) else 0}')
        hashes = [entry.get('output_sha256') for entry in evidence]
        if any(not isinstance(value, str) or not SHA256_RE.fullmatch(value) for value in hashes):
            raise ValueError(f'{key}: evidence contains a missing or malformed SHA-256')
        out[key] = hashes
    if set(out) != EXPECTED_KEYS:
        raise ValueError(f'{path}: prompt/temperature coverage mismatch; missing={sorted(EXPECTED_KEYS-set(out))}, unexpected={sorted(set(out)-EXPECTED_KEYS)}')
    return data, out


def main(argv=None):
    global OUT
    rr.parse_config(__doc__ or 'Ordered main/MTP/main receipt', 'bench-openai-ordered', argv)
    OUT = rr.OUT
    preflight = rr.gpu_preflight()
    env, runtime_environment = rr.benchmark_env()
    specs = [
        ('openai-main-a', rr.BASE / 'zig-out/native/bin/tensorfold-native', False, False),
        ('openai-mtp3', rr.PROJECT / 'zig-out/native/bin/tensorfold-native', True, False),
        ('openai-mtp3-tree', rr.PROJECT / 'zig-out/native/bin/tensorfold-native', True, True),
        ('openai-main-b', rr.BASE / 'zig-out/native/bin/tensorfold-native', False, False),
    ]
    results = {}
    for label, binary, mtp, tree_mode in specs:
        print(f'BEGIN {label} mtp={mtp} tree_mode={tree_mode}', flush=True)
        proc, f, mon, url, ready, log = start_variant(label, binary, mtp, tree_mode, env)
        item = {'startup_s': ready, 'server_log': str(log)}
        try:
            output = run_client(label, url)
            item['report'] = str(output)
        finally:
            item['memory'] = rr.stop_server(proc, f, mon)
        item['mtp_requested'] = mtp
        item['mtp_tree_mode'] = tree_mode
        results[label] = item
        print(f'END {label}', flush=True)
    reports = {name: read_hashes(OUT/f'{name}.openai.json') for name, _, _, _ in specs}
    a, b, m, tree = (reports[name][1] for name in ('openai-main-a','openai-main-b','openai-mtp3','openai-mtp3-tree'))
    checks = []
    for key in sorted(EXPECTED_KEYS, key=str):
        checks.append({'key': list(key), 'main_a_equals_mtp': key in a and key in m and a[key] == m[key],
                       'main_b_equals_mtp': key in b and key in m and b[key] == m[key],
                       'main_a_equals_mtp_tree': key in a and key in tree and a[key] == tree[key],
                       'main_b_equals_mtp_tree': key in b and key in tree and b[key] == tree[key],
                       'reps': len(a[key]), 'main_a_sha256': a[key], 'mtp_sha256': m[key],
                       'mtp_tree_sha256': tree[key], 'main_b_sha256': b[key]})
    summary = {'gpu': preflight,
               'baseline_commit': rr.git(rr.BASE, 'rev-parse', 'HEAD'),
               'candidate_commit': rr.git(rr.PROJECT, 'rev-parse', 'HEAD'),
               'baseline_git_state': rr.git_state(rr.BASE),
               'candidate_git_state': rr.git_state(rr.PROJECT),
               'checkpoint': str(rr.TARGET),
               'checkpoint_metadata_sha256': rr.checkpoint_metadata_hashes(),
               'binary_sha256': {label: rr.sha256(binary) for label, binary, _, _ in specs},
               'runtime_environment': runtime_environment,
               'order': [x[0] for x in specs], 'settings': {'tokens':128,'reps':3,
               'temperatures':[0,1.0], 'top_p':0.95,'top_k':20,'context':1024,'parallel':PARALLEL,'prompt_cache_gib':0},
               'results': results, 'hash_checks': checks,
               'hashes_equal': bool(checks) and all(x['main_a_equals_mtp'] and x['main_b_equals_mtp'] and
                                                    x['main_a_equals_mtp_tree'] and x['main_b_equals_mtp_tree'] for x in checks),
               'decode': {name: [{'prompt':row['prompt'],'temperature':row['temperature'],
                                  'decode_tps_median':row['decode_tps_median'],
                                  'decode_tps_all':row['decode_tps_all'],
                                  'ttft_s_median':row['ttft_s_median'],
                                  'thinking':row.get('thinking'),
                                  'evidence':row['evidence']}
                          for row in reports[name][0]]
                          for name,_,_,_ in specs}}
    (OUT/'bench_openai_ordered.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# `bench_openai.py` main / MTP / main receipt','',
           '- 128 tokens, 3 repetitions, greedy and seeded temperature 1.0, parallel=1, context 1024, cache off.','',
           '- Runtime environment explicitly sets MTP batch=1 and solo chunk rows=4096; inherited overrides are recorded in JSON.','',
           '- Includes MTP chain mode and opt-in sampled tree mode (`TF_MTP_TREES=1`).','',
           '| Run | Prompt | Temp | Median decode tok/s | Samples |', '|---|---|---:|---:|---|']
    for name, rows in summary['decode'].items():
        for row in rows:
            lines.append(f"| {name} | {row['prompt']} | {row['temperature']} | {row['decode_tps_median']} | {row['decode_tps_all']} |")
    lines += ['', f"- Output hash comparison across all rows: `{summary['hashes_equal']}` ({len(checks)} prompt/temperature cells).", '']
    (OUT/'bench_openai_ordered.md').write_text('\n'.join(lines))
    print(f"OPENAI_HASHES_EQUAL={summary['hashes_equal']}", flush=True)
    return 0 if summary['hashes_equal'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        (OUT/'openai_repeat_failure.txt').write_text(f'{type(e).__name__}: {e}\n')
        print(f'OPENAI REPEAT FAILED: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        raise SystemExit(1)
