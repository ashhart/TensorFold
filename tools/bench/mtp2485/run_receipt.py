#!/usr/bin/env python3
"""Build main/head and capture a local same-GPU Qwen dense-MTP serving receipt."""
from __future__ import annotations
import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

if __package__:
    from .receipt_monitor import Monitor, compute_apps, nvidia
else:
    from receipt_monitor import Monitor, compute_apps, nvidia

DEFAULT_PROJECT = Path(__file__).resolve().parents[3]
PROJECT = Path(os.environ.get('TF2485_PROJECT', DEFAULT_PROJECT)).resolve()
BASE = Path(os.environ['TF2485_BASE_WORKTREE']).resolve() if os.environ.get('TF2485_BASE_WORKTREE') else None
TARGET = Path(os.environ['TF2485_CHECKPOINT']).resolve() if os.environ.get('TF2485_CHECKPOINT') else None
OUT = Path(os.environ.get('TF2485_OUTPUT_DIR', PROJECT / 'zig-out/bench/mtp2485-serving'))
NVCC = os.environ.get('TF2485_NVCC') or shutil.which('nvcc')
MODEL = 'qwen-nvfp4-mtp2485'
BASELINE_REV = os.environ.get('TF2485_BASELINE_REV')
CANDIDATE_REV = os.environ.get('TF2485_CANDIDATE_REV')
BASE_URL = 'http://127.0.0.1'
PROMPTS = [
    {'name': 'code', 'kind': 'completion',
     'prompt': 'Write a short Python function that computes the Fibonacci sequence and explain it.'},
    {'name': 'chat', 'kind': 'chat',
     'prompt': 'Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.'},
]
BENCH_ENV_CLEAR = (
    'TENSORFOLD_MEMORY_RESERVE_GIB', 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB', 'TF_TIMING',
    'TF_MTP_DRAFTS', 'TF_MTP_TREES', 'TF_QWEN_KV_PAGES', 'TF_QWEN_KV_PAGES_ALL',
    'TF_QWEN_KV_SHARE', 'TF2485_DFLASH2_DRAFTER',
)


def benchmark_env(solo_chunk_rows: int = 4096) -> tuple[dict, dict]:
    """Return a controlled Qwen benchmark environment and its provenance."""
    if solo_chunk_rows not in (4096, 8192):
        raise ValueError('solo_chunk_rows must be 4096 or 8192')
    env = os.environ.copy()
    removed = {name: name in env for name in BENCH_ENV_CLEAR}
    for name in BENCH_ENV_CLEAR:
        env.pop(name, None)
    effective = {
        'TF_QWEN_NVFP4_TC': '1',
        'TF_MTP_BATCH': '1',
        'TF_QWEN35_CHUNK_SOLO': str(solo_chunk_rows),
    }
    overridden = {name: env.get(name) for name in effective}
    env.update(effective)
    return env, {
        'effective': effective,
        'inherited_settings_removed': removed,
        'inherited_values_overridden': overridden,
    }


def add_config_arguments(parser: argparse.ArgumentParser, output_name: str) -> argparse.ArgumentParser:
    """Add reusable checkout/checkpoint/output settings; no user or machine paths are pinned."""
    parser.add_argument('--project', default=os.environ.get('TF2485_PROJECT', str(DEFAULT_PROJECT)),
                        help='candidate TensorFold checkout (default: this script’s repository)')
    parser.add_argument('--base-worktree', default=os.environ.get('TF2485_BASE_WORKTREE'),
                        help='main/base worktree, already updated to the desired comparison revision')
    parser.add_argument('--checkpoint', default=os.environ.get('TF2485_CHECKPOINT'),
                        help='local checkpoint directory used for every variant')
    parser.add_argument('--output-dir', default=os.environ.get('TF2485_OUTPUT_DIR'),
                        help=f'output directory (default: zig-out/bench/mtp2485-serving/{output_name})')
    parser.add_argument('--nvcc', default=os.environ.get('TF2485_NVCC') or shutil.which('nvcc'),
                        help='CUDA compiler path required by run_receipt.py; defaults to nvcc on PATH')
    parser.add_argument('--model', default=os.environ.get('TF2485_MODEL', MODEL),
                        help='local server model alias')
    parser.add_argument('--baseline-rev', default=os.environ.get('TF2485_BASELINE_REV'),
                        help='optional expected base revision; recorded and checked if supplied')
    parser.add_argument('--candidate-rev', default=os.environ.get('TF2485_CANDIDATE_REV'),
                        help='optional expected candidate revision; recorded and checked if supplied')
    return parser


def configure(args: argparse.Namespace, output_name: str) -> None:
    global PROJECT, BASE, TARGET, OUT, NVCC, MODEL, BASELINE_REV, CANDIDATE_REV
    PROJECT = Path(args.project).expanduser().resolve()
    BASE = Path(args.base_worktree).expanduser().resolve() if args.base_worktree else None
    TARGET = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    OUT = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
           PROJECT / 'zig-out' / 'bench' / 'mtp2485-serving' / output_name)
    NVCC = args.nvcc
    MODEL = args.model
    BASELINE_REV = args.baseline_rev
    CANDIDATE_REV = args.candidate_rev
    if not PROJECT.is_dir():
        raise ValueError(f'candidate project directory does not exist: {PROJECT}')
    if BASE is None or not BASE.is_dir():
        raise ValueError('pass --base-worktree (or TF2485_BASE_WORKTREE)')
    if TARGET is None or not TARGET.is_dir():
        raise ValueError('pass --checkpoint (or TF2485_CHECKPOINT) as an existing local directory')
    OUT.mkdir(parents=True, exist_ok=True)


def parse_config(description: str, output_name: str, argv=None) -> argparse.Namespace:
    parser = add_config_arguments(argparse.ArgumentParser(description=description), output_name)
    args = parser.parse_args(argv)
    try:
        configure(args, output_name)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_metadata_hashes() -> dict:
    names = ('config.json', 'tokenizer.json', 'model.safetensors.index.json')
    return {name: sha256(TARGET / name) if (TARGET / name).is_file() else None for name in names}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE).stdout.strip()


def git_state(cwd: Path) -> dict:
    status = git(cwd, 'status', '--porcelain=v1')
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=cwd, check=True,
                          stdout=subprocess.PIPE).stdout
    return {'commit': git(cwd, 'rev-parse', 'HEAD'), 'dirty': bool(status),
            'status_porcelain': status.splitlines(),
            'tracked_worktree_diff_sha256': hashlib.sha256(diff).hexdigest()}


def run_logged(cmd: list[str], cwd: Path, log: Path, env=None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN: {" ".join(cmd)}', flush=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            print(line, end='', flush=True)
        rc = p.wait()
    if rc:
        raise RuntimeError(f'command exited {rc}: {cmd[0]} (see {log})')


def gpu_preflight() -> dict:
    info = nvidia('--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu',
                  '--format=csv,noheader,nounits')
    apps = compute_apps()
    busy = [a for a in apps if a['used_mib'] is not None and a['used_mib'] > 512]
    if busy:
        raise RuntimeError(f'GPU already has a compute process using >512 MiB; not disturbing it: {busy}')
    return {'gpu_query': info, 'compute_apps_before': apps}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def readiness(url: str, proc: subprocess.Popen, timeout: float = 360) -> float:
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited during startup with {proc.returncode}')
        try:
            with urllib.request.urlopen(url + '/v1/models', timeout=2) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f'server did not become ready within {timeout}s: {url}')


def stream_capacity(server_log: Path) -> int | None:
    text = server_log.read_text(encoding='utf-8', errors='replace') if server_log.exists() else ''
    match = re.search(r'\b(\d+)\s+streams?\s+at once\b', text)
    return int(match.group(1)) if match else None


def memory_capacity_refusal(server_log: Path) -> bool:
    text = server_log.read_text(encoding='utf-8', errors='replace').lower() if server_log.exists() else ''
    if 'memory budget fits no stream' in text:
        return True
    needs_parallel_capacity = re.search(r'--parallel\s+\d+\s+needs\s+[0-9.]+\s+gib\b', text)
    return bool(needs_parallel_capacity and any(
        marker in text for marker in ('memory budget', 'lower --parallel', 'serve --parallel')))


def suggested_parallel_capacity(server_log: Path) -> int | None:
    """Return the server's explicit lower --parallel recommendation, if present."""
    text = server_log.read_text(encoding='utf-8', errors='replace').lower() if server_log.exists() else ''
    if not memory_capacity_refusal(server_log):
        return None
    matches = re.findall(r'(?:serve\s+--parallel|lower\s+--parallel)\s+(\d+)', text)
    if not matches:
        return None
    return int(matches[-1])


def server_args(binary: Path, port: int, context: int, parallel: int, mtp: bool,
                drafter_path: Path | None = None) -> list[str]:
    cmd = [str(binary), 'serve', str(TARGET), '--host', '127.0.0.1', '--port', str(port),
           '--name', MODEL, '--backend', 'cuda', '--context', str(context), '--parallel', str(parallel),
           '--prompt-cache-gib', '0', '--max-tokens', '192', '--temperature', '1.0',
           '--top-p', '0.95', '--top-k', '20', '--no-thinking', '--no-update-check']
    if mtp:
        cmd += ['--drafter', str(TARGET), '--drafter-bits', '4', '--mtp-drafts', '3']
    elif drafter_path is not None:
        cmd += ['--drafter', str(drafter_path)]
    else:
        cmd += ['--no-drafts']
    return cmd


def telemetry(url: str, mtp: bool) -> dict:
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': PROMPTS[0]['prompt']}],
            'max_tokens': 64, 'temperature': 0, 'seed': 1234, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    msg = data.get('choices', [{}])[0].get('message', {})
    content = msg.get('content') or ''
    runtime = data.get('tensorfold') or {}
    spec = data.get('speculative') or {}
    return {'drafts': runtime.get('drafts'), 'tokens_per_second': runtime.get('tokens_per_second'),
            'seconds': runtime.get('seconds'), 'prefill_seconds': runtime.get('prefill_seconds'),
            'time_to_first_token': runtime.get('time_to_first_token'), 'sampling': runtime.get('sampling'),
            'token_sha256': runtime.get('token_sha256'), 'response_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'response_chars': len(content), 'speculative': spec,
            'reported_mtp_enabled': bool(runtime.get('drafts')) and (not mtp or spec.get('drafted', 0) > 0)}


def start_server(label: str, binary: Path, mtp: bool, context: int, parallel: int, env: dict,
                 drafter_path: Path | None = None):
    port = free_port()
    url = f'{BASE_URL}:{port}'
    log = OUT / f'{label}.server.log'
    f = log.open('w', encoding='utf-8')
    cmd = server_args(binary, port, context, parallel, mtp, drafter_path)
    print(f'START {label}: context={context} parallel={parallel} mtp={mtp} port={port}', flush=True)
    proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT,
                           start_new_session=True)
    mon = Monitor(proc.pid, OUT / f'{label}.memory.jsonl')
    mon.start()
    try:
        seconds = readiness(url, proc)
        if stream_capacity(log) is None:
            raise RuntimeError(f'server became ready without a parseable stream-capacity receipt: {log}')
        if mtp and 'drafts: MTP' not in log.read_text(encoding='utf-8', errors='replace'):
            raise RuntimeError(f'MTP was requested but the server startup receipt does not confirm it: {log}')
        if drafter_path is not None and 'drafts: DFlash2' not in log.read_text(encoding='utf-8', errors='replace'):
            raise RuntimeError(f'DFlash2 was requested but the server startup receipt does not confirm it: {log}')
        return proc, f, mon, url, seconds, log
    except Exception:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        f.close()
        mon.finish()
        raise


def stop_server(proc, f, mon) -> dict:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    f.flush()
    f.close()
    return mon.finish()


def bench(label: str, url: str, output: Path, stream_levels: list[int]) -> None:
    cmd = [sys.executable, str(PROJECT / 'tools/bench_concurrent.py'), url, MODEL,
           '--levels', ','.join(str(level) for level in stream_levels),
           '--tokens', '128', '--reps', '2', '--seed', '1234',
           '--temperatures', '0,1.0', '--alone', '--serial', '--label', label,
           '--output', str(output)]
    run_logged(cmd, PROJECT, OUT / f'{label}.decode.log')


def cold_bench(label: str, url: str, output: Path) -> None:
    cmd = [sys.executable, str(PROJECT / 'tools/bench_cache.py'), url, MODEL,
           '--tokens', '64', '--repeats', '1', '--streams', '1', '--require-cold',
           '--output', str(output)]
    run_logged(cmd, PROJECT, OUT / f'{label}.cold.log')


def run_variant(label: str, binary: Path, mtp: bool, phase: str, context: int, parallel: int, env: dict) -> dict:
    attempts = []
    started = None
    requested = parallel
    while requested >= 1:
        attempt_label = f'{label}-parallel{requested}'
        try:
            started = start_server(attempt_label, binary, mtp, context, requested, env)
        except Exception as exc:
            attempt_log = OUT / f'{attempt_label}.server.log'
            attempts.append({'parallel_requested': requested, 'status': 'failed',
                             'error': f'{type(exc).__name__}: {exc}',
                             'server_log': str(attempt_log)})
            if requested > 1 and memory_capacity_refusal(attempt_log):
                suggested = suggested_parallel_capacity(attempt_log)
                requested = suggested if suggested is not None and 0 < suggested < requested else requested - 1
                continue
            raise
        attempts.append({'parallel_requested': requested,
                         'streams_admitted': stream_capacity(started[5]),
                         'status': 'started', 'server_log': str(started[5])})
        break
    if started is None:
        raise RuntimeError(f'no server capacity admitted for {label}: {attempts}')
    proc, f, mon, url, ready_s, log = started
    admitted = stream_capacity(log)
    assert admitted is not None
    stream_levels = [level for level in (1, 2, 3, 4) if level <= admitted]
    result = {'label': label, 'binary': str(binary), 'mtp_requested': mtp,
              'context': context, 'parallel_requested': parallel, 'parallel_admitted': admitted,
              'supported_stream_levels': stream_levels, 'parallel_attempts': attempts,
              'startup_s': round(ready_s, 3), 'server_log': str(log), 'url_local_only': url}
    try:
        if phase == 'decode':
            result['telemetry'] = telemetry(url, mtp)
            output = OUT / f'{label}.decode.json'
            bench(label, url, output, stream_levels)
            result['decode_json'] = str(output)
        elif phase == 'cold':
            # Do not warm the process with telemetry; the benchmark client's first request is primary.
            output = OUT / f'{label}.cold.json'
            cold_bench(label, url, output)
            result['cold_json'] = str(output)
        else:
            raise ValueError(phase)
    finally:
        result['memory'] = stop_server(proc, f, mon)
    return result


def map_receipt(path: Path) -> dict:
    report = json.loads(path.read_text())
    rows = {}
    for cell in report['cells']:
        for rep_i, rep in enumerate(cell['reps']):
            for request_i, pair in enumerate(rep.get('comparisons', [])):
                key = (cell['prompt'], cell['temperature'], cell['streams'], rep_i, request_i)
                if key in rows:
                    raise ValueError(f'duplicate decode receipt cell: {key}')
                rows[key] = {role: pair.get(role) or {} for role in ('concurrent', 'alone', 'serial')}
    return rows


def expected_decode_keys(stream_levels: list[int] | tuple[int, ...] = (1, 2, 4)) -> set[tuple]:
    return {
        (prompt['name'], temperature, streams, rep_i, request_i)
        for prompt in PROMPTS
        for temperature in (0.0, 1.0)
        for streams in stream_levels
        for rep_i in range(2)
        for request_i in range(streams)
    }


def token_equal(a: dict, b: dict) -> bool | None:
    if not a.get('complete') or not b.get('complete') or a.get('error') or b.get('error'):
        return None
    ah, bh = a.get('token_sha256'), b.get('token_sha256')
    return ah == bh if ah and bh else None


def cross_compare(base_file: Path, plain_file: Path, mtp_file: Path,
                  stream_levels: list[int] | tuple[int, ...] = (1, 2, 4)) -> dict:
    base, plain, mtp = map_receipt(base_file), map_receipt(plain_file), map_receipt(mtp_file)
    expected = expected_decode_keys(stream_levels)
    checks = []
    def compare(a_map, a_role, b_map, b_role, name):
        for key in sorted(expected, key=str):
            if key not in a_map or key not in b_map:
                checks.append({'comparison': name, 'key': list(key), 'equal': False,
                               'missing_left': key not in a_map, 'missing_right': key not in b_map})
                continue
            equal = token_equal(a_map[key][a_role], b_map[key][b_role])
            checks.append({'comparison': name, 'key': list(key), 'equal': equal,
                           'left_sha256': a_map[key][a_role].get('token_sha256'),
                           'right_sha256': b_map[key][b_role].get('token_sha256')})
        if not expected.issubset(a_map) or not expected.issubset(b_map):
            checks.append({'comparison': name, 'key': 'coverage', 'equal': False,
                           'left_missing': len(expected - set(a_map)),
                           'right_missing': len(expected - set(b_map)),
                           'left_extra_skipped': len(set(a_map) - expected),
                           'right_extra_skipped': len(set(b_map) - expected)})
    for role in ('concurrent', 'alone'):
        compare(base, role, plain, role, f'main_no_drafts_vs_branch_no_drafts_{role}')
        compare(base, role, mtp, role, f'main_no_drafts_vs_branch_mtp_{role}')
    compare(base, 'alone', mtp, 'serial', 'main_no_drafts_vs_branch_mtp_serial_control')
    return {'checks': checks, 'equal': sum(c.get('equal') is True for c in checks),
            'unequal': sum(c.get('equal') is False for c in checks),
            'unverified': sum(c.get('equal') is None for c in checks),
            'all_equal': bool(checks) and all(c.get('equal') is True for c in checks)}


def compact_decode(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for c in data.get('cells', []):
        out.append({'prompt': c['prompt'], 'temperature': c['temperature'], 'streams': c['streams'],
                    'aggregate_tps': c.get('aggregate_tps'), 'per_stream_tps': c.get('per_stream_tps'),
                    'ttft_s_max': c.get('ttft_s_max'), 'alone': c.get('alone'), 'serial': c.get('serial'),
                    'failed': c.get('failed'), 'unmeasured': c.get('unmeasured')})
    return out


def compact_cold(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not data.get('rows'):
        raise ValueError(f'cold benchmark has no request rows: {path}')
    first = data['rows'][0]
    n = first.get('prompt_tokens')
    ttft = first.get('ttft_s')
    return {'qualified': data.get('qualified'), 'prompt_tokens': n, 'ttft_s': ttft,
            'prompt_tokens_over_client_ttft_tps': round(n / ttft, 3) if n and ttft else None,
            'cache_state': first.get('cache_state'), 'cached_tokens': first.get('cached_tokens'),
            'decode_tps': first.get('decode_tps'), 'token_sha256': first.get('token_sha256')}


def functional_qualification(equality: dict, cold_summary: dict,
                             variants: dict, common_streams: list[int]) -> bool:
    return {1, 2}.issubset(common_streams) and equality['all_equal'] and all(
        item['qualified'] for item in cold_summary.values()) and all(
        variants[name]['telemetry'].get('reported_mtp_enabled', False) == expected
        for name, expected in (('main_no_drafts', False), ('branch_no_drafts', False),
                               ('branch_mtp3', True)))


def pr_qualification_fields() -> dict:
    return {'pr_qualification_status': 'not_assessed',
            'pr_qualification_missing_gates': [
                'cold 2048/8192 prompt-speed A/B',
                'resumed-prefix exactness receipt',
                'full build, test, and lean-check receipt',
            ]}


def main(argv=None) -> int:
    args = parse_config(__doc__ or 'Dense MTP serving receipt', 'receipt', argv)
    if NVCC is None or not Path(NVCC).is_file():
        raise RuntimeError('pass --nvcc or put the desired CUDA compiler on PATH')
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = gpu_preflight()
    base_state = git_state(BASE)
    head_state = git_state(PROJECT)
    base_commit = base_state['commit']
    head_commit = head_state['commit']
    if BASELINE_REV and base_commit != BASELINE_REV:
        raise RuntimeError(f'base revision differs from --baseline-rev: {base_commit}')
    if CANDIDATE_REV and head_commit != CANDIDATE_REV:
        raise RuntimeError(f'candidate revision differs from --candidate-rev: {head_commit}')
    build_targets = [(BASE, 'main'), (PROJECT, 'branch')]
    for repo, label in build_targets:
        run_logged(['zig', 'build', '-j4', 'native', f'-Dnvcc={NVCC}', '-Dsm=120'],
                   repo, OUT / f'{label}.native-build.log')
    binaries = {'main': BASE / 'zig-out/native/bin/tensorfold-native',
                'branch': PROJECT / 'zig-out/native/bin/tensorfold-native'}
    for name, binary in binaries.items():
        if not binary.is_file():
            raise RuntimeError(f'{name} native binary not produced: {binary}')
    ck_hashes = checkpoint_metadata_hashes()
    env, runtime_environment = benchmark_env()
    common = {'checkpoint': str(TARGET), 'checkpoint_metadata_sha256': ck_hashes,
              'baseline_commit': base_commit, 'candidate_commit': head_commit,
              'baseline_git_state': base_state, 'candidate_git_state': head_state,
              'expected_baseline_commit': BASELINE_REV, 'expected_candidate_commit': CANDIDATE_REV,
              'baseline_binary_sha256': sha256(binaries['main']),
              'candidate_binary_sha256': sha256(binaries['branch']),
              'zig_version': subprocess.run(['zig', 'version'], check=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
              'python_version': sys.version.split()[0], 'nvcc': NVCC,
              'gpu': preflight, 'prompts': PROMPTS,
              'decode_settings': {'thinking': 'off', 'temperatures': [0, 1.0], 'top_p': 0.95,
                                  'top_k': 20, 'seed': 1234, 'tokens': 128, 'reps': 2,
                                  'streams_requested': [1, 2, 4], 'context': 1024, 'prompt_cache_gib': 0},
              'cold_settings': {'tokens': 64, 'repeats': 1, 'streams': 1,
                                'context': 4096, 'prompt_cache_gib': 0},
              'env_overrides': {'TF_QWEN_NVFP4_TC': '1', 'TF_MTP_BATCH': '1',
                                'TF_QWEN35_CHUNK_SOLO': '4096', 'TF_TIMING': 'unset for functional timing',
                                'TENSORFOLD_MEMORY_RESERVE_GIB': 'unset/default',
                                'TENSORFOLD_CUDA_MEMORY_LIMIT_GB': 'unset/default'},
              'runtime_environment': runtime_environment}
    (OUT / 'run.json').write_text(json.dumps(common, indent=2) + '\n')
    variants = {}
    try:
        # Fresh server per configuration. No background server is stopped; only each process started here is terminated.
        variants['main_no_drafts'] = run_variant('main-no-drafts', binaries['main'], False, 'decode', 1024, 4, env)
        variants['branch_no_drafts'] = run_variant('branch-no-drafts', binaries['branch'], False, 'decode', 1024, 4, env)
        variants['branch_mtp3'] = run_variant('branch-mtp3', binaries['branch'], True, 'decode', 1024, 4, env)
        variants['main_no_drafts_cold'] = run_variant('main-no-drafts-cold', binaries['main'], False, 'cold', 4096, 1, env)
        variants['branch_no_drafts_cold'] = run_variant('branch-no-drafts-cold', binaries['branch'], False, 'cold', 4096, 1, env)
        variants['branch_mtp3_cold'] = run_variant('branch-mtp3-cold', binaries['branch'], True, 'cold', 4096, 1, env)
        common_streams = sorted(set.intersection(*[
            set(variants[name]['supported_stream_levels'])
            for name in ('main_no_drafts', 'branch_no_drafts', 'branch_mtp3')
        ]))
        equality = cross_compare(OUT / 'main-no-drafts.decode.json',
                                OUT / 'branch-no-drafts.decode.json',
                                OUT / 'branch-mtp3.decode.json', common_streams)
        report = {'metadata': common, 'variants': variants, 'token_equality': equality,
                  'concurrency_coverage': {
                      'requested_stream_levels': [1, 2, 4],
                      'common_stream_levels_tested': common_streams,
                      'four_stream_status': 'tested' if 4 in common_streams else 'skipped_server_capacity',
                      'per_variant': {
                          name: {'status': 'tested' if 4 in variants[name]['supported_stream_levels'] else 'skipped',
                                 'reason': None if 4 in variants[name]['supported_stream_levels'] else
                                     f"server admitted {variants[name]['parallel_admitted']} streams; see startup log and attempts",
                                 'streams_admitted': variants[name]['parallel_admitted'],
                                 'server_log': variants[name]['server_log'],
                                 'attempts': variants[name]['parallel_attempts']}
                          for name in ('main_no_drafts', 'branch_no_drafts', 'branch_mtp3')
                      },
                  },
                  'decode_summary': {k: compact_decode(OUT / f'{file}.decode.json') for k, file in (
                      ('main_no_drafts', 'main-no-drafts'), ('branch_no_drafts', 'branch-no-drafts'),
                      ('branch_mtp3', 'branch-mtp3'))},
                  'cold_summary': {k: compact_cold(OUT / f'{file}.cold.json') for k, file in (
                      ('main_no_drafts', 'main-no-drafts-cold'), ('branch_no_drafts', 'branch-no-drafts-cold'),
                      ('branch_mtp3', 'branch-mtp3-cold'))}}
        report['functional_qualified'] = functional_qualification(
            equality, report['cold_summary'], variants, common_streams)
        report.update(pr_qualification_fields())
        (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
        lines = [
            '# Dense MTP main-versus-branch inference receipt', '',
            f"- Baseline: `{base_commit}`; candidate: `{head_commit}`.",
            f"- GPU: {preflight['gpu_query']}.",
            f"- Checkpoint: `{TARGET}`; config/tokenizer/index hashes are in `run.json`.",
                  f"- Decode: fresh servers, context 1024, prompt cache 0, no thinking, 128 requested tokens, 2 repetitions; common stream levels tested: {common_streams}. Greedy and seeded temperature 1.0 (top-p 0.95, top-k 20).",
                  f"- Four-stream coverage: `{report['concurrency_coverage']['four_stream_status']}`; attempted stream-capacity refusals are included in the receipt.",
            '- Cold prompt: fresh process per mode, context 4096, prompt cache 0; first request only is primary. Repeated and concurrent rows remain in raw reports; client TTFT is not kernel prefill time.',
            '- Runtime environment explicitly sets `TF_QWEN_NVFP4_TC=1`, `TF_MTP_BATCH=1`, and `TF_QWEN35_CHUNK_SOLO=4096`; cleared and replaced inherited values are recorded in `run.json`.', '',
            '## Decode cells', '',
            '| Mode | Prompt | Temp | Streams | Aggregate tok/s | Per-stream tok/s | Solo vs draft-off |',
            '|---|---|---:|---:|---:|---:|---|'
        ]
        for mode, rows in report['decode_summary'].items():
            for c in rows:
                lines.append(f"| {mode} | {c['prompt']} | {c['temperature']} | {c['streams']} | {c['aggregate_tps']} | {c['per_stream_tps']} | {c['serial']} |")
        lines += ['', '## Cold prompt', '', '| Mode | Prompt tokens | TTFT s | Tokens / client TTFT s | Cache state |', '|---|---:|---:|---:|---|']
        for mode, c in report['cold_summary'].items():
            lines.append(f"| {mode} | {c['prompt_tokens']} | {c['ttft_s']} | {c['prompt_tokens_over_client_ttft_tps']} | {c['cache_state']} |")
        lines += ['', '## Token equality and memory', '',
                  f"- Exact cross-run comparisons: {equality['equal']} equal, {equality['unequal']} unequal, {equality['unverified']} unverified; all equal: `{equality['all_equal']}`."]
        for key, v in variants.items():
            m = v['memory']
            lines.append(f"- {key}: process GPU peak {m['gpu_peak_process_mib']} MiB; process RSS peak {m['process_peak_rss_mib']} MiB; total GPU peak {m['gpu_peak_used_mib']} MiB.")
        lines += ['', f"- Functional receipt qualified: `{report['functional_qualified']}`.",
                  '- PR qualification is not assessed by this functional receipt; required missing gates are listed in `receipt.json`.',
                  '- Raw per-request hashes, logs, telemetry, and one-second GPU/RSS samples are retained alongside this summary.', '']
        (OUT / 'summary.md').write_text('\n'.join(lines))
        print(f"TOKEN_EQUALITY equal={equality['equal']} unequal={equality['unequal']} unverified={equality['unverified']}", flush=True)
        print(f"FUNCTIONAL_QUALIFIED={report['functional_qualified']} PR_QUALIFICATION=not_assessed RECEIPT={OUT / 'receipt.json'}", flush=True)
        return 0 if report['functional_qualified'] else 2
    except Exception as e:
        (OUT / 'failure.txt').write_text(f'{type(e).__name__}: {e}\n')
        print(f'FAILED: {type(e).__name__}: {e}', flush=True)
        raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'RECEIPT ERROR: {e}', file=sys.stderr, flush=True)
        raise SystemExit(1)
