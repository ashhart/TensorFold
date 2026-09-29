"""Benchmark, inspect and publish without requiring a repository checkout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys
import uuid

from tensorfold.benchmark import publish, receipt
from tensorfold.benchmark.protocol import SUITE_ID, SUITE_MANIFEST, SUITE_SHA256, summary

_ACTIONS = {'publish', 'inspect', 'login', 'logout', 'suites'}


def add_parser(commands) -> None:
    parser = commands.add_parser('benchmark', help='run a repeatable benchmark and optionally publish its receipt',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target', nargs='?', help='model ID, or publish/inspect/login/logout/suites')
    parser.add_argument('file', nargs='?', help='receipt JSON for publish or inspect')
    parser.add_argument('--backend', choices=('auto', 'mlx', 'cuda'), default='auto')
    parser.add_argument('--server', help='attach to an existing server; result stays unranked without a server manifest')
    parser.add_argument('--model-id', help='model alias to request from an attached server')
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--reps', type=int, default=5, help='measured repetitions per workload and temperature')
    parser.add_argument('--temperatures', default='1.0,0', help='comma-separated sampling temperatures')
    parser.add_argument('--serial', action='store_true', help='use the same engine with drafts disabled')
    parser.add_argument('--context', type=int, help='explicit prompt plus reply capacity')
    parser.add_argument('--drafter', default='auto')
    parser.add_argument('--download', action='store_true', help='allow missing checkpoint weights to download')
    parser.add_argument('--timeout', type=float, default=600, help='request timeout in seconds')
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--output', type=Path, help='receipt path; default ~/.tensorfold/benchmarks/RUN_ID.json')
    parser.add_argument('--publish', action='store_true', help='review and upload the saved receipt')
    parser.add_argument('--yes', action='store_true', help='explicitly approve public upload without the interactive preview prompt')
    parser.add_argument('--upload-url', default=publish.DEFAULT_API, help='benchmark API base URL')
    parser.add_argument('--token-stdin', action='store_true', help='for login, read a contributor token from stdin')
    parser.set_defaults(func=command)


def _show(value: dict) -> None:
    print(f"Run {value['run_id']} | {value['runtime']['backend']} | {value['model']['repo_id'] or 'local model'}")
    print('Workload       Temp    Stream delivery tok/s    First text, s    Status')
    for cell in summary(value['samples'], expected_repetitions=value['settings']['repetitions']):
        speed = cell.get('delivery_tps')
        first = cell.get('ttft_seconds')
        rate = f"{speed['median']:.2f}" if speed else 'not measured'
        ttft = f"{first['median']:.3f}" if first else 'not measured'
        status = 'complete' if cell['protocol_complete'] else 'custom/incomplete'
        print(f"{cell['fixture_id']:<14} {cell['temperature']:<6g} {rate:<25} {ttft:<16} {status}")
    print('Stream delivery includes transport buffering; it is not an engine token clock.')
    if not value['settings']['managed']:
        print('Attached-server hardware and checkpoint revisions are unknown; this result is unranked.')


def _publish(value: dict, args) -> int:
    publish.verify_public_models(value)
    print('Public upload preview:')
    print(json.dumps(value, indent=2, allow_nan=False))
    print('New submissions are community-reported and pending moderation.')
    if not args.yes:
        if not sys.stdin.isatty():
            raise ValueError('Publishing needs explicit consent; inspect the receipt and pass --yes')
        if input('Upload these public fields to the benchmark receiver? [y/N] ').strip().lower() not in ('y', 'yes'):
            print('Upload cancelled; the receipt remains local.')
            return 0
    result = publish.upload(value, args.upload_url)
    print(f"Submission {result['id']}: {result['status']}")
    if result.get('results_url'):
        print(result['results_url'])
    return 0


def _attached_metadata(target: str, backend: str) -> dict:
    from tensorfold import hub
    public = hub.is_repo_id(target)
    return {'model': {'repo_id': target if public else None, 'revision': None, 'family': None,
                      'config_sha256': None, 'tokenizer_sha256': None, 'quantization': None,
                      'drafter_repo_id': None, 'local': not public},
            'runtime': {'tensorfold_version': 'unknown', 'backend': backend, 'python_version': 'unknown',
                        'platform': 'macos' if backend == 'mlx' else 'linux', 'dependencies': {}}}


def command(args) -> int:
    try:
        return _command(args)
    except KeyboardInterrupt:
        print('Benchmark cancelled.', file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        # Exceptions in networking/metadata contain fixed messages, never uploaded server errors.
        print(f'tensorfold benchmark: {exc}', file=sys.stderr)
        return 1


def _command(args) -> int:
    if not args.target:
        raise ValueError('Choose a model ID, or benchmark publish/inspect/login/logout/suites')
    if args.target in _ACTIONS:
        if args.target in ('publish', 'inspect'):
            if not args.file:
                raise ValueError('This action needs a receipt JSON file')
            value = receipt.load(Path(args.file).expanduser())
            _show(value)
            return _publish(value, args) if args.target == 'publish' else 0
        if args.file:
            raise ValueError('This action does not take a receipt file')
        if args.target == 'suites':
            print(json.dumps({**SUITE_MANIFEST, 'sha256': SUITE_SHA256}, indent=2))
            return 0
        if args.target == 'logout':
            publish.auth_path().unlink(missing_ok=True)
            print('Local benchmark upload credentials removed.')
            return 0
        if args.target == 'login':
            if args.token_stdin:
                token = sys.stdin.readline().strip()
            elif sys.stdin.isatty():
                print('Use a contributor upload token issued by the TensorFold benchmark administrator.')
                token = getpass.getpass('Upload token: ')
            else:
                raise ValueError('Use login --token-stdin to provide a contributor token without exposing it in arguments')
            profile = publish.login(token, args.upload_url)
            print(f"Benchmark uploads enabled for {profile.get('display_name', 'contributor')}.")
            return 0
    if args.file or args.token_stdin:
        raise ValueError('Unexpected benchmark argument')
    try:
        temperatures = [float(item) for item in args.temperatures.split(',')]
    except ValueError:
        raise ValueError('Temperatures must be comma-separated finite numbers') from None
    import math
    if not 1 <= len(temperatures) <= 4 or len(set(temperatures)) != len(temperatures):
        raise ValueError('Choose one to four distinct temperatures')
    if any(not math.isfinite(x) or not 0 <= x <= 2 for x in temperatures):
        raise ValueError('Temperatures must be finite numbers between 0 and 2')
    if not 2 <= args.tokens <= 8192 or not 1 <= args.reps <= 20:
        raise ValueError('Tokens must be 2..8192 and repetitions 1..20')
    if args.context is not None and not 1 <= args.context <= 10000000:
        raise ValueError('Context must be a positive token count')
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        raise ValueError('Timeouts must be finite positive seconds')
    backend = args.backend if args.backend != 'auto' else ('mlx' if sys.platform == 'darwin' else 'cuda')
    from tensorfold.benchmark import hardware, metadata
    from tensorfold.benchmark.runner import run_suite
    run_id = str(uuid.uuid4())
    settings = {'tokens': args.tokens, 'repetitions': args.reps, 'temperatures': temperatures,
                'serial': args.serial, 'context_tokens': args.context, 'managed': not bool(args.server),
                'rank_count': 1 if not args.server else None}
    print(f'{SUITE_ID}: code and chat, {len(temperatures)} temperatures, {args.reps} repeats after warm-up.')
    print('This generates model replies; use Ctrl-C to cancel. Nothing is uploaded unless you publish.')

    def progress(sample):
        phase = 'warm-up' if sample['repeat'] < 0 else f"repeat {sample['repeat'] + 1}/{args.reps}"
        rate = sample['delivery_tps']
        measured = f" {rate:.2f} delivery tok/s" if rate is not None else ''
        print(f"{sample['fixture_id']} T={sample['temperature']:g} {phase}: {sample['status']}{measured}", flush=True)

    if args.server:
        if args.backend == 'auto':
            raise ValueError('--server needs an explicit --backend mlx or cuda; client hardware does not identify a remote server')
        from contextlib import nullcontext
        import urllib.parse
        parsed = urllib.parse.urlsplit(args.server)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Server URL must be HTTP(S) without credentials, query or fragment')
        if args.download:
            raise ValueError('--download applies only to a benchmark-owned server')
        manager = nullcontext(None)
        base = args.server.rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3]
        served = args.model_id or args.target
    else:
        from tensorfold.benchmark.process import ManagedServer
        if args.model_id:
            raise ValueError('--model-id applies only to an attached server')
        manager = ManagedServer(args.target, backend=backend, download=args.download, context=args.context,
                                drafter=args.drafter, startup_timeout=args.startup_timeout,
                                output_tokens=args.tokens, serial=args.serial)
        base = served = ''
    with manager as server:
        if server is not None:
            base, served = server.base_url, server.served_name
        samples = run_suite(base, served, tokens=args.tokens, repetitions=args.reps,
                            temperatures=temperatures, serial=args.serial, timeout=args.timeout, progress=progress)
        details = metadata.collect(args.target, backend, drafter='none' if args.serial else args.drafter) if server is not None else _attached_metadata(args.target, backend)
        devices = hardware.detect(backend) if server is not None else {
            'cpu_model': None, 'system_memory_bytes': None, 'memory_type': 'unknown', 'gpus': [],
            'gpu_count': None, 'source': 'unavailable'}
    value = {'schema_version': 1, 'suite': {'id': SUITE_ID, 'sha256': SUITE_SHA256}, 'run_id': run_id,
             'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
             **details, 'hardware': devices, 'settings': settings, 'samples': samples}
    destination = args.output or Path.home()/'.tensorfold'/'benchmarks'/f'{run_id}.json'
    receipt.save(value, destination)
    _show(value)
    print(f'Receipt saved: {destination}')
    if any(sample['error_code'] == 'cancelled' for sample in samples):
        return 130
    if args.publish:
        _publish(value, args)
    return 0 if all(sample['status'] == 'ok' for sample in samples) else 2
