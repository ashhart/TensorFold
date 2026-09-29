"""Strict portable receipts; public data never contains paths, prompts or credentials."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid
from typing import Any

MAX_BYTES = 1024 * 1024
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+ -]{0,79}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,119}\Z")
_TOP = {'schema_version', 'suite', 'run_id', 'created_at', 'runtime', 'model', 'hardware', 'settings', 'samples'}
_SAMPLE = {'fixture_id', 'temperature', 'repeat', 'seed', 'status', 'prompt_tokens', 'completion_tokens',
           'cached_tokens', 'delivery_seconds', 'delivery_tps', 'ttft_seconds', 'end_to_end_seconds',
           'server_decode_tps', 'server_decode_seconds', 'prefill_seconds', 'token_sha', 'error_code'}


def _object(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f'Invalid {label} fields')
    return value


def _number(value: Any, *, nullable: bool = False, integer: bool = False, minimum: float = 0,
            maximum: float = 1e15) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Invalid numeric benchmark field')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Invalid numeric benchmark field')
    if integer and not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError('Benchmark number is outside its allowed range')


def _string(value: Any, regex: re.Pattern, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or len(value) > 200 or not regex.fullmatch(value):
        raise ValueError('Invalid public benchmark identifier')


def _hex(value: Any, nullable: bool = True, lengths: tuple[int, ...] = (64,)) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or len(value) not in lengths or not re.fullmatch('[a-f0-9]+', value):
        raise ValueError('Invalid benchmark digest')


def validate(receipt: Any, *, publishing: bool = False) -> dict:
    from tensorfold.benchmark.protocol import SUITE_ID, SUITE_SHA256
    _object(receipt, _TOP, 'receipt')
    if receipt['schema_version'] != 1 or isinstance(receipt['schema_version'], bool):
        raise ValueError('Unsupported benchmark schema version')
    suite = _object(receipt['suite'], {'id', 'sha256'}, 'suite')
    if suite != {'id': SUITE_ID, 'sha256': SUITE_SHA256}:
        raise ValueError('Unsupported benchmark suite or fixture digest')
    try:
        if str(uuid.UUID(receipt['run_id'])) != receipt['run_id']:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Invalid run identifier') from None
    from datetime import datetime
    try:
        stamp = datetime.fromisoformat(receipt['created_at'].replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise ValueError('Invalid benchmark timestamp') from None
    runtime = _object(receipt['runtime'], {'tensorfold_version', 'backend', 'python_version', 'platform', 'dependencies'},
                      'runtime')
    if runtime['backend'] not in ('mlx', 'cuda') or runtime['platform'] not in ('macos', 'linux'):
        raise ValueError('Unsupported benchmark runtime')
    for key in ('tensorfold_version', 'python_version'):
        _string(runtime[key], _VERSION)
    deps = runtime['dependencies']
    if not isinstance(deps, dict) or set(deps) - {'mlx', 'mlx_lm', 'torch', 'triton', 'cuda', 'driver'}:
        raise ValueError('Invalid runtime dependency fields')
    for value in deps.values():
        _string(value, _VERSION, nullable=True)
    model = _object(receipt['model'], {'repo_id', 'revision', 'family', 'config_sha256', 'tokenizer_sha256',
                                     'quantization', 'drafter_repo_id', 'local'}, 'model')
    for key in ('repo_id', 'drafter_repo_id'):
        _string(model[key], _REPO, nullable=True)
    for key in ('config_sha256', 'tokenizer_sha256'):
        _hex(model[key])
    _hex(model['revision'], lengths=(40, 64))
    for key in ('family', 'quantization'):
        _string(model[key], _SAFE_NAME, nullable=True)
    if type(model['local']) is not bool:
        raise ValueError('Invalid local-model flag')
    if publishing and (model['local'] or not model['repo_id']):
        raise ValueError('Local or private model labels cannot be published; benchmark a public checkpoint ID')
    hardware = _object(receipt['hardware'], {'cpu_model', 'system_memory_bytes', 'memory_type', 'gpus', 'gpu_count', 'source'},
                       'hardware')
    _string(hardware['cpu_model'], _SAFE_NAME, nullable=True)
    _number(hardware['system_memory_bytes'], integer=True, nullable=True)
    _number(hardware['gpu_count'], integer=True, nullable=True, maximum=256)
    if hardware['memory_type'] not in ('unified', 'dedicated', 'unknown'):
        raise ValueError('Invalid hardware memory type')
    if hardware['source'] not in ('detected', 'partial', 'unavailable'):
        raise ValueError('Invalid hardware source')
    if not isinstance(hardware['gpus'], list) or len(hardware['gpus']) > 256:
        raise ValueError('Invalid GPU list')
    for gpu in hardware['gpus']:
        _object(gpu, {'name', 'memory_bytes', 'cores'}, 'GPU')
        _string(gpu['name'], _SAFE_NAME)
        _number(gpu['memory_bytes'], integer=True, nullable=True)
        _number(gpu['cores'], integer=True, nullable=True, maximum=100000)
    if hardware['gpu_count'] is not None and hardware['gpu_count'] != len(hardware['gpus']):
        raise ValueError('GPU count differs from its manifest')
    settings = _object(receipt['settings'], {'tokens', 'repetitions', 'temperatures', 'serial', 'context_tokens', 'managed', 'rank_count'},
                       'settings')
    _number(settings['tokens'], integer=True, minimum=2, maximum=8192)
    _number(settings['repetitions'], integer=True, minimum=1, maximum=20)
    _number(settings['rank_count'], integer=True, nullable=True, minimum=1, maximum=2)
    _number(settings['context_tokens'], integer=True, nullable=True, minimum=1, maximum=10000000)
    if type(settings['serial']) is not bool or type(settings['managed']) is not bool:
        raise ValueError('Invalid benchmark execution flags')
    temps = settings['temperatures']
    if not isinstance(temps, list) or not 1 <= len(temps) <= 4:
        raise ValueError('Invalid benchmark temperatures')
    for value in temps:
        _number(value, maximum=2)
    if len(set(temps)) != len(temps):
        raise ValueError('Invalid benchmark temperatures')
    samples = receipt['samples']
    if not isinstance(samples, list) or not 1 <= len(samples) <= 200:
        raise ValueError('Invalid benchmark sample count')
    seen = set()
    for sample in samples:
        _object(sample, _SAMPLE, 'sample')
        _number(sample['temperature'], maximum=2)
        if sample['fixture_id'] not in ('code', 'chat') or sample['temperature'] not in temps:
            raise ValueError('Sample does not belong to the requested suite cells')
        _number(sample['repeat'], integer=True, minimum=-1, maximum=settings['repetitions']-1)
        _number(sample['seed'], integer=True, maximum=2**32-1)
        key = sample['fixture_id'], sample['temperature'], sample['repeat']
        if key in seen:
            raise ValueError('Duplicate benchmark repetition')
        seen.add(key)
        if sample['status'] not in ('ok', 'early_eos', 'error', 'unmeasured'):
            raise ValueError('Invalid benchmark sample status')
        for name in ('prompt_tokens', 'completion_tokens', 'cached_tokens'):
            _number(sample[name], integer=True, nullable=True, maximum=10000000)
        for name in ('delivery_seconds', 'delivery_tps', 'ttft_seconds', 'end_to_end_seconds',
                     'server_decode_tps', 'server_decode_seconds', 'prefill_seconds'):
            _number(sample[name], nullable=True, maximum=1e9)
        _hex(sample['token_sha'], lengths=(12, 64))
        from tensorfold.benchmark.protocol import ERROR_CODES
        if sample['error_code'] is not None and (not isinstance(sample['error_code'], str) or sample['error_code'] not in ERROR_CODES):
            raise ValueError('Invalid benchmark error code')
        if sample['repeat'] == -1 and sample['status'] != 'error':
            raise ValueError('Only warm-up failures use the warm-up repetition marker')
        count, seconds, rate = sample['completion_tokens'], sample['delivery_seconds'], sample['delivery_tps']
        if rate is not None:
            if count is None or count < 2 or seconds is None or seconds <= 0:
                raise ValueError('A measured rate needs a token count and positive duration')
            expected = (count-1) / seconds
            if not math.isclose(rate, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('Delivery rate does not match sample counts and timing')
        if sample['status'] == 'ok' and (count != settings['tokens'] or rate is None or sample['error_code'] is not None):
            raise ValueError('A complete sample must contain its requested output and timings')
        if sample['cached_tokens'] is not None and sample['prompt_tokens'] is not None:
            if sample['cached_tokens'] > sample['prompt_tokens']:
                raise ValueError('Cached count exceeds the prompt')
    encoded = canonical(receipt)
    if len(encoded) > MAX_BYTES:
        raise ValueError('Benchmark receipt exceeds 1 MiB')
    return receipt


def canonical(receipt: dict) -> bytes:
    try:
        return json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (ValueError, TypeError):
        raise ValueError('Receipt cannot be encoded as finite JSON') from None


def digest(receipt: dict) -> str:
    return hashlib.sha256(canonical(receipt)).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON field')
        value[key] = item
    return value


def load(path: Path) -> dict:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Benchmark receipt exceeds 1 MiB')
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError('Invalid benchmark JSON') from None
    return validate(value)


def save(receipt: dict, path: Path) -> None:
    validate(receipt)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.benchmark-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(canonical(receipt) + b'\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
