"""Validate compact-CUDA parity and the unchanged fixed six-pair decode experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = 'f350cd3ea253a042e205746928271f9d8f71a82d'
CANDIDATE = '8d00b490f5ddf45721546e9b9816a083a46e5fa2'
PROTOCOL_SHA256 = '175a60d4809e4be585e555b991d86ae7f7c40126db304474b3a9d08bc6c1c2a2'
PARITY_SHA256 = 'c2ddeefbdc938118f229c60c088dbbaee1da230edf4624002a1dab787a06e5e5'
PINS = {
    'decode-equivalence/client.py': '0fcbb2f52a94a88e74a34bacb232ed0d729a72e5261136280277f4b37ce2db0b',
    'decode-equivalence/compare.py': '3f32ac9f64dabedb967912eb6876e2cc892efbd7cbb8749b3e05c9d644d12a02',
    'decode-equivalence/protocol.json': 'f86dae04e9f80398d198204380048b5efeaebf828bf71866bf0c077d313ff181',
    'model/client.py': 'e3e531b83e5099c223d984a9f5b7879df6e856e77ac7e6fc4274c7dc177bde92',
}


def require(condition: bool, message: str) -> None:
    """Stop on any missing, malformed or unequal evidence."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Hash an exact source or receipt file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str) -> Any:
    """Import a locally pinned helper without invoking its command line entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'cannot import ' + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen() -> tuple[Any, Any]:
    """Check new protocol pins while retaining the historical comparator's original inputs."""
    require(digest(HERE / 'protocol.json') == PROTOCOL_SHA256, 'compact protocol hash differs')
    for relative, expected in PINS.items():
        require(digest(HERE.parent / relative) == expected, 'reused source hash differs: ' + relative)
    require(digest(HERE / 'parity.py') == PARITY_SHA256, 'parity client hash differs')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    require(protocol['base'] == BASE and protocol['candidate'] == CANDIDATE, 'protocol revision differs')
    require(protocol['reuse_sha256'] == PINS and protocol['parity_client_sha256'] == PARITY_SHA256,
            'protocol source pins differ')
    return (load(HERE.parent / 'decode-equivalence/compare.py', 'compact_reused_compare'),
            load(HERE / 'parity.py', 'compact_parity_client'))


def validate_identity(path: Path, old: Any) -> dict[str, Any]:
    """Require the compact implementation and unchanged baseline, GPU assets and checkpoint."""
    data = json.loads(path.read_text())
    historical = json.loads((HERE.parent / 'model/environment.json').read_text())
    require(data.get('base') == BASE and data.get('candidate') == CANDIDATE, 'identity revisions differ')
    for key in ('shared_fatbin_sha256', 'checkpoint_assets_sha256', 'image_id', 'driver', 'client_sha256'):
        require(data.get(key) == historical[key], 'identity changed historical ' + key)
    native = data.get('native_sha256')
    require(isinstance(native, dict) and set(native) == {'base', 'candidate'}, 'native sides missing')
    require(native['base'] == historical['native_sha256']['base'], 'baseline executable changed')
    require(isinstance(native['candidate'], dict) and
            set(native['candidate']) == set(native['base']), 'candidate native file set')
    require(all(isinstance(value, str) and old.HASH.fullmatch(value)
                for value in native['candidate'].values()), 'candidate native hashes malformed')
    require(native['candidate'] != historical['native_sha256']['candidate'], 'historical candidate relabelled')
    for key, expected in (('parity_client_sha256', PARITY_SHA256),
                          ('decode_client_sha256', PINS['decode-equivalence/client.py']),
                          ('protocol_sha256', PROTOCOL_SHA256)):
        require(data.get(key) == expected, 'identity ' + key + ' differs')
    for key in ('controller_sha256', 'guard_sha256', 'restore_sha256'):
        require(isinstance(data.get(key), str) and old.HASH.fullmatch(data[key]), 'identity ' + key)
    for key, expected in (('context', 8192), ('parallel', 2), ('prompt_cache_gib', 0),
                          ('reserve_gib', 9), ('guard_min_available_gib', 6), ('load_threads', 1)):
        require(data.get(key) == expected, 'identity setting ' + key)
    return data


def validate_parity(data: dict[str, Any], old: Any, parity: Any) -> dict[tuple, tuple]:
    """Recompute eighteen requests, six tokenizations and all nine drafted/plain comparisons."""
    require(data.get('label') == 'P0' and data.get('schema_version') == 1, 'parity header')
    require(data.get('block') == 0 and data.get('side') == 'P' and
            data.get('candidate') == CANDIDATE, 'parity phase revision')
    require(data.get('phase') == 'candidate_drafted_plain_parity', 'parity phase name')
    require(data.get('qualified') is True and data.get('errors') == [], 'parity unqualified')
    require(data.get('scope') == parity.SCOPE and data.get('levels') == [1], 'parity scope/grid')
    require(data.get('repetitions') == 1 and data.get('temperatures') == [0.0, 1.0], 'parity repetitions')
    require(data.get('inference_responses_expected') == data.get('inference_responses') == 18,
            'parity response counts')
    require(data.get('first_cold_counted_in_pairs') is True and data.get('references') == [] and
            data.get('cells') == [], 'parity response accounting')
    require(data.get('solo_references', []) == [], 'unexpected extra parity references')
    require(data.get('pair_order') == [list(key) for key in parity.KEYS], 'parity schedule')
    require(data.get('client_sha256') == PARITY_SHA256 and
            data.get('shared_client_sha256') == PINS['model/client.py'], 'parity source hash')
    old.close(data.get('client_deadline_seconds'), 900.0, 'parity deadline')
    old.close(data.get('requested_reply_tokens'), 256, 'parity reply size')
    metadata = {**data, 'label': 'P0', 'repetitions': 2, 'levels': [1, 2]}
    prompts = old.validate_metadata(metadata, 'P0')
    public, _ = old.frozen_inputs()
    require(all(data['fixtures'][name]['messages'] == public[name] for name in old.FIXTURES),
            'parity public fixtures differ')
    pairs = data.get('pairs')
    require(isinstance(pairs, list) and len(pairs) == 9, 'nine parity pairs required')
    previous = 0.0
    seen = {}
    for key, pair in zip(parity.KEYS, pairs):
        require((pair.get('fixture'), pair.get('temperature'), pair.get('seed')) == key, 'parity pair key')
        require(pair.get('equal') is True, 'parity equality flag')
        for field, draft in (('drafted', True), ('plain', False)):
            run = pair[field]
            old.validate_request(run, (*key, draft), prompts[key[0]], 'parity ' + str(key))
            require(run['sent'] >= previous, 'parity serial chronology')
            previous = run['completed']
        require(parity.fingerprint(pair['drafted']) == parity.fingerprint(pair['plain']),
                'parity drafted/plain output differs')
        seen[key] = parity.fingerprint(pair['drafted'])
    require(data.get('first_cold') == pairs[0]['drafted'], 'cold request must be counted once')
    old.close(data.get('elapsed_s'), data['elapsed_s'], 'parity elapsed')
    require(0 < data['elapsed_s'] <= 900.0, 'parity elapsed bound')
    return seen


def compare_candidate(data: dict[str, Any], parity_data: dict[str, Any], outputs: dict, parity: Any) -> None:
    """Compare all available drafted candidate requests with their validated plain references."""
    for field in ('fixtures', 'helper_sha256', 'tokenize', 'shared_client_sha256'):
        require(data[field] == parity_data[field], 'parity/decode ' + field + ' differs')
    requests = [data['first_cold']] + [ref['run'] for ref in data['solo_references']]
    requests.extend(run for cell in data['cells'] for rep in cell['reps'] for run in rep['runs'])
    for run in requests:
        key = (run['fixture'], run['temperature'], run['seed'])
        require(key in outputs and parity.fingerprint(run) == outputs[key],
                'candidate decode/plain output differs ' + str(key))


def validate_windows(root: Path, expected: dict, manifest: Path, labels: set[str], partial: bool) -> dict:
    """Validate completed window captures with distinct parity and decode client identities."""
    schedule = {0: ['P0'], 1: ['A1', 'B1', 'B2', 'A2'],
                2: ['B3', 'A3', 'A4', 'B4'], 3: ['A5', 'B5', 'B6', 'A6']}
    checked = {}
    canonical = None
    for number, names in schedule.items():
        needed = number == 0 or bool(labels.intersection(names))
        if partial and not needed:
            continue
        directory = root / ('window-' + str(number))
        initial = json.loads((directory / 'initial-identity.json').read_text())
        phase = 'parity_client_sha256' if number == 0 else 'decode_client_sha256'
        for field in ('driver', 'image_id', 'checkpoint_assets_sha256', 'controller_sha256',
                      'guard_sha256', 'restore_sha256', 'protocol_sha256'):
            require(initial[field] == expected[field], 'window identity ' + field)
        require(initial['client_sha256'] == expected[phase], 'window phase client differs')
        require(initial['shared_client_sha256'] == expected['client_sha256'], 'window shared client differs')
        require(initial['expected_environment_sha256'] == digest(manifest), 'window manifest bytes differ')
        for side in ('base', 'candidate'):
            require(initial[side]['commit'] == expected[side], 'window source revision differs')
            require(initial[side]['native_sha256'] == expected['native_sha256'][side], 'window native differs')
            require(initial[side]['fatbin_sha256'] == expected['shared_fatbin_sha256'], 'window fatbins differ')
        normalized = {key: value for key, value in initial.items() if key != 'client_sha256'}
        require(canonical is None or normalized == canonical, 'cross-window artifact identity differs')
        canonical = normalized
        settings = json.loads((directory / 'settings.json').read_text())
        require(settings['window'] == number and settings['labels'] == names, 'window schedule differs')
        for field, value in (('context', 8192), ('parallel', 2), ('prompt_cache_gib', 0),
                             ('memory_reserve_gib', 9), ('min_available_gib', 6), ('load_threads', 1)):
            require(settings[field] == value, 'window setting differs ' + field)
        outcome = json.loads((directory / 'outcome.json').read_text())
        require(outcome.get('success') is True and outcome.get('failure') is None and
                outcome.get('cleanup_errors') == [], 'window did not restore successfully')
        require(not list(directory.glob('*-guard.event.json')), 'window guard event recorded')
        identities = {'initial-identity.json'} | {
            label + '-' + stage + '-identity.json' for label in names for stage in ('before', 'after')}
        actual = {path.name for path in directory.glob('*-identity.json')}
        require(actual == identities, 'missing or unexpected window boundary identity captures')
        for name in sorted(identities):
            path = directory / name
            require(json.loads(path.read_text()) == initial, 'within-window artifact identity differs')
        checked[str(number)] = {'phase_client_sha256': initial['client_sha256'], 'success': True}
    return checked


def verify(args: argparse.Namespace, old: Any, parity: Any) -> dict[str, Any]:
    """Check exactness before delegating fixed-sample statistics to the unchanged comparator."""
    expected = validate_identity(args.identity, old)
    parity_data = json.loads(args.parity.read_text())
    outputs = validate_parity(parity_data, old, parity)
    paths = {} if args.directory is None else {p.stem: p for p in args.directory.glob('[AB]*.json')}
    if args.validate_parity:
        require(not paths, 'parity-only mode does not accept decode receipts')
    else:
        require(args.directory is not None, 'decode directory required')
        require(args.partial or args.windows is not None, '--windows required for the complete gate')
        for label, path in paths.items():
            data = json.loads(path.read_text())
            old.validate_run(data, label)
            require(data['first_cold']['sent'] >= parity_data['pairs'][-1]['plain']['completed'],
                    label + ': decode must follow the parity phase')
            for field in ('fixtures', 'helper_sha256', 'tokenize', 'shared_client_sha256'):
                require(data[field] == parity_data[field], label + ': parity/decode ' + field)
            if label.startswith('B'):
                compare_candidate(data, parity_data, outputs, parity)
    windows = (validate_windows(args.windows, expected, args.identity, set(paths),
                                args.partial or args.validate_parity)
               if args.windows is not None else None)
    if args.validate_parity:
        result: dict[str, Any] = {'evidence_valid': True, 'analysis_complete': False,
                                  'status': 'validated_parity_only_no_intervals'}
    else:
        result = old.verify(args.directory, args.partial)
        result['reused_comparator_protocol_sha256'] = result.pop('protocol_sha256')
        result['decode_phase_plain_256_test'] = result.pop('plain_256_test')
    result.update({
        'base': BASE, 'candidate': CANDIDATE, 'protocol_sha256': PROTOCOL_SHA256,
        'parity_client_sha256': PARITY_SHA256, 'reused_sources_sha256': PINS,
        'identity_manifest_sha256': digest(args.identity), 'parity_input_sha256': digest(args.parity),
        'parity_responses': 18, 'parity_keys': 9, 'drafted_plain_equality': True,
        'available_candidate_decode_plain_equality': True,
        'candidate_decode_runs_compared': sorted(label for label in paths if label.startswith('B')),
        'window_captures': windows, 'cached_resume': 'not_run; prompt cache disabled',
    })
    return result


def synthetic_parity(old: Any, parity: Any) -> dict[str, Any]:
    """Build a deterministic eighteen-response oracle from existing synthetic request evidence."""
    source = old.synthetic_run('B1')
    requests = [source['first_cold']] + [ref['run'] for ref in source['solo_references']]
    requests.extend(run for cell in source['cells'] for rep in cell['reps'] for run in rep['runs'])
    data = copy.deepcopy(source)
    data.update(label='P0', phase='candidate_drafted_plain_parity', block=0, side='P', candidate=CANDIDATE,
                scope=parity.SCOPE, levels=[1], repetitions=1, inference_responses_expected=18,
                inference_responses=18, first_cold_counted_in_pairs=True, pair_order=[list(k) for k in parity.KEYS],
                client_sha256=PARITY_SHA256, references=[], cells=[], pairs=[], elapsed_s=100.0)
    data.pop('solo_references')
    clock = 10.0
    for key in parity.KEYS:
        original = next(run for run in requests if (run['fixture'], run['temperature'], run['seed']) == key)
        pair = dict(zip(('fixture', 'temperature', 'seed'), key))
        for field, draft in (('drafted', True), ('plain', False)):
            run = copy.deepcopy(original)
            offset = clock - run['sent']
            for name in ('sent', 'first', 'last', 'completed'):
                run[name] += offset
            for piece in run['pieces']:
                piece[0] += offset
            run['draft'] = draft
            pair[field] = run
            clock = run['completed'] + 0.1
        pair['equal'] = True
        data['pairs'].append(pair)
    data['first_cold'] = data['pairs'][0]['drafted']
    return data


def self_test(old: Any, parity: Any) -> dict[str, Any]:
    """Exercise reused statistics, parity validation and cross-phase output rejection offline."""
    reused = old.self_test()
    data = synthetic_parity(old, parity)
    outputs = validate_parity(data, old, parity)
    compare_candidate(old.synthetic_run('B1'), data, outputs, parity)
    mutations = (
        lambda d: d['pairs'].pop(),
        lambda d: d['pairs'][1]['plain'].update(cached_tokens=1),
        lambda d: d['pairs'][1]['plain'].update(complete=False),
        lambda d: d['pairs'][1]['plain'].update(output_sha256='f' * 64),
        lambda d: d['pairs'][1]['plain'].update(token_sha256='f' * 64),
        lambda d: d['pairs'][1]['plain'].update(draft=True),
        lambda d: d['pairs'][1]['plain'].update(first=d['pairs'][1]['plain']['last']),
        lambda d: d.update(inference_responses=19),
        lambda d: d.update(candidate=BASE),
    )
    for mutate in mutations:
        changed = copy.deepcopy(data)
        mutate(changed)
        try:
            validate_parity(changed, old, parity)
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError('malformed parity accepted')
    changed = old.synthetic_run('B1')
    changed['first_cold']['token_ids'][0] = 999
    try:
        compare_candidate(changed, data, outputs, parity)
    except ValueError:
        pass
    else:
        raise AssertionError('cross-phase mismatch accepted')
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        manifest = directory / 'environment.json'
        expected = json.loads((HERE.parent / 'model/environment.json').read_text())
        expected.update(candidate=CANDIDATE, parity_client_sha256=PARITY_SHA256,
                        decode_client_sha256=PINS['decode-equivalence/client.py'],
                        protocol_sha256=PROTOCOL_SHA256, controller_sha256='c' * 64,
                        guard_sha256='d' * 64, restore_sha256='e' * 64)
        expected['native_sha256']['candidate'] = {'native/bin/tensorfold-native': 'f' * 64}
        manifest.write_text(json.dumps(expected))
        validate_identity(manifest, old)
        parity_path = directory / 'P0.json'
        parity_path.write_text(json.dumps(data))
        decode = directory / 'decode'
        decode.mkdir()
        args = argparse.Namespace(identity=manifest, parity=parity_path, directory=None,
                                  validate_parity=True, windows=None, partial=True)
        require(verify(args, old, parity)['status'] == 'validated_parity_only_no_intervals',
                'parity-only mode')
        for label in old.LABELS:
            (decode / (label + '.json')).write_text(json.dumps(old.synthetic_run(label)))
        args.directory, args.validate_parity = decode, False
        partial = verify(args, old, parity)
        require('cells' not in partial and 'gate_pass' not in partial, 'partial emitted interval')
        require(len(partial['candidate_decode_runs_compared']) == 6, 'partial candidate parity count')
        window_root = directory / 'windows'
        for number, labels in enumerate((['P0'], ['A1', 'B1', 'B2', 'A2'],
                                         ['B3', 'A3', 'A4', 'B4'], ['A5', 'B5', 'B6', 'A6'])):
            window = window_root / ('window-' + str(number))
            window.mkdir(parents=True)
            actual = {key: expected[key] for key in ('driver', 'image_id', 'checkpoint_assets_sha256',
                      'controller_sha256', 'guard_sha256', 'restore_sha256', 'protocol_sha256')}
            actual.update(client_sha256=PARITY_SHA256 if number == 0 else PINS['decode-equivalence/client.py'],
                          shared_client_sha256=expected['client_sha256'],
                          expected_environment_sha256=digest(manifest))
            for side in ('base', 'candidate'):
                actual[side] = {'commit': expected[side], 'native_sha256': expected['native_sha256'][side],
                                'fatbin_sha256': expected['shared_fatbin_sha256'], 'source_sha256': {}}
            names = ['initial-identity.json'] + [label + '-' + stage + '-identity.json'
                                               for label in labels for stage in ('before', 'after')]
            for name in names:
                (window / name).write_text(json.dumps(actual))
            (window / 'settings.json').write_text(json.dumps({
                'window': number, 'labels': labels, 'context': 8192, 'parallel': 2, 'prompt_cache_gib': 0,
                'memory_reserve_gib': 9, 'min_available_gib': 6, 'load_threads': 1}))
            (window / 'outcome.json').write_text(json.dumps({
                'success': True, 'failure': None, 'cleanup_errors': []}))
        args.partial, args.windows = False, window_root
        require(verify(args, old, parity)['gate_pass'] is True, 'full wrapper gate')
        (window_root / 'window-1/A1-before-identity.json').unlink()
        try:
            verify(args, old, parity)
        except ValueError:
            pass
        else:
            raise AssertionError('missing boundary identity accepted')
        expected['candidate'] = BASE
        manifest.write_text(json.dumps(expected))
        try:
            validate_identity(manifest, old)
        except ValueError:
            pass
        else:
            raise AssertionError('wrong implementation identity accepted')
    return {'status': 'passed', 'reused_comparator': reused, 'parity_responses': 18,
            'parity_keys': len(outputs), 'parity_bad_receipts_rejected': len(mutations),
            'candidate_cross_phase_mismatch_rejected': True,
            'wrapper_modes': ['parity_only', 'partial_no_intervals', 'complete_fixed_sample'],
            'wrong_implementation_identity_rejected': True, 'missing_boundary_identity_rejected': True,
            'phase_specific_client_identities_validated': True}


def main() -> int:
    """Validate partial evidence without intervals or the complete frozen paired sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path)
    parser.add_argument('--parity', type=Path)
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--windows', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--validate-parity', action='store_true')
    parser.add_argument('--validate-run', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        old, parity = frozen()
        if args.self_test:
            result = self_test(old, parity)
        elif args.validate_run:
            checked = old.validate_run(json.loads(args.validate_run.read_text()), args.validate_run.stem)
            result = {'status': 'validated_single_run_only_no_intervals', 'responses': checked[2]}
        else:
            require(args.identity is not None and args.parity is not None, '--identity and --parity required')
            result = verify(args, old, parity)
        code = 2 if result.get('gate_pass') is False else 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {'evidence_valid': False, 'status': 'invalid', 'error': str(error)}
        code = 1
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
