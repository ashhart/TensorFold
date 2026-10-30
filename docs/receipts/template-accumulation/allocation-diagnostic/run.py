"""Attribute selected allocations during bounded template renders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from instrument import instrument

CORPUS_SHA256 = 'f5ac1de9e7226a701beda7887c280ee090c0694d3f6118945e3c39167b78e778'
TEMPLATE_SHA256 = '7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e'
COMMITS = {'base': 'f350cd3ea253a042e205746928271f9d8f71a82d',
           'candidate': '9dac03d6be8871d2d4c9404e53e9d44c07d7cd6a'}
ORDER = (('base', 'control'), ('base', 'instrumented'),
         ('candidate', 'control'), ('candidate', 'instrumented'))


def digest(path: Path) -> str:
    """Hash a file without embedding the external template in this receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_revision(repo: Path, zig: str, workspace: Path, side: str, mode: str, kind: str, harness: bytes) -> Path:
    """Build a standalone renderer test from a pinned revision."""
    source = workspace / (side + '-' + mode + '-' + kind)
    source.mkdir()
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', COMMITS[side],
                                       'zig/src/core/template', 'zig/src/platform'])
    subprocess.run(['tar', '-xf', '-', '-C', str(source)], input=archive, check=True)
    (source / 'zig/src/core/template/render_test.zig').write_bytes(harness)
    (source / 'zig/src/core/template/diagnostics.zig').write_bytes(Path(__file__).with_name('diagnostics.zig').read_bytes())
    if kind == 'instrumented':
        instrument(source, side)
    binary = workspace / (side + '-' + mode + '-' + kind + '.bin')
    subprocess.run([zig, 'test', '-O', mode, '--test-filter', 'external template case', '--test-no-exec',
                    f'-femit-bin={binary}', '--dep', 'sys', '-Mroot=zig/src/core/template/render_test.zig',
                    '-Msys=zig/src/platform/sys.zig', '-lc'], cwd=source, check=True)
    return binary


def render(binary: Path, template: Path, fixture: Path, log: Path) -> dict:
    """Run a fresh process and preserve correctness and allocator evidence."""
    env = os.environ.copy()
    env.update(TF_TEMPLATE=str(template), TF_CASE=str(fixture))
    try:
        result = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b''
        stderr = error.stderr or b''
        text = (stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout) + (
            stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr)
        log.write_text(text + '\nTIMEOUT after 180 seconds\n')
        return {'exit_code': None, 'valid': False, 'status': 'timeout', 'samples': []}
    text = result.stdout + result.stderr
    log.write_text(text)
    compiles = re.findall(r'COMPILE standing=(\d+) peak=(\d+)', text)
    samples = [(int(rep), int(peak), int(size), int(spans), sha) for rep, peak, size, spans, sha in re.findall(
        r'RENDER rep=(\d+) scratch_peak=(\d+) bytes=(\d+) spans=(\d+) spans_sha256=([0-9a-f]{64})', text)]
    errors = re.findall(r'(COMPILE_ERROR|RENDER_ERROR|MISMATCH) ([^\n]+)', text)
    valid = (result.returncode == 0 and len(compiles) == 1 and len(samples) == 1 and samples[0][0] == 0)
    capped = (result.returncode == 1 and 'CAP live=' in text and len(errors) == 1
              and errors[0][0] in ('RENDER_ERROR', 'COMPILE_ERROR')
              and errors[0][1].startswith('OutOfMemory ')
              and not any(marker in text.lower() for marker in ('scratchnotreleased', 'leaked', 'panic', 'mismatch')))
    status = 'valid' if valid else ('cap_exceeded' if capped else 'unexpected_failure')
    record = {'exit_code': result.returncode, 'valid': valid, 'status': status, 'errors': errors,
              'compile_standing': int(compiles[0][0]) if compiles else None,
              'compile_peak': int(compiles[0][1]) if compiles else None,
              'samples': [{'rep': rep, 'scratch_peak': peak, 'bytes': size, 'spans': spans,
                           'spans_sha256': sha} for rep, peak, size, spans, sha in samples]}
    sizes = re.findall(r'SIZES str=(\d+) value=(\d+)', text)
    joins = re.findall(r'JOINS ((?:\d+ ?){9})', text)
    allocs = re.findall(r'ALLOC (string|list) ((?:\d+ ?){15})', text)
    if valid and (len(sizes) != 1 or len(joins) != 1 or sorted(label for label, _ in allocs) != ['list', 'string']):
        record['status'] = 'unexpected_failure'
        record['valid'] = False
    record['sizes'] = list(map(int, sizes[0])) if sizes else None
    record['joins'] = list(map(int, joins[0].split())) if joins else None
    record['allocations'] = {label: list(map(int, values.split())) for label, values in allocs}
    if record['joins'] is not None:
        j = record['joins']
        if j[0] != j[2] + j[3] or j[1] > j[0] or j[5] > j[4] or j[7] > j[6]:
            raise RuntimeError('inconsistent join counters')
        if valid and (j[0] != j[1] or j[6] != j[7]):
            raise RuntimeError('successful render has incomplete concatenations')
    for counts in record['allocations'].values():
        if (counts[0] != counts[2] + counts[4] or counts[1] < counts[3]
                or counts[5] != counts[7] + counts[9] or counts[6] < counts[8]
                or counts[10] != counts[12] + counts[14] or counts[11] < counts[13]):
            raise RuntimeError('inconsistent allocator counters')
    return record


def main() -> None:
    """Verify input identities and run Debug, ReleaseFast and ReleaseSafe control and instrumented cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--zig', default='zig')
    parser.add_argument('--case', action='append', help='run only this corpus case ID; repeatable')
    args = parser.parse_args()
    template = args.template.resolve(strict=True)
    if digest(template) != TEMPLATE_SHA256:
        raise ValueError('template checksum mismatch')
    if digest(args.corpus) != CORPUS_SHA256:
        raise ValueError('frozen corpus checksum mismatch')
    corpus = json.loads(args.corpus.read_text())
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list of case objects')
    if any(not isinstance(c, dict) or not isinstance(c.get('id'), str) or not isinstance(c.get('expected'), str)
           for c in corpus) or len({c['id'] for c in corpus}) != len(corpus):
        raise ValueError('invalid or duplicate corpus cases')
    cases = [c for c in corpus if args.case is None or c['id'] in args.case]
    if not cases or (args.case is not None and set(args.case) != {c['id'] for c in cases}):
        raise ValueError('requested case absent from corpus')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    harness = Path(__file__).with_name('render_test.zig').read_bytes()
    records = []
    zig_version = subprocess.check_output([args.zig, 'version'], text=True).strip()
    if zig_version != '0.17.0':
        raise RuntimeError(f'required Zig 0.17.0, got {zig_version}')
    metadata = {
        'template_sha256': TEMPLATE_SHA256, 'corpus_sha256': digest(args.corpus),
        'case_expected_sha256': {c['id']: hashlib.sha256(c['expected'].encode('utf-8')).hexdigest() for c in cases},
        'harness_sha256': hashlib.sha256(harness).hexdigest(),
        'instrument_sha256': digest(Path(__file__).with_name('instrument.py')),
        'diagnostics_sha256': digest(Path(__file__).with_name('diagnostics.zig')),
        'runner_sha256': digest(Path(__file__)), 'zig_version': zig_version,
        'commits': COMMITS, 'order': ORDER, 'renders_per_process': 1,
        'allocation_scope': 'logical requests at evaluator string joins and list + only; not upstream peak',
        'join_counters': ['attempts', 'completed', 'reuse_attempts', 'new_attempts', 'lhs_seed_bytes_attempted',
                          'lhs_seed_bytes_completed', 'list_attempts', 'list_payloads_copied', 'value_elements_copied'],
        'allocation_counters': ['alloc_calls', 'alloc_bytes_requested', 'alloc_success', 'alloc_bytes_success',
                                'alloc_failed', 'resize_calls', 'resize_growth_requested', 'resize_success',
                                'resize_growth_success', 'resize_failed', 'remap_calls', 'remap_growth_requested',
                                'remap_success', 'remap_growth_success', 'remap_failed'],
        'limitations': ['LHS seeding excludes resize or remap relocation copies.',
                        'String join allocator includes builder, text and span allocations.',
                        'List allocator excludes List headers, maps, frames, list literals and list slices.',
                        'A list payload can succeed before its List header allocation fails.',
                        'Prefix completed counts successful whole prefix appends, not partial failed appends.',
                        'Resize and remap counts are logical growth and can overlap other request totals.'],
        'combined_compile_render_cap_bytes': 240 * 1024 * 1024,
        'provenance': 'ordered, nonempty, disjoint, in-bounds; SHA256 of decimal start:end newline per span',
        'binaries': {},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='deepseek-template-') as tmp:
        workspace = Path(tmp)
        binaries = {}
        for mode in ('Debug', 'ReleaseFast', 'ReleaseSafe'):
            for side, kind in ORDER:
                binary = compile_revision(args.repo.resolve(), args.zig, workspace, side, mode, kind, harness)
                binaries[side, mode, kind] = binary
                metadata['binaries'][side + '-' + mode + '-' + kind] = digest(binary)
                (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for index, case in enumerate(cases):
            fixture = workspace / f'case-{index}.json'
            fixture.write_text(json.dumps(case, ensure_ascii=False))
            case_template = template
            if 'template_source' in case:
                if not isinstance(case['template_source'], str):
                    raise ValueError('synthetic template source must be a string')
                case_template = workspace / f'template-{index}.jinja'
                case_template.write_text(case['template_source'])
                metadata.setdefault('synthetic_templates_sha256', {})[case['id']] = digest(case_template)
                (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            for mode in ('Debug', 'ReleaseFast', 'ReleaseSafe'):
                for position, (side, kind) in enumerate(ORDER):
                    print(f'RUN {case["id"]} {mode} {side} {kind}', flush=True)
                    item = render(binaries[side, mode, kind], case_template, fixture, out / f'{index}-{mode}-{position}.log')
                    item.update(case=case['id'], mode=mode, side=side, kind=kind, position=position)
                    if item['valid'] and kind == 'control' and (any(item['joins']) or
                            any(any(values) for values in item['allocations'].values())):
                        raise RuntimeError('unchanged control has nonzero instrumentation counters')
                    if item['valid'] and kind == 'instrumented':
                        info = case.get('diagnostic', {})
                        expected_calls = info.get('expected_list_concat_calls')
                        expected_elements = info.get('expected_list_concat_elements')
                        if expected_calls is not None and item['joins'][6:8] != [expected_calls, expected_calls]:
                            raise RuntimeError(f'list concat count differs: {case["id"]}')
                        if expected_elements is not None and item['joins'][8] != expected_elements:
                            raise RuntimeError(f'list copied element count differs: {case["id"]}')
                        item['list_value_copy_bytes'] = item['joins'][8] * item['sizes'][1]
                    records.append(item)
                    (out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    comparisons = []
    for case in cases:
        for mode in ('Debug', 'ReleaseFast', 'ReleaseSafe'):
            for side in COMMITS:
                pair = {row['kind']: row for row in records
                        if row['case'] == case['id'] and row['mode'] == mode and row['side'] == side}
                control, measured = pair['control'], pair['instrumented']
                entry = {'case': case['id'], 'mode': mode, 'side': side,
                         'same_status': control['status'] == measured['status'],
                         'control_status': control['status'], 'instrumented_status': measured['status']}
                if control['valid'] and measured['valid']:
                    a, b = control['samples'][0], measured['samples'][0]
                    entry.update(same_span_hash=a['spans_sha256'] == b['spans_sha256'],
                                 control_scratch_peak=a['scratch_peak'], instrumented_scratch_peak=b['scratch_peak'],
                                 scratch_peak_delta=b['scratch_peak'] - a['scratch_peak'],
                                 compile_standing_delta=measured['compile_standing'] - control['compile_standing'])
                comparisons.append(entry)
    (out / 'control-comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    if any(not entry['same_status'] or entry.get('same_span_hash') is False for entry in comparisons):
        raise RuntimeError('control and instrumented results differ')
    for case in cases:
        successful = [row for row in records if row['case'] == case['id'] and row['valid']]
        span_values = {(row['samples'][0]['spans'], row['samples'][0]['spans_sha256']) for row in successful}
        if len(span_values) > 1:
            raise RuntimeError(f'provenance spans differ: {case["id"]}')
    invalid = sum(not row['valid'] for row in records)
    print(json.dumps({'cases': len(cases), 'runs': len(records), 'invalid_runs': invalid}))
    if any(row['status'] in ('timeout', 'unexpected_failure') for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
