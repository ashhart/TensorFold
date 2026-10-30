"""Compare three pinned revisions using unchanged bounded allocation and compatibility controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

BASE = 'f350cd3ea253a042e205746928271f9d8f71a82d'
PRIOR = 'd33e5e769d7a4bb801cc39e0a32400488cb89777'
TEMPLATE_SHA256 = '7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e'
CORPORA = {
    'allocation': ('f5ac1de9e7226a701beda7887c280ee090c0694d3f6118945e3c39167b78e778', 13),
    'compatibility': ('f5b8a240a9e852aaad7efbc3335942e8089812a2d7b5e5783b83f3c9dab4b74d', 21),
}
HARNESS_HASHES = {
    'render_test.zig': '9110f0b70bc6b82f8e172620b5179c21f0e54b90847adc7ba241232185586b4b',
    'diagnostics.zig': '9dd133ffa7bea5d34e01ca6c686bb254a8e81207eab0ff2a9edf631238a957f3',
}
MODES = ('Debug', 'ReleaseFast', 'ReleaseSafe')
SIDES = ('base', 'prior', 'revised')
CAP = 240 * 1024 * 1024
ALLOWED_BASE_CAPS = {
    'tool-cycles-128-object-8192', 'tool-cycles-128-string-8192',
    'tool-cycles-128-object-8192-catalog-240kib',
}
PROCESS_TIMEOUT = 180
WHOLE_TIMEOUT = 1800
HASH = re.compile(r'[0-9a-f]{40}\Z')


def require(condition: bool, message: str) -> None:
    """Reject incomplete or inconsistent evidence without silently changing the matrix."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Fingerprint original bytes while retaining only relative artifact names in public metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    """Checkpoint JSON atomically after each complete process."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_corpus(path: Path, name: str) -> list[dict[str, Any]]:
    """Require the existing frozen corpus rather than accepting replacements or selected cases."""
    checksum, count = CORPORA[name]
    require(digest(path) == checksum, name + ': frozen corpus checksum differs')
    cases = json.loads(path.read_bytes())
    require(isinstance(cases, list) and len(cases) == count, name + ': corpus count')
    require(all(isinstance(c, dict) and isinstance(c.get('id'), str) and
                isinstance(c.get('expected'), str) for c in cases), name + ': corpus records')
    require(len({c['id'] for c in cases}) == count, name + ': duplicate IDs')
    for case in cases:
        if 'template_source' in case:
            require(name == 'allocation' and case['id'] == 'synthetic-interleaved-breaker-control',
                    'unexpected synthetic template override')
            require(isinstance(case['template_source'], str), 'synthetic template must be a string')
    return cases


def remaining(deadline: float, maximum: int) -> float:
    """Apply one whole-matrix deadline as well as each subprocess bound."""
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError('whole matrix deadline')
    return min(value, maximum)


def build(
    repo: Path, zig: str, workspace: Path, side: str, commit: str, mode: str,
    harnesses: dict[str, bytes], output: Path, deadline: float,
) -> tuple[Path, dict[str, str]]:
    """Compile an unmodified production archive with the original control harness and zero probes."""
    directory = workspace / (side + '-' + mode)
    directory.mkdir()
    archive = subprocess.check_output(
        ['git', '-C', str(repo), 'archive', commit, 'zig/src/core/template', 'zig/src/platform'],
        timeout=remaining(deadline, PROCESS_TIMEOUT),
    )
    subprocess.run(['tar', '-xf', '-', '-C', str(directory)], input=archive, check=True,
                   timeout=remaining(deadline, PROCESS_TIMEOUT))
    source_hashes = {str(path.relative_to(directory)): digest(path)
                     for path in sorted(directory.rglob('*')) if path.is_file()}
    for name, source in harnesses.items():
        (directory / 'zig/src/core/template' / name).write_bytes(source)
    binary = workspace / (side + '-' + mode + '.bin')
    command = [zig, 'test', '-O', mode, '--test-filter', 'external template case', '--test-no-exec',
               '-femit-bin=' + str(binary), '--dep', 'sys',
               '-Mroot=zig/src/core/template/render_test.zig', '-Msys=zig/src/platform/sys.zig', '-lc']
    with (output / ('compile-' + side + '-' + mode + '.log')).open('wb') as log:
        subprocess.run(command, cwd=directory, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=remaining(deadline, PROCESS_TIMEOUT))
    return binary, source_hashes


def parse_log(text: str, code: int | None, timed_out: bool = False) -> dict[str, Any]:
    """Independently classify complete renders and preserve every clean cap refusal request."""
    compiles = re.findall(r'COMPILE standing=(\d+) peak=(\d+)', text)
    sizes = re.findall(r'SIZES str=(\d+) value=(\d+)', text)
    sample_matches = re.findall(
        r'RENDER rep=(\d+) scratch_peak=(\d+) bytes=(\d+) spans=(\d+) spans_sha256=([0-9a-f]{64})', text,
    )
    samples = [{'rep': int(rep), 'scratch_peak': int(peak), 'bytes': int(size),
                'spans': int(spans), 'spans_sha256': sha} for rep, peak, size, spans, sha in sample_matches]
    errors = [list(x) for x in re.findall(r'(COMPILE_ERROR|RENDER_ERROR|MISMATCH) ([^\n]+)', text)]
    caps = [{'live_bytes': int(live), 'request_bytes': int(request),
             'would_be_combined_bytes': int(live) + int(request), 'cap_bytes': CAP}
            for live, request in re.findall(r'CAP live=(\d+) request=(\d+)', text)]
    joins = re.findall(r'^JOINS ([0-9 ]+)$', text, re.M)
    allocations = re.findall(r'^ALLOC (string|list) ([0-9 ]+)$', text, re.M)
    zero = (len(joins) == 1 and len(joins[0].split()) == 9 and
            all(value == '0' for value in joins[0].split()) and
            sorted(name for name, _ in allocations) == ['list', 'string'] and
            all(len(values.split()) == 15 and all(v == '0' for v in values.split())
                for _, values in allocations))
    forbidden = re.search(r'(?i)ScratchNotReleased|leaked|panic|MISMATCH|ExpectedProvenanceDiffers', text)
    valid = (not timed_out and code == 0 and len(compiles) == len(sizes) == 1 and zero and
             len(samples) == 1 and samples[0]['rep'] == 0 and not errors and not caps and
             forbidden is None and 'All 2 tests passed.' in text)
    capped = (not timed_out and code == 1 and len(compiles) == len(sizes) == 1 and zero and
              not samples and caps and len(errors) == 1 and errors[0][0] == 'RENDER_ERROR' and
              re.fullmatch(r'OutOfMemory rep=0 peak=\d+', errors[0][1]) is not None and
              forbidden is None and '1 passed; 0 skipped; 1 failed.' in text)
    status = 'valid' if valid else ('cap_exceeded' if capped else 'timeout' if timed_out else 'unexpected_failure')
    row: dict[str, Any] = {
        'exit_code': code, 'valid': valid, 'status': status, 'errors': errors, 'samples': samples,
        'compile_standing': int(compiles[0][0]) if len(compiles) == 1 else None,
        'compile_peak': int(compiles[0][1]) if len(compiles) == 1 else None,
        'sizes': list(map(int, sizes[0])) if len(sizes) == 1 else None,
        'probe_counters_zero': zero, 'cap_refusals': caps,
    }
    if capped:
        row['refused_render_scratch_peak'] = int(errors[0][1].split('peak=')[1])
    return row


def render(binary: Path, template: Path, fixture: Path, log: Path, deadline: float) -> dict[str, Any]:
    """Launch one private control process without collecting a timing measurement."""
    env = os.environ.copy()
    env.update(TF_TEMPLATE=str(template), TF_CASE=str(fixture))
    try:
        process = subprocess.run([str(binary)], env=env, capture_output=True,
                                 timeout=remaining(deadline, PROCESS_TIMEOUT))
        text = (process.stdout + process.stderr).decode('utf-8', errors='replace')
        row = parse_log(text, process.returncode)
    except subprocess.TimeoutExpired as error:
        text = ((error.stdout or b'') + (error.stderr or b'')).decode('utf-8', errors='replace')
        row = parse_log(text, None, True)
    log.write_text(text)
    return row


def qualify(row: dict[str, Any], case: dict[str, Any], corpus: str, side: str) -> list[str]:
    """Enforce compact value layout, complete bytes, bounded scratch and justified refusal scope."""
    failures = []
    if row['sizes'] != ([48, 56] if side == 'prior' else [40, 48]):
        failures.append('unexpected_value_layout')
    if row['valid']:
        sample = row['samples'][0]
        if sample['bytes'] != len(case['expected'].encode('utf-8')):
            failures.append('unexpected_output_byte_count')
        if row['compile_standing'] + sample['scratch_peak'] > CAP:
            failures.append('combined_cap_exceeded')
    elif row['status'] == 'cap_exceeded':
        if not (corpus == 'allocation' and side == 'base' and case['id'] in ALLOWED_BASE_CAPS):
            failures.append('required_case_did_not_complete')
        peak = row['refused_render_scratch_peak'] + row['compile_standing']
        if peak > CAP or any(not (c['live_bytes'] <= peak and c['would_be_combined_bytes'] > CAP)
                             for c in row['cap_refusals']):
            failures.append('invalid_cap_accounting')
    else:
        failures.append('incomplete_or_unexpected_failure')
    return failures


def compare(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare all complete identities and show every three-way peak delta without timing claims."""
    cells: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    identities: dict[tuple[str, str], set[tuple[int, int, str]]] = {}
    for row in records:
        cells.setdefault((row['corpus'], row['case'], row['mode']), {})[row['side']] = row
        if row['valid']:
            s = row['samples'][0]
            identities.setdefault((row['corpus'], row['case']), set()).add((s['bytes'], s['spans'], s['spans_sha256']))
    require(all(len(values) == 1 for values in identities.values()), 'completed span identities differ')
    rows = []
    for key, group in cells.items():
        require(set(group) == set(SIDES), 'three-version cell incomplete')
        peaks = {side: row['samples'][0]['scratch_peak'] if row['valid'] else None
                 for side, row in group.items()}
        revised = peaks['revised']
        changes = {side: {'bytes': revised - peaks[side], 'percent': 100 * (revised / peaks[side] - 1)}
                   if revised is not None and peaks[side] is not None else None for side in ('base', 'prior')}
        rows.append({'corpus': key[0], 'case': key[1], 'mode': key[2], 'scratch_peaks': peaks,
                     'revised_vs': changes, 'status': {side: row['status'] for side, row in group.items()}})
    return {'cells': rows, 'completed_output_and_span_identity': True,
            'allocation_scope': 'Combined cap; incremental render scratch; output/input outside cap; not RSS.',
            'timing_scope': 'No timing observations or speed claims.'}


def self_test() -> None:
    """Exercise failure classification offline without compiling or launching any renderer."""
    prefix = ('COMPILE standing=100 peak=100\nSIZES str=40 value=48\nJOINS ' + ' '.join(['0'] * 9) +
              '\nALLOC string ' + ' '.join(['0'] * 15) + '\nALLOC list ' + ' '.join(['0'] * 15) + '\n')
    good = prefix + 'RENDER rep=0 scratch_peak=100 bytes=1 spans=1 spans_sha256=' + 'a' * 64 + '\n'
    good += 'All 2 tests passed.\n'
    require(parse_log(good, 0)['valid'], 'valid parser control')
    cap = (prefix + f'CAP live=200 request={CAP}\nRENDER_ERROR OutOfMemory rep=0 peak=100\n'
           '1 passed; 0 skipped; 1 failed.\n')
    require(parse_log(cap, 1)['status'] == 'cap_exceeded', 'clean cap parser control')
    for changed in (cap + 'ScratchNotReleased', cap + 'leaked', cap + 'panic', cap + 'MISMATCH',
                    good.replace('All 2 tests passed.', ''), good.replace('JOINS 0', 'JOINS 1')):
        require(parse_log(changed, 1 if changed.startswith(cap) else 0)['status'] == 'unexpected_failure',
                'malformed parser control accepted')
    require(parse_log(good, None, True)['status'] == 'timeout', 'timeout parser control')
    print('OFFLINE SELF-TEST PASS: valid, cap, six negative controls and timeout; no renderer executed.')


def main() -> int:
    """Run the exact frozen matrix only after an explicit complete revised commit is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent.parents[3])
    parser.add_argument('--zig', default='zig')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--allocation-corpus', type=Path)
    parser.add_argument('--compatibility-corpus', type=Path)
    parser.add_argument('--candidate')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(all(value is not None for value in (args.template, args.allocation_corpus,
            args.compatibility_corpus, args.candidate, args.output)), 'all matrix arguments are required')
    require(HASH.fullmatch(args.candidate) is not None, 'candidate must be a complete reviewed commit SHA')
    repo = args.repo.resolve(strict=True)
    commits = {'base': BASE, 'prior': PRIOR, 'revised': args.candidate}
    for commit in commits.values():
        subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', commit + '^{commit}'], check=True)
    subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', PRIOR, args.candidate], check=True)
    template = args.template.resolve(strict=True)
    require(digest(template) == TEMPLATE_SHA256, 'exact template checksum differs')
    corpora = {'allocation': load_corpus(args.allocation_corpus, 'allocation'),
               'compatibility': load_corpus(args.compatibility_corpus, 'compatibility')}
    source_directory = Path(__file__).resolve().parent.parent / 'allocation-diagnostic'
    harnesses = {name: (source_directory / name).read_bytes() for name in HARNESS_HASHES}
    require(all(hashlib.sha256(data).hexdigest() == HARNESS_HASHES[name] for name, data in harnesses.items()),
            'unchanged control harness checksum differs')
    require(subprocess.check_output([args.zig, 'version'], text=True).strip() == '0.17.0', 'Zig 0.17.0 required')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    metadata = {'commits': commits, 'modes': MODES, 'sides': SIDES, 'planned_processes': 306,
                'template_sha256': TEMPLATE_SHA256, 'harness_sha256': HARNESS_HASHES,
                'runner_sha256': digest(Path(__file__)), 'zig_version': '0.17.0', 'corpora': CORPORA,
                'combined_cap_bytes': CAP, 'process_timeout_seconds': PROCESS_TIMEOUT,
                'whole_timeout_seconds': WHOLE_TIMEOUT, 'renders_per_process': 1,
                'instrumentation': 'None; unchanged production snapshots; original zero-probe controls.',
                'source_sha256': {}, 'binary_sha256': {}}
    save(output / 'metadata.json', metadata)
    records: list[dict[str, Any]] = []
    deadline = time.monotonic() + WHOLE_TIMEOUT
    try:
        with tempfile.TemporaryDirectory(prefix='template-value-layout-') as temporary:
            workspace = Path(temporary)
            binaries = {}
            for mode in MODES:
                for side in SIDES:
                    binary, hashes = build(repo, args.zig, workspace, side, commits[side], mode,
                                           harnesses, output, deadline)
                    binaries[side, mode] = binary
                    require(side not in metadata['source_sha256'] or metadata['source_sha256'][side] == hashes,
                            'source archive differs between modes')
                    metadata['source_sha256'][side] = hashes
                    metadata['binary_sha256'][side + '-' + mode] = digest(binary)
                    save(output / 'metadata.json', metadata)
            for corpus, cases in corpora.items():
                for ci, case in enumerate(cases):
                    fixture = workspace / (corpus + '-' + str(ci) + '.json')
                    fixture.write_text(json.dumps(case, ensure_ascii=False))
                    source = template
                    if 'template_source' in case:
                        source = workspace / 'synthetic-control.jinja'
                        source.write_text(case['template_source'])
                    for mode in MODES:
                        for side in SIDES:
                            stem = f'{corpus}-{ci}-{mode}-{side}'
                            print('RUN', corpus, case['id'], mode, side, flush=True)
                            row = render(binaries[side, mode], source, fixture, output / (stem + '.log'), deadline)
                            row.update(corpus=corpus, case=case['id'], mode=mode, side=side,
                                       expected_sha256=hashlib.sha256(case['expected'].encode()).hexdigest(),
                                       exact_provenance_oracle='expected_request_spans' in case,
                                       template_sha256=digest(source))
                            row['qualification_errors'] = qualify(row, case, corpus, side)
                            records.append(row)
                            save(output / 'results.json', records)
        require(len(records) == 306, 'fixed matrix incomplete')
        summary = compare(records)
        summary.update(planned_processes=306, completed_processes=len(records),
                       valid_renders=sum(r['valid'] for r in records),
                       cap_refusals=sum(r['status'] == 'cap_exceeded' for r in records),
                       compared_output_bytes=sum(r['samples'][0]['bytes'] for r in records if r['valid']),
                       exact_provenance_checks=sum(r['valid'] and r['exact_provenance_oracle'] for r in records),
                       qualified=all(not r['qualification_errors'] for r in records))
        save(output / 'summary.json', summary)
        print(json.dumps({key: value for key, value in summary.items() if key != 'cells'}))
        return 0 if summary['qualified'] else 1
    except (ValueError, OSError, subprocess.SubprocessError, TimeoutError) as error:
        save(output / 'failure.json', {'qualified': False, 'error_type': type(error).__name__,
                                      'completed_processes': len(records), 'planned_processes': 306,
                                      'reason': 'Inspect private logs; incomplete matrices cannot qualify.'})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
