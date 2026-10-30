"""Run bounded renders of an independently supplied DeepSeek template corpus."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

TEMPLATE_SHA256 = '7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e'
COMMITS = {'base': 'f350cd3ea253a042e205746928271f9d8f71a82d',
           'candidate': 'f1c7c6d3a6632daed5f26c3d2d240f8d56926e1b'}
ORDER = ('base', 'candidate', 'candidate', 'base')


def digest(path: Path) -> str:
    """Hash a file without embedding the external template in this receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_revision(repo: Path, zig: str, workspace: Path, side: str, mode: str, harness: bytes) -> Path:
    """Build a standalone renderer test from a pinned revision."""
    source = workspace / (side + '-' + mode)
    source.mkdir()
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', COMMITS[side],
                                       'zig/src/core/template', 'zig/src/platform'])
    subprocess.run(['tar', '-xf', '-', '-C', str(source)], input=archive, check=True)
    (source / 'zig/src/core/template/render_test.zig').write_bytes(harness)
    binary = workspace / (side + '-' + mode + '.bin')
    subprocess.run([zig, 'test', '-O', mode, '--test-filter', 'external template case', '--test-no-exec',
                    f'-femit-bin={binary}', '--dep', 'sys', '-Mroot=zig/src/core/template/render_test.zig',
                    '-Msys=zig/src/platform/sys.zig', '-lc'], cwd=source, check=True)
    return binary


def render(binary: Path, template: Path, fixture: Path, log: Path) -> dict:
    """Run a fresh process and preserve correctness, timing and allocator evidence."""
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
    samples = [(int(rep), int(ns), int(peak), int(size), int(spans), sha) for rep, ns, peak, size, spans, sha in re.findall(
        r'RENDER rep=(\d+) ns=(\d+) scratch_peak=(\d+) bytes=(\d+) spans=(\d+) spans_sha256=([0-9a-f]{64})', text)]
    errors = re.findall(r'(COMPILE_ERROR|RENDER_ERROR|MISMATCH) ([^\n]+)', text)
    valid = (result.returncode == 0 and len(compiles) == 1 and len(samples) == 5
             and [x[0] for x in samples] == list(range(5))
             and len({(x[4], x[5]) for x in samples}) == 1)
    capped = (result.returncode == 1 and 'CAP live=' in text and len(errors) == 1
              and errors[0][0] in ('RENDER_ERROR', 'COMPILE_ERROR')
              and errors[0][1].startswith('OutOfMemory ')
              and not any(marker in text.lower() for marker in ('scratchnotreleased', 'leaked', 'panic', 'mismatch')))
    status = 'valid' if valid else ('cap_exceeded' if capped else 'unexpected_failure')
    record = {'exit_code': result.returncode, 'valid': valid, 'status': status, 'errors': errors,
              'compile_standing': int(compiles[0][0]) if compiles else None,
              'compile_peak': int(compiles[0][1]) if compiles else None,
              'samples': [{'rep': rep, 'ns': ns, 'scratch_peak': peak, 'bytes': size, 'spans': spans,
                           'spans_sha256': sha} for rep, ns, peak, size, spans, sha in samples]}
    if valid:
        record['median_ns'] = statistics.median(sample[1] for sample in samples[2:])
    return record


def main() -> None:
    """Verify input identities and run Debug and ReleaseFast ABBA blocks."""
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
        'runner_sha256': digest(Path(__file__)), 'zig_version': zig_version,
        'commits': COMMITS, 'order': ORDER, 'warmups': 2, 'measurements': 3,
        'combined_compile_render_cap_bytes': 240 * 1024 * 1024,
        'provenance': 'ordered, nonempty, disjoint, in-bounds; SHA256 of decimal start:end newline per span',
        'binaries': {},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='deepseek-template-') as tmp:
        workspace = Path(tmp)
        binaries = {}
        for mode in ('Debug', 'ReleaseFast'):
            for side in COMMITS:
                binary = compile_revision(args.repo.resolve(), args.zig, workspace, side, mode, harness)
                binaries[side, mode] = binary
                metadata['binaries'][side + '-' + mode] = digest(binary)
                (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for index, case in enumerate(cases):
            fixture = workspace / f'case-{index}.json'
            fixture.write_text(json.dumps(case, ensure_ascii=False))
            for mode in ('Debug', 'ReleaseFast'):
                for position, side in enumerate(ORDER):
                    print(f'RUN {case["id"]} {mode} {side} {position}', flush=True)
                    item = render(binaries[side, mode], template, fixture, out / f'{index}-{mode}-{position}.log')
                    item.update(case=case['id'], mode=mode, side=side, position=position)
                    records.append(item)
                    (out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
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
