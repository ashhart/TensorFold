"""Compare three template revisions with a fixed balanced host-render schedule."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

ORDER = (
    ('base', 'prior', 'revised'), ('revised', 'prior', 'base'),
    ('prior', 'revised', 'base'), ('base', 'revised', 'prior'),
    ('revised', 'base', 'prior'), ('prior', 'base', 'revised'),
)
CASES = (
    ('exact', 'scaled-0009-tools-none'),
    ('exact', 'scaled-2001-tools-none'),
    ('exact', 'scaled-2001-tools-synthetic-157'),
    ('allocation', 'tool-cycles-064-object-8192'),
)
CORPORA = {
    'exact': 'f5b8a240a9e852aaad7efbc3335942e8089812a2d7b5e5783b83f3c9dab4b74d',
    'allocation': 'f5ac1de9e7226a701beda7887c280ee090c0694d3f6118945e3c39167b78e778',
}


def digest(path: Path) -> str:
    """Return a content identity without publishing local paths."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Build first, then run the complete fixed schedule without interim acceptance decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--exact-corpus', type=Path, required=True)
    parser.add_argument('--allocation-corpus', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--zig', default='zig')
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    helper_path = Path(__file__).resolve().parent.parent / 'deepseek-template' / 'run.py'
    spec = importlib.util.spec_from_file_location('frozen_render', helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('missing frozen renderer helper')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if digest(helper_path) != '295d316fd988dd44b1933ed2ec53018f4bcbbbc01b404e822fd1e2b6a3d7eb12':
        raise ValueError('renderer helper changed')
    template = args.template.resolve(strict=True)
    if digest(template) != helper.TEMPLATE_SHA256:
        raise ValueError('template changed')
    if subprocess.check_output([args.zig, 'version'], text=True).strip() != '0.17.0':
        raise ValueError('Zig 0.17.0 required')
    candidate = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', '--verify', args.candidate + '^{commit}'], text=True).strip()
    helper.COMMITS = {
        'base': 'f350cd3ea253a042e205746928271f9d8f71a82d',
        'prior': 'd33e5e769d7a4bb801cc39e0a32400488cb89777',
        'revised': candidate,
    }
    corpus = {}
    for label, path in (('exact', args.exact_corpus), ('allocation', args.allocation_corpus)):
        if digest(path) != CORPORA[label]:
            raise ValueError('frozen corpus changed')
        corpus[label] = {case['id']: case for case in json.loads(path.read_text())}
    cases = [corpus[label][name] for label, name in CASES]
    harness_path = helper_path.with_name('render_test.zig')
    harness = harness_path.read_bytes()
    if digest(harness_path) != 'b1ecaf74b2fb8d2e9fbaf770301049133d57f2c5f01d2825deb98e5a9addec0b':
        raise ValueError('frozen timing harness changed')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    metadata = {
        'commits': helper.COMMITS, 'mode': 'ReleaseSafe', 'zig_version': '0.17.0',
        'template_sha256': helper.TEMPLATE_SHA256, 'corpus_sha256': CORPORA,
        'cases': CASES, 'order': ORDER, 'warmups': 2, 'timed_renders': 3,
        'runner_sha256': digest(Path(__file__)), 'helper_sha256': digest(helper_path),
        'harness_sha256': digest(harness_path), 'binaries': {},
        'combined_compile_render_cap_bytes': 240 * 1024 * 1024,
        'summary': 'median of six per-process medians; descriptive, no equivalence claim',
    }
    records = []
    with tempfile.TemporaryDirectory(prefix='template-layout-timing-') as tmp:
        workspace = Path(tmp)
        binaries = {}
        for side in helper.COMMITS:
            binary = helper.compile_revision(repo, args.zig, workspace, side, 'ReleaseSafe', harness)
            binaries[side] = binary
            metadata['binaries'][side] = digest(binary)
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for case_index, case in enumerate(cases):
            fixture = workspace / f'case-{case_index}.json'
            fixture.write_text(json.dumps(case, ensure_ascii=False))
            for block, order in enumerate(ORDER):
                for position, side in enumerate(order):
                    print(f'RUN {case["id"]} {block} {position} {side}', flush=True)
                    row = helper.render(binaries[side], template, fixture,
                                        out / f'{case_index}-{block}-{position}-{side}.log')
                    row.update(case=case['id'], block=block, position=position, side=side)
                    records.append(row)
                    (out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        if len(records) != 72 or any(not row['valid'] for row in records):
            raise RuntimeError('fixed schedule has an incomplete or failed render')
        summary = []
        for case in cases:
            rows = [row for row in records if row['case'] == case['id']]
            spans = {(sample['spans'], sample['spans_sha256']) for row in rows for sample in row['samples']}
            if len(spans) != 1:
                raise RuntimeError('provenance differs across revisions or runs')
            medians = {side: statistics.median(row['median_ns'] for row in rows if row['side'] == side)
                       for side in helper.COMMITS}
            summary.append({'case': case['id'], 'median_ns': medians,
                            'revised_vs_base_pct': 100 * (medians['revised'] / medians['base'] - 1),
                            'revised_vs_prior_pct': 100 * (medians['revised'] / medians['prior'] - 1)})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('PASS 72 processes, 360 validated renders')


if __name__ == '__main__':
    main()
