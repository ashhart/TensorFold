"""Compile two revisions and run a bounded, alternating template-render benchmark."""
import argparse
import json
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

CASES = ('short 8', 'interleaved 128', 'interleaved 256', 'interleaved 512')
REVISIONS = {'base': 'f350cd3ea', 'candidate': 'a8dd85606'}


def build(repo: Path, zig: str, workspace: Path, name: str, harness: bytes) -> Path:
    """Build a standalone ReleaseFast test binary from an isolated revision."""
    source = workspace / name
    source.mkdir()
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', REVISIONS[name],
                                       'zig/src/core/template', 'zig/src/platform'])
    subprocess.run(['tar', '-xf', '-', '-C', str(source)], input=archive, check=True)
    (source / 'zig/src/core/template/speed_test.zig').write_bytes(harness)
    binary = workspace / f'{name}.bin'
    subprocess.run([zig, 'test', '-O', 'ReleaseFast', '--test-filter', 'speed measurement',
                    '--test-no-exec', f'-femit-bin={binary}', '--dep', 'sys',
                    '-Mroot=zig/src/core/template/speed_test.zig',
                    '-Msys=zig/src/platform/sys.zig', '-lc'], cwd=source, check=True)
    return binary


def measure(binary: Path, revision: str, block: int, position: int, out: Path) -> dict:
    """Record validated results, allowing only the documented baseline 512 cap refusal."""
    proc = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    log = proc.stdout + proc.stderr
    (out / f'block-{block}-{position}-{revision}.log').write_text(log)
    values = {name: [] for name in CASES}
    for kind, n, _, ns in re.findall(r'SPEED (short|interleaved) (\d+) (\d+) (\d+)', log):
        values[f'{kind} {n}'].append(int(ns))
    capped = revision == 'base' and proc.returncode == 1 and 'INVALID 512 rep=0' in log and 'error=OutOfMemory' in log
    if proc.returncode != 0 and not capped:
        raise RuntimeError(f'{revision} failed: inspect block {block}, position {position}')
    if capped and '4 passed; 0 skipped; 1 failed.' not in log:
        raise RuntimeError('baseline failure was not isolated to the 512 cap')
    if not capped and 'All 5 tests passed.' not in log:
        raise RuntimeError(f'{revision} validation did not pass')
    for case in CASES:
        expected = 0 if capped and case == 'interleaved 512' else 3
        if len(values[case]) != expected:
            raise RuntimeError(f'incomplete measurements: {revision} {case}')
    return {'block': block, 'position': position, 'revision': revision,
            'returncode': proc.returncode, 'ns': values}


def main() -> None:
    """Run five alternating ABBA/BAAB blocks and save raw per-run results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--zig', default='zig')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    harness = Path(__file__).with_name('speed_test.zig').read_bytes()
    with tempfile.TemporaryDirectory(prefix='template-speed-') as tmp:
        workspace = Path(tmp)
        binaries = {name: build(repo, args.zig, workspace, name, harness) for name in REVISIONS}
        rows = []
        for block in range(5):
            order = ('base', 'candidate', 'candidate', 'base') if block % 2 == 0 else (
                'candidate', 'base', 'base', 'candidate')
            for position, name in enumerate(order):
                rows.append(measure(binaries[name], name, block, position, out))
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    for case in CASES:
        for name in REVISIONS:
            values = [statistics.median(row['ns'][case]) for row in rows
                      if row['revision'] == name and row['ns'][case]]
            print(case, name, 'runs', len(values), 'median_ns',
                  statistics.median(values) if values else 'unavailable')


if __name__ == '__main__':
    main()
