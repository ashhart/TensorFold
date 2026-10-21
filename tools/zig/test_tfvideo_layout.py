"""Exercise the actual Linux loader in temporary layouts with a locally compiled ABI stub."""
from pathlib import Path
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('the layout runtime cell requires Linux')
    source = Path(__file__).resolve().parents[2] / 'zig/tests/tfvideo_stub.c'
    probe = args.probe.resolve(strict=True)
    env = dict(os.environ)
    for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
        env.pop(name, None)
    cc = shlex.split(env.get('CC', 'cc'))
    for case in ('lib', 'beside', 'missing-symbol'):
        with tempfile.TemporaryDirectory(prefix='tfvideo-layout-') as name:
            root = Path(name)
            bindir, libdir, cwd = (root / p for p in ('bin', 'lib', 'cwd'))
            for path in (bindir, libdir, cwd):
                path.mkdir()
            child = bindir / 'probe'
            shutil.copy2(probe, child)
            library = (bindir if case == 'beside' else libdir) / 'libtfvideo.so'
            defines = ['-DTFVIDEO_STUB_NO_FRAMES'] if case == 'missing-symbol' else []
            subprocess.run(cc + ['-shared', '-fPIC', '-std=c11', '-Wall', '-Wextra', '-Werror',
                                 *defines, str(source), '-o', str(library)], env=env, check=True)
            result = subprocess.run([str(child), *(['missing-symbol'] if defines else [])],
                                    cwd=cwd, env=env, text=True, capture_output=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f'{case}: exit {result.returncode}\n{result.stdout}{result.stderr}')
            warning = 'libtfvideo.so is required for video inputs (--vision)'
            expected = 1 if defines else 0
            if result.stderr.count(warning) != expected:
                raise RuntimeError(f'{case}: expected {expected} warnings\n{result.stderr}')
            print(f'PASS tfvideo layout {case}', flush=True)


if __name__ == '__main__':
    main()
