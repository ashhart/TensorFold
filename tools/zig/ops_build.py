"""Compile the operator oracle with the serving container's pinned compiler and arithmetic flags."""

import ctypes
import hashlib
from pathlib import Path
import subprocess

FLAGS = ('-O3', '--fmad=false', '--ftz=false', '--expt-relaxed-constexpr', '-std=c++20',
         '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
         '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
         '-gencode=arch=compute_121,code=sm_121', '-shared', '-Xcompiler', '-fPIC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_operators(source, out, names):
    source, out = Path(source), Path(out)
    version = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True).stdout
    if 'V13.3.73' not in version:
        raise RuntimeError('Operator parity requires the serving container nvcc13.3.73')
    files = [source / name for name in names]
    target = out / 'operators.so'
    with (out / 'compile.log').open('w') as stream:
        subprocess.run(['nvcc', *FLAGS, *map(str, files), '-o', str(target)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=180)
    return ctypes.CDLL(str(target)), {'nvcc': version, 'flags': list(FLAGS), 'image_sha256': digest(target),
                                    'sources': {p.name: digest(p) for p in files},
                                    'headers': {p.name: digest(p) for p in sorted(source.glob('*.h'))}}
