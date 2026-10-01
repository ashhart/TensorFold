"""Build only the isolated native library, never a donor HTTP server or live binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .sources import PACKAGE, PIN, resolve_sources, verify_sources

CUDA_UNITS = (
    "ds4_cuda.cu",
    "cuda/mmq/ds4_ggml_stubs.cu",
    "cuda/mmq/ds4_mmq.cu",
    "cuda/mmq/ds4_mmq_d2r.cu",
    "cuda/mmq/quantize.cu",
    "cuda/mmq/mmid.cu",
    "cuda/mmq/mmvq.cu",
    "cuda/mmq/ds4_repack.cu",
)


def build(build_dir: Path, *, backend="cuda", jobs=2, source=None, offline=False):
    if backend not in ("cpu", "cuda") or not 1 <= int(jobs) <= 4:
        raise ValueError("backend must be cpu/cuda and jobs must be between 1 and 4")
    selected = source if source is not None else os.environ.get("TENSORFOLD_DS4_SOURCE")
    build_dir = Path(build_dir).expanduser().resolve()
    if selected:
        local = Path(selected).expanduser().resolve()
        if build_dir == local or local in build_dir.parents:
            raise ValueError("native build directory must be outside the selected source tree")
    source = resolve_sources(source, offline=offline)
    if build_dir == source or source in build_dir.parents:
        raise ValueError("native build directory must be outside the pinned source tree")
    manifest = verify_sources(source)
    cc = shutil.which(os.environ.get("CC", "cc"))
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = cuda_home / "bin/nvcc"
    if cc is None or (backend == "cuda" and not nvcc.is_file()):
        raise RuntimeError("native build needs a C compiler and, for CUDA, nvcc")
    build_dir.mkdir(parents=True, exist_ok=True)
    cflags = [
        "-O3",
        "-ffast-math",
        "-fno-finite-math-only",
        "-fPIC",
        "-march=native",
        "-pthread",
        "-D_GNU_SOURCE",
        "-std=c99",
        "-I",
        str(source),
        f'-DTF_DS4_REVISION="{PIN}"',
    ]
    if backend == "cpu":
        cflags.append("-DDS4_NO_GPU")
    units = [
        (PACKAGE / "native/shim.c", build_dir / "shim.o"),
        (source / "ds4_distributed.c", build_dir / "distributed.o"),
    ]
    nvflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode",
        "arch=compute_121a,code=sm_121a",
        "-DDS4_CUDA_HAVE_MXF4=1",
        "-Xcompiler",
        "-fPIC",
        "-Xcompiler",
        "-march=native",
        "-Xcompiler",
        "-pthread",
        "-I",
        str(source / "cuda/mmq"),
    ]
    if backend == "cuda":
        units += [(source / name, build_dir / (Path(name).stem + ".o")) for name in CUDA_UNITS]

    def compile_unit(unit):
        file, obj = unit
        command = [str(nvcc), *nvflags] if file.suffix == ".cu" else [cc, *cflags]
        subprocess.run([*command, "-c", str(file), "-o", str(obj)], check=True)

    with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
        list(pool.map(compile_unit, units))
    output = build_dir / "libtensorfold_ds4.so"
    linker = [str(nvcc)] if backend == "cuda" else [cc]
    libs = (
        ["-lm", "-pthread"]
        if backend == "cpu"
        else [
            "-lm",
            "-Xcompiler",
            "-pthread",
            "-L",
            str(cuda_home / "targets/sbsa-linux/lib"),
            "-L",
            str(cuda_home / "lib64"),
            "-lcudart",
            "-lcublas",
            "-lcuda",
        ]
    )
    temporary = output.with_suffix(".tmp.so")
    subprocess.run([*linker, "-shared", "-o", str(temporary), *(str(o) for _, o in units), *libs], check=True)
    temporary.replace(output)
    (build_dir / "native-build.json").write_text(
        json.dumps(
            {
                "abi": 2,
                "revision": PIN,
                "backend": backend,
                "source_sha256": manifest["sha256"],
                "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "shim_sha256": hashlib.sha256((PACKAGE / "native/shim.c").read_bytes()).hexdigest(),
                "cc": subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
                "cuda_arch": "sm_121a" if backend == "cuda" else None,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--ds4-source", type=Path, help="existing ds4 source tree or Git checkout; never modified")
    parser.add_argument("--offline", action="store_true", help="refuse to fetch missing native sources")
    args = parser.parse_args()
    print(build(args.build_dir, backend=args.backend, jobs=args.jobs, source=args.ds4_source, offline=args.offline))


if __name__ == "__main__":
    main()
