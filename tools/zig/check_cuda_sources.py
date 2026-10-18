"""Resolve CUDA source names through their real includes without generating PTX or invoking nvcc."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADERS = Path(__file__).with_name("cuda_host")
ARCHES = (800, 890, 900, 1200, 1210)


def command(compiler, source, arch):
    """Use Clang's semantic parser only, with a fixed SDK declaration surface."""
    flags = [compiler, "-std=c++20", "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
             "-fsyntax-only", "-ferror-limit=0", "-Wno-unknown-pragmas", "-I", str(HEADERS),
             "-include", str(HEADERS / "cuda_runtime.h"), "-U__CUDA_ARCH__", f"-D__CUDA_ARCH__={arch}"]
    if arch >= 1200:
        flags += [f"-D__CUDA_ARCH_FEAT_SM{arch // 10}_ALL"]
    return flags + [str(source)]


def check(compiler, source, arch):
    """Return every diagnostic, without ignoring unresolved project identifiers."""
    run = subprocess.run(command(compiler, source, arch), capture_output=True, text=True)
    return dict(source=str(source.relative_to(ROOT)), arch=arch, returncode=run.returncode,
                diagnostics=run.stdout + run.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CLANGXX", "clang++"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--arch", type=int, choices=ARCHES, action="append")
    parser.add_argument("sources", nargs="*", help="Paths relative to the repository; default: every kernel .cu")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("clang++ is required; this check never falls back to nvcc")
    version = subprocess.check_output([compiler, "--version"], text=True)
    if "clang" not in version.lower():
        parser.error("the semantic check requires clang++")
    sources = [ROOT / p for p in args.sources] if args.sources else sorted((ROOT / "zig/kernels/cuda").rglob("*.cu"))
    if not sources:
        parser.error("no CUDA sources found")
    results = []
    for arch in args.arch or ARCHES:
        for source in sources:
            result = check(compiler, source, arch)
            results.append(result)
            if result["returncode"]:
                print(f"FAIL {result['source']} arch={arch}\n{result['diagnostics']}", flush=True)
        print(f"Checked {len(sources)} translation units at architecture branch {arch}", flush=True)
    failures = sum(r["returncode"] != 0 for r in results)
    receipt = dict(compiler=version.splitlines()[0], source_count=len(sources), checks=len(results),
                   failures=failures, mode="semantic-only; no PTX, object, assembly, linker or device execution",
                   sdk="fixed declarations in tools/zig/cuda_host; no generated project declarations", results=results)
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{len(results) - failures}/{len(results)} CUDA source/include checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
