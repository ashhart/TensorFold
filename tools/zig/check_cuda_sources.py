"""Resolve CUDA source names through their real includes without generating PTX or invoking nvcc."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADERS = Path(__file__).with_name("cuda_host")
ARCHES = (800, 860, 890, 900, 1200, 1210)


def unit_floors():
    """Read literal per-unit floors from the build catalogs; unknown sources remain checked at every SM."""
    floors = {}
    for name in ("cuda_kernels.zig", "dsv41_units.zig"):
        text = (ROOT / "zig/build" / name).read_text()
        for entry in re.finditer(r'\.\{\s*\.name\s*=\s*"([^"]+)"(.*?)(?=\n\s*\.\{|^\};)', text, re.S | re.M):
            body = entry[2]
            src = re.search(r'\.src\s*=\s*"([^"]+)"', body)
            floor = re.search(r'\.min_sm\s*=\s*(\d+)', body)
            if ".min_sm" in body and floor is None:
                raise ValueError("nonliteral SM floor: " + entry[1])
            key = "zig/kernels/cuda/" + (src[1] if src else entry[1]) + ".cu"
            value = int(floor[1]) * 10 if floor else 0
            if key in floors and floors[key] != value:
                raise ValueError("inconsistent SM floor: " + key)
            floors[key] = value
    return floors


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
    diagnostics = run.stdout + run.stderr
    code = run.returncode
    if arch < 900:
        args = command(compiler, source, arch)
        args[args.index("-fsyntax-only")] = "-E"
        prep = subprocess.run(args, capture_output=True, text=True)
        if prep.returncode:
            code = code or prep.returncode
            diagnostics += prep.stderr
        else:
            assemblies = re.finditer(r'\b(?:asm|__asm__)\s*(?:volatile|__volatile__)?\s*\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)', prep.stdout)
            for assembly in assemblies:
                text = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', assembly[1]))
                text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
                for op in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:[.:][a-zA-Z0-9_:]+)*', text):
                    floor = 0
                    if op.startswith("cvt.") and {"e4m3x2", "e5m2x2"} & set(op.split(".")):
                        floor = 890
                    if op.startswith("barrier.cluster.") or op == "mapa" or op.startswith("mapa.") or "::cluster" in op:
                        floor = 900
                    if arch < floor:
                        diagnostics += f"SM-FLOOR: {op} requires sm_{floor // 10}+, requested sm_{arch // 10}\n"
                        code = 1
    return dict(source=str(source.relative_to(ROOT)), arch=arch, returncode=code, diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CLANGXX", "clang++"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--arch", type=int, choices=ARCHES, action="append")
    parser.add_argument("sources", nargs="*", help="Explicit paths force every requested SM; default: all .cu at build-eligible SMs")
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
    excluded = []
    floors = unit_floors()
    for arch in args.arch or ARCHES:
        for source in sources:
            floor = floors.get(str(source.relative_to(ROOT)), 0)
            if not args.sources and arch < floor:
                excluded.append(dict(source=str(source.relative_to(ROOT)), arch=arch, min_sm=floor // 10))
                continue
            result = check(compiler, source, arch)
            results.append(result)
            if result["returncode"]:
                print(f"FAIL {result['source']} arch={arch}\n{result['diagnostics']}", flush=True)
        print(f"Checked {len(sources)} translation units at architecture branch {arch}", flush=True)
    failures = sum(r["returncode"] != 0 for r in results)
    receipt = dict(compiler=version.splitlines()[0], source_count=len(sources), checks=len(results),
                   failures=failures, mode="semantic-only; no PTX, object, assembly, linker or device execution",
                   sdk="fixed declarations in tools/zig/cuda_host; no generated project declarations",
                   excluded_by_build_floor=excluded, results=results)
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{len(results) - failures}/{len(results)} CUDA source/include checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
