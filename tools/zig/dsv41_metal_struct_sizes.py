#!/usr/bin/env python3
"""The DeepSeek V4.1 Metal struct manifest.

Every struct the family's Metal sources define is compiled once, in the same translation unit the engine
compiles it in (the three run-time joins plus one unit a source), and its C size is read from the Metal
compiler's own record-layout dump.  The result is a checked-in manifest (`struct_sizes.txt`), and this
tool fails the build when:

  * a struct's size differs from the manifest (a Metal-side edit, or a compiler layout change),
  * a kernel argument struct has no `static_assert(sizeof(T) == N)` in its unit,
  * a struct defined in a source has no record layout in the unit that compiles it.

Usage: dsv41_metal_struct_sizes.py [--update] [DIR]

DIR defaults to the repository's zig/kernels/metal/deepseek_v41.  `--update` rewrites the manifest.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

# The engine's run-time joins (csa2.zig:91, csa2_index.zig:164, metal/pipelines.zig:122), as
# check_dsv41_metal_mirrors.sh reproduces them.  A source that only compiles inside a join is assigned to it below.
JOINS = [
    ("csa2-kv-join", ["kv.metal", "csa2.metal", "csa2_nax.metal"], None),
    ("index-kv-join", ["kv.metal", "index_exl3.metal"], None),
    ("route-defines", ["route.metal"], "#define TF_DSV41_E 384\n#define TF_DSV41_TOPK 6\n"),
]
JOIN_ONLY = {"csa2.metal", "csa2_nax.metal", "index_exl3.metal", "route.metal"}

STRUCT_DEF = re.compile(r"\bstruct\s+([A-Za-z_]\w*)\s*\{")
KERNEL = re.compile(r"(?:\[\[\s*kernel[^\]]*\]\]\s*|\bkernel\s+)void\s+([A-Za-z_]\w*)\s*\(")
ARG = re.compile(r"\b(?:constant|device)\s+(?:const\s+)?([A-Za-z_]\w*)\s*[&*]")
LAYOUT = re.compile(r"^\s*\|\s*\[sizeof=(\d+),")
LAYOUT_NAME = re.compile(r"^\s*\d+\s*\|\s*struct\s+([A-Za-z_]\w*)\s*$")


def unit_texts(directory):
    """The text of every compiled unit: each standalone source plus the three joins."""
    texts = {}
    for name, parts, prefix in JOINS:
        body = "".join(open(os.path.join(directory, p), encoding="utf-8").read() for p in parts)
        texts[name] = (prefix or "") + body
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".metal") or entry in JOIN_ONLY:
            continue
        texts[entry[:-6]] = open(os.path.join(directory, entry), encoding="utf-8").read()
    return texts


def unit_of_source(source, sources):
    """The unit name that compiles `source` (a join if it is join-only, else its own)."""
    for name, parts, _ in JOINS:
        if source in parts:
            return name
    assert source in sources, source
    return source[:-6]


def dump(directory, work, unit, text):
    """Compile one unit with the record-layout dump; return (sizes, error text or None)."""
    path = os.path.join(work, "deepseek_v41", unit + ".metal")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    run = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-Xclang", "-fdump-record-layouts", "-c", path,
         "-o", os.path.join(work, "deepseek_v41", unit + ".air")],
        capture_output=True, text=True)
    if run.returncode != 0:
        return None, run.stderr
    sizes = {}
    current = None
    for line in run.stdout.splitlines():
        match = LAYOUT_NAME.match(line)
        if match:
            current = match.group(1)
            continue
        match = LAYOUT.match(line)
        if match and current:
            sizes[current] = int(match.group(1))
            current = None
    return sizes, None


def kernel_arg_structs(text, defined):
    """Every struct type a kernel in `text` takes by reference or pointer."""
    found = set()
    for match in KERNEL.finditer(text):
        start = match.end()
        depth, at = 1, start
        while at < len(text) and depth:
            if text[at] == "(":
                depth += 1
            elif text[at] == ")":
                depth -= 1
            at += 1
        for arg in ARG.finditer(text[start:at]):
            if arg.group(1) in defined:
                found.add(arg.group(1))
    return found


def main(argv):
    update = "--update" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = args[0] if args else os.path.join(root, "zig", "kernels", "metal", "deepseek_v41")
    manifest_path = os.path.join(directory, "struct_sizes.txt")
    if not os.path.isdir(directory):
        print(f"metal struct sizes: {directory} not found")
        return 2
    if not shutil.which("xcrun"):
        print("metal struct sizes: xcrun not on PATH")
        return 2

    sources = [e for e in sorted(os.listdir(directory)) if e.endswith(".metal")]
    texts = unit_texts(directory)
    work = tempfile.mkdtemp(prefix="dsv41-struct-sizes.")
    try:
        nax = os.path.join(root, "zig", "kernels", "metal", "nax.h")
        if not os.path.exists(nax):
            nax = os.path.join(directory, "..", "nax.h")
        shutil.copy(nax, os.path.join(work, "nax.h"))
        # The sources `#include "../nax.h"` and at least one includes another source by name, so the
        # unit is compiled where the engine's own sources sit: a directory one level under `nax.h`.
        os.makedirs(os.path.join(work, "deepseek_v41"), exist_ok=True)
        for source in sources:
            shutil.copy(os.path.join(directory, source), os.path.join(work, "deepseek_v41", source))

        layouts, failures = {}, {}
        for unit, text in texts.items():
            sizes, error = dump(directory, work, unit, text)
            if sizes is None:
                failures[unit] = error
            else:
                layouts[unit] = sizes

        # The manifest: one line a struct, keyed by the file that defines it.
        lines = {}
        for source in sources:
            unit = unit_of_source(source, sources)
            if unit not in layouts:
                continue
            text = open(os.path.join(directory, source), encoding="utf-8").read()
            for name in STRUCT_DEF.findall(text):
                if name not in layouts[unit]:
                    failures.setdefault(unit, f"no record layout for struct {name} ({source})")
                    continue
                lines[(source, name)] = layouts[unit][name]
        manifest = "".join(f"{source} {name} {lines[(source, name)]}\n" for source, name in sorted(lines))

        fail = 0
        for unit, error in sorted(failures.items()):
            print(f"FAIL {unit}")
            for line in (error or "").splitlines()[:12]:
                print("  " + line)
            fail = 1

        # Every kernel argument struct must assert its own size in the unit that compiles it.
        for unit, text in texts.items():
            if unit not in layouts:
                continue
            defined = set(STRUCT_DEF.findall(text))
            asserted = set(re.findall(r"static_assert\s*\(\s*sizeof\s*\(\s*([A-Za-z_]\w*)\s*\)", text))
            for name in sorted(kernel_arg_structs(text, defined) - asserted):
                print(f"FAIL {unit}: kernel argument struct `{name}` has no "
                      f"static_assert(sizeof({name}) == N) in this unit")
                fail = 1

        if update and not fail:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write("# DeepSeek V4.1 Metal struct sizes: one `source struct bytes` a line.\n")
                f.write("# Generated by tools/zig/dsv41_metal_struct_sizes.py --update from the Metal\n")
                f.write("# compiler's own record layouts. `zig build` fails when this file goes stale.\n")
                f.write(manifest)
            print(f"metal struct sizes: wrote {manifest_path} ({len(lines)} structs)")
            return 0

        try:
            current = open(manifest_path, encoding="utf-8").read()
        except FileNotFoundError:
            print(f"FAIL metal struct sizes: {manifest_path} is missing (regenerate with --update)")
            return 1
        recorded = {}
        for line in current.splitlines():
            if not line or line.startswith("#"):
                continue
            source, name, size = line.split()
            recorded[(source, name)] = int(size)
        failed = {unit for unit in failures}
        def known(key):
            return unit_of_source(key[0], sources) not in failed
        keys = {k for k in set(recorded) | set(lines) if known(k)}
        for key in sorted(keys):
            if key not in lines:
                print(f"FAIL struct sizes: {key[0]} {key[1]} is in the manifest but no longer defined")
                fail = 1
            elif key not in recorded:
                print(f"FAIL struct sizes: {key[0]} {key[1]} {lines[key]} is new (regenerate the manifest)")
                fail = 1
            elif recorded[key] != lines[key]:
                print(f"FAIL struct sizes: {key[0]} {key[1]} {recorded[key]} -> {lines[key]}")
                fail = 1
        if fail:
            print("metal struct sizes: a shared struct's size moved, or a kernel argument has no assert")
            return 1
        print(f"metal struct sizes: {len(lines)} structs match the manifest; every kernel argument asserts its size")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
