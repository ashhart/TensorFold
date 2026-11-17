#!/usr/bin/env python3
"""pdl_early_loads.py CUOBJDUMP IMAGE...: global reads a dependent-launch kernel issues before its wait (sm_90 on).

ptxas may schedule a non-coherent load (const __restrict__, __ldg) above griddepcontrol.wait (ACQBULK); under a
dependent launch that reads the predecessor's data early. Every function compiled for sm_90 or later must issue no
global read before its first ACQBULK, and wait at all unless it is a plain-only entry (PLAIN); only the
weight-first bodies (qmmf_kernel, glm_mm's matmuls) may start their weights' async copies there. Exit 1 on any early
read, a failed disassembly, or images with no such function.
"""
import re
import subprocess
import sys

READ = re.compile(r"\b(LDG|LD|ULDG|TEX|TLD|TLD4|TXD|TMML|SULD)\b(?=[.\s])")
ASYNC = re.compile(r"\b(LDGSTS|UBLKCP|UTMALDG|UBLKPF)\b")
WEIGHT_FIRST = re.compile(r"qmmf_kernel|^glm_mm\d+_(\d+|rows)$")
PLAIN = re.compile(r"qmmf_prompt_k32_kernel|^tf_qmmf_shared|^tf_qmmf_native_fp4_check$")  # launched plain only


def scan(text):
    """Problems in one `cuobjdump -sass` dump, and how many sm_90+ functions it checked (plain-only entries are not)."""
    problems, functions = [], 0
    arch, fn, waited, early = 0, None, False, []

    def close():
        nonlocal functions
        if fn is None or arch < 90 or PLAIN.search(fn):
            return
        functions += 1
        problems.extend(early if waited else [f"sm_{arch} {fn}: no dependency wait, yet not a plain-only entry"])

    for line in text.splitlines():
        m = re.search(r"code for sm_(\d+)", line)
        if m:
            close()
            arch, fn = int(m.group(1)), None
            continue
        m = re.search(r"Function : (\S+)", line)
        if m:
            close()
            fn, waited, early = m.group(1), False, []
            continue
        if fn is None or arch < 90 or waited:
            continue
        if "ACQBULK" in line:
            waited = True
        elif READ.search(line):
            early.append(f"sm_{arch} {fn}: global read before the wait: {' '.join(line.split())[:80]}")
        elif ASYNC.search(line) and not WEIGHT_FIRST.search(fn):
            what = "async copy before the wait outside a weight-first body"
            early.append(f"sm_{arch} {fn}: {what}: {' '.join(line.split())[:80]}")
    close()
    return problems, functions


def main(argv):
    if len(argv) < 3:
        print("usage: pdl_early_loads.py CUOBJDUMP IMAGE...", file=sys.stderr)
        return 2
    tool, images = argv[1], argv[2:]
    problems, functions = [], 0
    for img in images:
        r = subprocess.run([tool, "-sass", img], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{img}: {tool} failed: {r.stderr.strip()[:200]}", file=sys.stderr)
            return 1
        p, n = scan(r.stdout)
        if n == 0:
            print(f"{img}: no sm_90 or later function to check", file=sys.stderr)
            return 1
        problems += [f"{img.split('/')[-1]} {x}" for x in dict.fromkeys(p)]
        functions += n
    for x in problems:
        print("EARLY", x)
    print(f"{len(problems)} early read(s) in {functions} function(s) of {len(images)} image(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
