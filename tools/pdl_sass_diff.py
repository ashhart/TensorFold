#!/usr/bin/env python3
"""Compare every untouched image's SASS exactly; audit all four modified images for early input loads."""
from pathlib import Path
import subprocess
import sys
from pdl_early_loads import scan

TOUCHED = {"flashnext_hc", "flashnext_batch", "flashnext_attn", "route_rows"}

def compare(before, after):
    problems = []
    if set(before) != set(after):
        problems.append("image inventories differ")
    for name in sorted(set(before) & set(after)):
        if Path(name).stem in TOUCHED:
            bad, count = scan(after[name])
            problems.extend(f"{name}: {p}" for p in bad)
            if not count:
                problems.append(f"{name}: no sm90+ guarded function")
        elif before[name] != after[name]:
            problems.append(f"{name}: untouched SASS differs")
    missing = TOUCHED - {Path(n).stem for n in after}
    if missing:
        problems.append("missing guarded images: " + ", ".join(sorted(missing)))
    return problems

def dumps(tool, directory):
    root = Path(directory).resolve()
    images = sorted(root.rglob("*.fatbin"))
    if not images:
        raise ValueError("empty fatbin inventory")
    return {str(p.relative_to(root)): subprocess.check_output(
        [str(Path(tool).resolve()), "-sass", p.name], cwd=p.parent, text=True) for p in images}

def main(argv):
    if len(argv) != 4:
        print("usage: pdl_sass_diff.py CUOBJDUMP BASE_FATBINS HEAD_FATBINS", file=sys.stderr)
        return 1
    try:
        before, after = dumps(argv[1], argv[2]), dumps(argv[1], argv[3])
        problems = compare(before, after)
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        print(str(e), file=sys.stderr)
        return 1
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{len(after)} images compared; {len(problems)} problems")
    return bool(problems)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
