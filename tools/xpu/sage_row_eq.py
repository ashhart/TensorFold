#!/usr/bin/env python3
"""Run xpu-sage MODEL --row-eq. The program compares hidden bytes. This script only starts it."""
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sage_row_eq.py XPU_SAGE MODEL_DIR", file=sys.stderr)
        return 2
    return subprocess.run([sys.argv[1], sys.argv[2], "--row-eq"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
