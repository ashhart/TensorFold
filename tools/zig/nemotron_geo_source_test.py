"""Require the Nemotron geometry adapter to reject duplicate reservations."""
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    zig, *fixtures = sys.argv[1:]
    failures = 0
    with tempfile.TemporaryDirectory() as cache:
        for fixture in fixtures:
            label = Path(fixture).stem
            result = subprocess.run(
                [zig, "test", fixture, "-fno-emit-bin", "--cache-dir", cache],
                capture_output=True, text=True,
            )
            diagnostic = result.stdout + result.stderr
            if result.returncode == 0 or "captured geometry threadgroup reservation changed" not in diagnostic:
                failures += 1
                print(f"Nemotron geometry control was not rejected: {label}")
            else:
                print(f"Nemotron geometry rejected duplicate reservation: {label}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
