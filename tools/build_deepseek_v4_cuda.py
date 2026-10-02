#!/usr/bin/env python3
"""Install the candidate native TensorFold runtime; preflight is read-only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorfold.families.deepseek_v4.cuda.build_runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
