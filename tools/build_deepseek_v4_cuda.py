#!/usr/bin/env python3
"""Install native TensorFold and prepare local GGUF sidecars; no donor engine or weight downloads."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_deepseek_v4_gguf import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--drafter", default=os.environ.get("TF_DEEPSEEK_DRAFTER", ""))
    p.add_argument("--context", type=int, default=163840)
    p.add_argument("--retained-prefix", type=int, default=int(os.environ.get("TF_DEEPSEEK_RETAINED_PREFIX", "131072")))
    p.add_argument("--companion-reserve-gib", type=float, default=3)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--tokenizer-dir")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.jobs <= 4 or not 0 <= a.retained_prefix < a.context:
        p.error("jobs must be 1..4 and retained prefix below the positive context")
    os.environ["MAX_JOBS"] = str(a.jobs)
    if a.tokenizer_dir:
        p.error("native GGUF uses its embedded tokenizer; remove --tokenizer-dir")
    os.environ["TF_DEEPSEEK_COMPANION_RESERVE_GIB"] = str(a.companion_reserve_gib)
    from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine

    candidate = DeepSeekEngine.__new__(DeepSeekEngine)
    candidate.limit = a.context
    candidate._admit(a.gguf, a.drafter)
    if a.preflight_only:
        print(f"native GGUF admission passed: context {a.context}, retained prefix {a.retained_prefix}")
        return
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(root)], check=True
    )
    from tensorfold.cuda.experts import _ext

    _ext()  # Compile TensorFold's own planner before large weights occupy unified memory.
    prepare(a.gguf, a.model_dir, context=a.context, drafter=a.drafter, retained_prefix=a.retained_prefix)
    print("native TensorFold installed; own CUDA kernels compile on first use")


if __name__ == "__main__":
    main()
