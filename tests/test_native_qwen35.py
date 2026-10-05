"""Native Qwen admission and real-model checks skip cleanly without their compiled tools, weights or MLX."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def program(name):
    path = ROOT / "zig-out" / ("native/bin" if name == "tensorfold-native" else "bin") / name
    if not path.is_file():
        pytest.skip(f"build {name} first")
    return path


def checkpoint():
    mlx = pytest.importorskip("mlx.core")
    if not mlx.metal.is_available():
        pytest.skip("Apple Metal is unavailable")
    path = os.environ.get("TENSORFOLD_QWEN35_MODEL")
    if not path or not (Path(path) / "model.safetensors").is_file():
        pytest.skip("TENSORFOLD_QWEN35_MODEL must name the pinned MLX affine 4-bit checkpoint")
    return Path(path)


def test_native_qwen_family_is_advertised():
    result = subprocess.run([program("tensorfold-native"), "capabilities", "--json"], check=True, capture_output=True, text=True)
    caps = json.loads(result.stdout)
    assert caps["families"]["qwen3_5"] == ["mlx-q4g64"]
    assert caps["families"]["nemotron_h"] == ["mlx-q4g64"]


def test_qwen_generated_kernels_are_current():
    subprocess.run([sys.executable, ROOT / "tools/zig/gen_qwen35_kernels.py", "--check"], check=True)


def test_native_qwen_operations(tmp_path):
    model, binary = checkpoint(), program("tf-qwen35-check")
    subprocess.run([sys.executable, ROOT / "tools/zig/capture_qwen35.py", model, tmp_path], check=True)
    subprocess.run([binary, model, tmp_path], check=True)


def test_native_qwen_windows_and_committed_states():
    subprocess.run([program("tf-qwen35-exact"), checkpoint()], check=True)
