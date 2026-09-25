"""Tests of the lane kernels need Metal 4 tensor units (M5-generation GPUs); elsewhere they are skipped."""

from __future__ import annotations

import pytest

TENSOR_UNIT_TESTS = {
    "test_lane_qmm.py", "test_lane_attention.py", "test_lane_tree.py", "test_lane_fuse.py", "test_lane_glue_norm.py",
    "test_dflash_draft_vocab.py",
}


def _tensor_units() -> bool:
    try:
        from tensorfold.families.qwen3_5 import tensor_units

        return tensor_units()
    except Exception:  # noqa: BLE001 - no MLX or no Metal: no tensor units
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _tensor_units():
        return
    skip = pytest.mark.skip(reason="needs Metal 4 tensor units (an M5-generation GPU)")
    for item in items:
        if item.path.name in TENSOR_UNIT_TESTS:
            item.add_marker(skip)
