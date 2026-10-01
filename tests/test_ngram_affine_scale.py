"""Affine n-gram shards retain their separate post-dequantization scale."""

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.host_table import open_table, shard_keys
from tests.test_ple_ssd import _checkpoint


@pytest.mark.parametrize("ssd", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.0002002716064453125])
def test_affine_table_retains_global_scale_without_changing_packed_rows(tmp_path, scale, ssd):
    files, rows = _checkpoint(tmp_path)
    shards = [(entry[0].name, f"emb.shard_{i}") for i, entry in enumerate(files)]
    table = open_table(tmp_path, shards, lambda name: scale, ssd=ssd)
    assert getattr(table, "weight_scale", 1.0) == scale
    ids = np.array([0, 17, table.rows - 1])
    for actual, expected in zip(table.gather(ids), rows):
        assert np.array_equal(actual, expected[ids])


def test_nested_shard_names_from_omlx_resolve_like_flat_names():
    for pattern in ("table.shard_{}", "table.shards.{}"):
        keys = [pattern.format(i) for i in range(3)]
        assert shard_keys("table", 3, {key + ".weight" for key in keys}) == keys
