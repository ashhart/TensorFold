"""The bounded lookup benchmark uses real table readers and checks exact output bytes."""

from __future__ import annotations

import json

import pytest

from tools.bench_ngram_ssd import _index, _scale, _shards, run, synthetic_checkpoint


@pytest.mark.parametrize("layout,row_width", [("nvfp4", 80), ("fp8", 160), ("affine", 80), ("bf16", 320)])
def test_bounded_lookup_measures_same_rows_and_exact_bytes(tmp_path, layout, row_width):
    table = synthetic_checkpoint(tmp_path, layout, 64)
    report = run(tmp_path, table, (4, 16), 2)
    assert report["rows"] == 64
    assert report["row_width_bytes"] == row_width
    assert [case["row_ids"] for case in report["samples"]] == [4, 16]
    for case in report["samples"]:
        assert case["exact_bytes"] is True
        assert case["ssd"]["pread_bytes"] > 0
        assert case["ssd"]["pread_calls"] > 0
        assert case["resident"]["pread_bytes"] == 0
        for mode in ("ssd", "resident"):
            assert case[mode]["first_observed_ms"] >= 0
            assert case[mode]["repeated_p50_ms"] >= 0
            assert case[mode]["repeated_p95_ms"] >= 0


def test_index_places_nvfp4_block_and_global_scales_in_separate_files(tmp_path):
    table = synthetic_checkpoint(tmp_path, "nvfp4", 32)
    where = _index(tmp_path)
    shards = _shards(where, table)
    assert len(shards) == 1
    _, shard = shards[0]
    assert where[shard + ".weight"] != where[shard + ".weight_scale"]
    assert where[table + ".weight_scale_2"] not in {
        where[shard + ".weight"], where[shard + ".weight_scale"]}
    assert _scale(tmp_path, where, table, "weight_scale_2") == pytest.approx(0.0371)
    assert _scale(tmp_path, where, table, "absent") == 1.0


def test_missing_or_noncontiguous_shards_fail_before_lookup(tmp_path):
    table = synthetic_checkpoint(tmp_path, "nvfp4", 32)
    index = tmp_path / "model.safetensors.index.json"
    contents = json.loads(index.read_text())
    weight = table + ".shard_0.weight"
    contents["weight_map"][table + ".shard_2.weight"] = contents["weight_map"].pop(weight)
    index.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="noncontiguous"):
        run(tmp_path, table, (4,), 2)


def test_rejects_sample_larger_than_table(tmp_path):
    table = synthetic_checkpoint(tmp_path, "nvfp4", 8)
    with pytest.raises(ValueError, match="exceeds table size"):
        run(tmp_path, table, (16,), 2)
