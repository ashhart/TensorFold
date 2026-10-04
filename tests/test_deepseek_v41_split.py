"""DeepSeek-V4.1-Flash's split rules: one rule for every checkpoint tensor, each rank's part, the bytes admission counts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from dsv41_inventory import inventory

from tensorfold.cuda import capacity
from tensorfold.families.deepseek_v41.cuda import split

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41" / "config.json"
GIB = 1 << 30

WHOLE_MAIN_PROJ = 78_233_715_016      # a rank's checkpoint bytes with mtp.0.main_proj held whole
STORED = WHOLE_MAIN_PROJ - (78_643_200 + 76_800) // 2     # the engine splits its rows (nsplit)
MX8_ROW_REPEAT = 118_639_232            # N*K/32 scale bytes instead of N*K/1024, every FP8 32x32 weight
NVFP4_SCALES = 212_336_640 + 3 * 128 * 3 * 4   # e4m3 per 16 inputs (twice E8M0 per 32), an fp32 per matrix
CONFIDENCE_FP32 = 5376 * 2                     # the confidence head held in fp32
RESIDENT = STORED + MX8_ROW_REPEAT + NVFP4_SCALES + CONFIDENCE_FP32


@pytest.fixture(scope="module")
def names() -> dict[str, dict]:
    return inventory(json.loads(FIXTURE.read_text()))


def test_every_name_of_the_full_checkpoint_matches_exactly_one_rule(names):
    assert len(names) == 188_246
    assert sum(name.endswith(".trellis") for name in names) == 46_080
    assert sum(name.startswith("mtp.") and name.endswith(".w1.weight") and ".experts." in name for name in names) == 384
    kinds = {name: split.rule(name) for name in names}
    assert set(kinds.values()) == set(split.RULES)


@pytest.mark.parametrize("name, kind", [
    ("embed.weight", "rep"), ("layers.0.attn.wq_a.scale", "rep"), ("mtp.1.attn.wkv.weight", "rep"),
    ("layers.20.attn.compressor.wkv.weight", "rep"), ("layers.36.attn.indexer.wq_b.weight", "rep"),
    ("layers.8.attn.indexer.wk.weight", "rep"), ("layers.3.hc_ffn_fn", "rep"), ("mtp.2.ffn.gate.bias_vl", "rep"),
    ("layers.0.ffn.experts.0.w1.suh", "rep"), ("layers.0.ffn.experts.0.w2.svh", "rep"),
    ("layers.0.ffn.experts.383.w3.mcg", "rep"), ("layers.14.engram.q_weight", "rep"),
    ("mtp.2.markov_head.embed.weight", "rep"), ("mtp.2.confidence_head.proj.weight", "rep"),
    ("mtp.0.main_norm.weight", "rep"), ("mtp.2.norm.weight", "rep"),
    ("layers.0.ffn.experts.7.w1.trellis", "dim1"), ("layers.39.ffn.experts.7.w3.trellis", "dim1"),
    ("layers.0.ffn.experts.7.w1.svh", "row"), ("layers.0.ffn.experts.7.w2.trellis", "row"),
    ("layers.0.ffn.experts.7.w2.suh", "row"), ("layers.5.ffn.shared_experts.w3.scale", "row"),
    ("mtp.0.ffn.experts.127.w1.weight", "row"), ("mtp.0.ffn.experts.127.w3.scale", "row"),
    ("layers.5.ffn.shared_experts.w2.weight", "col"), ("layers.5.attn.wo_b.scale", "col"),
    ("mtp.1.ffn.experts.0.w2.weight", "col"), ("mtp.1.ffn.experts.0.w2.scale", "col"),
    ("layers.2.attn.wq_b.weight", "heads"), ("mtp.0.attn.attn_sink", "heads"), ("layers.2.attn.wo_a.scale", "groups"),
    ("lm_head.weight", "vocab"), ("lm_head.weight_scale", "vocab"), ("mtp.2.markov_head.head.weight", "vocab"),
    ("layers.1.engram.wkv.scale", "nsplit"), ("mtp.0.main_proj.weight", "nsplit"),
    ("layers.14.engram.embed.weight", "engram"), ("layers.1.engram.embed.scale", "engram"),
    ("vision.blocks.0.attn.wqkv.weight", "drop"), ("aligner.w2.bias", "drop"), ("image_newline", "drop"),
])
def test_each_tensor_family_has_the_designed_rule(name, kind):
    assert split.rule(name) == kind


@pytest.mark.parametrize("name", ["layers.0.attn.wq_c.weight", "mtp.0.ffn.experts.0.w1.trellis",
                                  "mtp.0.attn.compressor.wkv.weight", "layers.0.ffn.experts.0.w1.weight",
                                  "head.weight", "layers.1.engram.embed.bias", "model.layers.0.attn.wq_a.weight"])
def test_unknown_names_raise(name):
    with pytest.raises(ValueError, match="ambiguous or missing"):
        split.rule(name)


def test_the_two_ranks_tile_every_loaded_tensor(names):
    for name, info in names.items():
        kind, shape = split.rule(name), info["shape"]
        if kind in ("engram", "drop"):
            continue
        full = tuple(slice(0, n) for n in shape)
        assert split.slice_for(0, 1, name, shape) == full
        parts = [split.slice_for(r, 2, name, shape) for r in (0, 1)]
        if kind == "rep":
            assert parts == [full, full], name
            continue
        axis, half = split.AXIS[kind], shape[split.AXIS[kind]] // 2
        for r, part in enumerate(parts):
            assert part[axis] == slice(r * half, (r + 1) * half), name
            assert part[:axis] + part[axis + 1:] == full[:axis] + full[axis + 1:], name


@pytest.mark.parametrize("name, shape, rank, want", [
    ("layers.0.attn.wq_b.weight", [32768, 1280], 1, (slice(16384, 32768), slice(0, 1280))),
    ("layers.0.attn.wq_b.scale", [1024, 40], 1, (slice(512, 1024), slice(0, 40))),
    ("layers.0.attn.wo_a.weight", [8192, 4096], 1, (slice(4096, 8192), slice(0, 4096))),
    ("layers.0.attn.wo_b.scale", [160, 256], 1, (slice(0, 160), slice(128, 256))),
    ("layers.0.attn.attn_sink", [64], 1, (slice(32, 64),)),
    ("layers.0.ffn.experts.0.w1.trellis", [320, 144, 32], 1, (slice(0, 320), slice(72, 144), slice(0, 32))),
    ("layers.0.ffn.experts.0.w2.trellis", [144, 320, 32], 1, (slice(72, 144), slice(0, 320), slice(0, 32))),
    ("layers.0.ffn.experts.0.w1.suh", [5120], 1, (slice(0, 5120),)),
    ("layers.0.ffn.experts.0.w3.svh", [2304], 1, (slice(1152, 2304),)),
    ("layers.0.ffn.shared_experts.w2.scale", [160, 72], 1, (slice(0, 160), slice(36, 72))),
    ("mtp.0.ffn.experts.0.w2.weight", [5120, 1152], 1, (slice(0, 5120), slice(576, 1152))),
    ("mtp.0.ffn.experts.0.w2.scale", [5120, 72], 0, (slice(0, 5120), slice(0, 36))),
    ("lm_head.weight_scale", [129280, 160], 1, (slice(64640, 129280), slice(0, 160))),
    ("layers.14.engram.wkv.weight", [25600, 6144], 1, (slice(12800, 25600), slice(0, 6144))),
    ("mtp.0.main_proj.scale", [160, 480], 0, (slice(0, 80), slice(0, 480))),
    ("layers.0.ffn.experts.0.w1.mcg", [], 1, ()),
])
def test_rank_parts_follow_the_design(name, shape, rank, want):
    assert split.slice_for(rank, 2, name, shape) == want


@pytest.mark.parametrize("name, shape, rank, world", [
    ("layers.14.engram.embed.weight", [10, 256], 0, 2), ("vision.norm.weight", [1024], 0, 2),
    ("layers.0.attn.attn_sink", [63], 0, 2), ("layers.0.attn.attn_sink", [64], 2, 2),
])
def test_unloadable_parts_raise(name, shape, rank, world):
    with pytest.raises(ValueError):
        split.slice_for(rank, world, name, shape)


def test_tables_and_vision_cost_nothing(names):
    for name, info in names.items():
        if split.rule(name) in ("engram", "drop"):
            assert split.weights_estimate(name, info) == (0, 0), name


def test_per_rank_resident_total(names):
    stored = resident = 0
    for name, info in names.items():
        size, mapped = split.weights_estimate(name, info)
        assert mapped == 0
        resident += size
        if split.rule(name) not in ("engram", "drop"):
            part = split.slice_for(0, 2, name, info["shape"])
            stored += math.prod(s.stop - s.start for s in part) * capacity.itemsize(info, name)
    assert stored == STORED
    assert resident == RESIDENT == WHOLE_MAIN_PROJ + 118_639_232 + 212_336_640 - 39_344_640


@pytest.mark.skipif(not os.environ.get("TF_DSV41_MODEL"), reason="set TF_DSV41_MODEL to the checkpoint")
def test_real_headers_reproduce_the_total():
    model = Path(os.environ["TF_DSV41_MODEL"])
    found = capacity.headers(model)
    want = inventory(json.loads((model / "config.json").read_text()))
    assert {k: (v["dtype"], v["shape"]) for k, v in found.items()} == {k: (v["dtype"], v["shape"]) for k, v in want.items()}
    weights = capacity.estimate_weights(model, split.weights_estimate)
    assert (weights.resident, weights.mapped) == (RESIDENT, 0)
    assert weights.staging <= 5.4 * GIB             # three copies of the largest layer (14, with Engram's wkv)
