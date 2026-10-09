"""The expert pool's file map and residency plan, without a GPU: synthetic checkpoints, real bytes."""

import json
import struct

import pytest

from tensorfold.families.qwen3_5_moe.cuda import pool

LAYERS, EXPERTS = 2, 4
# a real checkpoint's routed tensors: U32 words for the weights, BF16 scales and biases (two bytes a value)
VALUES = {"weight": ("U32", 8, 4), "scales": ("BF16", 2, 2), "biases": ("BF16", 2, 2)}


def write_checkpoint(path, *, layers=LAYERS, experts=EXPERTS, parts=pool.PARTS, drop=None, extra=()):
    """A real safetensors shard whose each expert's bytes are its (layer, expert, part) signature."""

    entries, blobs, offset = {}, [], 0
    for layer in range(layers):
        for proj in pool.PROJS:
            for part in parts:
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}.{part}"
                if drop is not None and drop(layer, proj, part):
                    continue
                dtype, columns, size = VALUES[part]
                stride = experts * columns * size                # one expert's bytes, each signed with its id
                body = b"".join(bytes([layer, e, pool.PARTS.index(part), pool.PROJS.index(proj)])
                                + bytes([e]) * (stride - 4) for e in range(experts))
                entries[name] = {"dtype": dtype, "shape": [experts, experts, columns],
                                 "data_offsets": [offset, offset + len(body)]}
                offset += len(body)
                blobs.append(body)
    for name, body in extra:
        entries[name] = {"dtype": "U32", "shape": [1, len(body) // 4, 1], "data_offsets": [offset, offset + len(body)]}
        offset += len(body)
        blobs.append(body)
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def test_a_source_spans_one_expert_of_its_projection(tmp_path):
    write_checkpoint(tmp_path)
    found = pool.sources(tmp_path, LAYERS, EXPERTS)
    assert set(found) == {(layer, proj, part) for layer in range(LAYERS) for proj in pool.PROJS
                          for part in pool.PARTS}
    for key, source in found.items():
        layer, proj, part = key
        raw = source.path.read_bytes()
        for expert in range(EXPERTS):
            at, count = source.span(expert)
            body = raw[at:at + count]
            assert count == source.per_expert
            assert body[0] == layer and body[1] == expert
            assert body[2] == pool.PARTS.index(part) and body[3] == pool.PROJS.index(proj)
            assert len(set(body[4:])) == 1 and body[4] == expert


def test_a_source_covers_exactly_one_expert_of_its_stack(tmp_path):
    write_checkpoint(tmp_path)
    declared = {name: entry for _, (_, entries) in pool.shards(tmp_path).items() for name, entry in entries.items()}
    for (layer, proj, part), source in pool.sources(tmp_path, LAYERS, EXPERTS).items():
        entry = declared[f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}.{part}"]
        assert source.per_expert * EXPERTS == entry["data_offsets"][1] - entry["data_offsets"][0]
        assert source.per_expert == pool.per_expert_bytes(f"{layer}.{proj}.{part}", entry)


def test_sources_refuse_an_incomplete_checkpoint(tmp_path):
    write_checkpoint(tmp_path, drop=lambda layer, proj, part: (layer, proj, part) == (1, "down_proj", "scales"))
    with pytest.raises(ValueError, match=r"incomplete: 1 tensors missing.*layers\.1\.mlp\.switch_mlp\.down_proj\.scales"):
        pool.sources(tmp_path, LAYERS, EXPERTS)


def test_sources_ignore_the_router_and_the_shared_expert(tmp_path):
    # mlp.gate is the router and mlp.shared_expert.gate_proj is the always-on expert: neither is streamed
    extra = [("language_model.model.layers.0.mlp.gate.weight", b"\x01" * 16),
             ("language_model.model.layers.0.mlp.shared_expert.gate_proj.weight", b"\x02" * 16)]
    write_checkpoint(tmp_path, extra=extra)
    found = pool.sources(tmp_path, LAYERS, EXPERTS)
    assert len(found) == LAYERS * len(pool.PROJS) * len(pool.PARTS)


def test_sources_refuse_a_stack_of_the_wrong_expert_count(tmp_path):
    write_checkpoint(tmp_path, experts=EXPERTS)
    with pytest.raises(ValueError, match=r"is not \[8, rows, columns\]"):
        pool.sources(tmp_path, LAYERS, 8)


def test_assign_keeps_a_row_order_and_leaves_the_shared_slot_alone():
    slots = pool.Slots(EXPERTS, 4)
    got = slots.assign(0, [3, 1, 3, 0])
    assert got.slots(0, [3, 1, 0]) == [got.slot_of[(0, 3)], got.slot_of[(0, 1)], got.slot_of[(0, 0)]]
    assert [e for _, e in got.load] == [3, 1, 0]          # one read per distinct id, in first-seen order
    assert slots.shared not in got.slot_of.values()       # slot 3 belongs to the shared expert alone
    assert set(got.slot_of.values()) == {0, 1, 2}


def test_a_resident_expert_is_a_hit_and_is_not_read_again():
    slots = pool.Slots(EXPERTS, 4)
    slots.assign(0, [2, 1])
    again = slots.assign(0, [1, 2])
    assert again.load == [] and slots.hits == 2 and slots.loads == 2


def test_eviction_takes_the_least_recently_used_slot():
    slots = pool.Slots(EXPERTS, 3)                         # slots 0, 1 hold experts; slot 2 is the shared one
    first = slots.assign(0, [0, 1])
    slots.assign(0, [0])                                   # (0, 0) is now the most recently used
    third = slots.assign(0, [2])
    assert third.load == [(0, 2)] and third.evict == [first.slot_of[(0, 1)]]
    assert slots.held[(0, 0)] == first.slot_of[(0, 0)]      # the recently used expert kept its slot
    assert (0, 1) not in slots.held


def test_a_layer_of_more_experts_than_the_pool_still_assigns_every_id():
    # a prompt chunk of 2,048 rows can pick every one of the 256 experts: the pool cycles slots and the
    # remap hands each id a slot, so no id is left unassigned even when residency is far smaller
    slots = pool.Slots(256, 9)
    got = slots.assign(7, list(range(256)))
    assert len(set(got.slot_of.values())) == 8
    assert len(got.evict) == 256 - 8
    assert slots.shared not in got.slot_of.values()


def test_an_id_outside_the_checkpoint_is_refused():
    with pytest.raises(ValueError, match="outside this checkpoint's 256"):
        pool.Slots(256, 4).assign(0, [256])


def test_stats_report_the_hit_rate_over_every_requested_id():
    slots = pool.Slots(EXPERTS, 4)
    slots.assign(0, [0, 1])
    slots.assign(0, [1, 2])
    stats = slots.stats()
    assert stats["slots"] == 4 and stats["residency"] == 3 and stats["loads"] == 3 and stats["hits"] == 1
    assert stats["hit_rate"] == pytest.approx(0.25)
