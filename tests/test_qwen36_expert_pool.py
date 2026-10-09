"""The expert pool's file map and residency plan, without a GPU: synthetic checkpoints, real bytes."""

import json
import struct

import pytest

from tensorfold.families.qwen3_5_moe.cuda import pool

LAYERS, EXPERTS = 2, 4
# a real checkpoint's routed tensors: U32 words for the weights, BF16 scales and biases (two bytes a value)
VALUES = {"weight": ("U32", 8, 4), "scales": ("BF16", 2, 2), "biases": ("BF16", 2, 2)}


def shared_tensors(layers=LAYERS, drop=None):
    """A layer's shared expert: single-expert tensors beside the routed stacks (shape has no expert axis)."""

    out = []
    for layer in range(layers):
        for proj in pool.PROJS:
            for part in pool.PARTS:
                if drop is not None and drop(layer, proj, part):
                    continue
                dtype, columns, size = VALUES[part]
                rows = 3
                body = bytes([layer, pool.PARTS.index(part), pool.PROJS.index(proj)]) * (rows * columns * size // 3)
                out.append((f"language_model.model.layers.{layer}.mlp.shared_expert.{proj}.{part}", dtype,
                            [rows, columns], body))
    return out


def write_checkpoint(path, *, layers=LAYERS, experts=EXPERTS, parts=pool.PARTS, drop=None, extra=(),
                     shared=False, shared_drop=None):
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
    for name, dtype, shape, body in (shared_tensors(layers, shared_drop) if shared else ()):
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(body)]}
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


def test_a_call_that_fills_the_pool_exactly_assigns_every_id():
    # the widest call a pool can serve: one routed id a slot, the shared expert's own slot left alone
    slots = pool.Slots(256, 9)
    got = slots.assign(7, list(range(8)))
    assert len(set(got.slot_of.values())) == 8 and slots.shared not in got.slot_of.values()
    assert len(got.load) == 8 and got.evict == []
    with pytest.raises(ValueError, match="routing 9 distinct experts"):
        slots.assign(7, list(range(9)))


def test_an_id_outside_the_checkpoint_is_refused():
    with pytest.raises(ValueError, match="outside this checkpoint's 256"):
        pool.Slots(256, 4).assign(0, [256])


def test_the_reserved_shared_slots_are_never_handed_out():
    slots = pool.Slots(256, 45, reserved=40)
    got = slots.assign(3, list(range(5)))
    assert set(got.slot_of.values()) == set(range(5))          # the 40 shared slots are never handed out
    assert slots.shared_slot(0) == 5 and slots.shared_slot(39) == 44
    with pytest.raises(ValueError, match=r"routing 6 distinct experts does not fit the pool's 5"):
        slots.assign(3, list(range(6)))
    with pytest.raises(ValueError, match="no shared slot"):
        slots.shared_slot(40)


def test_shared_sources_read_one_expert_a_layer(tmp_path):
    write_checkpoint(tmp_path, shared=True)
    shared = pool.shared_sources(tmp_path, LAYERS)
    assert set(shared) == {(layer, proj, part) for layer in range(LAYERS) for proj in pool.PROJS
                           for part in pool.PARTS}
    routed = pool.sources(tmp_path, LAYERS, EXPERTS)
    for key, source in shared.items():                        # a shared expert is its own tensor, not a stack
        assert source.offset != routed[key].offset and source.rows != routed[key].rows
    for (layer, proj, part), source in sorted(shared.items()):
        at, count = source.span(0)
        body = source.path.read_bytes()[at:at + count]
        assert count == source.per_expert == 3 * source.columns * pool.itemsize({"dtype": source.dtype}, "")
        assert body[0] == layer and body[1] == pool.PARTS.index(part)


def test_shared_sources_refuse_an_incomplete_checkpoint(tmp_path):
    write_checkpoint(tmp_path, shared=True,
                     shared_drop=lambda layer, proj, part: (layer, proj, part) == (1, "up_proj", "weight"))
    with pytest.raises(ValueError, match=r"shared experts are incomplete: 1 tensors missing.*"
                                         r"layers\.1\.mlp\.shared_expert\.up_proj\.weight"):
        pool.shared_sources(tmp_path, LAYERS)


def test_a_call_wider_than_the_pool_is_refused_not_thrashed():
    # one call's experts must all be resident at once, so the pool refuses rather than evicting what it just read
    slots = pool.Slots(256, 45, reserved=40)
    with pytest.raises(ValueError, match=r"routing 6 distinct experts does not fit the pool's 5 routed slots.*"
                                         r"raise --expert-pool"):
        slots.assign(0, [1, 2, 3, 4, 5, 6])
    assert slots.assign(0, [1, 2, 3, 4, 5]).load == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]


def test_stats_report_the_hit_rate_over_every_requested_id():
    slots = pool.Slots(EXPERTS, 4)
    slots.assign(0, [0, 1])
    slots.assign(0, [1, 2])
    stats = slots.stats()
    assert stats["slots"] == 4 and stats["residency"] == 3 and stats["loads"] == 3 and stats["hits"] == 1
    assert stats["hit_rate"] == pytest.approx(0.25)


QWEN36 = {"num_hidden_layers": 40, "full_attention_interval": 4, "hidden_size": 2048,
          "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
          "linear_num_key_heads": 16, "linear_num_value_heads": 32,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
          "moe_intermediate_size": 512, "vocab_size": 151936, "num_experts_per_tok": 8}


def test_verify_rows_bounds_a_pooled_round_and_leaves_the_dense_default_alone():
    """A round verifies the carried token plus at most ``depth`` drafts; the 128-row default is a tree engine's."""

    from tensorfold.families.qwen3_5_moe.cuda.engine import verify_rows

    assert verify_rows(3, True) == 4 and verify_rows(0, True) == 1
    assert verify_rows(3, False) is None


def test_narrow_verify_rows_cut_the_pooled_floor_without_touching_the_cache():
    """The cutoff is the activation over-provision: the KV and the recurrent state stay charged in full."""

    from tensorfold.cuda import geometry
    from tensorfold.families.qwen3_5_moe.cuda.engine import verify_rows

    def needed(rows):
        return geometry.gdn_geometry(QWEN36, 1, 4, mtp=True, rows=rows).needed(8192)

    wide, narrow = needed(128), needed(verify_rows(3, True))
    assert wide - narrow > 1.5 * 2**30                     # the 128-row activation and replay arrays, gone
    assert (wide - narrow) / wide > 0.6                    # and they are most of what the estimate charged
    # what remains still covers the engine's real holdings: an 11-layer bf16 KV at 8192 rows
    assert narrow > (10 + 1) * 8192 * 2 * 256 * 4
