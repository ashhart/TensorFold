"""The CUDA expert-pool startup plan and Qwen3.6's routed-expert hook, without models or devices."""

import json
import math
import struct
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.cuda import capacity
from tensorfold.families import qwen3_5_moe

GIB = capacity.GIB
TITLE = "Qwen3.6 MoE"


def checkpoint(path, config, tensors):
    """A checkpoint's config and safetensors headers; the byte counts are declared, never written."""

    (path / "config.json").write_text(json.dumps(config))
    entries = {}
    offset = 0
    for name, dtype, shape, size in tensors:
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)


def expert_transform(pool: bool):
    """(resident, mapped[, streamed]): a routed stack is file-served when a pool serves it."""

    def transform(name, info):
        size = math.prod(info["shape"]) * capacity.itemsize(info, name)
        if pool and qwen3_5_moe.ROUTED_EXPERTS.search(name):
            return 0, 0, size
        return size, 0

    return transform


# one layer of the family's layout, small enough to declare in a header: non-experts, then two routed stacks
TENSORS = [("model.embed_tokens.weight", "U32", [1, 100], 400),
           ("model.layers.0.mlp.gate.weight", "U32", [1, 20], 80),
           ("model.layers.0.mlp.shared_expert.gate_proj.weight", "U32", [1, 10], 40),
           ("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", "U32", [1, 4_800_000_000],
            19_200_000_000),
           ("model.layers.0.mlp.switch_mlp.up_proj.weight", "U32", [1, 1000], 4000)]
ROUTED = 19_200_000_000 + 4000
REST = 400 + 80 + 40


def weights(**overrides):
    fields = {"resident": 2 * GIB + 200 * 2**20, "staging": 300 * 2**20, "mapped": 0, "streamed": 18 * GIB}
    return capacity.Weights(**(fields | overrides))


def test_estimate_weights_keeps_streamed_experts_out_of_the_resident_set(tmp_path):
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    resident = capacity.estimate_weights(tmp_path, expert_transform(False))
    pooled = capacity.estimate_weights(tmp_path, expert_transform(True))
    assert (resident.resident, resident.streamed) == (REST + ROUTED, 0)
    assert (pooled.resident, pooled.streamed) == (REST, ROUTED)


def test_a_pool_slot_is_the_landing_buffer_so_staging_drops_with_the_experts(tmp_path):
    # counting the stack in staging too would charge the same experts twice: once in the pool, once at load
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    resident = capacity.estimate_weights(tmp_path, expert_transform(False))
    pooled = capacity.estimate_weights(tmp_path, expert_transform(True))
    assert resident.staging == 3 * (ROUTED + 120)  # the layer, experts and all, is what loads without a pool
    assert pooled.staging == 3 * 400               # with one, the layer's non-experts are the load peak


def test_a_transform_still_returning_two_items_plans_exactly_as_before(tmp_path):
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    two = capacity.estimate_weights(tmp_path, lambda name, info: (math.prod(info["shape"]) * 4, 0))
    assert (two.resident, two.streamed) == (REST + ROUTED, 0)


@pytest.mark.parametrize("gib, expected", [(8.0, 8 * GIB), (64.0, 18 * GIB), (0.5, GIB // 2), (None, 0)])
def test_pool_bytes_is_capped_by_the_experts_it_serves(gib, expected):
    assert capacity.pool_bytes(gib, 18 * GIB) == expected


@pytest.mark.parametrize("gib", [0, -1, float("nan"), float("inf")])
def test_pool_bytes_refuses_a_size_that_is_not_a_positive_number(gib):
    with pytest.raises(ValueError, match="needs a positive number of GiB"):
        capacity.pool_bytes(gib, 18 * GIB)


def test_pool_bytes_refuses_a_checkpoint_whose_experts_stay_resident():
    with pytest.raises(ValueError, match="cannot be served from files"):
        capacity.pool_bytes(8.0, 0)


def test_a_pool_fits_a_window_the_resident_plan_refuses():
    budget = 4 * GIB
    geometry = capacity.Geometry(lambda slots: slots * 20_480, 0)      # 20 KB a token, as the family's page says
    held = weights(resident=20 * GIB + 700 * 2**20, staging=1700 * 2**20, streamed=0)
    resident_plan = capacity.make_plan(262144, 32768, True, budget, held, geometry)
    assert resident_plan.fitting == 0
    with pytest.raises(ValueError, match="--expert-pool"):
        capacity.choose(resident_plan)

    pooled = weights()
    plan = capacity.make_plan(262144, 32768, True, budget, pooled, geometry, pool=900 * 2**20)
    assert plan.pool == 900 * 2**20
    assert capacity.choose(plan) == 32768
    assert plan.largest > 32768 and plan.largest < 262144


def test_a_refusal_that_already_streams_does_not_offer_the_pool_again():
    geometry = capacity.Geometry(lambda slots: slots * 20_480, 0)
    plan = capacity.make_plan(262144, 262144, True, 4 * GIB, weights(), geometry, pool=900 * 2**20)
    with pytest.raises(ValueError) as caught:
        capacity.choose(plan)
    assert "--expert-pool" not in str(caught.value)
    assert "free memory" in str(caught.value)


def test_the_receipt_reports_the_pool_and_the_streamed_experts():
    geometry = capacity.Geometry(lambda slots: slots * 20_480, 0)
    plan = capacity.make_plan(262144, 32768, True, 4 * GIB, weights(), geometry, pool=900 * 2**20)
    receipt = plan.receipt(32768)
    assert receipt["expert_pool_bytes"] == 900 * 2**20
    assert receipt["streamed_weight_bytes"] == 18 * GIB
    assert receipt["weight_bytes_estimate"] == plan.weights.resident
    held = plan.weights.resident + plan.pool
    assert receipt["startup_peak_bytes_estimate"] == held + plan.weights.staging
    assert receipt["total_bytes_estimate"] == held + max(plan.weights.staging, geometry.needed(32768))


def test_the_family_hook_counts_only_the_routed_stacks(tmp_path):
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    assert qwen3_5_moe.cuda_expert_bytes(tmp_path) == ROUTED       # router and shared expert stay resident


def test_the_family_hook_agrees_with_the_profile_the_engine_plans(tmp_path):
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    assert capacity.estimate_weights(tmp_path, expert_transform(True)).streamed == \
        qwen3_5_moe.cuda_expert_bytes(tmp_path)


def test_cli_refuses_the_pool_with_the_checkpoint_and_pool_numbers(tmp_path):
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    package = SimpleNamespace(cuda_engine=lambda *a, **kw: SimpleNamespace(),
                              cuda_expert_bytes=qwen3_5_moe.cuda_expert_bytes)
    family = SimpleNamespace(title=TITLE, model_type="qwen3_5_moe", package=package)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--expert-pool", "8"])
    with pytest.raises(ValueError) as caught:
        cli._serve_cuda(args, family, tmp_path, 262144)
    text = str(caught.value)
    assert "--expert-pool 8.0: Qwen3.6 MoE's routed experts are 17.88 GiB" in text
    assert "a pool of 8.00 GiB would serve the rest" in text
    assert "does not fill a pool yet" in text


def test_cli_refuses_the_pool_for_a_family_that_holds_every_expert(tmp_path):
    checkpoint(tmp_path, {"max_position_embeddings": 262144}, TENSORS)
    family = SimpleNamespace(title="Test", model_type="test",
                             package=SimpleNamespace(cuda_engine=lambda *a, **kw: SimpleNamespace()))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--expert-pool", "8"])
    with pytest.raises(ValueError) as caught:
        cli._serve_cuda(args, family, tmp_path, 262144)
    assert "the flag applies to a family whose experts a pool can serve" in str(caught.value)


def test_the_pool_flag_reaches_the_family_engine(monkeypatch):
    """--expert-pool must survive the family's ``cuda_engine``: a served run caught it going missing here."""

    from tensorfold.families import qwen3_5_moe
    from tensorfold.families.qwen3_5_moe.cuda import engine as module

    seen = {}

    class Recorder:
        def __init__(self, path, **options):
            seen.update(options)
            seen["path"] = path

    monkeypatch.setattr(module, "Qwen36Engine", Recorder)
    qwen3_5_moe.cuda_engine("/tmp/checkpoint", context=32768, expert_pool=2.0)
    assert seen["expert_pool"] == 2.0 and seen["context"] == 32768
    qwen3_5_moe.cuda_engine("/tmp/checkpoint", context=32768)
    assert seen["expert_pool"] is None          # without the flag the engine keeps every expert resident


def test_serving_without_the_pool_reaches_the_engine_as_before(tmp_path, monkeypatch):
    from tensorfold.cuda import server

    observed = []
    family = SimpleNamespace(title=TITLE, model_type="qwen3_5_moe",
                             package=SimpleNamespace(cuda_engine=lambda path, **options: observed.append(options)
                                                     or SimpleNamespace(context_window=8192)))
    monkeypatch.setattr(server, "App", lambda *a, **kw: SimpleNamespace(effective_context_window=a[0].context_window))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    cli._serve_cuda(args, family, tmp_path, 262144)
    assert "expert_pool" not in observed[0]
    assert observed[0]["context"] == 262144
