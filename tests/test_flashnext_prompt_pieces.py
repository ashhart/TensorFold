"""Idle prompt pieces use spare budget without changing the admitted window or delaying live replies."""

import copy

import pytest

from tensorfold.cuda.geometry import indexed_prompt_bytes
from tensorfold.families.qwen4_exp.cuda.prompt_plan import choose, pass_limit
from tests.test_cuda_capacity import small_config


def text():
    return {**small_config(), "hidden_size": 2560, "num_experts": 512, "num_experts_per_tok": 10,
            "moe_intermediate_size": 640, "hc_count": 4, "hc_lowrank": 320,
            "_quantization": {"bits": 4, "group_size": 32, "mode": "affine"}}


def receipt(t, spare, *, loading=0):
    fixed = indexed_prompt_bytes(t, 2048)
    total = max(fixed, loading) + 1000
    return {"context_window": 65536, "cache_slots": 65543, "largest_window": 131072,
            "cache_workspace_bytes_estimate": fixed, "serving_peak_bytes_estimate": fixed + 1000,
            "startup_peak_bytes_estimate": loading + 1000, "total_bytes_estimate": total,
            "mapped_table_bytes": 500, "full_mapped_working_set_bytes_estimate": total + 500,
            "budget_bytes": total + spare}


@pytest.mark.parametrize("loading", [0, 1 << 40])
def test_larger_pieces_count_every_extra_byte_without_reducing_the_window(loading):
    t = text()
    extra = indexed_prompt_bytes(t, 2048)
    plan = receipt(t, extra, loading=loading)
    old = copy.deepcopy(plan)
    rows, workspace = choose(plan, t, (12, 1))
    assert rows == 4096 and workspace == indexed_prompt_bytes(t, rows)
    assert plan["prefill_rows"] == rows and plan["prompt_workspace_bytes_estimate"] == workspace
    for key in ("context_window", "cache_slots", "largest_window", "budget_bytes"):
        assert plan[key] == old[key]
    assert plan["cache_workspace_bytes_estimate"] == old["cache_workspace_bytes_estimate"] + extra
    assert plan["serving_peak_bytes_estimate"] == old["serving_peak_bytes_estimate"] + extra
    peak = max(plan["serving_peak_bytes_estimate"], old["startup_peak_bytes_estimate"])
    assert plan["total_bytes_estimate"] == peak <= plan["budget_bytes"]
    assert plan["full_mapped_working_set_bytes_estimate"] == peak + plan["mapped_table_bytes"]


def test_insufficient_headroom_keeps_the_original_window_and_estimates():
    t = text()
    plan = receipt(t, indexed_prompt_bytes(t, 2048) - 1)
    old = copy.deepcopy(plan)
    assert choose(plan, t, (12, 1)) == (2048, 0)
    assert plan == old


def test_opt_in_fp8_keeps_the_released_piece_size():
    plan = {"untouched": True}
    assert choose(plan, text(), (12, 1), fp8=True) == (2048, 0)
    assert plan == {"untouched": True}


@pytest.mark.parametrize("change,capability,world,vision", [
    ({}, (9, 0), 1, False), ({}, (12, 0), 1, False), ({}, (12, 1), 2, False),
    ({}, (12, 1), 1, True), ({"hidden_size": 512}, (12, 1), 1, False),
    ({"num_experts": 128}, (12, 1), 1, False),
    ({"_quantization": {"bits": 4, "group_size": 64}}, (12, 1), 1, False),
    ({"_quantization": {}}, (12, 1), 1, False),
])
def test_unqualified_models_and_devices_keep_released_pieces(change, capability, world, vision):
    t = {**text(), **change}
    plan = {"untouched": True}
    assert choose(plan, t, capability, world=world, vision=vision) == (2048, 0)
    assert plan == {"untouched": True}


@pytest.mark.parametrize("share,round_s,row_s", [(0, None, None), (0.5, None, None), (0.5, 0.1, 0.0001)])
def test_idle_pieces_are_wide_and_live_pieces_never_exceed_the_released_limit(share, round_s, row_s):
    assert pass_limit(4096, False, share, round_s, row_s) == 4096
    assert 128 <= pass_limit(4096, True, share, round_s, row_s) <= 2048


def test_decode_share_and_small_test_buffers_keep_their_bounds():
    assert pass_limit(4096, True, 0.5, 0.05, 0.0001) == 960
    assert pass_limit(4096, True, 0.5, 0.0001, 0.0001) == 128
    assert pass_limit(16, True, 0.5, 0.0001, 0.0001) == 16


@pytest.mark.torch
def test_both_planner_paths_bound_live_pieces_and_restore_idle_width():
    from types import SimpleNamespace

    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    dec = object.__new__(MultiDecoder)
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 4096, 0.0, None, None
    dec.streams = {}
    prompt = SimpleNamespace(sid=1, prompt=[1] * 9000, background=False)
    dec.filling = [prompt]
    dec.fills = {1: [SimpleNamespace(stops=[]), False, 0, None]}
    assert dec._pieces() == [(prompt, 0, 4096)]
    reply = SimpleNamespace(done=False)
    dec.streams[2] = reply
    assert dec._pieces() == dec._pieces(dec._pass_rows()) == [(prompt, 0, 2048)]
    dec.share, dec.round_s, dec.row_s = 0.5, 0.05, 0.0001
    assert dec._pieces() == [(prompt, 0, 960)]
    reply.done = True
    assert dec._pieces() == [(prompt, 0, 4096)]
