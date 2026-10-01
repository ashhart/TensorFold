"""DeepSeek-V4.1-Flash family: discovery, config parsing and CLI refusals, without torch or weights."""

import json
import shutil
from pathlib import Path

import pytest

from tensorfold import families
from tensorfold.families import deepseek_v41
from tensorfold.families.deepseek_v41.config import Config

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41"


def test_family_is_discovered_for_cuda():
    family = families.families()["deepseek_v41"]
    assert families.backends_of(family) == ("cuda",)


def test_config_parses_the_released_checkpoint():
    c = Config.from_dict(json.loads((FIXTURE / "config.json").read_text()))
    assert (c.hidden_size, c.num_hidden_layers, c.n_routed_experts) == (5120, 40, 384)
    assert len(c.layer_ratios) == 40 and c.layer_ratios[:3] == (0, 0, 2) and c.layer_ratios[-1] == 1
    assert c.engram_layer_ids == (1, 14) and c.dspark_target_layer_ids == (37, 38, 39)
    assert (c.rope_factor, c.rope_original) == (16.0, 65536)


def test_config_names_a_missing_setting():
    config = json.loads((FIXTURE / "config.json").read_text())
    del config["text_config"]["index_topk"]
    with pytest.raises(KeyError, match="index_topk"):
        Config.from_dict(config)


def test_check_accepts_exl3(tmp_path, capsys):
    shutil.copy(FIXTURE / "config.json", tmp_path / "config.json")
    deepseek_v41.check(tmp_path)
    assert "two DGX Sparks" in capsys.readouterr().out


def test_engine_needs_two_ranks(tmp_path):
    with pytest.raises(ValueError, match="two GPUs"):
        deepseek_v41.cuda_engine(tmp_path, tp=1)
    with pytest.raises(ValueError, match="--master"):
        deepseek_v41.cuda_engine(tmp_path, tp=2)


def test_engram_dir_override(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_DSV41_ENGRAM_DIR", raising=False)
    assert deepseek_v41.engram_dir(tmp_path) == tmp_path / "engram"
    monkeypatch.setenv("TF_DSV41_ENGRAM_DIR", "/x/engram-src")
    assert deepseek_v41.engram_dir(tmp_path) == Path("/x/engram-src")


def test_split_rules():
    from tensorfold.families.deepseek_v41.split import rank_bytes, rule

    assert rule("layers.3.attn.wq_b.trellis") == "cols"
    assert rule("layers.3.attn.wo_b.suh") == "rows"
    assert rule("layers.3.ffn.experts.17.w2.trellis") == "rows"
    assert rule("layers.3.ffn.shared_experts.w1.svh") == "cols"
    assert rule("layers.3.attn.wq_a.trellis") == "rep"
    assert rule("vision.blocks.0.attn.wo.weight") == "skip"
    assert rank_bytes("layers.3.attn.wo_a.slice.2.trellis", 100, 0) == 100
    assert rank_bytes("layers.3.attn.wo_a.slice.5.trellis", 100, 0) == 0
    assert rank_bytes("head.trellis", 100, 1) == 50


def test_part_kinds_keep_exl3_tiles_and_scales_together():
    from tensorfold.families.deepseek_v41.split import part_kind

    assert [part_kind(f"layers.3.attn.wq_b.{p}") for p in ("trellis", "suh", "svh", "mul1")] == \
        ["dim1", "rep", "row", "rep"]
    assert [part_kind(f"layers.3.ffn.experts.9.w2.{p}") for p in ("trellis", "suh", "svh", "mul1")] == \
        ["row", "row", "rep", "rep"]
    assert part_kind("layers.3.attn.attn_sink") == "row"
    assert part_kind("layers.3.attn.wo_a.slice.6.trellis") == "rep"
    assert part_kind("embed.weight") == "rep"
    assert part_kind("vision.norm.weight") == "drop"


def test_v41_dsml_tool_calls_parse():
    """V4.1's template writes DSML with a space after the marker (``<｜DSML｜ calls>``); V4's form still parses."""

    from tensorfold.server.tools import parse_tool_calls_from_content

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]
    for sp, tag in ((" ", " calls"), ("", "tool_calls")):
        text = (f"Let me check.<｜DSML｜{tag}>\n<｜DSML｜{sp}invoke name=\"get_weather\">\n"
                f"<｜DSML｜{sp}parameter name=\"city\" string=\"true\">Warsaw</｜DSML｜{sp}parameter>\n"
                f"<｜DSML｜{sp}parameter name=\"days\" string=\"false\">3</｜DSML｜{sp}parameter>\n"
                f"</｜DSML｜{sp}invoke>\n</｜DSML｜{tag}>")
        content, calls = parse_tool_calls_from_content(text, tools)
        assert content.strip() == "Let me check." and calls is not None and len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        import json

        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Warsaw", "days": 3}


def test_v41_dsml_through_the_cuda_server_parser():
    """The CUDA server's reply parser and stream hider read V4.1's DSML blocks."""

    import json

    from tensorfold.cuda.reply_text import hide_tool_calls, parse_tool_calls

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}}]
    text = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="city" string="true">'
            'Warsaw</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    content, calls = parse_tool_calls("Sure. " + text, tools)
    assert content == "Sure." and json.loads(calls[0]["function"]["arguments"]) == {"city": "Warsaw"}
    assert hide_tool_calls("Sure. " + text, finished=True) == "Sure. "
    assert hide_tool_calls("Sure. <｜DSML｜ ca", finished=False) == "Sure. "      # a partial opener is held back
