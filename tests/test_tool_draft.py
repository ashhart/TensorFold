"""Tool-call drafts follow the schema; system-block snapshots survive a restart."""

from __future__ import annotations

import pytest

from tensorfold.engine.tool_draft import ToolCallProposer, tool_schema

TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {}, "offset": {}, "limit": {}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit", "parameters": {
        "type": "object", "properties": {"path": {}, "old_text": {}, "new_text": {}},
        "required": ["path", "old_text", "new_text"]}}},
]


class _Tok:
    def convert_tokens_to_ids(self, token: str) -> int:
        return 7


def _next(text: str) -> str | None:
    return ToolCallProposer(_Tok(), TOOLS, 0).structure(text)


def test_schema_puts_required_parameters_first():
    assert tool_schema(TOOLS) == {"read": ["path", "offset", "limit"], "edit": ["path", "old_text", "new_text"]}


def test_structure_follows_the_call():
    call = "<tool_call>\n<function="
    assert _next("") == "<tool_call>\n<function="
    assert _next(call) is None                      # the model picks the tool
    assert _next(call + "edit") == ">\n<parameter=path>\n"
    assert _next(call + "edit>") == "\n<parameter=path>\n"
    value = call + "edit>\n<parameter=path>\ncalc.py\n"
    assert _next(value) == "</parameter>\n<parameter=old_text>\n"
    assert _next(value + "</parameter>") == "\n<parameter=old_text>\n"
    done = call + "read>\n<parameter=path>\ncalc.py\n"
    assert _next(done) == "</parameter>\n<parameter=offset>\n"
    assert _next(call + "read>\n<parameter=path>\ncalc.py\n</parameter>\n</function>") == "\n</tool_call>"
    assert _next("Some prose first") is None
    # a multi-line value is never closed early
    assert _next(call + "edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=old_text>\ndef f():\n") is None


def test_snapshot_round_trip(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import ArraysCache, KVCache

    from tensorfold.engine.prefix_snapshots import load_snapshots, save_snapshot

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.full((1, 2, 5, 4), 2.0))
    gdn = ArraysCache(2)
    gdn.cache = [mx.arange(6).reshape(1, 2, 3), mx.full((1, 2, 2), 0.5)]
    tokens = list(range(40))
    assert save_snapshot(tmp_path, "model-a", tokens, [kv, gdn]) is not None
    assert save_snapshot(tmp_path, "model-a", tokens, [kv, gdn]) is None  # already there
    assert list(load_snapshots(tmp_path, "model-b")) == []
    (got_tokens, (kv2, gdn2)), = load_snapshots(tmp_path, "model-a")
    assert got_tokens == tokens
    assert type(kv2) is KVCache and kv2.offset == 5
    assert mx.array_equal(kv2.keys[..., :5, :], kv.keys[..., :5, :]).item()
    assert type(gdn2) is ArraysCache and mx.array_equal(gdn2.cache[0], gdn.cache[0]).item()
    k, v = kv2.update_and_fetch(mx.zeros((1, 2, 1, 4)), mx.zeros((1, 2, 1, 4)))
    assert k.shape[2] == 6


def test_streamed_tool_call_arguments_equal_the_parsed_call():
    import json

    from tensorfold.server.http import parse_tool_calls_from_content
    from tensorfold.engine.tool_draft import ToolCallStreamer

    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {}, "content": {}}, "required": ["path", "content"]}}}]
    full = ('Writing it.\n<tool_call>\n<function=write>\n<parameter=path>\nsite/index.html\n</parameter>\n'
            '<parameter=content>\n<!DOCTYPE html>\n<p class="x">Hi "there" \\ ok</p>\n  \n</parameter>\n'
            '</function>\n</tool_call>')
    streamer = ToolCallStreamer(tools)
    deltas = []
    for n in range(1, len(full) + 1, 5):
        deltas += streamer.feed(full[:n])
    deltas += streamer.feed(full)
    parts = [d["tool_calls"][0]["function"] for d in deltas]
    assert parts[0]["name"] == "write" and streamer.streamed
    arguments = "".join(p.get("arguments", "") for p in parts)
    _, calls = parse_tool_calls_from_content(full, tools)
    assert json.loads(arguments) == json.loads(calls[0]["function"]["arguments"])


def test_unknown_tool_is_not_streamed():
    from tensorfold.engine.tool_draft import ToolCallStreamer

    streamer = ToolCallStreamer([{"type": "function", "function": {"name": "read", "parameters": {}}}])
    assert streamer.feed("<tool_call>\n<function=bash>\n<parameter=command>\nls\n") == []
    assert not streamer.streamed


def test_parameter_values_keep_their_own_whitespace():
    """One framing newline a side is the template's; the rest belongs to the value.

    A written file's last newline (and an edit's indentation) must reach the client, and
    the client's resent history must then render to the tokens the model wrote: stripping
    the value re-prefilled a whole 15,007-token file write on the next turn (51 s).
    """
    import json

    from tensorfold.server.http import parse_tool_calls_from_content
    from tensorfold.engine.tool_draft import ToolCallStreamer

    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {}, "content": {}}}}}]
    values = {"path": "site/index.html", "content": "  <html>\n    <p>x</p>\n</html>\n"}
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in values.items())
    full = f"Writing it.\n\n<tool_call>\n<function=write>\n{body}</function>\n</tool_call>"
    _, calls = parse_tool_calls_from_content(full, tools)
    assert json.loads(calls[0]["function"]["arguments"]) == values
    for step in (1, 3, 7):          # the framing newline may arrive after its tag
        streamer = ToolCallStreamer(tools)
        deltas = []
        for n in range(1, len(full) + 1, step):
            deltas += streamer.feed(full[:n])
        deltas += streamer.feed(full)
        arguments = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
        assert json.loads(arguments) == values, step


def test_disk_blocks_read_the_longest_stored_prefix(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import os

    from mlx_lm.models.cache import KVCache

    from tensorfold.engine.prefix_snapshots import DiskBlocks, load_snapshot, save_snapshot

    def cache(n):
        kv = KVCache()
        kv.update_and_fetch(mx.ones((1, 2, n, 4)), mx.ones((1, 2, n, 4)))
        return [kv]

    short, long = list(range(30)), list(range(60))
    save_snapshot(tmp_path, "model-a", long, cache(60))
    save_snapshot(tmp_path, "model-a", short, cache(30))
    save_snapshot(tmp_path, "model-b", list(range(90)), cache(90))    # other kernels: never offered
    blocks = DiskBlocks(tmp_path, "model-a")
    prompt = list(range(100))
    path, tokens = blocks.best(prompt, 0)
    assert tokens == long
    assert blocks.best(prompt, 60) is None                 # the store already has as much
    assert blocks.best(list(range(59)), 0)[1] == short     # a strict prefix only
    assert blocks.best([7] + prompt, 0) is None
    got_tokens, (kv,) = load_snapshot(path, "model-a")
    assert got_tokens == long and kv.offset == 60
    os.utime(path, (1, 1))
    blocks.blocks()
    blocks.touch(long)
    assert path.stat().st_mtime > 1


def test_target_candidates_reach_the_capture_sidecar(tmp_path):
    """sample_rows hands back the candidates it drew among; the tool-call proposer forwards them to the
    DFlash proposer, which appends them next to its capture file (read back by position)."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from tensorfold.drafters.dflash_drafter import DFlashProposer, _capture_writer
    from tensorfold.engine.exact_sampling import Sampling, choose, sample_rows
    from tensorfold.engine.tool_draft import ToolCallProposer

    logits = mx.array(np.random.default_rng(0).normal(size=(3, 300)).astype(np.float32)).astype(mx.bfloat16)
    s = Sampling(seed=7)
    keep = {}
    toks = sample_rows(logits, [10, 11, 12], s, keep=keep)
    assert keep["cand"].shape[0] == 3 and keep["cand"].shape == keep["vals"].shape
    assert [choose(keep["vals"][r], keep["cand"][r].astype(np.int64), 10 + r, s) for r in range(3)] == toks

    inner = DFlashProposer.__new__(DFlashProposer)
    inner.capture_dir = str(tmp_path)
    inner._capture_file = tmp_path / "stream.bin"
    outer = ToolCallProposer(None, [], 0, fallback=inner)
    outer.capture_target([11, 12], keep["cand"][[0, 1]], keep["vals"][[0, 1]])
    _capture_writer().put((tmp_path / "flush-marker", b""))
    import time
    deadline = time.time() + 5
    while not (tmp_path / "flush-marker").exists() and time.time() < deadline:
        time.sleep(0.01)
    data = (tmp_path / "stream.logits").read_bytes()
    count, k = np.frombuffer(data, np.int32, 2, 0)
    positions = np.frombuffer(data, np.int64, count, 8)
    ids = np.frombuffer(data, np.int32, count * k, 8 + 8 * count).reshape(count, k)
    assert list(positions) == [11, 12] and (ids[1] == keep["cand"][1]).all()
