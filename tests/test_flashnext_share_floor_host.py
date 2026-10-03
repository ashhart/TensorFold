"""Exercise the production pass policy and piece widths without importing accelerator runtimes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _decoder():
    root = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/cuda"
    plan = ast.parse((root / "prompt_plan.py").read_text())
    source = ast.parse((root / "multi_fill.py").read_text())
    methods = {"_pass_rows", "_pieces", "_timed", "_pass"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in plan.body if isinstance(node, ast.FunctionDef) and node.name == "pass_limit")
    nodes.extend(node for node in source.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "PASS_MIN" for target in node.targets))
    owner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PromptPasses")
    body = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    nodes.append(ast.ClassDef(name="Decoder", bases=[], keywords=[], body=body, decorator_list=[], type_params=[]))
    namespace = {"PREFILL_ROWS": 2048, "ENDS": 16}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(root / "multi_fill.py"), "exec"), namespace)
    dec = namespace["Decoder"]()
    dec.streams = {0: SimpleNamespace(done=False)}
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 2048, 0.25, None, None
    return dec, namespace


@pytest.mark.parametrize("rows,live,share,round_s,row_s,expected", [
    (2048, True, 0.25, None, None, 2048),
    (2048, True, 0.25, 0.001, 0.001, 512),
    (2048, True, 0.25, 0.08, 0.0005, 640),
    (2048, True, 0.25, 0.1, 0.0004, 960),
    (4096, True, 0.25, 1.0, 0.0001, 2048),
    (256, True, 0.25, 0.001, 0.001, 256),
    (4096, True, 0.0, 0.001, 0.001, 2048),
    (4096, False, 0.25, 0.001, 0.001, 4096),
])
def test_floor_preserves_calibration_and_workspace_ceiling(rows, live, share, round_s, row_s, expected):
    dec, _ = _decoder()
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = rows, share, round_s, row_s
    dec.streams[0].done = not live
    assert dec._pass_rows() == expected
    assert 0 < expected <= rows


@pytest.mark.parametrize("round_s,row_s,width", [(0.001, 0.001, 512), (0.08, 0.0005, 640), (0.1, 0.0004, 960)])
def test_actual_piece_rows_use_the_floor_without_turning_it_into_a_ceiling(round_s, row_s, width):
    dec, _ = _decoder()
    dec.round_s, dec.row_s = round_s, row_s
    streams = [SimpleNamespace(sid=1, prompt=list(range(100))), SimpleNamespace(sid=2, prompt=list(range(4096)))]
    dec._order = lambda: streams
    dec.fills = {stream.sid: [SimpleNamespace(stops=[]), False, 0, None] for stream in streams}
    pieces = dec._pieces()
    assert [(stream.sid, start, rows) for stream, start, rows in pieces] == [(1, 0, 100), (2, 0, width - 100)]
    assert sum(rows for _, _, rows in pieces) == width
    dec.fills[2][0].stops = [200]
    pieces = dec._pieces()
    assert sum(rows for _, _, rows in pieces) == 300


def test_shared_prompt_row_floors_host():
    dec, _ = _decoder()
    for round_s, row_s, expected in [(0.001, 0.001, 512), (0.08, 0.0005, 640), (0.1, 0.0004, 960)]:
        dec.round_s, dec.row_s = round_s, row_s
        assert dec._pass_rows() == expected


@pytest.mark.parametrize("converged,failed,prior", [(False, False, None), (False, False, 0.002),
                                                  (True, False, None), (False, True, None)])
def test_standalone_calibration_uses_completed_nonconvergent_work(converged, failed, prior):
    dec, namespace = _decoder()
    stream = SimpleNamespace(st=object(), prompt=list(range(512)))
    pieces = [(stream, 0, 128), (stream, 128, 384)]
    dec.converged, dec.row_s, dec.w, dec.pbuf = converged, prior, object(), object()
    dec.pass_plan, dec.pass_index = None, 0
    dec._pieces, dec._note_passed = lambda *_: pieces, lambda _: None
    dec._prompt_candidates = lambda *_: None
    dec._end_rows, dec._cuts, dec._absorb = lambda *_: [], lambda *_: [], lambda *_: []
    dec._joined, dec._failed = lambda *args: ("joined", args[-2]), lambda *_: ("failed", None)
    namespace["stage"] = lambda *_: []
    def compute(*args, **kwargs):
        if failed:
            raise RuntimeError("synthetic pass failure")
    namespace["compute"] = compute
    ticks = iter([10.0, 10.512, 10.512])
    namespace["time"] = SimpleNamespace(perf_counter=lambda: next(ticks))
    result = dec._pass()
    assert result[0] == ("failed" if failed else "joined")
    if failed or converged:
        assert dec.row_s == prior
    else:
        assert dec.row_s == pytest.approx(0.001 if prior is None else 0.0017)
    if not failed:
        assert result[1] == pytest.approx(0.256)
