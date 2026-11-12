import json
import os
import stat
import subprocess
import sys
import time

import pytest

from tensorfold_web import bench, compare, report, tooleval
from tensorfold_web.mock_engine import serve as mock_serve

# a stand-in for tool-eval-bench: its progress events on stderr, its result file, as version 2.8.0 writes them
FAKE = r'''
import json, sys, time
args = sys.argv[1:]
if args == ["--version"]:
    print("tool-eval-bench 2.8.0"); sys.exit(0)
out = args[args.index("--json-file") + 1]
assert "--no-think" in args and "--short" in args and args[args.index("--seed") + 1] == "42"
import os; assert os.environ.get("TOOL_EVAL_API_KEY") == "tf-key"
ids = [("TC-01", "A", "pass", 2), ("TC-02", "A", "partial", 1), ("TC-03", "B", "fail", 0)]
for i, (sid, cat, st, pts) in enumerate(ids):
    print(json.dumps({"event": "scenario_start", "scenario_id": sid, "title": "t", "category": cat, "index": i, "total": 3}), file=sys.stderr, flush=True)
    print(json.dumps({"event": "scenario_result", "scenario_id": sid, "status": st, "points": pts, "index": i, "total": 3, "duration_seconds": 0.1}), file=sys.stderr, flush=True)
doc = {"tool_eval_bench_version": "2.8.0", "total_scenarios": 3, "safety_warnings": [],
       "scores": {"final_score": 50, "rating": "fair", "total_points": 3, "max_points": 6, "deployability": 70,
                  "responsiveness": 60, "median_turn_ms": 900.0, "worst_category": "B Parameter Precision (0%)",
                  "category_scores": [{"category": "A", "label": "Tool Selection", "earned": 3, "max": 4, "percent": 75, "pass_count": 1, "partial_count": 1, "fail_count": 0},
                                      {"category": "B", "label": "Parameter Precision", "earned": 0, "max": 2, "percent": 0, "pass_count": 0, "partial_count": 0, "fail_count": 1}],
                  "scenario_results": [{"scenario_id": s, "status": st, "points": p, "summary": "did " + s, "expected_behavior": "do " + s,
                                        "duration_seconds": 0.1, "ttft_ms": 120.0, "completion_tokens": 40} for s, _, st, p in ids]},
       "metadata": {"hostname": "secret-host"}}
open(out, "w").write(json.dumps(doc))
'''


@pytest.fixture
def fake_tool(tmp_path, monkeypatch):
    script = tmp_path / "tool-eval-bench"
    script.write_text(f"#!{sys.executable}\n{FAKE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("TOOL_EVAL_BENCH", str(script))
    return script


@pytest.mark.skipif(os.name == "nt", reason="the stand-in is a script with a shebang")
def test_tool_calls_run_through_tool_eval_bench_and_compare_scenario_by_scenario(tmp_path, fake_tool):
    server = mock_serve(port=0, rate_tps=2000.0)
    try:
        store = bench.Store(tmp_path / "bench.sqlite")
        runner = bench.Bench(store, lambda: (f"http://127.0.0.1:{server.server_address[1]}", "tf-key", "Qwen3.8-27B"))
        runner.power = False
        run = runner.start("tools")
        end = time.time() + 60
        while runner.busy() and time.time() < end:
            time.sleep(0.05)
    finally:
        server.shutdown()
    got = store.run(run["id"])
    assert got["status"] == "done", got["error"]
    assert got["tools"]["final_score"] == 50 and got["suites_version"]["digest"] == "teb-2.8.0-short-s42"
    assert [(r["prompt"], r["passed"], r["category"]) for r in got["results"]] == [
        ("TC-01", True, "Tool Selection"), ("TC-02", False, "Tool Selection"), ("TC-03", False, "Parameter Precision")]
    assert got["summary"]["quality"]["categories"]["Tool Selection"] == {"items": 2, "passed": 1}
    assert "secret-host" not in json.dumps(got) and "tools_exe" not in got
    assert report.findings(got)[0].startswith("tool-eval-bench 2.8.0: 50 of 100 (3 of 6 points), 1 scenarios passed")
    c = compare.compare(got, got)
    assert c["exactness"]["checked"] == 3 and any(f.startswith("Quality: the same outcome") for f in c["findings"])


def test_without_the_tool_the_mode_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOL_EVAL_BENCH", "")
    monkeypatch.setattr(tooleval, "CANDIDATES", ())
    monkeypatch.setattr(tooleval.shutil, "which", lambda name: None)
    runner = bench.Bench(bench.Store(tmp_path / "bench.sqlite"), lambda: ("http://127.0.0.1:9", None, "m"))
    with pytest.raises(ValueError, match="uv tool install tool-eval-bench"):
        runner.start("tools")


def test_the_install_button_runs_uv_with_the_checked_version(tmp_path, monkeypatch):
    calls = []
    exe = tmp_path / "tool-eval-bench"
    monkeypatch.setattr(tooleval.shutil, "which", lambda name: "/x/uv" if name == "uv" else None)
    monkeypatch.setattr(tooleval, "CANDIDATES", (str(exe),))
    monkeypatch.setenv("TOOL_EVAL_BENCH", "")
    monkeypatch.setattr(tooleval, "version", lambda e: tooleval.VERSION)

    def run(cmd, **kw):
        calls.append(cmd)
        exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    assert tooleval.install(run) == {"path": str(exe), "version": tooleval.VERSION}
    assert calls == [["/x/uv", "tool", "install", f"tool-eval-bench=={tooleval.VERSION}"]]
    with pytest.raises(RuntimeError, match="could not install it: error no network"):
        tooleval.install(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "error\nno network"))
    monkeypatch.setattr(tooleval.shutil, "which", lambda name: None)
    monkeypatch.setattr(tooleval, "UV_CANDIDATES", ())
    with pytest.raises(RuntimeError, match="uv is needed"):
        tooleval.install(run)
