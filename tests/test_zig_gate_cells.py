"""The native engine's gate cells: a same-engine run passes, a planted token or slowdown fails on its own check."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from tensorfold.native import contract

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "zig"
HARNESS = os.environ.get("QUAL_HARNESS", "")       # the release qualification harness: qual_work.py and its kin
sys.path.insert(0, str(TOOLS))

import gate_cells  # noqa: E402
import gate_plant  # noqa: E402

needs_harness = pytest.mark.skipif(not (Path(HARNESS) / "qual_work.py").is_file(),
                                   reason="QUAL_HARNESS names no release qualification harness")
FAKE_SERVER = r'''
import argparse, hashlib, json, os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
for flag in ("command", "model", "--name", "--host", "--engine"):
    p.add_argument(flag)
p.add_argument("--port", type=int)
args, _ = p.parse_known_args()
plant = os.environ.get("FAKE_PLANT", "") if args.engine == "zig" else ""
if plant == "refuse":
    sys.exit(print("tensorfold: --engine zig: the native engine 0.6.6 lacks --snapshot-dir", file=sys.stderr) or 1)
print(f"[tensorfold] engine: {args.engine} (--engine {args.engine})", flush=True)
seen, lock = [], threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        data = json.dumps({"data": [{"id": args.name}]} if self.path.startswith("/v1/models") else {"ok": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        key = {k: body.get(k) for k in ("messages", "seed", "temperature", "max_tokens", "chat_template_kwargs")}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        words = [digest[i % 60:i % 60 + 4] for i in range(int(body["max_tokens"]))]
        if plant == "token" and body.get("temperature") == 0.0 and "Fibonacci" in body["messages"][-1]["content"]:
            words[5] = "zzzz"
        with lock:
            cached = 100 if any(body["messages"][:len(m)] == m and len(body["messages"]) > len(m) for m in seen) else 0
            seen.append(body["messages"])
        time.sleep(0.004)                       # a prompt pass long enough to time steadily
        self.send_response(200)
        self.end_headers()
        send = lambda chunk: (self.wfile.write(b"data: " + json.dumps({"model": args.name, **chunk}).encode() + b"\n\n"),
                              self.wfile.flush())
        for word in words:
            send({"choices": [{"delta": {"content": word + " "}}]})
            time.sleep(0.006 if plant == "slow" else 0.001)
        sha = hashlib.sha256(",".join(words).encode()).hexdigest()[:12]
        send({"choices": [{"delta": {}, "finish_reason": "length"}], "tensorfold": {"token_sha": sha}})
        send({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": len(words),
                                       "prompt_tokens_details": {"cached_tokens": cached}}})
        self.wfile.write(b"data: [DONE]\n\n")


server = ThreadingHTTPServer((args.host, args.port), Handler)
signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
server.serve_forever()
'''
FAKE_ZIG_GATE = ("import json, sys; a = sys.argv; print(json.dumps({'family': 'nemotron_h', 'format': 'mlx-q4g64', "
                 "'backend': 'metal', 'chip': 'apple-m5', 'run': a[a.index('--run') + 1], 'bundle': '0' * 64}))")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def plan(tmp_path):
    """A short gate plan over a fake `tensorfold serve` that answers like the Python engine."""

    package = tmp_path / "fake" / "tensorfold"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(FAKE_SERVER)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    (tmp_path / "zig_gate.py").write_text(FAKE_ZIG_GATE)
    return {"model_dir": str(tmp_path / "model"), "python": sys.executable, "src": str(tmp_path / "fake"),
            "zig_gate": str(tmp_path / "zig_gate.py"), "port": free_port(), "ready_s": 30, "args": [], "env": {},
            "phases": ["serial9", "cells", "sweep", "resume", "ladder"], "serial9": ["x1-code-off-g", "x2-code-off-s"],
            "resume": ["prefix-thinking-False"], "cell_tokens": [8], "max_tokens": 8, "prompts": 2, "seeds": [1234],
            "sweep": [1, 2], "ladder": ["1k"], "ladder_tokens": 8, "margin": 0.5, "steady": 0.5}


def gate(tmp_path: Path, plan: dict, plant: str = "") -> tuple[int, dict, Path]:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**plan, "env": {"FAKE_PLANT": plant}}))
    out = tmp_path / "run"
    done = subprocess.run([sys.executable, "-B", str(TOOLS / "gate_cells.py"), "run", "--plan", str(path),
                           "--qual", HARNESS, "--out", str(out), "--run", "rel000t", "--cells", str(tmp_path / "cells")],
                          capture_output=True, text=True, timeout=600)
    assert (out / "verdict.json").is_file(), done.stdout[-2000:] + done.stderr[-2000:]
    return done.returncode, json.loads((out / "verdict.json").read_text()), out


def failing(result: dict) -> list[dict]:
    return [check for check in result["checks"] if not check["pass"]]


@needs_harness
def test_the_same_answers_pass_and_the_pass_is_entered(tmp_path, plan):
    code, result, out = gate(tmp_path, plan)
    assert code == 0 and result["pass"], failing(result)
    names = {check["check"] for check in result["checks"]}
    assert {"sha==python", "python==python", "drafted==serial", "stream==solo", "fresh==resumed", "resumed-cached",
            "clean-stop", "engine", "speed", "steady"} <= names
    entry = json.loads((out / "entry.json").read_text())
    assert entry["run"] == "rel000t" and (tmp_path / "cells" / "nemotron_h-mlx-q4g64-metal-apple-m5.json").is_file()


@needs_harness
def test_one_changed_token_fails_on_python_s_tokens_alone(tmp_path, plan):
    code, result, out = gate(tmp_path, plan, plant="token")
    bad = failing(result)
    assert code == 1 and {check["check"] for check in bad} == {"sha==python"}
    assert sorted(check["item"].split()[1] for check in bad) == ["1", "2"]
    assert all("/code-g-8/p0/" in check["item"] for check in bad) and not (out / "entry.json").exists()


@needs_harness
def test_a_slower_engine_fails_on_speed_alone(tmp_path, plan):
    code, result, _ = gate(tmp_path, {**plan, "margin": 0.98}, plant="slow")
    bad = failing(result)
    assert code == 1 and {check["check"] for check in bad} == {"speed"}
    slow = {check["item"] for check in bad}
    assert {"single stream code-g-8", "single stream chat-s-8", "concurrency 2", "decode at 1k"} <= slow


@needs_harness
def test_a_native_engine_that_never_answers_is_named(tmp_path, plan):
    code, result, _ = gate(tmp_path, plan, plant="refuse")
    bad = failing(result)
    visit = [check for check in bad if check["check"] == "visit"]
    assert code == 1 and len(visit) == 1 and "lacks --snapshot-dir" in visit[0]["detail"]
    assert {check["check"] for check in bad} >= {"visits", "visit", "engine"}


@needs_harness
def test_speed_fails_on_a_clear_loss_and_on_an_unsteady_arm():
    sys.path.insert(0, HARNESS)
    import gate_judge

    def visits(*values):
        return [{"rows": [{"phase": "cells", "cell": "c", "unit": "p0", "decode_tps": v, "error": None}]} for v in values]

    for base, cand, slower, steady in (((100, 101), (99.5, 100.4), False, True), ((100, 101), (90, 91), True, True),
                                       ((100, 80), (99, 98), False, False)):
        verdict = gate_judge.Verdict()
        rows = gate_judge.speed(visits(*base), visits(*cand), {"margin": 0.98, "steady": 0.95}, verdict)
        assert rows[0]["slower"] is slower
        assert all(check["pass"] for check in verdict.checks if check["check"] == "steady") is steady


@needs_harness
def test_a_cuda_box_container_keeps_the_box_harness_limits(plan):
    sys.path.insert(0, HARNESS)
    import gate_serve

    box = {**gate_cells.GATE, **plan, "python": "python", "src": "/gate/src", "model_dir": "/models/m",
           "launch": {"kind": "docker", "name": "tf-gate", "image": "img:1", "memory": "80g", "timeout": 600,
                      "mounts": ["/data:/models:ro"], "workdir": "/gate"}}
    serve = gate_serve.serve_argv(box, "zig")
    argv = gate_serve.launcher(box)._argv(serve, True)
    at = argv.index("--entrypoint")
    assert argv[:3] == ["docker", "run", "--rm"] and argv[at + 1:at + 4] == ["timeout", "img:1", "600"]
    assert argv[at + 4:] == serve and serve[serve.index("--engine") + 1] == "zig"
    assert {"tensorfold.gate=tf-gate", "PYTHONPATH=/gate/src", "/data:/models:ro", "80g", "--read-only"} <= set(argv)


def test_a_plan_is_refused_before_any_server_starts(tmp_path, plan):
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        busy = {**plan, "order": ["base", "cand"], "args": ["--port", "1"], "port": other.getsockname()[1]}
        found = gate_cells.problems({**gate_cells.GATE, **busy}, None)
    assert any("two visits" in p for p in found) and any("sets --engine" in p for p in found)
    assert any("--run LABEL" in p for p in found) and any("is taken" in p for p in found)
    assert not gate_cells.problems({**gate_cells.GATE, **plan, "engines": {"base": "python", "cand": "python"}}, None)


def test_a_gate_plan_comes_from_the_release_plan(tmp_path, capsys):
    release = {"model_dir": "~/m", "port": 8491, "ready_s": 1500, "cell_tokens": [64], "quick": False,
               "sweep": [2, 4], "ladder": ["2k", "8k", "32k"], "ladder_tokens": 64, "candidate": "cand",
               "arms": {"cand": {"python": "~/py", "src": "~/src", "env": {"HF_HUB_OFFLINE": "1"},
                                 "args": ["--no-thinking"], "phases": ["serial9", "cells"]}}}
    (tmp_path / "release.json").write_text(json.dumps(release))
    assert gate_cells.main(["plan", "--release", str(tmp_path / "release.json"), "--src", ""]) == 0
    made = json.loads(capsys.readouterr().out)
    assert made["src"] is None and made["sweep"] == [1, 2, 4, 8, 16] and made["order"] == ["base", "cand", "cand", "base"]
    assert made["args"] == ["--no-thinking"] and made["engines"] == {"base": "python", "cand": "zig"}


def test_the_stand_in_reports_capabilities_the_switch_accepts(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_NO_LIVE", "1")
    found = gate_plant.capabilities({"families": {"nemotron_h": ["mlx-q4g64"]}})
    contract.check(found, contract.schema("capabilities"))
    assert "--snapshot-dir" in found["serve"]["flags"] and "--engine" not in found["serve"]["flags"]
    assert "TENSORFOLD_NO_LIVE" in found["env"]


def test_the_planted_character_lands_however_the_pieces_split():
    whole = "".join(f"piece{i} " for i in range(9))
    for size in (1, 3, 7, 50):
        got: list[str] = []
        swap = gate_plant.Swap(got.append)
        for at in range(0, len(whole), size):
            swap(whole[at:at + size])
        assert "".join(got) == gate_plant.swap(whole, gate_plant.OFFSET) != whole
