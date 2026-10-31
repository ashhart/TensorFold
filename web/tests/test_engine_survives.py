import json
import sys
import time

import pytest

from tensorfold_web import engine as engine_mod
from tensorfold_web import engines
from tensorfold_web.engine import Engine, _redrawn
from tensorfold_web.settings import Settings

# an engine stand-in: a few lines, a progress bar redrawn with carriage returns, then it waits
SCRIPT = ("import sys, time\n"
          "print('[tensorfold] context window 4,096 tokens: the most one request can use', flush=True)\n"
          "sys.stdout.write('loading 10%\\rloading 50%\\rloading 100%\\n'); sys.stdout.flush()\n"
          "print('ready', flush=True)\n"
          "time.sleep(60)\n")


@pytest.fixture
def fake_launch(monkeypatch):
    class L:
        kind = "native"

        def launch(self, model, s, drafter=None, slide=None):
            return engines.Launch(kind="native", cmd=[sys.executable, "-c", SCRIPT], env={}, notes=[])

    monkeypatch.setattr(engines, "choose", lambda native_model: L())
    monkeypatch.setattr(Engine, "health", lambda self, *a, **k: engine_mod.Health(ok=False, status="unreachable"))


def _until(cond, timeout=10.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.05)
    return cond()


def test_the_engine_outlives_the_studio_and_is_found_again(tmp_path, fake_launch):
    first = Engine(Settings(engine_url="http://127.0.0.1:9"), state_dir=tmp_path)
    first.start("Org/Model")
    assert _until(lambda: any(line.endswith(" ready") for line in first.log))
    assert any(line.endswith("loading 100%") for line in first.log)            # the bar shows as its last state
    state = json.loads((tmp_path / "engine.json").read_text())
    assert state["model"] == "Org/Model" and state["pid"] == first.proc.pid
    assert oct((tmp_path / "engine.json").stat().st_mode & 0o777) == "0o600"

    # a new TensorFold Web: the first one's process is still there, found by pid and start time, its log read back
    second = Engine(Settings(engine_url="http://127.0.0.1:9"), state_dir=tmp_path)
    assert second.adopt() and second.owned and second.model == "Org/Model" and second.kind == "native"
    assert _until(lambda: any("previous TensorFold Web" in line for line in second.log))
    assert any(line == "--:--:-- ready" for line in second.log)
    assert second.startup.get("context_tokens") == 4096.0
    with pytest.raises(RuntimeError, match="already running"):
        second.start("Org/Other")

    second.stop(timeout=5)
    assert not second.owned and not first.owned
    assert _until(lambda: not (tmp_path / "engine.json").exists())
    assert not Engine(Settings(), state_dir=tmp_path).adopt()


def test_a_pid_given_to_another_process_is_not_taken_for_the_engine(tmp_path):
    import os

    (tmp_path / "engine.json").write_text(json.dumps({"pid": os.getpid(), "create_time": 1.0, "model": "x"}))
    assert not Engine(Settings(), state_dir=tmp_path).adopt()
    assert not (tmp_path / "engine.json").exists()


def test_progress_bars_show_their_last_state():
    assert _redrawn("a 10%\rb 20%\rc 30%") == "c 30%"
    assert _redrawn("plain line\r") == "plain line"
    assert len(_redrawn("x" * 20000)) == engine_mod.LINE_MAX


def test_a_stop_asked_for_is_not_a_failure_whatever_the_exit_code(tmp_path):
    """A binary without a SIGTERM handler exits -15 on the stop TensorFold Web asked for: it stopped, it did not fail."""

    import subprocess
    import sys

    from tensorfold_web import engine as engine_mod

    eng = Engine(Settings(engine_url="http://127.0.0.1:9", remote=False), state_dir=tmp_path)
    popen = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    child = engine_mod._Child(popen)
    eng.proc = child
    log = tmp_path / "engine.log"
    log.write_text("")
    eng._stopping = True                                                 # as stop() sets it before terminate()
    popen.terminate()
    eng._pump(child, log, False)
    assert popen.returncode == -15 and eng.exit_code == 0 and eng.status().status != "failed"


def test_a_kill_after_a_stop_that_took_too_long_is_a_stop_too(tmp_path):
    """stop() kills an engine that has not left after 20 s: -9 then, but TensorFold Web asked for it, so no OOM finding."""

    import subprocess
    import sys

    from tensorfold_web import engine as engine_mod

    eng = Engine(Settings(engine_url="http://127.0.0.1:9", remote=False), state_dir=tmp_path)
    popen = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    child = engine_mod._Child(popen)
    eng.proc = child
    log = tmp_path / "engine.log"
    log.write_text("")
    eng._stopping = True
    popen.kill()
    eng._pump(child, log, False)
    assert popen.returncode == -9 and eng.exit_code == 0 and eng.status().status != "failed"
