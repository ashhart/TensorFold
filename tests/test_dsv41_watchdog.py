"""deploy/dsv41-tp2/watchdog.sh against stub docker / ssh / curl / make: when it stands down, what counts as a bad
tick, when it heals (in the background, under the lock, with TF_DSV41_LOCKED so the Makefile does not lock again),
the rate limit, alert-only mode, and that the API key never reaches a command line."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "dsv41-tp2" / "watchdog.sh"
DEPLOY = ROOT / "deploy" / "dsv41-tp2"

pytestmark = pytest.mark.skipif(not all(shutil.which(t) for t in ("bash", "flock", "jq", "setsid")),
                                reason="needs bash, flock, jq and setsid")

STUBS = {
    # docker: a side's state file (head_state / worker_state: running, exited, absent), started, ps, logs
    "docker": r"""#!/usr/bin/env bash
side=${STUB_SIDE:-head}
case "$1" in
  inspect) [[ "$3" == *StartedAt* ]] && { cat "$STUB/started"; exit 0; }
           s=$(cat "$STUB/${side}_state" 2>/dev/null || echo absent); [[ $s == absent ]] && exit 1; echo "$s" ;;
  ps) cat "$STUB/${side}_ps" 2>/dev/null; true ;;
  logs) echo "logs of $side" ;;
esac
""",
    "ssh": r"""#!/usr/bin/env bash
[[ -f "$STUB/worker_unreachable" ]] && exit 255
STUB_SIDE=worker bash -c "${@: -1}"
""",
    "curl": r"""#!/usr/bin/env bash
echo "$*" >> "$STUB/curl_argv"
for a in "$@"; do [[ $a == @* ]] && cat "${a:1}" >> "$STUB/curl_headers"; done
case "$*" in
  */health*) printf '%s\n%s' "$(cat "$STUB/health_body" 2>/dev/null)" "$(cat "$STUB/health_code" 2>/dev/null || echo 000)" ;;
  */chat/completions*) echo probe >> "$STUB/probes"; exit "$(cat "$STUB/probe_rc" 2>/dev/null || echo 0)" ;;
esac
""",
    "make": r"""#!/usr/bin/env bash
echo "TF_DSV41_LOCKED=${TF_DSV41_LOCKED:-} $*" >> "$STUB/make_calls"
""",
}

HEALTHY = '{"ok": true, "requests_running": 0, "fatal": null, "stalled": false, "call_age_s": null}'


class Pair:
    def __init__(self, tmp: Path):
        self.stub, self.state, self.dir = tmp / "stub", tmp / "state" / "tensorfold-dsv41", tmp / "deploy"
        self.stub.mkdir()
        self.dir.mkdir()
        (self.dir / ".env").write_text("PORT=8888\nTF_API_KEY=sk-secret-123\nSERVED_NAME=DS\n")
        bin_ = tmp / "bin"
        bin_.mkdir()
        for name, text in STUBS.items():
            (bin_ / name).write_text(text)
            (bin_ / name).chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "STUB": str(self.stub),
                    "XDG_STATE_HOME": str(tmp / "state"), "WATCH_DIR": str(self.dir), "WATCH_PROBE_S": "600"}
        self.set(head_state="running", worker_state="running", health_code="200", health_body=HEALTHY,
                 started=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())

    def set(self, **files):
        for name, value in files.items():
            path = self.stub / name
            if value is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(str(value))

    def tick(self, **env) -> tuple[int, str]:
        done = subprocess.run(["bash", str(SCRIPT)], env={**self.env, **env}, capture_output=True, text=True,
                              timeout=60)
        return done.returncode, done.stdout + done.stderr

    def read(self, name: str, where: Path | None = None) -> str:
        path = (where or self.state) / name
        return path.read_text() if path.exists() else ""

    def heals(self) -> list[str]:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not self.read("make_calls", self.stub):
            time.sleep(0.05)
        return self.read("make_calls", self.stub).splitlines()


@pytest.fixture
def pair(tmp_path):
    return Pair(tmp_path)


def test_a_healthy_pair_is_probed_once_while_idle(pair):
    assert pair.tick() == (0, "") and pair.read("fails") == "0\n"
    assert pair.tick()[0] == 0 and pair.read("probes", pair.stub).count("probe") == 1          # every 600 s
    argv, headers = pair.read("curl_argv", pair.stub), pair.read("curl_headers", pair.stub)
    assert "sk-secret-123" not in argv and "Authorization: Bearer sk-secret-123" in headers   # never in argv


@pytest.mark.parametrize("case,files,said", [
    ("lease", {}, "lease until"),
    ("stopped", {"head_state": "absent", "worker_state": "absent"}, "stopped on purpose"),
    ("vllm", {"worker_ps": "dsv41-exl3-worker\n"}, "vLLM holds the pair"),
    ("loading", {"health_code": "000", "started": datetime.now(timezone.utc).isoformat()}, "loading"),
])
def test_it_stands_down(pair, case, files, said):
    pair.state.mkdir(parents=True)
    if case == "lease":
        (pair.state / "lease").touch()
        future = time.time() + 600
        os.utime(pair.state / "lease", (future, future))
    if case == "stopped":
        (pair.state / "stopped").write_text("boot 1\n")
    pair.set(**{"health_code": "503", **files})
    rc, out = pair.tick()
    assert rc == 0 and said in out and pair.read("fails") == "0\n"


def test_it_stands_down_while_the_lock_is_held(pair):
    pair.state.mkdir(parents=True)
    pair.set(head_state="exited")
    with open(pair.state / "lock", "w") as held:
        holder = subprocess.Popen(["flock", str(pair.state / "lock"), "sleep", "5"], stdin=held)
        time.sleep(0.2)
        try:
            rc, out = pair.tick()
        finally:
            holder.kill()
    assert rc == 0 and "standing down" in out


@pytest.mark.parametrize("files,said,weight", [
    ({"health_code": "503", "health_body": '{"ok": false, "fatal": "RuntimeError()"}'}, "/health 503", 1),
    ({"health_body": '{"ok": false, "fatal": "RuntimeError(\'a round failed\')"}'}, "fatal", 1),
    ({"health_body": '{"ok": false, "fatal": null, "stalled": true, "call_age_s": 400}'}, "stalled", 1),
    ({"probe_rc": "28"}, "idle probe failed", 1),
    ({"worker_state": "exited"}, "worker container exited", 2),
    ({"head_state": "absent", "worker_state": "running"}, "head container absent", 2),
    ({"head_state": "absent", "worker_state": "absent"}, "no stop marker", 1),
])
def test_bad_ticks(pair, files, said, weight):
    pair.set(**files)
    rc, out = pair.tick()
    assert rc == 1 and said in out and pair.read("fails") == f"{weight}\n"


def test_an_unreachable_worker_is_not_counted(pair):
    pair.set(worker_unreachable="1")
    assert pair.tick()[0] == 0 and pair.read("fails") in ("", "0\n")


def test_three_bad_ticks_heal_under_the_lock_then_the_rate_limit_holds(pair):
    pair.set(health_body='{"ok": false, "fatal": null, "stalled": true, "call_age_s": 400}')
    for k in (1, 2):
        assert pair.tick(WATCH_HEAL="1")[0] == 1 and pair.read("fails") == f"{k}\n"
    rc, out = pair.tick(WATCH_HEAL="1")
    assert rc == 1 and "restarting both ranks" in out
    calls = pair.heals()
    assert calls == [f"TF_DSV41_LOCKED=1 -C {pair.dir} --no-print-directory restart"]
    assert "logs of head" in pair.read(next(p.name for p in pair.state.glob("heal-*-head.log")))
    assert "logs of worker" in pair.read(next(p.name for p in pair.state.glob("heal-*-worker.log")))
    for _ in range(3):                                        # a second failure within WATCH_MIN_HEAL only alerts
        rc, out = pair.tick(WATCH_HEAL="1")
    assert "healed" in out and "waiting" in out and len(pair.heals()) == 1


def test_alert_only_by_default(pair, tmp_path):
    alert = tmp_path / "alert.sh"
    alert.write_text(f'#!/usr/bin/env bash\necho "$*" >> {tmp_path}/alerts\n')
    alert.chmod(0o755)
    pair.set(worker_state="exited")
    pair.tick(WATCH_ALERT=str(alert))
    rc, out = pair.tick(WATCH_ALERT=str(alert))                # 2 + 2 >= 3
    assert rc == 1 and "WATCH_HEAL=0, not restarting" in out
    assert "worker container exited" in (tmp_path / "alerts").read_text()
    time.sleep(0.3)
    assert not pair.read("make_calls", pair.stub)


# -- the Makefile's lock ------------------------------------------------------------------------------------------------

def make_n(target: str, tmp_path: Path, **env) -> str:
    done = subprocess.run(["make", "-n", "-s", "-f", str(DEPLOY / "Makefile"), "-p", target], cwd=tmp_path,
                          env={**os.environ, "XDG_STATE_HOME": str(tmp_path), **env}, capture_output=True,
                          text=True, timeout=60)
    return done.stdout


@pytest.mark.skipif(not shutil.which("make"), reason="needs make")
def test_up_down_restart_lock_once_and_never_run_in_parallel(tmp_path):
    db = make_n("help", tmp_path)
    assert ".NOTPARALLEL:" in db
    lock = next(line for line in db.splitlines() if line.startswith("LOCK = "))
    assert "$(if $(TF_DSV41_LOCKED),,flock -n -E 75 $(STATE)/lock env TF_DSV41_LOCKED=1)" in lock
    assert "_restart: _down _up" in db


@pytest.mark.skipif(not shutil.which("make"), reason="needs make")
def test_a_heal_holding_the_lock_restarts_without_waiting_for_itself(tmp_path):
    """The heal holds the lock and runs `make restart` with TF_DSV41_LOCKED=1: it goes ahead (dry run here); a
    plain `make restart` meanwhile is refused at once instead of queueing behind it."""

    state = tmp_path / "tensorfold-dsv41"
    state.mkdir()

    def restart(**env):
        return subprocess.run(["make", "-n", "-C", str(DEPLOY), "restart"], cwd=tmp_path,
                              env={**os.environ, "XDG_STATE_HOME": str(tmp_path), **env}, capture_output=True,
                              text=True, timeout=60)

    with open(state / "lock", "w") as held:
        holder = subprocess.Popen(["flock", str(state / "lock"), "sleep", "20"], stdin=held)
        time.sleep(0.2)
        try:
            refused = restart()
            healed = restart(TF_DSV41_LOCKED="1")
        finally:
            holder.kill()
    assert refused.returncode != 0 and "holds" in refused.stdout and "Error 75" in refused.stderr
    assert healed.returncode == 0, healed.stdout + healed.stderr
    assert f"{state}/stopped" in healed.stdout and f"rm -f {state}/stopped" in healed.stdout     # down, then up
