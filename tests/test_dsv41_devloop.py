"""The DeepSeek-V4.1 dev loop on the host (notes/dsv41/DEV.md): suite definitions and their fresh command lines
(tools/dsv41_suite.py through dsv41_serial_run.py's own parser), results / baseline / prove, the memory guard's plan,
the run watcher (tools/dsv41_memwatch.sh against a fake /proc/meminfo and stand-in processes), dsv41_run2.sh's
wiring, tf-dev's rebuilt docker run flags, and the compose deployment's hot source (deploy/dsv41-tp2)."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DEPLOY = ROOT / "deploy" / "dsv41-tp2"


def tool(name: str):
    """tools/<name>.py as a module (the serial runner imports the suite and guard modules beside it the same way)."""

    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    return importlib.import_module(name)


SU = tool("dsv41_suite")
MG = tool("dsv41_memguard")
BASE = ["/models/m", "/models/g", "--rank", "0", "--master", "10.42.0.1"]


# -- suites ---------------------------------------------------------------------------------------------------------
def test_tiers_and_groups():
    quick = SU.resolve("quick")
    assert [t.group for t in quick] == ["q"] * 5 + ["fp8"] * 2
    assert {t.name for t in quick} >= {"multi-test", "step-test", "resume-test", "views-test", "chunk-prefill",
                                       "fp8-multi-test", "fp8-resume-test"}
    full = SU.resolve("full")
    assert [t.name for t in full[:len(quick)]] == [t.name for t in quick]
    assert SU.groups_of(full) == ["q", "fp8", "long"]
    assert {t.name for t in SU.resolve("long")} == {"decode-bench", "needle", "tf-compare", "prefill-bench"}
    assert [t.name for t in SU.resolve("resume-test,multi-test,multi-test")] == ["multi-test", "resume-test"]
    with pytest.raises(ValueError, match="unknown"):
        SU.resolve("quick,nope")
    # the long group holds a 131072-token needle and its reply; quick stays at short context
    assert SU.GROUPS["long"].cap > 131072 + 64 and SU.GROUPS["q"].cap <= 16384
    assert SU.check_env("fp8", {}) and SU.check_env("fp8", {"TF_DSV41_KV": "fp8"}) is None
    assert SU.mode_env(SU.BY_NAME["multi-test"], {"TF_CHUNK_PREFILL": "1", "X": "1"}) == {"X": "1"}
    assert SU.mode_env(SU.BY_NAME["chunk-prefill"], {})["TF_CHUNK_PREFILL"] == "1"


def test_every_suite_test_is_a_fresh_command_of_the_serial_runner():
    """Each test's argument list parses with dsv41_serial_run.py's parser, selects the test's mode, and a fresh run of
    those flags (with run2.sh's neutral ones) is recognised as that test, so both print the same [result] name."""

    R = tool("dsv41_serial_run")
    ap = R.build_parser()
    for t in SU.TESTS:
        args = ap.parse_args(BASE + ["--run-tag", "run1-2", "--mem-cap-gib", "0"] + t.argv())
        assert R.single_mode(args) == t.mode, t.name
        assert t.mode in R.MODES
        env = dict((*SU.GROUPS[t.group].env, *t.env))
        assert R.match_test(ap, args, env) == t.name
        g = SU.GROUPS[t.group]
        assert (args.cap, args.slots, args.graph, args.dspark) == (g.cap, g.slots, g.graph, g.dspark)
    multi = ap.parse_args(BASE + SU.BY_NAME["multi-test"].argv())
    assert R.match_test(ap, multi, {"TF_DSV41_KV": "fp8"}) == "fp8-multi-test"
    assert R.match_test(ap, multi, {"TF_CHUNK_PREFILL": "1"}) is None          # another mode environment
    assert R.match_test(ap, ap.parse_args(BASE + ["--multi-test", "24"]), {}) is None   # another construction
    assert R.single_mode(ap.parse_args(BASE + ["--decoder-test", "1", "--multi-test", "2"])) is None


def test_fresh_commands_and_env(capsys):
    assert SU.main(["fresh", "chunk-prefill"]) == 0
    assert capsys.readouterr().out.strip() == ("chunk-prefill\tTF_CHUNK_PREFILL=1 tools/dsv41_run2.sh --cap 16384 "
                                               "--slots 2 --graph --dspark 3 --chunk-test 6000,1000,3000,4500")
    assert SU.main(["env", "fp8"]) == 0 and capsys.readouterr().out == "TF_DSV41_KV=fp8\n"
    assert SU.main(["groups", "full"]) == 0 and capsys.readouterr().out == "q fp8 long\n"
    assert SU.main(["testenv", "fp8-resume-test"]) == 0 and capsys.readouterr().out == "TF_DSV41_KV=fp8\n"


def test_results_summary_and_fingerprints():
    a = SU.Result("multi-test", "PASS", {"alone": [[1, 2]], "x": 0.5}, "ok", seconds=12.4)
    b = SU.Result("multi-test", "PASS", {"x": 0.5, "alone": [[1, 2]]}, "other words", {"ms": 3.0})
    assert a.fp == b.fp and len(a.fp) == 12
    assert SU.Result("t", "PASS", {"x": 0.5000001}).fp != a.fp
    line = SU.line(a)
    assert line == f"[result] multi-test PASS fp={a.fp} 12s ok"
    log = "\n".join(["noise", line, "[rank 1] " + SU.line(SU.Result("views-test", "FAIL", [1])),
                     SU.line(SU.Result("decode-bench", "INFO", None, "L=1 2 ms")),
                     "[baseline] " + SU.line(SU.Result("decode-bench", "WARN", None, "slower"))])
    rows = SU.parse_lines(log)
    assert [(r["name"], r["status"]) for r in rows] == [("multi-test", "PASS"), ("decode-bench", "WARN")]
    text, ok = SU.summary(rows)
    assert ok and text.startswith("[suite] PASS")
    text, ok = SU.summary([a, SU.Result("step-test", "FAIL"), SU.Result("x", "ERROR")])
    assert not ok and "failed: step-test, x" in text
    assert not SU.summary([])[1]


def test_baseline_judges_tf_compare_on_the_same_documents_and_warns_on_slow_decode():
    def tf(docs):
        return SU.Result("tf-compare", "INFO", {}, "NLL", {"docs": docs})

    base = SU.to_json([tf({"32768/golden": {"nll": 1.0, "top1": 70.0, "sha": "a"},
                           "32768/code0": {"nll": 1.0, "top1": 70.0, "sha": "b"}}),
                       SU.Result("decode-bench", "INFO", {}, "", {"ms": {"32768/1": 10.0}})])
    same = tf({"32768/golden": {"nll": 1.005, "top1": 69.5, "sha": "a"},
               "32768/code0": {"nll": 9.0, "top1": 0.0, "sha": "CHANGED"}})          # another tree's code: skipped
    worse = tf({"32768/golden": {"nll": 1.02, "top1": 70.0, "sha": "a"}})
    slow = SU.Result("decode-bench", "INFO", {}, "", {"ms": {"32768/1": 11.5}})
    fine = SU.Result("decode-bench", "INFO", {}, "", {"ms": {"32768/1": 10.5}})
    for group in ([same, slow], [worse, fine]):
        SU.compare(group, json.loads(json.dumps(base)))
    assert same.status == "PASS" and "1 documents" in same.summary
    assert worse.status == "FAIL" and "32768/golden NLL 1.0000->1.0200" in worse.summary
    assert slow.status == "WARN" and fine.status == "INFO"


def test_prove_compares_fingerprints(tmp_path):
    r = SU.Result("multi-test", "PASS", {"a": 1})
    suite = SU.line(r) + "\n" + SU.line(SU.Result("step-test", "PASS", {"b": 2}))
    lines, ok = SU.prove(suite, ["x\n" + SU.line(r) + "\n[rank 1] whatever"])
    assert ok and lines == [f"multi-test: same fp={r.fp} (PASS)"]
    lines, ok = SU.prove(suite, [SU.line(SU.Result("step-test", "PASS", {"b": 3}))])
    assert not ok and "DIFFERENT" in lines[0]
    assert not SU.prove(suite, [SU.line(SU.Result("needle", "PASS", 1))])[1]
    (tmp_path / "g.log").write_text(suite + "\n" + SU.line(SU.Result("views-test", "FAIL", [0])))
    assert SU.main(["merge", str(tmp_path / "g.log")]) == 1


# -- memory guard ---------------------------------------------------------------------------------------------------
GIB = 1 << 30


def test_memguard_plan():
    cap, why = MG.plan(115 * GIB)
    assert cap == 109 * GIB and "reserve 3 - slack 3" in why
    assert MG.plan(115 * GIB, cap="0")[0] is None
    assert MG.plan(115 * GIB, cap="off")[0] is None
    assert MG.plan(115 * GIB, cap="100.5")[0] == int(100.5 * GIB)
    assert "above what is available" in MG.plan(105 * GIB, cap="104")[1]
    with pytest.raises(MemoryError, match="another job holds"):
        MG.plan(6 * GIB)                                       # a GPU job of 109 GB running beside
    assert MG.plan(6 * GIB, min_start=0, reserve=1, slack=1)[0] == 4 * GIB
    with pytest.raises(MemoryError, match="nothing above"):
        MG.plan(5 * GIB, min_start=0)
    with pytest.raises(ValueError):
        MG.plan(115 * GIB, cap="-2")


def test_memguard_settings_and_meminfo(tmp_path):
    assert MG.setting("cap", None, {}) == "auto"
    assert MG.setting("reserve", None, {"TF_MEM_RESERVE_GIB": "5"}) == "5"
    assert MG.setting("reserve", 2.0, {"TF_MEM_RESERVE_GIB": "5"}) == "2.0"      # the command line wins
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       127535340 kB\nMemAvailable:    6272828 kB\nHugePages_Total:       0\n")
    m = MG.meminfo(str(f))
    assert m["MemAvailable"] == 6272828 * 1024 and m["HugePages_Total"] == 0
    with pytest.raises(MemoryError):
        MG.guard(0, path=str(f), log=lambda *a, **k: None)


# -- the run watcher ------------------------------------------------------------------------------------------------
WATCH = TOOLS / "dsv41_memwatch.sh"
needs_procps = pytest.mark.skipif(not all(shutil.which(t) for t in ("bash", "pgrep", "pkill", "awk")),
                                  reason="needs bash, pgrep, pkill and awk")


def stand_in(tag: str, seconds: int = 30) -> subprocess.Popen:
    """A process whose command line is a dev run's: '... dsv41_serial_run.py ... --run-tag TAG'."""

    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})", "dsv41_serial_run.py", "x",
                             "--run-tag", tag])


def meminfo(path: Path, gib: float) -> None:
    path.write_text(f"MemTotal: 127535340 kB\nMemAvailable: {int(gib * 1048576)} kB\n")


def watch_env(tmp_path, **kw):
    return {**os.environ, "TF_MEMWATCH_MEMINFO": str(tmp_path / "meminfo"), "TF_MEMWATCH_PERIOD": "0.05",
            "TF_MEMWATCH_WAIT": "5", **kw}


@needs_procps
def test_watcher_kills_only_its_run_below_the_floor(tmp_path):
    tag = f"t{os.getpid()}a"
    meminfo(tmp_path / "meminfo", 50)
    mine, other = stand_in(tag), stand_in(tag + "x")             # another run whose tag starts the same
    log = tmp_path / "w.log"
    try:
        w = subprocess.Popen(["bash", str(WATCH), tag, "3", str(log)], env=watch_env(tmp_path),
                             stderr=subprocess.PIPE, text=True)
        time.sleep(1.5)
        assert w.poll() is None and mine.poll() is None
        meminfo(tmp_path / "meminfo", 2.5)
        assert w.wait(timeout=10) == 3
        assert mine.wait(timeout=5) == -9
        assert other.poll() is None                              # not this run: left alone
        text = log.read_text()
        assert f"memwatch {tag}: KILL: MemAvailable 2.50 GiB < 3 GiB" in text and str(mine.pid) in text
    finally:
        for p in (mine, other):
            p.kill()


@needs_procps
def test_watcher_ends_with_the_run_and_without_one(tmp_path):
    tag = f"t{os.getpid()}b"
    meminfo(tmp_path / "meminfo", 50)
    log = tmp_path / "w.log"
    run = stand_in(tag, seconds=2)
    w = subprocess.run(["bash", str(WATCH), tag, "3", str(log)], env=watch_env(tmp_path), capture_output=True,
                       text=True, timeout=20, check=False)
    assert w.returncode == 0 and "the run ended; lowest MemAvailable 50.0 GiB" in log.read_text()
    run.wait()
    t0 = time.time()
    w = subprocess.run(["bash", str(WATCH), tag + "none", "3", str(log)], capture_output=True, text=True, timeout=20, check=False,
                       env=watch_env(tmp_path, TF_MEMWATCH_WAIT="1"))
    assert w.returncode == 0 and time.time() - t0 < 5 and "no process of the run within 1s" in log.read_text()


@needs_procps
def test_stop_kills_the_runs_leftovers_and_the_watcher(tmp_path):
    tag = f"t{os.getpid()}c"
    meminfo(tmp_path / "meminfo", 50)
    log = tmp_path / "w.log"
    run = stand_in(tag)
    try:
        w = subprocess.Popen(["bash", str(WATCH), tag, "3", str(log)], env=watch_env(tmp_path),
                             stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        stop = subprocess.run(["bash", str(WATCH), "--stop", tag, "1", str(log)], capture_output=True, text=True,
                              timeout=20, check=False)
        assert stop.returncode == 0
        assert run.wait(timeout=5) == -9 and w.wait(timeout=5) == 0
        assert "stop: killed the run's processes still there after 1s" in log.read_text()
        text = log.read_text()                                   # TERM, or it saw the run go first
        assert f"memwatch {tag}: stopped" in text or f"memwatch {tag}: the run ended" in text
    finally:
        run.kill()


def test_run2_wires_the_guard_without_touching_the_rank_lines():
    text = (TOOLS / "dsv41_run2.sh").read_text()
    assert subprocess.run(["bash", "-n", str(TOOLS / "dsv41_run2.sh")], check=False).returncode == 0
    assert 'set -- $MEMARGS "$@"' in text and 'MEMARGS="--run-tag $TAG"' in text
    assert "docker exec -d tf-dev bash /tf/tools/dsv41_memwatch.sh $TAG $MW_GIB" in text
    assert "dsv41_memwatch.sh --stop $TAG $GRACE" in text and "trap finish EXIT" in text
    assert 'if [ -z "${TF_DSV41_PREPARED+x}" ]' in text
    ranks = [ln for ln in text.splitlines() if "tools/dsv41_serial_run.py $M $G --rank" in ln]
    assert len(ranks) == 2 and all("$*" in ln and "-e TF_DSV41_PREPARED=${TF_DSV41_PREPARED:-}" in ln for ln in ranks)


# -- tf-dev with one more mount ---------------------------------------------------------------------------------------
TFDEV = {
    "Name": "/tf-dev",
    "Config": {"Image": "nvcr.io/nvidia/pytorch:26.07-py3", "Cmd": ["sleep", "infinity"], "WorkingDir": "/tf",
               "Env": ["PATH=/usr/bin", "NVIDIA_VISIBLE_DEVICES=all", "EXTRA=1"], "Tty": False, "OpenStdin": False,
               "Labels": {"com.nvidia.volumes.needed": "nvidia_driver"}},
    "HostConfig": {"Runtime": "nvidia", "NetworkMode": "host", "IpcMode": "host", "ShmSize": 67108864,
                   "Ulimits": [{"Name": "memlock", "Hard": -1, "Soft": -1}],
                   "Devices": [{"PathOnHost": "/dev/infiniband", "PathInContainer": "/dev/infiniband",
                                "CgroupPermissions": "rwm"}],
                   "CapAdd": ["CAP_IPC_LOCK"], "SecurityOpt": ["label=disable"], "Privileged": False,
                   "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}, "Memory": 0, "MemorySwap": 0,
                   "DeviceRequests": [{"Driver": "", "Count": -1, "DeviceIDs": None, "Capabilities": [["gpu"]],
                                       "Options": {}}],
                   "LogConfig": {"Type": "json-file", "Config": {}}, "Dns": [], "PidsLimit": None},
    "Mounts": [{"Type": "bind", "Source": "/home/u/models", "Destination": "/models", "RW": False},
               {"Type": "bind", "Source": "/home/u/tensorfold", "Destination": "/tf", "RW": True}],
}


def test_tfdev_run_flags_rebuild_the_container_with_the_extra_mount():
    T = tool("dsv41_tfdev_recreate")
    cmds, missing = T.plan(TFDEV, ["PATH=/usr/bin", "NVIDIA_VISIBLE_DEVICES=all"], "S",
                           ["-v", "/home/u/.cache/tensorfold-prepared:/prepared:ro"])
    assert missing == []
    assert cmds[:3] == [["docker", "commit", "tf-dev", "tf-dev-snapshot:S"], ["docker", "rename", "tf-dev",
                        "tf-dev-old-S"], ["docker", "stop", "-t", "10", "tf-dev-old-S"]]
    assert " ".join(cmds[3]) == (
        "docker run -d --name tf-dev --runtime nvidia --gpus all --network host --ipc host --ulimit memlock=-1:-1 "
        "--device /dev/infiniband:/dev/infiniband:rwm --cap-add CAP_IPC_LOCK --security-opt label=disable "
        "-v /home/u/models:/models:ro -v /home/u/tensorfold:/tf -e EXTRA=1 -w /tf "
        "-v /home/u/.cache/tensorfold-prepared:/prepared:ro tf-dev-snapshot:S sleep infinity")


def test_tfdev_refuses_what_it_cannot_reproduce():
    T = tool("dsv41_tfdev_recreate")
    c = json.loads(json.dumps(TFDEV))
    c["HostConfig"]["CpusetCpus"] = "0-3"
    c["Config"]["Labels"]["com.docker.compose.project"] = "x"
    _, missing = T.run_args(c, [], ["-v", "/a:/tf:ro"])
    assert any("CpusetCpus" in m for m in missing) and any("compose" in m for m in missing)
    assert any("/tf is mounted already" in m for m in missing)


# -- the compose deployment's hot source ------------------------------------------------------------------------------
needs_make = pytest.mark.skipif(not shutil.which("make"), reason="needs make")


def deploy_copy(tmp_path, hot: bool) -> Path:
    d = tmp_path / "deploy"
    d.mkdir()
    for f in ("Makefile", "docker-compose.yaml", "docker-compose.hot.yaml", "rank0.env", "rank1.env"):
        shutil.copy(DEPLOY / f, d / f)
    (d / ".env").write_text("CACHE_DIR=/c\nPREPARED_DIR=/p\nPORT=8999\n")
    if hot:
        (d / "build" / "hot").mkdir(parents=True)
        (d / "build" / "hot" / "REVISION").write_text("abc123def456\n")
    return d


def make_db(d: Path, target: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "-pn", "-C", str(d), target], capture_output=True, text=True, timeout=60,
                          check=False, env={**os.environ, "XDG_STATE_HOME": str(d / "state"), **env})


@needs_make
def test_default_compose_commands_are_the_image_ones(tmp_path):
    db = make_db(deploy_copy(tmp_path, hot=False), "help").stdout
    assert "\nCOMPOSE0 := docker compose --env-file .env --env-file rank0.env\n" in db
    assert "\nCOMPOSE1 := docker compose --env-file .env --env-file rank1.env\n" in db
    assert "\nHOT_MOUNT := \n" in db
    d = tmp_path / "deploy"
    prepare = subprocess.run(["make", "-n", "-C", str(d), "prepare"], capture_output=True, text=True, timeout=60, check=False)
    assert prepare.returncode == 0 and "PREPARE_HOT" not in prepare.stdout
    assert "docker compose --env-file .env --env-file rank0.env run --rm --no-deps serve" in prepare.stdout


@needs_make
def test_hot_marker_layers_the_override_everywhere(tmp_path):
    d = deploy_copy(tmp_path, hot=True)
    db = make_db(d, "help").stdout
    assert ("\nCOMPOSE0 := TF_HOT_REV=abc123def456 docker compose -f docker-compose.yaml -f docker-compose.hot.yaml "
            "--env-file .env --env-file rank0.env\n") in db
    pre = make_db(d, "prebuild").stdout
    assert f"-v {d}/build/hot/src:/opt/tensorfold/src:ro tensorfold-dsv41:latest python /prebuild.py" in pre
    refused = subprocess.run(["make", "-C", str(d), "prepare"], capture_output=True, text=True, timeout=60, check=False)
    assert refused.returncode != 0 and "PREPARE_HOT=1" in refused.stdout
    db = make_db(d, "help").stdout                     # the rules as make read them (-n would still recurse)
    rule = db[db.index("\nhot: hot-stage sync\n"):]
    rule = rule[:rule.index("\n\n")]
    assert [ln.split()[-1] for ln in rule.splitlines() if "$(MAKE)" in ln] == ["down", "prebuild", "up"]
    stage = db[db.index("\nhot-stage:\n"):]
    assert "$(SRC)/src/ build/hot/src/" in stage[:stage.index("\n\n")]
    cold = db[db.index("\ncold:\n"):]
    assert "rm -rf build/hot" in cold[:cold.index("\n\n")]


def test_hot_override_mounts_the_source_over_the_editable_install():
    yaml = pytest.importorskip("yaml")
    hot = yaml.safe_load((DEPLOY / "docker-compose.hot.yaml").read_text())["services"]["serve"]
    assert hot["volumes"] == ["./build/hot/src:/opt/tensorfold/src:ro"]
    assert hot["environment"]["TF_REVISION"].startswith("hot-${TF_HOT_REV:?")
    assert "pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/tensorfold" in \
        (DEPLOY / "Dockerfile").read_text()
    assert "COPY tensorfold /opt/tensorfold" in (DEPLOY / "Dockerfile").read_text()


def test_merge_fails_a_tier_test_without_a_result(tmp_path, capsys):
    """A group killed before all its results (memwatch, OOM, NCCL init) must not read as PASS."""

    names = [t.name for t in SU.resolve("quick")]
    log = tmp_path / "q.log"
    log.write_text(SU.line(SU.Result(names[0], "PASS", {"a": 1})) + "\n")
    (tmp_path / "fp8.log").write_text("Traceback (most recent call last):\n")
    assert SU.main(["merge", str(log), str(tmp_path / "fp8.log")]) == 0          # without --spec: what was printed
    assert SU.main(["merge", "--spec", "quick", str(log), str(tmp_path / "fp8.log")]) == 1
    out = capsys.readouterr().out
    assert f"{names[1]} ERROR" in out and out.rstrip().splitlines()[-1].startswith("[suite] FAIL")


def test_prove_fails_a_fresh_run_without_a_result():
    r = SU.Result("multi-test", "PASS", {"a": 1})
    lines, ok = SU.prove(SU.line(r), [SU.line(r), "Traceback (most recent call last):\n"])
    assert not ok and any("no [result] line" in s for s in lines)
