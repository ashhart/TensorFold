"""The native engine: finding the binary, reading what it says, building its command."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from tensorfold_web import native
from tensorfold_web.engine import Engine, tensorfold_command
from tensorfold_web.settings import Settings

CAPS = {"schema": 1, "engine": "zig", "version": "1.0.2", "chip": "apple-m6", "backends": ["metal"],
        "families": {"nemotron_h": ["mlx-q4g64"]},
        "serve": {"flags": {"--host": {}, "--port": {}, "--context": {}, "--no-thinking": {}, "--thinking": {},
                            "--dashboard": {}, "--keep-warm": {}}},
        "env": ["TENSORFOLD_API_KEY"]}

INFO = """family: Nemotron 3.5 Lightning (nemotron_h)
format: 4-bit, group 64
context: 262144
weights: 17.27 GiB (18544234457 bytes)
memory floor: 19.27 GiB (weights plus a 2.00 GiB runtime margin)
"""

SERVING = ("[tensorfold] serving NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit at http://127.0.0.1:8080/v1 "
           "(sampling: temperature 1.0, top_p 0.95; drafts: on; context: 262144; loaded in 13.9s)")


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    """A stand-in `tensorfold-native` that answers --version, capabilities --json and info, as 1.0.2 does."""

    script = tmp_path / "tensorfold-native"
    script.write_text(f"""#!{sys.executable}
import sys
args = sys.argv[1:]
if args == ["--version"]:
    print("tensorfold-native 1.0.2")
elif args[:2] == ["capabilities", "--json"]:
    print({json.dumps(json.dumps(CAPS))})
elif args[:1] == ["info"] and args[1].startswith("TensorFold/"):
    print({INFO!r}, end="")
else:
    print(args[-1] + " is not in the Hugging Face cache", file=sys.stderr)
    sys.exit(1)
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("TENSORFOLD_BIN", str(script))
    native.forget()
    yield str(script)
    native.forget()


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_finds_and_reads_the_binary(fake_binary):
    assert native.find_binary() == fake_binary
    assert native.version_of(fake_binary) == "1.0.2"
    assert native.capabilities(fake_binary)["chip"] == "apple-m6"
    got = native.info(fake_binary, "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    assert got == {"family": "Nemotron 3.5 Lightning (nemotron_h)", "format": "4-bit, group 64", "context": 262144,
                   "weights_bytes": 18544234457, "floor_bytes": int(19.27 * 1024**3)}
    assert native.info(fake_binary, "nobody/nothing") is None


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_python_line_command_is_not_the_native_binary(fake_binary, monkeypatch, tmp_path):
    """Homebrew names the native binary `tensorfold`: on PATH, it must not be taken for the Python line's command."""

    (tmp_path / "tensorfold").symlink_to(fake_binary)
    monkeypatch.setenv("PATH", str(tmp_path))
    cmd = tensorfold_command()
    assert cmd is None or cmd[0] != str(tmp_path / "tensorfold")


def test_a_binary_that_is_not_native_is_not_found(tmp_path, monkeypatch):
    other = tmp_path / "tensorfold"
    other.write_text("#!/bin/sh\necho 'tensorfold 0.6.6'\n")
    other.chmod(0o755)
    monkeypatch.setenv("TENSORFOLD_BIN", str(other))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(native, "CANDIDATES", ())
    native.forget()
    try:
        assert native.find_binary() is None
    finally:
        native.forget()


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_command_keeps_only_the_flags_the_binary_lists(fake_binary):
    s = Settings(context=32768, thinking="off", budget_fraction=0.8, api_key="secret-key",
                 extra_args=["--keep-warm", "600"])
    cmd, env, notes = native.build_command(fake_binary, "TensorFold/Nemotron", s)
    assert cmd[:3] == [fake_binary, "serve", "TensorFold/Nemotron"]
    assert cmd[cmd.index("--context") + 1] == "32768"
    assert "--no-thinking" in cmd and "--dashboard" in cmd and cmd[-2:] == ["--keep-warm", "600"]
    assert "secret-key" not in " ".join(cmd) and env["TENSORFOLD_API_KEY"] == "secret-key"
    assert any("memory budget" in n for n in notes)            # the native engine sizes itself


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_a_flag_the_binary_lacks_is_left_out_and_said(fake_binary):
    native.capabilities(fake_binary)["serve"]["flags"].pop("--context")
    cmd, _, notes = native.build_command(fake_binary, "m", Settings(context=8192))
    assert "--context" not in cmd and any("--context" in n for n in notes)


def test_startup_line_gives_window_and_load_time():
    assert native.parse_startup(SERVING) == {"context_tokens": 262144.0, "loaded_s": 13.9, "drafts": 1.0}
    assert native.parse_startup("[tensorfold] open-file limit raised") == {}


def test_engine_reads_the_native_startup_line():
    eng = Engine(Settings())
    eng._emit(SERVING)
    assert eng.startup["context_tokens"] == 262144.0 and eng.startup["loaded_s"] == 13.9


def test_a_native_model_without_any_engine_is_refused(monkeypatch):
    monkeypatch.setattr(native, "find_binary", lambda: None)
    monkeypatch.setattr("tensorfold_web.engine.tensorfold_command", lambda: None)
    eng = Engine(Settings(engine_url="http://127.0.0.1:9"))
    with pytest.raises(RuntimeError, match="native engine"):
        eng.start("TensorFold/Nemotron", native_model=True)


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_catalog_shows_what_the_engine_reads_of_a_native_checkpoint(fake_binary):
    from tensorfold_web.catalog import Catalog

    cat = Catalog()
    nemotron = next(e for e in cat.entries if e.native)
    assert cat.engine_facts(nemotron)["floor_bytes"] == int(19.27 * 1024**3)
    other = next(e for e in cat.entries if not e.native)
    assert cat.engine_facts(other) is None                   # the Python line's models are not asked


# -- the serve flags as data (10 Oct 2026, TensorFold 1.0.2's usage line) -------------------------------------------
USAGE = ("usage: tensorfold serve [-h] [--host HOST] [--port PORT] [--api-key API_KEY] [--dashboard] [--context CONTEXT] "
         "[--speed-up SETTINGS] [--temperature TEMPERATURE] [--thinking | --no-thinking] "
         "[--reasoning-effort {low,medium,high,xhigh}] [--no-drafts] [--parallel PARALLEL] model")


def test_the_usage_line_gives_each_flags_arity():
    from tensorfold_web.native import parse_usage

    u = parse_usage(USAGE)
    assert u["--port"] == {"value": True, "choices": None} and u["--no-drafts"] == {"value": False, "choices": None}
    assert u["--thinking"]["value"] is False and u["--no-thinking"]["value"] is False
    assert u["--reasoning-effort"]["choices"] == ["low", "medium", "high", "xhigh"]


def test_extra_flags_are_checked_against_what_the_engine_offers():
    from tensorfold_web.native import check_extra, parse_usage

    flags = {n: {**v, "managed": None} for n, v in parse_usage(USAGE).items()}
    flags["--keep-warm"] = {"value": None, "choices": None, "managed": None}        # offered, not in the usage line
    assert check_extra(["--no-drafts", "--parallel", "4", "--reasoning-effort=high", "--keep-warm"], flags) == \
        ["--no-drafts", "--parallel", "4", "--reasoning-effort", "high", "--keep-warm"]
    for args, said in ((["--paral", "4"], "short for --parallel"), (["--turbo"], "not a flag this engine offers"),
                       (["--api-key", "x"], "secret"), (["--port", "9"], "set by TensorFold Web"),
                       (["--parallel"], "needs a value"), (["--no-drafts=1"], "takes no value"),
                       (["--reasoning-effort", "max"], "one of low"), (["4"], "not a flag")):
        with pytest.raises(ValueError, match=said):
            check_extra(args, flags)



def test_the_cuda_engines_sizing_line_is_read():
    # TensorFold 1.0.4 on the GX10's GB10, 10 Oct 2026
    line = ("[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 17.99 GiB; 8 streams "
            "at once, 0.18 GiB each at a 16384-token window, of 83.8 GiB left after a 12.2 GiB reserve; prompts in "
            "2048-row chunks; own glue kernels")
    assert native.parse_startup(line) == {"resident_gib": 17.99, "streams": 8.0, "stream_gib": 0.18, "left_gib": 83.8,
                                          "reserve_gib": 12.2}


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_an_upgrade_behind_the_same_link_is_read_afresh(fake_binary, tmp_path):
    link = tmp_path / "bin-link"
    link.symlink_to(fake_binary)
    assert native.version_of(str(link)) == "1.0.2" and native.capabilities(str(link))["version"] == "1.0.2"
    newer = tmp_path / "tensorfold-native-105"
    newer.write_text(Path(fake_binary).read_text().replace("1.0.2", "1.0.5"))
    newer.chmod(newer.stat().st_mode | stat.S_IEXEC)
    link.unlink()
    link.symlink_to(newer)                       # as a new archive linked in place, or `brew upgrade`
    assert native.version_of(str(link)) == "1.0.5" and native.capabilities(str(link))["version"] == "1.0.5"


def test_the_prompt_cache_is_read_from_both_engines_startup_lines():
    metal = ("info: prompt cache: 1.8 GiB from 1.8 GiB free under the 70% cap (18.6 GiB in use once loaded, 32 GiB "
             "of RAM, 2 GiB kept for prompt buffers)")
    assert native.parse_startup(metal) == {"prompt_cache_gib": 1.8, "prompt_cache_free_gib": 1.8, "in_use_gib": 18.6}
    cuda = ("[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 17.99 GiB; 8 streams "
            "at once, 0.18 GiB each at a 16384-token window, of 85.4 GiB left after a 12.2 GiB reserve; prompts in "
            "2048-row chunks, a 16.0 GiB prompt cache; own glue kernels")
    got = native.parse_startup(cuda)
    assert got["streams"] == 8 and got["reserve_gib"] == 12.2 and got["prompt_cache_gib"] == 16.0
    # a request's own cache line is not the startup budget
    assert native.parse_startup("info: prompt cache: 8701 tokens, resumed at 8554, kept 0; 1 states") == {}


def test_what_the_engine_serves_comes_from_its_capabilities(fake_binary, monkeypatch):
    from tensorfold_web.catalog import Catalog

    cat = Catalog()
    qwen = next(e for e in cat.entries if e.id == "TensorFold/Qwen3.8-27B-MLX-4bit")
    flash = next(e for e in cat.entries if e.family == "qwen4_exp")
    assert not cat.native_here(qwen)                          # this engine lists Nemotron only
    caps = {**CAPS, "families": {"nemotron_h": ["mlx-q4g64"], "qwen3_5": ["mlx-q4g64"], "qwen4_exp": ["mlx-q6g32"]}}
    monkeypatch.setattr(native, "capabilities", lambda _b: caps)
    assert cat.native_here(qwen)                              # 1.0.3 and later on a Mac
    assert not cat.native_here(flash)                         # the family, but not this checkpoint's format
    monkeypatch.setenv("TENSORFOLD_BIN", "/nonexistent")
    native.forget()
    assert cat.native_here(qwen) == qwen.native               # no engine to ask: the catalog's flag


def test_a_native_drafter_is_named_on_the_command(fake_binary):
    caps = {**CAPS, "serve": {"flags": {**CAPS["serve"]["flags"], "--drafter": {}}}}
    native._caps.clear()
    native._caps[native._stamp(fake_binary)] = caps
    cmd, _env, notes = native.build_command(fake_binary, "TensorFold/Qwen3.8-27B-MLX-4bit", Settings(),
                                            drafter="z-lab/Qwen3.8-27B-DFlash2")
    assert cmd[cmd.index("--drafter") + 1] == "z-lab/Qwen3.8-27B-DFlash2" and not notes


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_a_model_the_engine_refuses_on_this_chip_goes_to_the_python_line(fake_binary, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    caps = {**CAPS, "families": {"nemotron_h": ["mlx-q4g64"], "qwen3_5": ["mlx-q4g64"]}}
    monkeypatch.setattr(native, "capabilities", lambda _b: caps)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False)
    eng, cat = Engine(s), Catalog()
    cat.refresh_sizes_async = lambda: None
    starts = []

    def start(model, *, native_model=False, drafter=None):
        starts.append((native_model, drafter))
        if native_model:                                   # 1.0.6 on an M6: refused before any weight is read
            eng.log.append("10:08:18 tensorfold: the native Qwen3.8-27B engine cannot load /x/snapshots/abc "
                           "(QwenMetalQualificationRequired)")
            eng.exit_code = 1
        return {"status": "starting"}
    monkeypatch.setattr(eng, "start", start)
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    qwen = "TensorFold/Qwen3.8-27B-MLX-4bit"
    assert c.post("/api/engine/start", json={"model": qwen}).status_code == 200
    assert starts == [(True, "z-lab/Qwen3.8-27B-DFlash2"), (False, None)]
    assert native.refused(fake_binary, qwen) == "QwenMetalQualificationRequired"
    assert not cat.native_here(next(e for e in cat.entries if e.id == qwen))    # the next start goes straight there


def test_a_replaced_engines_log_reader_never_marks_the_new_one(tmp_path):
    import subprocess

    from tensorfold_web.engine import _Child

    eng = Engine(Settings())
    log = tmp_path / "engine.log"
    log.write_text("tensorfold: the native Qwen3.8-27B engine cannot load /x (QwenMetalQualificationRequired)\n")
    old = _Child(subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"]))
    old.popen.wait()
    eng.proc = _Child(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"]))   # started in its place
    try:
        eng.exit_code = None
        eng._pump(old, log, False)                    # the old reader ends late
        assert eng.exit_code is None and not any("cannot load" in line for line in eng.log)
    finally:
        eng.proc.popen.kill()


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_a_refusal_said_just_as_the_engine_exits_still_moves_it(fake_binary, tmp_path, monkeypatch):
    import threading
    import time as _time

    from fastapi.testclient import TestClient

    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    caps = {**CAPS, "families": {"nemotron_h": ["mlx-q4g64"], "qwen3_5": ["mlx-q4g64"]}}
    monkeypatch.setattr(native, "capabilities", lambda _b: caps)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False)
    eng, cat = Engine(s), Catalog()
    cat.refresh_sizes_async = lambda: None
    starts = []

    def start(model, *, native_model=False, drafter=None):
        starts.append(native_model)
        if native_model:
            def later():                                   # the exit is seen before the last line is read
                eng.exit_code = 1
                _time.sleep(0.3)
                eng.log.append("tensorfold: the native Qwen3.8-27B engine cannot load /x (QwenMetalQualificationRequired)")
            eng._reader = threading.Thread(target=later)
            eng._reader.start()
        return {"status": "starting"}
    monkeypatch.setattr(eng, "start", start)
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    assert c.post("/api/engine/start", json={"model": "TensorFold/Qwen3.8-27B-MLX-4bit"}).status_code == 200
    assert starts == [True, False]


def test_a_replacement_during_the_old_readers_last_step_is_never_marked(tmp_path):
    import subprocess
    import threading

    from tensorfold_web.engine import _Child

    eng = Engine(Settings())
    log = tmp_path / "engine.log"
    log.write_text("last words\n")
    old = _Child(subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"]))
    old.popen.wait()
    eng.proc = old
    new = _Child(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"]))
    try:
        with eng._lock:                                   # a start replacing the engine holds the lock...
            reader = threading.Thread(target=eng._pump, args=(old, log, False))
            reader.start()                                # ...while the old reader reaches its last step
            reader.join(0.5)
            eng.proc, eng.exit_code = new, None
            eng.log.clear()
        reader.join(5)
        assert eng.exit_code is None and not any("last words" in x or "exited" in x for x in eng.log)
    finally:
        new.popen.kill()


def test_managed_and_secret_flags_never_reach_a_command_line():
    from tensorfold_web import native

    kept, dropped = native.safe_extra(["--host", "0.0.0.0", "--keep-warm", "5", "--api-key=x", "--port", "1", "--loop-guard"])
    assert kept == ["--keep-warm", "5", "--loop-guard"] and dropped == ["--host", "--api-key", "--port"]


def _release(folder, version):
    """A stand-in native binary that only answers --version."""

    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "tensorfold"
    exe.write_text(f"#!/bin/sh\necho 'tensorfold-native {version}'\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.skipif(os.name == "nt", reason="the stand-in binary is a script with a shebang")
def test_the_newest_of_two_installs_is_found(tmp_path, monkeypatch):
    """The installer's ~/.tensorfold beside an older Homebrew one: the newer serves, whichever is on PATH first;
    TENSORFOLD_BIN still picks one by hand."""

    brew = _release(tmp_path / "brew", "1.0.9")
    installer = _release(tmp_path / "home" / ".tensorfold" / "bin", "1.0.11")
    monkeypatch.delenv("TENSORFOLD_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "brew"))
    monkeypatch.setattr(native, "CANDIDATES", (installer, brew))
    native.forget()
    try:
        assert native.find_binary() == installer
        assert native.version_tuple("1.0.9") < native.version_tuple("1.0.11")
        native.forget()
        monkeypatch.setenv("TENSORFOLD_BIN", brew)
        assert native.find_binary() == brew
    finally:
        native.forget()
