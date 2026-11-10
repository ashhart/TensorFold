import plistlib
from pathlib import Path

import pytest

from tensorfold_web import service
from tensorfold_web import settings as settings_mod


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    conf = tmp_path / "Application Support" / "TensorFold Web"       # a path with a space, as on a Mac
    monkeypatch.setattr(settings_mod, "config_dir", lambda: conf)
    monkeypatch.setattr(service, "lingering", lambda: False)
    calls: list[tuple] = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def run(*cmd):
        calls.append(cmd)
        r = Done()
        r.returncode = 1 if cmd[:2] == ("launchctl", "print") else 0       # not loaded yet
        return r

    monkeypatch.setattr(service, "_run", run)
    monkeypatch.setattr(service, "_stop_hand_started", lambda: False)
    return tmp_path, conf, calls


def test_read_env(tmp_path):
    f = tmp_path / "web.env"
    f.write_text("# models apart\nHF_HUB_CACHE=/srv/s/.cache/tensorfold-hf/hub\n\nA='b c'\nnot a line\n")
    assert service.read_env(f) == {"HF_HUB_CACHE": "/srv/s/.cache/tensorfold-hf/hub", "A": "b c"}
    assert service.read_env(tmp_path / "none") == {}


def test_linux_unit_keeps_the_engine_and_says_lingering(home, monkeypatch):
    tmp, conf, calls = home
    monkeypatch.setattr(service, "platform_name", lambda: "linux")
    p = service.plan()
    assert p.path == tmp / ".config" / "systemd" / "user" / "tensorfold-web.service"
    assert "KillMode=process" in p.content and f"EnvironmentFile=-{conf / 'web.env'}" in p.content
    assert "StartLimitBurst=5" in p.content and "RestartPreventExitStatus=4" in p.content   # no restart loop for ever
    assert f'--token-file "{conf / "web-token"}"' in p.content          # quoted for systemd
    assert "/opt/homebrew" not in p.content
    assert any("lingering is off" in n for n in p.notes)
    assert service.main(["install", "--check"]) == 0 and not p.path.exists() and not calls   # --check does nothing

    done = service.install(p)
    assert p.path.read_text() == p.content
    token = (conf / "web-token")
    assert oct(token.stat().st_mode & 0o777) == "0o600" and len(token.read_text().strip()) > 30
    assert ("systemctl", "--user", "restart", "tensorfold-web.service") in calls
    assert done[0].startswith("wrote")
    calls.clear()
    assert not any(d.startswith("wrote") for d in service.install(p))   # unchanged: not written again
    assert service.uninstall(p)[-1].startswith("removed") and not p.path.exists()


def test_macos_agent(home, monkeypatch):
    tmp, conf, calls = home
    monkeypatch.setattr(service, "platform_name", lambda: "macos")
    conf.mkdir(parents=True)
    (conf / "web.env").write_text("HF_HUB_CACHE=/Volumes/models\n")
    p = service.plan(port=8091)
    doc = plistlib.loads(p.content.encode())
    assert p.path == tmp / "Library" / "LaunchAgents" / "com.tensorfold.web.plist"
    assert doc["RunAtLoad"] and doc["AbandonProcessGroup"] and doc["KeepAlive"] == {"SuccessfulExit": False}
    assert doc["ProgramArguments"][-4:] == ["--token-file", str(conf / "web-token"), "--port", "8091"]
    assert doc["EnvironmentVariables"]["HF_HUB_CACHE"] == "/Volumes/models"
    assert "/opt/homebrew/bin" in doc["EnvironmentVariables"]["PATH"]
    assert str(tmp / ".tensorfold" / "bin") in doc["EnvironmentVariables"]["PATH"]   # the official installer's
    service.install(p)
    assert any(c[:2] == ("launchctl", "bootstrap") for c in calls)


def test_other_platforms_refused(monkeypatch):
    monkeypatch.setattr(service, "platform_name", lambda: "win32")
    with pytest.raises(RuntimeError):
        service.plan()


def test_a_token_kept_elsewhere_leaves_its_folder_alone(home, monkeypatch):
    tmp, conf, _calls = home
    monkeypatch.setattr(service, "platform_name", lambda: "linux")
    shared = tmp / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    service.install(service.plan(token_file=shared / "token"))
    assert shared.stat().st_mode & 0o777 == 0o755
    assert (shared / "token").stat().st_mode & 0o777 == 0o600
    assert conf.is_dir() and conf.stat().st_mode & 0o777 == 0o700


def test_a_service_is_not_recorded_from_uvx_s_temporary_environment(monkeypatch, capsys):
    assert service.temporary_environment("/srv/u/.cache/uv/archive-v0/AbC123")
    assert not service.temporary_environment("/srv/u/.local/share/uv/tools/tensorfold-web")
    monkeypatch.setenv("UV_CACHE_DIR", "/scratch/uvc")
    assert service.temporary_environment("/scratch/uvc/builds-v0/x")
    monkeypatch.setattr(service, "temporary_environment", lambda prefix=None: True)
    monkeypatch.setattr(service, "install", lambda *a, **k: (_ for _ in ()).throw(AssertionError("installed")))
    assert service.main(["install"]) == 2 and "uv tool install" in capsys.readouterr().err
