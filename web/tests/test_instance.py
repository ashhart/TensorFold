import subprocess
import sys

from tensorfold_web import instance


def test_a_second_studio_does_not_get_the_lock(tmp_path):
    path = tmp_path / "web.lock"
    first = instance.hold(path=path)
    assert first is not None and instance.holder(path) != ""
    code = "import sys; from pathlib import Path; from tensorfold_web import instance; " \
           f"sys.exit(0 if instance.hold(wait=0.5, path=Path({str(path)!r})) is None else 3)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
    first.close()
    code = code.replace("is None else 3", "is not None else 3")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_engine_url_is_refused_while_an_adopted_engine_of_ours_runs(tmp_path, monkeypatch, capsys):
    import types

    from tensorfold_web import cli
    from tensorfold_web import settings as settings_mod

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)

    def adopted(self):                                  # a surviving local engine of ours
        self.proc, self.model = types.SimpleNamespace(alive=lambda: True), "org/model"
        return True
    monkeypatch.setattr(cli.Engine, "adopt", adopted)
    monkeypatch.setattr(cli, "create_app", lambda **k: (_ for _ in ()).throw(AssertionError("app built")))
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("served")))
    assert cli.main(["--engine-url", "http://spark.local:8080", "--no-browser"]) == 1
    assert "stop it" in capsys.readouterr().err


def test_a_held_lock_exits_with_its_own_code_and_cleanly_under_launchd(tmp_path, monkeypatch, capsys):
    from tensorfold_web import cli, instance

    monkeypatch.setattr(instance, "hold", lambda *a, **k: None)
    monkeypatch.setattr(instance, "holder", lambda *a, **k: "123")
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    assert cli.main(["--no-browser"]) == 4 and "another TensorFold Web is running" in capsys.readouterr().err
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.tensorfold.web")
    assert cli.main(["--no-browser"]) == 0
