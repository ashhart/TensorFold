import json
import socket
from collections import namedtuple

from fastapi.testclient import TestClient

from tensorfold_web import diagnostic
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app

Usage = namedtuple("Usage", "total used free")


def test_backend_and_install_say_how_to_fix():
    mac = {"backend": "mlx", "chip": "Apple M6", "ram_bytes": 32 * diagnostic.GB, "gpu_ceiling_bytes": 25 * diagnostic.GB,
           "os": "macOS", "arch": "arm64", "tensorfold_version": None}
    assert diagnostic.backend_check(mac).status == "pass" and "25 GB of 32" in diagnostic.backend_check(mac).detail
    missing = diagnostic.engine_installed_check(mac, remote=False)
    assert missing.status == "fail" and missing.fix == "curl -fsSL https://tensorfold.dev/install.sh | sh"
    spark = {**mac, "os": "Linux", "arch": "aarch64", "tensorfold_version": "0.6.6 (Python)"}
    old = diagnostic.engine_installed_check(spark, remote=False)
    assert old.status == "warn" and "tensorfold.dev/install.sh" in old.fix
    assert diagnostic.engine_installed_check(spark, remote=True).status == "skip"
    none = diagnostic.backend_check({"backend": "none", "backend_note": "RTX 3060 is compute capability 8.6"})
    assert none.status in ("fail", "warn") and none.fix


def test_port_held_by_another_program():
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        c = diagnostic.port_check(f"http://127.0.0.1:{port}", "stopped", remote=False)
        assert c.status == "fail" and f"http://127.0.0.1:{port + 1}" in c.fix
        assert diagnostic.port_check(f"http://127.0.0.1:{port}", "running", remote=False).status == "pass"
    finally:
        holder.close()
    assert diagnostic.port_check(f"http://127.0.0.1:{port}", "stopped", remote=False).status == "pass"


def test_model_folder_by_free_space(tmp_path, monkeypatch):
    for free, status in ((2, "fail"), (20, "warn"), (300, "pass")):
        monkeypatch.setattr(diagnostic.shutil, "disk_usage", lambda _p, f=free: Usage(0, 0, f * diagnostic.GB))
        c = diagnostic.folder_check(tmp_path / "hub", remote=False)
        assert c.status == status, (free, c)
        assert ("HF_HUB_CACHE" in c.fix) == (status != "pass")


def test_service_states():
    assert diagnostic.service_check(None).status == "skip"
    assert diagnostic.service_check({"platform": "macos", "installed": True, "running": True}).status == "pass"
    linger_off = diagnostic.service_check({"platform": "linux", "installed": True, "running": True, "lingering": False})
    assert linger_off.status == "warn" and "enable-linger" in linger_off.fix
    assert diagnostic.service_check({"platform": "linux", "installed": False, "running": False,
                                     "lingering": True}).fix == "tensorfold-web service install"


def test_redact():
    text = 'Authorization: Bearer sk-abc123 and token=xyz789 {"api_key": "k3y-value", "api_key_set": true, ' \
           '"max_tokens": 400} the-secret-token'
    out = diagnostic.redact(text, ["the-secret-token"])
    for secret in ("sk-abc123", "xyz789", "k3y-value", "the-secret-token"):
        assert secret not in out
    assert '"api_key_set": true' in out and '"max_tokens": 400' in out


def test_checks_and_report_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    server = mock_serve(port=0)
    try:
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{server.server_address[1]}", remote=True,
                                  api_key="engine-key-123456")
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="web-token-987654"),
                       headers={"x-studio-token": "web-token-987654"})
        checks = c.get("/api/diagnostic").json()["checks"]
        assert [x["id"] for x in checks] == ["backend", "installed", "port", "folder", "model", "engine", "memory",
                                             "access", "service", "cortheon"]
        assert {x["id"]: x["status"] for x in checks}["installed"] == "skip"       # remote engine
        r = c.get("/api/diagnostic/report")
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        assert "engine-key-123456" not in r.text and "web-token-987654" not in r.text
        doc = json.loads(r.text)
        assert doc["checks"] and doc["machine"] and "api_key" not in doc["settings"]
    finally:
        server.shutdown()


def test_urls_with_credentials_are_refused_and_redacted():
    import pytest as _pt

    from tensorfold_web import fleet
    from tensorfold_web import settings as settings_mod

    with _pt.raises(ValueError):
        settings_mod.validate({"engine_url": "http://alice:password123@spark:8080"})
    with _pt.raises(ValueError):
        fleet.normalize_url("http://alice:password123@spark:8090")
    out = diagnostic.redact('{"engine_url": "http://alice:password123@spark:8080"}', [])
    assert "password123" not in out and "alice" not in out and "@spark:8080" in out


def test_a_native_engine_without_its_grammar_library_is_said(tmp_path, monkeypatch):
    from tensorfold_web import native

    opt = tmp_path / "opt" / "tensorfold-1.0.6"
    (opt / "bin").mkdir(parents=True)
    binary = opt / "bin" / "tensorfold-native"
    binary.write_text("")
    link = tmp_path / "tensorfold-native"
    link.symlink_to(binary)                               # linked from ~/.local/bin, as on the Sparks
    monkeypatch.setattr(native, "find_binary", lambda: str(link))
    monkeypatch.setattr(native, "version_of", lambda _b: "1.0.6")
    m = {"tensorfold_version": "1.0.6", "os": "Linux", "arch": "aarch64"}
    c = diagnostic.engine_installed_check(m, remote=False)
    assert c.status == "warn" and "libtfgrammar" in c.detail
    (opt / "lib").mkdir()
    (opt / "lib" / "libtfgrammar.so").write_text("")
    assert diagnostic.engine_installed_check(m, remote=False).status == "pass"
    monkeypatch.setattr(native, "version_of", lambda _b: "1.0.5")       # before it shipped: nothing to look for
    (opt / "lib" / "libtfgrammar.so").unlink()
    assert diagnostic.engine_installed_check(m, remote=False).status == "pass"


def test_the_diagnostic_answers_on_a_machine_without_systemctl(tmp_path, monkeypatch):
    from tensorfold_web import service
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(service, "status", lambda _p: (_ for _ in ()).throw(FileNotFoundError("systemctl")))
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    assert c.get("/api/diagnostic").status_code == 200


def test_sizes_are_refreshed_by_a_post_not_by_a_get_a_page_elsewhere_can_make(tmp_path, monkeypatch):
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    calls = []
    monkeypatch.setattr(cat, "refresh_sizes", lambda force=False: calls.append(force))
    c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    assert c.get("/api/catalog?refresh=true").status_code == 200 and calls == []
    assert c.post("/api/catalog/refresh").status_code == 200 and calls == [True]


def test_web_research_on_an_engine_before_its_security_release_is_flagged():
    check = diagnostic.cortheon_check
    assert check({"cortheon": False}, "1.0.9", remote=False).status == "skip"
    assert check({"cortheon": True}, "1.0.9", remote=True).status == "skip"     # another machine's release decides
    old = check({"cortheon": True}, "1.0.9", remote=False)
    assert old.status == "warn" and "1.0.11" in old.detail and "install.sh" in old.fix
    assert check({"cortheon": True}, "1.0.11", remote=False).status == "pass"
    assert check({"cortheon": True}, "1.0.12", remote=False).status == "pass"   # compared as numbers, not text
