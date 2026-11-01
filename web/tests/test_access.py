"""Getting into a TensorFold Web that has a token: the token shown on its machine, and one-time links."""

import time

from fastapi.testclient import TestClient

from tensorfold_web import access
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.server import create_app


def _app(tmp_path, monkeypatch, token):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token=token))


def test_a_code_is_good_once_and_for_a_while(monkeypatch):
    links = access.Links()
    code = links.make()["code"]
    assert links.spend(code) and not links.spend(code)
    late = links.make()["code"]
    later = time.time() + access.LINK_SECONDS + 1
    monkeypatch.setattr(access.time, "time", lambda: later)
    assert not links.spend(late)
    assert not links.spend("made-up")


def test_links_are_made_for_someone_already_in_and_traded_for_the_token(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch, "the-token")
    assert c.post("/api/access/link").status_code == 401                 # not without the token
    code = c.post("/api/access/link", headers={"x-studio-token": "the-token"}).json()["code"]
    assert c.post("/api/access/redeem", json={"code": code}).json() == {"token": "the-token"}
    assert c.post("/api/access/redeem", json={"code": code}).status_code == 403
    assert c.post("/api/access/redeem", json={"code": "guess"}).status_code == 403
    assert c.get("/api/settings").status_code == 401                     # the exemption is for that one route only
    open_studio = _app(tmp_path, monkeypatch, None)
    assert open_studio.post("/api/access/link").status_code == 409       # no token, nothing to link


def test_the_token_command_shows_copies_and_renews(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    assert access.main([]) == 1 and "No token file" in capsys.readouterr().out
    access.token_path().write_text("secret-1\n")
    assert access.main([]) == 0 and capsys.readouterr().out.splitlines()[0] == "secret-1"
    copied = []
    monkeypatch.setattr(access, "_copy", lambda text: copied.append(text) or True)
    assert access.main(["--copy"]) == 0 and copied == ["secret-1"] and "secret-1" not in capsys.readouterr().out
    assert access.main(["--new"]) == 0
    new = access.token_path().read_text().strip()
    assert new != "secret-1" and len(new) > 30 and oct(access.token_path().stat().st_mode & 0o777) == "0o600"
