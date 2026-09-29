"""CUDA server model ids: ``--alias`` is listed and answered to, as on the MLX server (any machine)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold import cli
from tensorfold.cuda import server
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, app_for, events, request


def models(port):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", "/v1/models")
        return [entry["id"] for entry in json.loads(connection.getresponse().read())["data"]]
    finally:
        connection.close()


def test_models_lists_the_name_then_each_alias_once(tmp_path):
    app = app_for(tmp_path)
    app.aliases = ("legacy-name", "fake-cuda", "legacy-name")
    with http_server(app) as port:
        assert models(port) == ["fake-cuda", "legacy-name"]


def test_without_aliases_models_lists_the_name(tmp_path):
    app = app_for(tmp_path)                     # built without __init__, as older callers do: no aliases attribute
    with http_server(app) as port:
        assert models(port) == ["fake-cuda"]


@pytest.mark.parametrize("asked, named", [("legacy-name", "legacy-name"), ("fake-cuda", "fake-cuda"),
                                          ("someone-else", "fake-cuda"), (None, "fake-cuda")])
def test_a_reply_names_the_id_it_was_asked_for_when_it_answers_to_it(tmp_path, asked, named):
    app = app_for(tmp_path)
    app.aliases = ("legacy-name",)
    body = {"messages": HI, "max_tokens": 8, **({"model": asked} if asked else {})}
    with http_server(app) as port:
        status, _, text = request(port, body)
        assert status == 200 and json.loads(text)["model"] == named
        status, _, text = request(port, {**body, "stream": True})
        assert status == 200
        assert {event["model"] for event in events(text) if isinstance(event, dict)} == {named}
        status, _, text = request(port, {"prompt": "Hi", "max_tokens": 8, **({"model": asked} if asked else {})},
                                  chat=False)
        assert status == 200 and json.loads(text)["model"] == named


def test_serve_hands_the_aliases_to_the_cuda_app(tmp_path, monkeypatch):
    made = []
    family = SimpleNamespace(title="Test family", model_type="test",
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: SimpleNamespace(max_len=8192)))
    monkeypatch.setattr(server, "App", lambda *a, **k: made.append(k) or SimpleNamespace(effective_context_window=8192))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts",
                                          "--name", "qwen3.6", "--alias", "qwen3.8-27b-fp4", "--alias", "chat"])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert made[0]["aliases"] == ["qwen3.8-27b-fp4", "chat"]
