"""``--name-priority ID=background``: a request naming that id with no priority of its own is treated as background;
the request's own ``priority`` field always wins. CPU only (a stand-in concurrent engine, no GPU work)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold import cli
from tensorfold.serve_options import check as _check_serve_options
from tensorfold.serve_options import name_priority
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, app_for


class BgEngine:
    """A concurrent stand-in: it records the ``background`` flag ``App.run`` computed for each request."""

    eos = (0,)
    concurrent = True

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, background=False):
        self.calls.append(background)
        on_tokens([self.eos[0]])
        return {}


def _app(tmp_path):
    app = app_for(tmp_path)
    app.engine = BgEngine()
    app.aliases = ("qwen3.6-bg",)
    app.background_ids = frozenset({"qwen3.6-bg"})
    return app


def _post(port, body):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        return connection.getresponse().status
    finally:
        connection.close()


@pytest.mark.parametrize("model, priority, expected", [
    ("qwen3.6-bg", None, True),                 # named id, no priority of its own: the default applies
    ("fake-cuda", None, False),                 # a different id: no default
    ("qwen3.6-bg", "background", True),         # explicit background, redundant with the default
    ("qwen3.6-bg", "not-background", False),    # the request's own (non-background) priority wins over the default
    ("fake-cuda", "background", True),          # explicit background reaches a plain id too
])
def test_the_named_ids_default_priority_applies_unless_the_request_sets_its_own(tmp_path, model, priority, expected):
    app = _app(tmp_path)
    body = {"model": model, "messages": HI, "max_tokens": 4, **({"priority": priority} if priority else {})}
    with http_server(app) as port:
        assert _post(port, body) == 200
    assert app.engine.calls == [expected]


def test_name_priority_parses_id_equals_background():
    args = SimpleNamespace(name_priority=["qwen3.6-bg=background"])
    assert name_priority(args) == {"qwen3.6-bg": "background"}
    assert name_priority(SimpleNamespace(name_priority=[])) == {}
    assert name_priority(SimpleNamespace()) == {}


@pytest.mark.parametrize("entry", ["qwen3.6-bg", "qwen3.6-bg=foreground", "=background", ""])
def test_name_priority_refuses_anything_but_id_equals_background(entry):
    with pytest.raises(ValueError, match="ID=background"):
        name_priority(SimpleNamespace(name_priority=[entry]))


def test_name_priority_is_case_and_whitespace_insensitive_on_the_word():
    assert name_priority(SimpleNamespace(name_priority=["qwen3.6-bg=Background "])) == {"qwen3.6-bg": "background"}


def test_serve_options_refuses_an_id_that_is_not_the_name_or_an_alias(tmp_path):
    family = SimpleNamespace(title="Test family", package=SimpleNamespace(cuda_engine=lambda *a, **k: None),
                             model_type="test")
    args = SimpleNamespace(name="qwen3.6", alias=["qwen3.6-bg"], model=str(tmp_path),
                           name_priority=["someone-else=background"])
    with pytest.raises(ValueError, match="not --name or an --alias"):
        _check_serve_options(args, family, "cuda")


def test_serve_options_refuses_name_priority_on_mlx(tmp_path):
    family = SimpleNamespace(title="Test family", package=SimpleNamespace(cuda_engine=lambda *a, **k: None),
                             model_type="test")
    args = SimpleNamespace(name="qwen3.6", alias=["qwen3.6-bg"], model=str(tmp_path),
                           name_priority=["qwen3.6-bg=background"])
    with pytest.raises(ValueError, match="CUDA server option"):
        _check_serve_options(args, family, "mlx")


def test_serve_options_accepts_a_valid_id(tmp_path):
    family = SimpleNamespace(title="Test family", package=SimpleNamespace(cuda_engine=lambda *a, **k: None),
                             model_type="test")
    args = SimpleNamespace(name="qwen3.6", alias=["qwen3.6-bg"], model=str(tmp_path),
                           name_priority=["qwen3.6-bg=background"])
    assert _check_serve_options(args, family, "cuda") is None


def test_serve_options_takes_the_served_name_of_a_relative_model_path(tmp_path, monkeypatch):
    family = SimpleNamespace(title="Test family", package=SimpleNamespace(cuda_engine=lambda *a, **k: None),
                             model_type="test")
    model = tmp_path / "my-model"
    model.mkdir()
    monkeypatch.chdir(model)
    args = SimpleNamespace(name="", alias=[], model=".", name_priority=["my-model=background"])
    assert _check_serve_options(args, family, "cuda") is None


def test_serve_hands_the_background_ids_to_the_cuda_app(tmp_path, monkeypatch):
    from tensorfold.cuda import server

    made = []
    family = SimpleNamespace(title="Test family", model_type="test",
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: SimpleNamespace(max_len=8192)))
    monkeypatch.setattr(server, "App", lambda *a, **k: made.append(k) or SimpleNamespace(effective_context_window=8192))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts",
                                          "--name", "qwen3.6", "--alias", "qwen3.6-bg",
                                          "--name-priority", "qwen3.6-bg=background"])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert made[0]["background_ids"] == frozenset({"qwen3.6-bg"})
