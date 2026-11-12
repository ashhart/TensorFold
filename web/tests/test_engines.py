"""The engine interface: choosing a launcher, and naming the engine behind an address."""

import httpx
import pytest

from tensorfold_web import engines, native


def _server(routes: dict[str, tuple[int, object]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes.get(request.url.path, (404, {"detail": "not found"}))
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def models(owner: str, *ids: str) -> tuple[int, dict]:
    return 200, {"object": "list", "data": [{"id": i, "object": "model", "owned_by": owner} for i in ids or ("m",)]}


@pytest.mark.parametrize(("routes", "engine", "version"), [
    ({"/v1/models": models("tensorfold", "NVIDIA-Nemotron")}, "TensorFold", None),
    ({"/v1/models": models("vllm"), "/version": (200, {"version": "0.11.2"})}, "vLLM", "0.11.2"),
    ({"/v1/models": models("library"), "/api/version": (200, {"version": "0.13.1"})}, "Ollama", "0.13.1"),
    ({"/v1/models": models("llamacpp")}, "llama.cpp", None),
    ({"/v1/models": models("organization_owner"), "/props": (200, {"build_info": "b7000", "default_generation_settings": {}})},
     "llama.cpp", "b7000"),
    ({"/v1/models": models("sglang")}, "SGLang", None),
    ({"/v1/models": models("someone")}, "OpenAI-compatible", None),
])
def test_identify_names_the_engine(routes, engine, version):
    ep = engines.identify("http://box:8080/v1", client=_server(routes))
    assert (ep.engine, ep.version, ep.url) == (engine, version, "http://box:8080")
    assert ep.models


def test_identify_reports_what_stopped_it():
    assert engines.identify("http://box:8080", client=_server({"/v1/models": (401, {})})).error == \
        "the server wants an API key"
    ep = engines.identify("http://box:8080", client=_server({}))
    assert ep.engine is None and "404" in (ep.error or "")


def test_a_native_model_never_falls_back_to_the_python_line(monkeypatch):
    monkeypatch.setattr(native, "find_binary", lambda: None)
    monkeypatch.setattr("tensorfold_web.engine.tensorfold_command", lambda: ["python", "-m", "tensorfold"])
    with pytest.raises(RuntimeError, match="native engine"):
        engines.choose(native_model=True)
    assert engines.choose(native_model=False).kind == "python"


def test_a_model_not_yet_ported_needs_the_python_line(monkeypatch):
    monkeypatch.setattr(native, "find_binary", lambda: "/opt/homebrew/bin/tensorfold")
    monkeypatch.setattr("tensorfold_web.engine.tensorfold_command", lambda: None)
    assert engines.choose(native_model=True).kind == "native"
    with pytest.raises(RuntimeError, match="Python line"):
        engines.choose(native_model=False)
