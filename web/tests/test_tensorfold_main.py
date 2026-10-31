"""TensorFold Web against TensorFold main's server, mocked from its source (d377fcc6a): no GPU, no engine binary."""

import asyncio
from urllib.parse import urlsplit

import httpx

from tensorfold_web.proxy import Proxy


def main_origin_rule(request: httpx.Request) -> bool:
    """zig/src/server/request_guard.zig's browser rule: no Origin passes; an Origin must be http and name the
    request's own Host and port (no --allowed-origin here)."""

    origins = [v for k, v in request.headers.multi_items() if k.lower() == "origin"]
    if not origins:
        return True
    if len(origins) != 1:
        return False
    o = urlsplit(origins[0])
    host = request.headers.get("host", "")
    name, _, port = host.partition(":")
    return o.scheme == "http" and (o.hostname or "") == name.lower() and (o.port or 80) == int(port or 80)


def guarded_engine(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if not main_origin_rule(request):
            msg = "Origin header does not match this server; use --allowed-origin ORIGIN for a trusted origin"
            return httpx.Response(403, json={"error": {"message": msg, "type": "invalid_request_error"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    return httpx.MockTransport(handler)


class _Req:
    def __init__(self, method: str, headers: dict, body: bytes = b""):
        self.method, self.query_params = method, httpx.QueryParams()
        self.headers = httpx.Headers(headers)
        self._body = body

    async def stream(self):
        yield self._body


def test_the_mocked_rule_refuses_the_studio_page_origin():
    # the failure being fixed: the page's own origin, forwarded as it was, is a foreign origin to the engine
    r = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions", headers={"origin": "http://127.0.0.1:8090"})
    assert not main_origin_rule(r)
    assert main_origin_rule(httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions"))


def test_chat_through_the_relay_reaches_main_without_the_browser_origin():
    seen: list[httpx.Request] = []
    p = Proxy(lambda: "http://127.0.0.1:8080", lambda: "KEY")
    p._client = httpx.AsyncClient(transport=guarded_engine(seen))
    req = _Req("POST", {"origin": "http://127.0.0.1:8090", "content-type": "application/json",
                        "host": "127.0.0.1:8090"}, b'{"messages": []}')
    resp = asyncio.run(p.handle(req, "v1/chat/completions"))
    assert resp.status_code == 200
    assert len(seen) == 1
    assert "origin" not in {k.lower() for k in seen[0].headers}
    assert seen[0].headers["host"] == "127.0.0.1:8080"          # the engine's own address, not the page's
    assert seen[0].headers["authorization"] == "Bearer KEY"


def test_upstream_headers_drop_origin_in_any_case():
    p = Proxy(lambda: "http://127.0.0.1:8080")
    out = p.upstream_headers([("Origin", "http://a.test"), ("oRiGiN", "http://b.test"), ("accept", "*/*")])
    assert {k.lower() for k in out} == {"accept", "x-tfs-relayed"}
