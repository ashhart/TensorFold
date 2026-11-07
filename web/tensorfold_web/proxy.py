
from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate"}
STUDIO_HEADERS = {"x-studio-token"}
# Web checks the browser Origin itself; the engine relay drops it to satisfy the engine's own Origin check.
BROWSER_ONLY = {"origin"}
STUDIO_PARAMS = {"token"}
ENGINE_AUTH = {"authorization", "x-api-key"}


RELAYED = "x-tfs-relayed"                     # set on what the relay sends: seen again, it is the relay itself
MAX_BODY = 64 * 1024 * 1024                   # a chat with images fits; a body meant to fill memory does not


class Proxy:
    def __init__(self, engine_url_getter: Callable[[], str], api_key_getter: Callable[[], str | None] | None = None,
                 target: Callable[[], tuple[str, str | None]] | None = None) -> None:
        self._url = engine_url_getter
        self._key = api_key_getter or (lambda: None)
        # address and key read together, once per request: a change of engine midway never sends one's key to the other
        self._target = target or (lambda: (self._url(), self._key()))
        self._client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(600.0, connect=5.0))

    def upstream_headers(self, incoming: list[tuple[str, str]], key: str | None = None, *,
                         snapshot: bool = False) -> dict[str, str]:

        headers = {k: v for k, v in incoming
                   if k.lower() not in HOP and k.lower() not in STUDIO_HEADERS and k.lower() not in BROWSER_ONLY}
        key = key if snapshot else self._key()
        headers[RELAYED] = "1"                        # a request that comes back to a relay is refused there (a loop)
        if key:
            headers = {k: v for k, v in headers.items() if k.lower() not in ENGINE_AUTH}
            headers["Authorization"] = f"Bearer {key}"        # the page never holds the engine's key itself
        return headers

    async def handle(self, request: Request, path: str) -> Response:
        base, key = self._target()
        url = base.rstrip("/") + "/" + path
        headers = self.upstream_headers(list(request.headers.items()), key, snapshot=True)
        params = [(k, v) for k, v in request.query_params.multi_items() if k not in STUDIO_PARAMS]
        body = bytearray()
        async for part in request.stream():           # read in parts: a body past the cap is refused before it is held
            body += part
            if len(body) > MAX_BODY:
                return Response(content='{"error":{"message":"request body over the relay\'s 64 MiB"}}',
                                status_code=413, media_type="application/json")
        try:
            upstream = self._client.build_request(request.method, url, headers=headers, content=bytes(body), params=params)
            resp = await self._client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            return Response(content=f'{{"error":{{"message":"engine unreachable: {exc}"}}}}', status_code=502,
                            media_type="application/json")
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}

        async def body_iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        # Close after any response, even if its body generator never started; repeated close is harmless.
        return StreamingResponse(body_iter(), status_code=resp.status_code, headers=out_headers,
                                 background=BackgroundTask(resp.aclose))
