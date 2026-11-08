"""Chat HTTP routes."""

from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    Response,
)

from . import (
    brands,
    conversations,
    replay,
)
from .server_http import (
    attachment,
)


def install(app, chats, makers, replays) -> None:
    @app.get("/api/conversations")
    def chat_list(q: str = "", offset: int = 0):
        rows, more = chats.page(q, offset=max(0, offset))
        return {"conversations": rows, "more": more}

    @app.post("/api/conversations")
    def chat_create(body: dict[str, Any]):
        try:
            return chats.create(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/conversations/compare")
    def chat_compare(a: str, b: str):
        ca, cb = chats.get(a), chats.get(b)
        if ca is None or cb is None:
            raise HTTPException(404, "unknown conversation")
        return replay.compare_replies(ca, cb)

    @app.post("/api/conversations/{cid}/replay")
    def chat_replay(cid: str, body: dict[str, Any]):
        target = body.get("target") or {"kind": "served", "drafts": True}
        if not isinstance(target, dict):
            raise HTTPException(400, "target must be an object")
        try:
            return replays.start(cid, target)
        except KeyError as exc:
            raise HTTPException(404, "unknown conversation") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/replay")
    def replay_progress():
        return {"busy": replays.busy(), "current": replays.current}

    @app.post("/api/replay/cancel")
    def replay_cancel():
        return {"cancelling": replays.cancel()}

    @app.get("/api/conversations/{cid}")
    def chat_get(cid: str):
        doc = chats.get(cid)
        if doc is None:
            raise HTTPException(404, "unknown conversation")
        # A streaming reply left unfinished in an old saved conversation has lost its browser connection.
        return conversations.settle(doc, live={cid} if time.time() - doc["updated"] < 60 else set())

    @app.put("/api/conversations/{cid}")
    def chat_update(cid: str, body: dict[str, Any]):
        rev = body.pop("rev", None)
        if rev is not None and (isinstance(rev, bool) or not isinstance(rev, int)):
            raise HTTPException(400, "rev must be the revision number the page read")
        if rev is None and ("messages" in body or "settings" in body):
            # a write of content says which revision it was made on, so it never overwrites one it has not seen
            raise HTTPException(400, "rev is required to change messages or settings: send the revision you read")
        try:
            return chats.update(cid, body, rev=rev)
        except conversations.Conflict as exc:
            return JSONResponse({"detail": str(exc), "rev": exc.current.get("rev", 0)}, status_code=409)
        except KeyError as exc:
            raise HTTPException(404, "unknown conversation") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/conversation-folders")
    def chat_folders():
        return {"folders": chats.folders()}

    @app.post("/api/conversation-folders")
    def chat_folder_make(body: dict[str, Any]):
        try:
            return chats.make_folder(body.get("name"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/conversation-folders/{name:path}")
    def chat_folder_rename(name: str, body: dict[str, Any]):
        try:
            return chats.rename_folder(name, body.get("name"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/conversation-folders/{name:path}")
    def chat_folder_remove(name: str):
        return {"removed": name, "moved_out": chats.remove_folder(name)}

    @app.get("/api/chat-presets")
    def chat_presets():
        return {"presets": chats.presets()}

    @app.post("/api/chat-presets")
    def chat_preset_save(body: dict[str, Any]):
        """Saves a preset; one of the same name is replaced only when `replace` says so (409 otherwise)."""
        try:
            return chats.save_preset(body.get("preset"), replace=bool(body.get("replace")))
        except conversations.Taken as exc:
            raise HTTPException(409, f"a preset named {exc} is kept already") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/chat-presets/{name:path}")
    def chat_preset_rename(name: str, body: dict[str, Any]):
        try:
            return chats.rename_preset(name, body.get("name"))
        except KeyError as exc:
            raise HTTPException(404, "unknown preset") from exc
        except conversations.Taken as exc:
            raise HTTPException(409, f"a preset named {exc} is kept already") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/chat-presets/{name:path}")
    def chat_preset_remove(name: str):
        if not chats.remove_preset(name):
            raise HTTPException(404, "unknown preset")
        return {"removed": name}

    @app.get("/api/brands")
    def brand_list(request: Request):
        # up to a hundred logos inline: sent again only when the file changed (the browser revalidates each time)
        tag = f'"{makers.version()}"'
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers={"ETag": tag, "Cache-Control": "no-cache"})
        return JSONResponse({"brands": makers.all()}, headers={"ETag": tag, "Cache-Control": "no-cache"})

    @app.put("/api/brands/{brand_id}")
    def brand_save(brand_id: str, body: dict[str, Any]):
        try:
            return makers.save(brand_id, body)
        except brands.Invalid as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/brands/{brand_id}")
    def brand_remove(brand_id: str, kind: str | None = None):
        if not makers.remove(brand_id, kind):
            raise HTTPException(404, "unknown maker")
        return {"removed": brand_id}

    @app.delete("/api/conversations/{cid}")
    def chat_delete(cid: str):
        if not chats.delete(cid):
            raise HTTPException(404, "unknown conversation")
        return {"deleted": cid}

    @app.get("/api/conversations/{cid}/export")
    def chat_export(cid: str, format: str = "md"):
        doc = chats.get(cid)
        if doc is None:
            raise HTTPException(404, "unknown conversation")
        if format == "json":
            return JSONResponse({"kind": "tensorfold-studio-conversation", **doc},
                                headers=attachment(doc["title"], "conversation", ".json"))
        return Response(conversations.markdown(doc), media_type="text/markdown",
                        headers=attachment(doc["title"], "conversation", ".md"))
