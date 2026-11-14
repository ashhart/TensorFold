"""Page HTTP routes."""

from __future__ import annotations

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from . import (
    __version__,
)
from .server_http import (
    UI_DIST,
)


def install(app) -> None:
    @app.get("/{path:path}", include_in_schema=False)
    def page(path: str):
        if path.startswith(("api/", "engine/")):
            return JSONResponse({"detail": "unknown route"}, status_code=404)
        index = UI_DIST / "index.html"
        if not index.is_file():
            return JSONResponse({"studio": __version__, "note": "the web UI is not built; run `npm run build` in ui/"})
        if path:
            candidate = (UI_DIST / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(UI_DIST):      # never a file outside the page's folder
                # hashed assets may be cached for good; the manifest names them and changes with each build
                headers = {"Cache-Control": "no-cache"} if path.endswith(".webmanifest") else {}
                return FileResponse(candidate, headers=headers)
            if "." in path.rsplit("/", 1)[-1]:                                   # an asset that does not exist
                return JSONResponse({"detail": "not found"}, status_code=404)
        # always revalidated: a page cached across a deploy would ask for assets that no longer exist
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
