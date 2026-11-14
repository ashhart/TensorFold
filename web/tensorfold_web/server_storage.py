"""Storage HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from . import settings as settings_mod
from . import (
    storage,
)


def install(app, catalog_repos, gate, in_use) -> None:
    @app.get("/api/storage")
    def storage_view():
        from . import pull

        cache = pull.cache_root()
        rows = storage.models(cache, catalog_repos())
        for m in rows:
            m["busy"] = in_use(m["repo"])
        return {"stores": storage.summary(settings_mod.config_dir()), "models": rows, "cache": str(cache)}

    @app.post("/api/storage/clean")
    def storage_clean(body: dict[str, Any]):
        days = body.get("older_than_days")
        if days is not None and (isinstance(days, bool) or not isinstance(days, (int, float))):
            raise HTTPException(400, "older_than_days must be a number of days, or null for everything")
        try:
            return storage.clean(settings_mod.config_dir(), str(body.get("kind")), days,
                                 dry_run=bool(body.get("dry_run", True)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/storage/models/remove")
    def storage_remove_model(body: dict[str, Any]):
        from . import pull

        repo = body.get("repo")
        if not isinstance(repo, str) or repo not in catalog_repos():
            raise HTTPException(400, "only the catalog's models are removed from here")
        try:
            with gate.hold("a model's removal"):          # nothing starts serving or downloading it meanwhile
                busy = in_use(repo)
                if busy:
                    raise HTTPException(409, f"{repo} is not removed: {busy}")
                return storage.remove_model(pull.cache_root(), repo, dry_run=bool(body.get("dry_run", True)))
        except KeyError as exc:
            raise HTTPException(404, f"{repo} is not in the cache") from exc
        except RuntimeError as exc:                      # Busy too
            raise HTTPException(409, str(exc)) from exc
