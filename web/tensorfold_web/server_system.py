"""System HTTP routes."""

from __future__ import annotations

from fastapi import HTTPException

from . import (
    __version__,
    fit,
    history,
    living,
    usage,
)
from .catalog import capacity_of
from .machine import detect


def install(app, cat, kept, s, used, eng):
    @app.on_event("startup")
    async def _startup():
        kept.start()

    @app.on_event("shutdown")
    async def _shutdown():
        kept.stop()

    @app.get("/api/history")
    def figures_history(window: int = 1800):
        if not 60 <= window <= int(history.KEEP_S):
            raise HTTPException(400, f"window must be 60 to {int(history.KEEP_S)} seconds")
        return kept.since(window)

    @app.get("/api/usage")
    def usage_view(days: int = 30):
        """What the engine served by day and by hour, by model, kept across engine and TensorFold Web restarts."""

        if not 1 <= days <= usage.KEEP_DAYS:
            raise HTTPException(400, f"days must be 1 to {usage.KEEP_DAYS}")
        return used.summary(days)

    @app.get("/api/machine")
    def machine():
        m = detect()
        cap = capacity_of(m)
        return {**m.to_dict(), "studio_version": __version__,
                **({"tensorfold_version": eng.mock_version} if getattr(eng, "mock_version", None) else {}),
                "budget": {"default_fraction": cap.fraction if cap else fit.DEFAULT_FRACTION,
                           "raised_fraction": cap.raised if cap else fit.RAISED_FRACTION,
                           "current_fraction": s.budget_fraction or (cap.fraction if cap else fit.DEFAULT_FRACTION),
                           "reserve_bytes": cap.reserve_bytes if cap else fit.PROCESS_RESERVE,
                           "ceiling_bytes": cap.ceiling_bytes if cap else None,
                           "capacity_bytes": cap.bytes if cap else None, "basis": cap.basis if cap else "none"}}

    @app.post("/api/catalog/discover")
    def catalog_discover():
        cat.refresh_sizes_async()
        return {"refreshing": True}

    @app.post("/api/catalog/refresh")
    def catalog_refresh():
        """Sizes read again from Hugging Face: a POST, so a page elsewhere cannot make TensorFold Web do it with a GET."""

        cat.refresh_sizes(force=True)
        return catalog_view()

    @app.get("/api/catalog")
    def catalog_view():
        m = detect()
        return {"models": cat.view(m, fraction=s.budget_fraction), "sizes_known": bool(cat._sizes),
                "refreshing": cat.refreshing, "refreshed_at": cat.refreshed_at}

    @app.get("/api/catalog/{ref}/verdicts")
    def copy_verdicts(ref: str):
        """A Living Weights copy (``living:<name>``) takes the memory of the model it was copied from."""

        if not ref.startswith(living.PREFIX):
            raise HTTPException(404, "unknown model")
        try:
            base = living.base_of(ref.removeprefix(living.PREFIX), [])
        except ValueError as exc:
            raise HTTPException(404, "unknown copy") from exc
        if not base or "/" not in base:
            raise HTTPException(404, "the model this copy came from is not known")
        return verdicts(*base.split("/", 1))

    @app.get("/api/catalog/{owner}/{name}/verdicts")
    def verdicts(owner: str, name: str):
        """The verdict at every budget from the default to the raised one, each one for exactly that budget."""

        m = detect()
        cap = capacity_of(m)
        repo = f"{owner}/{name}"
        entry = next((e for e in cat.entries if e.id == repo), None)
        if entry is None:
            raise HTTPException(404, "unknown model")
        if cap is None:
            raise HTTPException(409, "no supported GPU on this machine")
        weights, workspace, _ = cat.fit_inputs(entry)
        if not weights:
            raise HTTPException(409, "size unknown yet")
        out = []
        lo, hi = round(cap.fraction * 100), round(cap.raised * 100)
        for pct in range(lo, hi + 1, 5):
            v = cat.verdict_of(entry, m, weights, workspace, pct / 100, strict=True)
            out.append({"fraction": pct / 100, **v.to_dict()})
        return {"model": repo, "weights_bytes": weights, "verdicts": out}
    return machine
