"""Bench HTTP routes."""

from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from . import (
    bench,
    compare,
    report,
    series,
    tooleval,
)
from . import suites as custom_suites
from .server_http import (
    attachment,
)


def install(_two, app, bench_go, comparisons, concurrency, engine_named, quality_info, runner, s, served_name) -> None:
    @app.get("/api/bench/suites")
    def bench_suites():
        return {"modes": bench.MODES, "protocol": bench.PROTOCOL, "sizes": bench.SIZES, "about": runner.suites.get("about"),
                "version": runner.suites.get("version"), "digest": runner.suites.get("digest"),
                "suites": [{"id": x["id"], "name": x["name"], "prompts": len(x["prompts"]),
                            "quick": sum(1 for p in x["prompts"] if p.get("quick"))} for x in runner.suites["suites"]],
                "custom": [{**c, "digest": custom_suites.digest(c)} for c in runner.custom.list()],
                "quality": quality_info,
                "tools": {"installed": tooleval.find() is not None}}

    @app.post("/api/bench/tools/install")
    def tool_eval_install():
        """Installs tool-eval-bench for this user with uv, so the Tool calls run needs no command typed."""
        try:
            return tooleval.install()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/bench/suites")
    def custom_suite_create(body: dict[str, Any]):
        try:
            return runner.custom.save(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/bench/suites/copy/{builtin_id}")
    def custom_suite_copy(builtin_id: str):
        """A built-in suite copied into the user's, to change its prompts; the built-in one stays as it is."""
        suite = next((x for x in runner.suites["suites"] if x["id"] == builtin_id), None)
        if suite is None:
            raise HTTPException(404, "unknown built-in suite")
        return runner.custom.save(custom_suites.from_builtin(suite))

    @app.put("/api/bench/suites/{suite_id}")
    def custom_suite_update(suite_id: str, body: dict[str, Any]):
        try:
            return runner.custom.save(body, suite_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown suite") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/bench/suites/{suite_id}")
    def custom_suite_delete(suite_id: str):
        if not runner.custom.delete(suite_id):
            raise HTTPException(404, "unknown suite")
        return {"deleted": suite_id}

    @app.get("/api/bench/suites/{suite_id}/export")
    def custom_suite_export(suite_id: str):
        suite = runner.custom.get(suite_id)
        if suite is None:
            raise HTTPException(404, "unknown suite")
        return JSONResponse(custom_suites.export(suite), headers=attachment(
            suite["name"], "suite", ".json", prefix="suite-"))

    @app.get("/api/bench/runs")
    def bench_runs(limit: int = 50):
        return {"runs": runner.store.runs(limit), "busy": runner.busy(),
                "current": runner.current["id"] if runner.busy() and runner.current else None}

    @app.post("/api/bench/series")
    def series_start(body: dict[str, Any]):
        if s.remote:
            raise HTTPException(409, "the engine is remote: a comparison restarts it, which only its own TensorFold Web can do")
        try:
            return comparisons.start(body.get("configs"), mode=str(body.get("mode") or "profile"),
                                     suites=body.get("suites") or None, exactness=bool(body.get("exactness", True)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/bench/series")
    def series_list():
        return {"series": [series.matrix(d, runner.store.run) for d in comparisons.list()],
                "current": comparisons.doc["id"] if comparisons.busy() and comparisons.doc else None}

    @app.get("/api/bench/series/{sid}")
    def series_get(sid: str):
        doc = comparisons.doc if comparisons.busy() and comparisons.doc and comparisons.doc["id"] == sid \
            else comparisons.get(sid)
        if doc is None:
            raise HTTPException(404, "unknown comparison")
        return series.matrix(doc, runner.store.run)

    @app.post("/api/bench/series/{sid}/cancel")
    def series_cancel(sid: str):
        if not (comparisons.busy() and comparisons.doc and comparisons.doc["id"] == sid):
            raise HTTPException(409, "this comparison is not going")
        comparisons.cancel()
        return {"cancelling": True}

    @app.post("/api/bench/runs")
    def bench_start(body: dict[str, Any]):
        if comparisons.busy():
            raise HTTPException(409, "a comparison of configurations is going; cancel it or wait")
        if runner.busy():
            raise HTTPException(409, "a run is already going; cancel it or wait")
        suites = body.get("suites")
        if suites is not None and not (isinstance(suites, list) and all(isinstance(x, str) for x in suites)):
            raise HTTPException(400, "suites must be a list of suite ids")
        target = body.get("target")
        try:
            # KL references may use a plain local arm or an explicit endpoint, on generated replies or fixed text.
            reference, kl_load, kl_text, tags = None, 0, None, {}
            if body.get("mode") == "kl":
                ref = body.get("reference")
                if ref is not None:
                    r_ident, r_key, r_headers, r_peer = engine_named(ref, "reference")
                    if r_ident.error:
                        raise ValueError(f"{r_peer or r_ident.url}: {r_ident.error}")
                    r_model = ref.get("model") or next(iter(r_ident.models), None)
                    if r_model not in r_ident.models:
                        raise ValueError(f"{r_peer or r_ident.url} does not serve {r_model!r}")
                    reference = (r_ident.url, r_key, r_model, r_headers)
                    if r_peer:
                        tags["reference_peer"] = r_peer
                kl_load = body.get("kl_load") or 0
                if isinstance(kl_load, bool) or not isinstance(kl_load, int) or not 0 <= kl_load <= 32:
                    raise ValueError("kl_load must be 0 to 32 requests")
                kl_text = body.get("kl_text")
            if target is not None:
                # another engine: identified, its model the one asked or the first it lists
                ident, key, headers, peer = engine_named(target, "target")
                at = peer or ident.url
                if ident.error:
                    raise ValueError(f"{at}: {ident.error}")
                model = target.get("model") or next(iter(ident.models), None)
                if model not in ident.models:
                    raise ValueError(f"{at} does not serve {model!r}; it serves {', '.join(ident.models) or 'nothing'}")
                name = ident.engine + (f" {ident.version}" if ident.version else "")
                label = target.get("label")
                if isinstance(label, str) and label.strip() and ident.engine == "OpenAI-compatible":
                    name = label.strip()[:60]           # a server that does not say what it is, named by the user
                if peer:
                    tags["peer"] = peer
                max_context = body.get("max_context")
                if max_context is not None and not (isinstance(max_context, int) and max_context > 0):
                    raise ValueError("max_context must be a number of tokens")
                return bench_go(str(body.get("mode") or "quick"), suites=suites or None,
                                    exactness=bool(body.get("exactness", True)), engine_name=name,
                                    max_context=max_context, target=(ident.url, key, model, headers),
                                    max_concurrency=concurrency(body), full=bool(body.get("full")),
                                    reference=reference, kl_load=kl_load, kl_text=kl_text, tags=tags or None)
            name = served_name()
            max_context = body.get("max_context")
            if max_context is not None and not (isinstance(max_context, int) and max_context > 0):
                raise ValueError("max_context must be a number of tokens")
            return bench_go(str(body.get("mode") or "quick"), suites=suites or None,
                                exactness=bool(body.get("exactness", True)), engine_name=name, max_context=max_context,
                                max_concurrency=concurrency(body), full=bool(body.get("full")),
                                reference=reference, kl_load=kl_load, kl_text=kl_text, tags=tags or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/bench/identify")
    def bench_identify(body: dict[str, Any]):

        try:
            ep, *_ = engine_named(body, "the engine")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"url": ep.url, "engine": ep.engine, "version": ep.version, "models": ep.models, "error": ep.error}

    @app.get("/api/bench/runs/{run_id}")
    def bench_run(run_id: str):
        # the server's clock goes with the run, so the page counts elapsed time from it and not its own
        if runner.busy() and runner.current and runner.current["id"] == run_id:
            return {**runner.current, "results": runner.store.run(run_id)["results"], "server_time": time.time()}
        run = runner.store.run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return {**run, "server_time": time.time(), "findings": report.findings(run), "share_text": report.share_text(run)}

    @app.post("/api/bench/runs/{run_id}/cancel")
    def bench_cancel(run_id: str):
        if not (runner.busy() and runner.current and runner.current["id"] == run_id):
            raise HTTPException(409, "this run is not going")
        runner.cancel()
        return {"cancelling": True}

    @app.delete("/api/bench/runs/{run_id}")
    def bench_delete(run_id: str):
        if runner.busy() and runner.current and runner.current["id"] == run_id:
            raise HTTPException(409, "cancel the run before deleting it")
        if not runner.store.delete(run_id):
            raise HTTPException(404, "unknown run")
        return {"deleted": run_id}

    @app.get("/api/bench/runs/{run_id}/receipt")
    def bench_receipt(run_id: str):
        run = runner.store.run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return JSONResponse(bench.receipt(run), headers={
            "Content-Disposition": f'attachment; filename="bench-{run_id}.json"'})

    @app.get("/api/bench/compare")
    def bench_compare(a: str, b: str):
        ra, rb = _two(a, b)
        try:
            return compare.compare(ra, rb)
        except compare.Refused as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/bench/compare/report")
    def bench_compare_report(a: str, b: str):
        ra, rb = _two(a, b)
        try:
            c = compare.compare(ra, rb)
        except compare.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
        return HTMLResponse(compare.html_report(ra, rb, c), headers={
            "Content-Disposition": f'attachment; filename="bench-{a}-vs-{b}.html"'})

    @app.get("/api/bench/runs/{run_id}/report")
    def bench_report(run_id: str):
        run = runner.store.run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return HTMLResponse(report.html_report(run), headers={
            "Content-Disposition": f'attachment; filename="bench-{run_id}.html"'})
