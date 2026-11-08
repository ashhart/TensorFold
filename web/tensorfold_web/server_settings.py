"""Settings HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from . import settings as settings_mod
from .gate import Busy
from .server_http import (
    ENGINE_SETTINGS,
)


def install(app, eng, gate, s, save_settings, user_settings) -> None:
    @app.get("/api/settings")
    def get_settings():
        return user_settings()

    @app.put("/api/settings")
    def put_settings(body: dict[str, Any]):
        try:
            patch = settings_mod.validate(body)
            if patch.get("extra_args"):
                from . import native
                binary = native.find_binary()
                if binary:                               # checked against the flags this engine offers
                    patch["extra_args"] = native.check_extra(patch["extra_args"], native.serve_flags(binary))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        held = False
        if any(k in patch for k in ENGINE_SETTINGS):
            try:
                gate.take("a settings change")              # compared and applied under the gate, never before it
                held = True
            except Busy as exc:
                # something drives the engine: its settings stay as that leaves them; the rest of the patch applies
                if any(patch[k] != getattr(s, k) for k in ENGINE_SETTINGS if k in patch):
                    raise HTTPException(409, f"the engine's settings are not changed now: {exc}") from exc
                patch = {k: v for k, v in patch.items() if k not in ENGINE_SETTINGS}
        try:
            target_changed = any(k in patch and patch[k] != getattr(s, k) for k in ("engine_url", "remote"))
            if target_changed and eng.owned:            # its key and its process belong to the address it was given
                raise HTTPException(409, f"this TensorFold Web runs {eng.model} at {s.engine_url}: stop it before pointing "
                                         "TensorFold Web at another engine")
            with eng.target_lock:                            # address and key change together, as one new target
                before = (s.engine_url, s.api_key, s.remote)
                for key, value in patch.items():
                    setattr(s, key, value)
                if held and s.share and not s.api_key:
                    s.api_key = settings_mod.new_api_key()      # sharing without a key is never offered
                if (s.engine_url, s.api_key, s.remote) != before:
                    eng.retargeted()
                if target_changed:
                    eng.reset_target()                       # nothing read from the previous engine applies here
            save_settings()
        finally:
            if held:
                gate.give()
        return {**user_settings(), "key_pending_restart": eng.key_pending_restart()}

    @app.post("/api/settings/api-key")
    def rotate_api_key():
        """A new key for the engine's next start; an engine already running keeps the one it was started with."""

        try:
            with gate.hold("a new engine key"):              # a comparison restarting the engine would take it midway
                with eng.target_lock:
                    s.api_key = settings_mod.new_api_key()
                    eng.retargeted()
                save_settings()
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        pending = eng.key_pending_restart()
        return {"api_key_set": True, "key_pending_restart": pending,
                "note": "the running engine keeps its previous key until it restarts" if pending else
                        "applies to the engine at its next start"}
