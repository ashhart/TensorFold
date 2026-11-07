"""Engine launches and the gate held during start transactions."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import HTTPException

from . import (
    living,
)
from . import settings as settings_mod
from .gate import Busy


def build(cat, eng, gate, s, save_settings):
    def friendly_name(served: str | None) -> str | None:

        if not served:
            return None
        copy = living.name_of(served)
        if copy:
            return copy
        low = served.lower().rstrip("/")
        for e in cat.entries:
            if low in (e.id.lower(), e.id.split("/")[-1].lower()) or low.endswith("/" + e.id.lower()):
                return e.name
        return served

    def key_for_learning() -> None:

        if s.api_key:
            return
        with eng.target_lock:
            s.api_key = settings_mod.new_api_key()
            eng.retargeted()
        save_settings()

    def launch(model: str, entry: Any) -> dict:

        from . import native

        copy = living.name_of(model)
        if copy:                                            # a living copy: its folder, learning where the machine can
            path = living.folder(copy)
            if not path.is_dir() or living.is_unfinished(copy):
                raise RuntimeError(f"there is no living copy named {copy} on this machine")
            why = living.family_refusal(copy) or living.teachable_here(native.find_binary())
            if not why:
                key_for_learning()
            result = eng.start(model, native_model=True, serve=str(path),
                               slide=None if why else str(living.graph_path(copy)))
            if why:
                eng.note(f"{copy} serves without learning: {why}")
            return result
        here = bool(entry and cat.native_here(entry))
        result = eng.start(model, native_model=here, drafter=entry.drafter if here and entry else None)
        if not here:
            return result
        end = time.time() + 8
        while time.time() < end:                            # a refusal comes before any weight is read
            if native.refusal_in(list(eng.log)[-20:]) or eng.exit_code is not None or eng.health().ok:
                break
            time.sleep(0.25)
        if eng.owned and not native.refusal_in(list(eng.log)[-20:]):
            return result                                   # loading or serving: nothing was refused
        if not eng.wait_stopped(5):                         # gone, and its log read to the last line
            return result                                   # still there: nothing is started over it
        why = native.refusal_in(list(eng.log)[-40:])
        binary = native.find_binary()
        if not why or binary is None:
            return result
        native.remember_refusal(binary, model, why)
        result = eng.start(model, native_model=False)      # a start clears the log: the reason goes after it
        eng.note(f"the native engine does not load {model} on this machine ({why}); started on the Python line")
        return result

    async def held_in_thread(who: str, work: Any) -> Any:

        def run() -> Any:
            try:
                gate.take(who)
            except Busy as exc:
                raise HTTPException(409, str(exc)) from exc
            try:
                return work()
            finally:
                gate.give()
        return await asyncio.to_thread(run)

    def _start_held(patch: dict[str, Any]) -> dict:
        """The start itself, the gate held: settings applied for the launch, put back if it does not happen."""

        model = patch.get("model") or s.model
        if not model:
            raise HTTPException(400, "no model chosen")
        patch["model"] = model
        previous = {k: getattr(s, k) for k in patch}
        for k, v in patch.items():
            setattr(s, k, v)
        try:
            entry = next((e for e in cat.entries if e.id == model), None)
            result = launch(model, entry)
        except Exception as exc:
            for k, v in previous.items():                   # a start that did not happen changes nothing
                setattr(s, k, v)
            if isinstance(exc, RuntimeError):
                raise HTTPException(409, str(exc)) from exc
            raise HTTPException(500, f"could not launch the engine: {exc}") from exc
        save_settings()
        return {**result, **eng.info()}
    return friendly_name, launch, held_in_thread, _start_held
