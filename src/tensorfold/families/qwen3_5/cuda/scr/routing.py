"""The engine hooks, as functions the engine calls.

``MultiDecoder`` (``--parallel``) and the serial ``Qwen27Engine.generate`` route
through these at five points; every one is a no-op unless
``TENSORFOLD_SCR_ENABLED=1`` and falls back to plain serving on any error. The
engine's exact-resume path is untouched: planned turns keep no states in the
shared prefix cache, and a session's snapshot is that turn's resume.

  admission   match the prompt to a session, diff, attach a plan (never on two
              ranks or for image prompts)
  queue       a planned stream builds its ``State`` from the session snapshot
              (rows ``[0, base)`` verbatim, private buffers, forked linear state)
  fill        planned streams alternate segment prefills with splices; after a
              plan aborts they keep the no-keep route and prefill plainly
  snapshots   a finished turn's committed rows become its session's new snapshot
  generate    the serial path's planned turn

The one deliberate approximation: a planned turn's linear-attention layers run
from the snapshot's end-of-turn state (``fork``); attention rows are exact where
their position is unchanged. Dense models (Qwen3.8-27B) therefore serve planned
append-only turns bit-exactly; hybrid models take SCR's documented fork trade.
"""

import time

import torch

from .config import Config, _log
from .plan import build_plan, describe
from .sessions import SessionStore
from .splice import execute_splice
from .state import build_state_from_snapshot, snapshot_from_state


class Runtime:
    """Per-weights SCR state: config, session store, rotary frequencies."""

    def __init__(self, cfg: Config, w) -> None:
        self.cfg = cfg
        self.w = w
        self.store = SessionStore(cfg)
        self.inv_freq = getattr(w, "inv_freq", None)
        self.hybrid = any(getattr(l, "linear", False) for l in getattr(w, "layers", []))
        self.installed_at = time.perf_counter()

    def init_line(self) -> str:
        return (f"K={self.cfg.max_blocks} min_tail={self.cfg.min_tail} min_extend={self.cfg.min_extend} "
                f"min_base={self.cfg.min_base} min_gain={self.cfg.min_gain} "
                f"sessions={self.cfg.max_sessions} side_budget={self.cfg.side_gib} GiB "
                f"ssm={self.cfg.ssm_mode} hybrid={self.hybrid}")


_RT: Runtime | None = None


def _rt_for(w) -> Runtime:
    global _RT
    if _RT is None or _RT.w is not w:
        _RT = Runtime(Config(), w)
        _log(f"runtime ready: {_RT.init_line()}")
    return _RT


# ---------------------------------------------------------------------------
# MultiDecoder hooks (--parallel)
# ---------------------------------------------------------------------------
def admission(md, s) -> None:
    """Match the prompt to a session and attach a plan when one applies. The
    engine's admission continues either way; a planning bug must not break it."""
    if not _RT and not Config().enabled:
        return
    if md.world == 2 or getattr(s, "vision", None) is not None:
        return
    try:
        rt = _rt_for(md.w)
        sess, _ = rt.store.session_for(list(s.prompt))
        old_ids = list(sess.ids)
        s.scr_sid = sess.sid
        if sess.snap is not None:
            hit = md.cache.longest(s.prompt) if s.draft else None
            hit_len = len(hit[0]) if hit is not None else 0
            plan, why = build_plan(old_ids, list(s.prompt), sess.snap, sess.sid, rt.cfg,
                                   resume_len=hit_len, log=_log)
            sess.ids = list(s.prompt)         # SCR parity: the tracked ids follow the newest prompt
            if plan is not None:
                s.scr_plan = plan
                s.scr_route = True            # keeps off for the whole turn, even after an abort
                if rt.cfg.trace:
                    _log(f"plan: {describe(plan)} hit={hit_len}")
            elif rt.cfg.debug:
                _log(f"no plan: sid={sess.sid} {why} n_new={len(s.prompt)} n_old={len(old_ids)} hit={hit_len}")
        elif rt.cfg.debug:
            _log(f"cold: sid={sess.sid} n_new={len(s.prompt)}")
    except Exception as e:                    # noqa: BLE001  (a planning bug must not break serving)
        _log(f"plan error (serving plain): rid={getattr(s, 'sid', '?')} {e}")


def queue(md, s, hit) -> bool:
    """Queue a planned stream from its session snapshot. True: handled; False: the
    engine's own queueing continues (also after a state-build failure)."""
    plan = getattr(s, "scr_plan", None)
    if plan is None:
        return False
    try:
        rows = md._most(s) if md.memory_gate is None else md._first(s)
        s.st = build_state_from_snapshot(plan.snapshot, rows, md.w, pos=plan.base)
        drafter = md.draft if s.draft and md.drafts else None
        s.snap = None if drafter is None else ([None] * drafter.layers, [None] * drafter.layers, 0, 0)
        s.stops = []                          # planned turns keep no states in the shared cache
        s.cached = plan.base
        md.filling.append(s)
        return True
    except Exception as e:                    # noqa: BLE001  (serve plain rather than fail)
        _log(f"state build failed (serving plain): rid={getattr(s, 'sid', '?')} {e}")
        s.scr_plan = None
        s.scr_served = 0
        s.scr_base = 0
        return False


def hold_planned(md) -> list:
    """Planned streams fill alone (their steps interleave with splices): take them
    out of ``md.filling`` for one round; ``resume_planned`` puts them back."""
    planned = [x for x in md.filling if getattr(x, "scr_route", None) is True]
    if planned:
        md.filling = [x for x in md.filling if getattr(x, "scr_route", None) is not True]
    return planned


def resume_planned(md, planned: list) -> None:
    if planned:
        md.filling[:0] = planned


def _abort(s, why: str) -> None:
    plan = getattr(s, "scr_plan", None)
    if plan is not None:
        _log(f"plan aborted: rid={s.sid} {why} -> plain prefill of the tail")
        s.scr_base = plan.base
        s.scr_served = plan.served
    s.scr_plan = None                          # scr_route stays: no keeps on this turn


def fill(md) -> list | None:
    """One round of the filling loop for a planned stream, or None: the engine's
    own round continues."""
    from ..multi import STEP
    from tensorfold.cuda import streams as streams_mod

    s = streams_mod.next_fill(md.filling) if md.filling else None
    if s is None or getattr(s, "scr_route", None) is not True:
        return None
    rt = _rt_for(md.w)
    plan = getattr(s, "scr_plan", None)
    pos = s.st.pos

    sp = plan.pending_splice(pos) if plan is not None else None
    if sp is not None:
        try:
            i = plan.splices.index(sp) + 1
            ok = execute_splice(s.st, sp, plan.snapshot, s.prompt, rt.inv_freq, log=_log,
                                rid=s.sid, blocks=f"blocks={i}/{len(plan.splices)}")
            if ok:
                plan.served += sp.n
        except Exception as e:                # noqa: BLE001
            _log(f"splice error: rid={s.sid} {e}")
            ok = False
        if not ok:
            _abort(s, "splice failed")
        return []                             # a splice is sub-millisecond; the next round fills on

    n = len(s.prompt)
    stop = plan.segment_stop(pos, n) if plan is not None else n
    if pos >= n and (plan is None or plan.pending_splice(pos) is None):
        # unreachable via build_plan (base < len(prompt) is enforced), but never
        # wedge the scheduler: fail this stream cleanly instead of spinning
        s.error = RuntimeError("scr plan covers the whole prompt with nothing to prefill")
        s.done = True
        md.filling = [x for x in md.filling if x is not s]
        return [s]
    if s.background or any(not x.done for x in md.streams.values()):
        stop = min(stop, pos + STEP)
    if stop <= pos:
        return []
    from ..multi import FILL

    md._send([FILL, s.sid, stop])
    try:
        first = _step_planned(md, s, stop)
    except Exception as exc:                  # noqa: BLE001  (this request fails, the others go on)
        if md.world == 2:
            raise
        md.filling = [x for x in md.filling if x is not s]
        s.error, s.done = exc, True
        return [s]
    if first is None:
        return []
    s.take([first], md._ends(s))
    return [s] if s.done else []


def _step_planned(md, s, stop):
    """``MultiDecoder._step`` for a routed stream: same prefill and first-token
    sampling, but no state keeps in the shared prefix cache (a planned turn's
    rows may be relocated, and the session snapshot is that turn's resume)."""
    from ..multi import CopyIndex, first_token, prefill_state

    t0 = time.perf_counter()
    drafter = md.draft if s.draft and md.drafts else None
    try:
        if drafter is not None:
            drafter.restore(s.snap)
        n = len(s.prompt)
        out = prefill_state(md.w, s.prompt[:stop], s.st, tp=md.world == 2, draft=drafter, vision=None)
        normed = out
        if drafter is not None:
            s.snap = drafter.snapshot()
            drafter.skip(0)
        first = None if stop < n else first_token(md.w, normed, n, s.sampling, md.rank, md.world,
                                                  s.constraint)
    except Exception as exc:                  # noqa: BLE001
        if md.world == 2:
            md.broken = exc
        raise
    finally:
        s.prefill_s += time.perf_counter() - t0
    if first is None:
        return None
    s.copies = CopyIndex() if md.allow_copy and s.draft and md.rank == 0 else None
    s.context = list(s.prompt)
    s.started = time.perf_counter()
    md.filling = [x for x in md.filling if x is not s]
    md.streams[s.sid] = s                     # after every step that can fail: a failure leaves it queued
    return first


def snapshots(md, done) -> None:
    """A finished turn's committed rows become its session's new snapshot."""
    if md.world == 2:
        return
    rt = _rt_for(md.w)
    for s in done:
        sid = getattr(s, "scr_sid", None)
        if sid is not None and s.error is None and s.st is not None:
            try:
                snap = snapshot_from_state(s.st, list(s.context), log=_log)
                if snap is not None and snap.pos > 0:
                    rt.store.refresh(sid, snap)
                    if rt.cfg.trace:
                        _log(f"snapshot: sid={sid} rid={s.sid} pos={snap.pos} "
                             f"bytes={snap.bytes / 2**20:.1f} MiB")
            except Exception as e:            # noqa: BLE001
                _log(f"snapshot error: sid={sid} {e}")


def stats_fields(s) -> dict:
    """The stream's SCR accounting, into its ``stats`` dict (empty when unplanned)."""
    sid = getattr(s, "scr_sid", None)
    plan = getattr(s, "scr_plan", None)
    if sid is None and plan is None:
        return {}
    base = getattr(s, "scr_base", plan.base if plan else 0)
    relocated = getattr(s, "scr_served", plan.served if plan else 0)
    blocks = getattr(s, "scr_blocks", len(plan.splices) if plan else 0)
    return {"scr": True, "scr_sid": sid, "scr_base": base, "scr_relocated": relocated, "scr_blocks": blocks}


# ---------------------------------------------------------------------------
# Qwen27Engine.generate (no --parallel)
# ---------------------------------------------------------------------------
def generate(engine, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True, *,
             vision=None, constraint=None, background=False):
    """A planned turn runs a segmented prefill over the session snapshot; returns
    None when SCR is off, inapplicable, or no plan applied: the engine's own
    generate continues."""
    if not _RT and not Config().enabled:
        return None
    if engine.scheduler is not None or engine.tp == 2 or vision is not None:
        return None
    rt = _rt_for(engine.w)
    prompt = list(prompt)
    if len(prompt) >= engine.context_window:
        return None
    max_tokens = max(1, min(int(max_tokens), engine.context_window - len(prompt)))
    try:
        sess, _ = rt.store.session_for(prompt)
        old_ids = list(sess.ids)
        scr_sid = sess.sid
        plan = None
        hit_len = 0
        if sess.snap is not None:
            hit = engine._resume(prompt) if draft else None
            hit_len = len(hit[0]) if hit else 0
            plan, why = build_plan(old_ids, prompt, sess.snap, sess.sid, rt.cfg,
                                   resume_len=hit_len, log=_log)
            sess.ids = list(prompt)           # SCR parity: the tracked ids follow the newest prompt
            if plan is None and rt.cfg.debug:
                _log(f"no plan: {why} n_new={len(prompt)} n_old={len(old_ids)} hit={hit_len}")
    except Exception as e:                    # noqa: BLE001
        _log(f"plan error (serving plain): {e}")
        return None
    if plan is None:
        return None
    if rt.cfg.trace:
        _log(f"plan: hit={hit_len} {describe(plan)}")
    return _generate_planned(engine, prompt, max_tokens, sampling, on_tokens, stop_eos, constraint,
                             plan, rt, scr_sid)


def _generate_planned(engine, prompt, max_tokens, sampling, on_tokens, stop_eos, constraint,
                      plan, rt, scr_sid):
    from tensorfold.cuda.sampling import sample_rows
    from ..decode import draft_decode
    from ..forward import _mm
    from ..prefill import prefill_state

    t0 = time.perf_counter()
    base = plan.base
    st = build_state_from_snapshot(plan.snapshot, len(prompt) + max_tokens, engine.w, pos=plan.base)
    st.limit = engine.context_window
    n = len(prompt)
    pending = None
    served = 0

    def first_of(st, normed):
        logits = _mm(normed, engine.w.head)
        if constraint is not None:
            constraint.mask(logits)
        pending = sample_rows(logits, [n], sampling)[0]
        if constraint is not None:
            constraint.advance([pending])
        return pending

    while st.pos < n:
        sp = plan.pending_splice(st.pos) if plan is not None else None
        if sp is not None:
            try:
                i = plan.splices.index(sp) + 1
                ok = execute_splice(st, sp, plan.snapshot, prompt, rt.inv_freq, log=_log,
                                    rid=scr_sid, blocks=f"blocks={i}/{len(plan.splices)}")
                if ok:
                    served += sp.n
            except Exception as e:            # noqa: BLE001
                _log(f"splice error: {e}")
                ok = False
            if not ok:
                plan = None                   # plain prefill of the tail; rows so far stay valid
            continue
        stop = plan.segment_stop(st.pos, n) if plan is not None else n
        normed = prefill_state(engine.w, prompt[:stop], st, draft=None, vision=None)
        if stop >= n:
            pending = first_of(st, normed)
    if pending is None:                       # unreachable: every plan's last segment is a prefill
        raise RuntimeError("scr planned prefill produced no first token")
    prefill_s = time.perf_counter() - t0

    if on_tokens([pending]):
        return {"prefill_s": prefill_s, "cached": base, "scr": True,
                "scr_sid": scr_sid, "scr_base": base, "scr_relocated": served}
    t1 = time.perf_counter()
    result = draft_decode(engine.w, st, prompt, pending, max_tokens, sampling, None,
                          max_rows=engine.max_rows, tree_rows=engine.tree_rows, allow_copy=False,
                          stop_eos=stop_eos, on_tokens=on_tokens, inplace=True)
    decode_s = time.perf_counter() - t1
    try:
        snap = snapshot_from_state(st, list(prompt) + list(result.tokens), log=_log)
        if snap is not None and snap.pos > 0:
            rt.store.refresh(scr_sid, snap)
    except Exception as e:                    # noqa: BLE001
        _log(f"snapshot error: {e}")
    return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds,
            "cached": base, "drafts": False,
            "min_rows": min(result.widths, default=0), "drafted": result.drafted_rows,
            "accepted": result.accepted_drafts,
            "scr": True, "scr_sid": scr_sid, "scr_base": base, "scr_relocated": served}