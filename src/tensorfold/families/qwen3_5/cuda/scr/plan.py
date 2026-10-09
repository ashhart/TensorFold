"""Relocation plan: diff opcodes -> spans to relocate, with SCR's gates.

A plan serves a request as: copy snapshot rows ``[0, base)`` verbatim (positions
unchanged), prefill the new text between spans, and for each ``Splice`` copy the
span's rows from the snapshot to ``[at, at+n)`` — re-rotating keys by the constant
position shift — leaving the span's last ``cfg.min_extend`` rows (and everything
after the last span) to plain prefill, so linear-attention layers always run on
real tokens before the next step."""

from dataclasses import dataclass, field

from .diff import diff_opcodes


@dataclass
class Splice:
    at: int          # new-prompt position the relocated rows land at (== st.pos when it runs)
    old: int         # row index in the snapshot (== the old turn's token index)
    n: int           # relocated rows: span length minus the min_extend tail


@dataclass
class Plan:
    base: int                       # shared leading prefix, served from snapshot rows [0, base)
    splices: list                   # ordered by `at`, strictly increasing
    snapshot: object                # the pinned Snapshot object (its ids cover `old`)
    session_id: str
    relocated: int = 0              # sum of n over splices (plan-time)
    served: int = 0                 # rows actually relocated (splices executed so far)
    gain: int = 0                   # reused tokens over the engine's own resume: (base - resume_len) + relocated
    new_ids: list = field(default_factory=list)

    def pending_splice(self, pos: int):
        """The splice due exactly at `pos` (the fill reached it), or None."""
        for sp in self.splices:
            if sp.at == pos:
                return sp
            if sp.at > pos:
                return None
        return None

    def segment_stop(self, pos: int, prompt_len: int) -> int:
        """Where the next prefill segment ends: the next splice point or the prompt end."""
        for sp in self.splices:
            if sp.at > pos:
                return sp.at
        return prompt_len


def build_plan(old_ids, new_ids, snapshot, session_id: str, cfg, resume_len: int = 0,
               log=None) -> tuple:
    """``(Plan | None, reason)`` from the diff of the previous turn's committed ids
    against the new prompt. `snapshot.ids` must equal `old_ids` (the caller passes
    the snapshot's own ids)."""
    if snapshot is None or list(snapshot.ids) != list(old_ids):
        return None, "stale-session"

    ops = diff_opcodes(old_ids, new_ids, fastdiff=cfg.fastdiff, cdc_target=cfg.cdc_target,
                       fastdiff_min=cfg.fastdiff_min, log=log)
    if not ops or ops[0][0] != "equal" or ops[0][1] != 0 or ops[0][3] != 0:
        return None, "no-leading-run"
    base = ops[0][4]
    if base < cfg.min_base:
        return None, f"base<{cfg.min_base}"
    if base >= len(new_ids):
        # the snapshot covers the whole prompt: nothing to prefill and no first
        # token to sample (the engine's own resume leaves >= 1 token too)
        return None, "base-covers-prompt"

    # candidate spans: every equal opcode after the leading run
    kext = cfg.min_extend
    cands = []
    for k, (t, ci1, ci2, cj1, cj2) in enumerate(ops):
        if k == 0 or t != "equal":
            continue
        n_c = ci2 - ci1
        if n_c < cfg.min_tail or n_c - kext < cfg.min_tail:
            continue
        if len(new_ids) - (cj2 - kext) < kext:
            continue
        # identity guard: the rows relocated must equal the new prompt's tokens there
        if old_ids[ci1:ci2 - kext] != new_ids[cj1:cj2 - kext]:
            if log:
                log(f"TAIL ROWS MISALIGNED: old {ci1}:{ci2 - kext} != new {cj1}:{cj2 - kext} -> next candidate")
            continue
        cands.append((n_c, ci1, ci2, cj1, cj2))

    # K longest first, then back into prompt order; min gap between relocated spans
    cands.sort(key=lambda c: -c[0])
    acc = cands[:cfg.max_blocks]
    sched = sorted(acc, key=lambda c: c[3])
    if cfg.mb_min_gap > 0 and len(sched) > 1:
        keep, end_prev = [sched[0]], sched[0][4] - kext
        for b in sched[1:]:
            if b[3] - end_prev < cfg.mb_min_gap:
                continue
            keep.append(b)
            end_prev = b[4] - kext
        sched = keep

    splices = [Splice(at=cj1, old=ci1, n=ci2 - kext - ci1) for _, ci1, ci2, cj1, cj2 in sched]
    relocated = sum(sp.n for sp in splices)
    # gain over the engine's own exact resume: the leading run beyond it (e.g. the
    # previous reply, which TensorFold's prompt-state cache does not cover) plus the
    # relocated spans. A splice-free plan is a pure append/output-reuse turn.
    gain = relocated + base - resume_len
    if gain < cfg.min_gain:
        return None, f"gain<{cfg.min_gain}"

    return Plan(base=base, splices=splices, snapshot=snapshot, session_id=session_id,
                relocated=relocated, gain=gain, new_ids=list(new_ids)), "ok"


def describe(plan) -> str:
    return (f"sid={plan.session_id} base={plan.base} blocks={len(plan.splices)} "
            f"relocated={plan.relocated} gain={plan.gain} stops={[sp.at for sp in plan.splices]} "
            f"rows={[sp.n for sp in plan.splices]} n_new={len(plan.new_ids)} "
            f"n_old={len(plan.snapshot.ids)}")