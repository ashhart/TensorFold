"""Which drafter each GLM round uses, MTP chains or DFlash2 blocks, and the drafted decoding that follows it."""

from __future__ import annotations

import time

import torch

from tensorfold.engine.exact_sampling import Sampling

from .decode import DecodeResult, DepthPolicy, Engine, _sync, absorb, draft
from .forward import commit


def _other(arm: str) -> str:
    return "f" if arm == "m" else "m"


class DrafterChoice:
    """MTP chains ("m") or DFlash2 blocks ("f") a round, by tokens committed a model-ms; both ranks agree."""

    def __init__(self, costs: dict, *, first: str, explore: int = 2, every: int = 8, margin: float = 0.03,
                 window: int = 6, recheck: int = 3) -> None:
        self.costs = costs
        self.first = first
        self.explore, self.every, self.margin, self.window, self.recheck = explore, every, margin, window, recheck
        self.rounds: list[tuple[str, int, float]] = []       # (drafter, tokens committed, model ms)
        self.choice = first
        self.run = 0
        self.since = {"m": 0, "f": 0}      # a drafter's rate counts its rounds from here: left, it is judged afresh

    def cost(self, arm: str, rows: int, steps: int, backlog: int) -> float:
        """Model ms of a round: the verify window, then MTP steps or a DFlash2 block, plus the backlog's catch-up."""

        c = self.costs
        verify = c["verify"][min(rows, len(c["verify"])) - 1]
        if arm == "m":
            return verify + c["mtp"] + c["mtp_step"] * max(steps - 1, 0) + c["mtp_row"] * max(backlog - 1, 0)
        return verify + c["block"] + c["taps_row"] * backlog

    def rate(self, arm: str) -> float | None:
        rs = [r for r in self.rounds[self.since[arm]:] if r[0] == arm][-self.window:]
        return sum(r[1] for r in rs) / sum(r[2] for r in rs) if rs else None

    def pick(self) -> str:
        n = len(self.rounds)
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return _other(self.first)
        cur = self.choice
        rc, ro = self.rate(cur), self.rate(_other(cur))
        need = 1.0 if n == 2 * self.explore else 1.0 + self.margin      # no bias at the first choice
        if ro is not None and (rc is None or ro > rc * need):
            self.since[cur] = n            # a bad stretch that forced the switch no longer counts against it
            self.choice = cur = _other(cur)
            self.run = max(0, self.every - self.recheck) if self.recheck else 0     # and it is probed again soon
        if self.every and self.run >= self.every:
            self.run = 0
            return _other(cur)
        self.run += 1
        return cur

    def record(self, arm: str, rows: int, steps: int, backlog: int, keep: int) -> None:
        self.rounds.append((arm, keep, self.cost(arm, rows, steps, backlog)))


@torch.no_grad()
def auto_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                choice: DrafterChoice | None,
                m_policy: DepthPolicy, f_policy: DepthPolicy, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Drafted decoding with ``choice`` picking each round's drafter; drafts only propose, so it equals serial."""

    w, st, b = e.w, e.st, e.buf
    cap = 256                               # backlog rows a drafter may owe before it absorbs them anyway
    m_rows = torch.empty((cap, w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    m_rows[:1].copy_(e.last_hidden)
    m_next: list[int] = [pending]
    f_taps = None
    n_f = 0
    if drafter is not None:
        f_taps = torch.empty((cap, len(b.taps) * w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    arms: list[str] = []
    last = {"m": (0, 0), "f": (0, 0)}
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        arm = choice.pick() if choice is not None else "m"
        room = count - len(out)
        t0 = time.perf_counter()
        if arm == "m":
            backlog = len(m_next)
            depth = max(1, min(m_policy.next(*last["m"]), room))
            drafts = draft(e, m_rows[:backlog], m_next, st.pos + 1, depth, sampling, m_policy.confidence)
            steps = 1 + st.mtp_drafted
            m_next = []
        else:
            backlog = n_f
            if n_f:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            depth = max(1, min(f_policy.next(*last["f"]), room))
            drafts = drafter.propose(out[-1], depth, sampling, f_policy.confidence)
            steps = 0
        t1 = time.perf_counter()
        tokens = e.verify_window([out[-1]] + drafts)
        drafts = tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        # the kept rows join both backlogs (a full backlog is taken first)
        if len(m_next) + keep > cap:
            absorb(e, m_rows[:len(m_next)], m_next)
            m_next = []
        m_rows[len(m_next):len(m_next) + keep].copy_(e.main_hidden(slice(0, keep)))
        m_next.extend(sampled[:keep])
        if drafter is not None:
            if n_f + keep > cap:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            f_taps[n_f:n_f + keep].copy_(e.tap_rows(keep))
            n_f += keep
        t5 = time.perf_counter()
        if choice is not None:
            choice.record(arm, R, steps, backlog, keep)
        last[arm] = (len(drafts), keep - 1)
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        arms.append(arm)
        e.follow(sampled[:keep])
        out.extend(sampled[:keep])
        halt = on_tokens is not None and on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        if halt:                                # True: the caller stops the reply here
            break
    _sync(w)
    seconds = time.perf_counter() - start
    if drafter is not None and n_f:
        drafter.add_taps(f_taps[:n_f])          # DFlash2's context ends where the committed rows end
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, stages, depths, keeps, "".join(arms),
                        m_rows[:len(m_next)])
