"""Size draft windows and shared rounds from per-depth acceptance, node landing probabilities, and round costs."""

from __future__ import annotations

from typing import Any


def extend_costs(costs: dict[int, float], rows: int) -> dict[int, float]:
    """Interpolate forward costs between known widths and extrapolate wider rounds at the widest known cost per row."""

    if not costs:
        return {}
    measured = sorted(costs)
    widest = measured[-1]
    out: dict[int, float] = {}
    for total in range(1, int(rows) + 1):
        if total in costs:
            out[total] = costs[total]
        elif total > widest:
            out[total] = costs[widest] * total / widest
        else:
            above = next(w for w in measured if w > total)
            below = max((w for w in measured if w < total), default=None)
            out[total] = costs[above] if below is None else (
                costs[below] + (costs[above] - costs[below]) * (total - below) / (above - below))
    return out


class DraftDepth:
    """Draft depth and row allocation for ``FamilyRounds``."""

    # per-depth acceptance: the prior, the weight of the newest round, and rounds between probes one deeper
    depth_prior = (0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    depth_rate = 0.15
    depth_probe_every = 8
    # a round's wall time by depth: measured rounds replace the load-time estimate at this rate
    cost_rate = 0.2

    def _depth_rates(self, stream: Any) -> list[float]:
        state = self._depth_state.get(stream.stream_id)
        if state is None:
            state = {"p": [float(p) for p in self.depth_prior[:max(1, self.most_drafts)]], "rounds": 0}
            self._depth_state[stream.stream_id] = state
        return state["p"]

    def _observe_depth(self, stream: Any, proposed: int, accepted: int) -> None:
        """Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves toward whether it was kept."""

        rates = self._depth_rates(stream)
        for j in range(min(proposed, len(rates))):
            if accepted < j:
                break
            rates[j] += self.depth_rate * ((1.0 if accepted > j else 0.0) - rates[j])

    def _observe_cost(self, drafts: int, ms: float, *, initializing: bool = False) -> None:
        # a stream's first round carries the prefill-to-decode switch: its time stays in round_stats, not the costs
        if drafts <= 0 or initializing:
            return
        before = self._round_ms.get(drafts)
        self._round_ms[drafts] = ms if before is None else before + self.cost_rate * (ms - before)

    def _round_cost(self, drafts: int) -> float | None:
        """Use round wall time when available, else the load-time forward cost plus head steps."""

        if drafts in self._round_ms:
            return self._round_ms[drafts]
        forward = self.family_costs.get(drafts + 1)
        if forward is None:
            return None
        return forward + self.mtp_step_ms * drafts

    def _depth(self, stream: Any) -> int:
        """Choose the most expected tokens per unit cost and periodically probe one depth farther to refresh acceptance estimates."""

        # Verify at least one draft; commit and budget cuts discard tokens beyond the available room.
        most = min(self.most_drafts, max(1, stream.draft_room - 1))
        if most <= 0:
            return 0
        rates = self._depth_rates(stream)
        if not self.family_costs:
            rate = rates[0]
            return max(1, min(most, 1 if rate < 0.8 else 2 if rate < 0.9 else 3))
        best, best_rate = 1, -1.0
        expected = run = 1.0
        for d in range(1, most + 1):
            cost = self._round_cost(d)
            if cost is None:
                break
            run *= rates[d - 1] if d - 1 < len(rates) else rates[-1]
            expected += run
            if expected / cost > best_rate:
                best, best_rate = d, expected / cost
        state = self._depth_state[stream.stream_id]
        state["rounds"] += 1
        if best < most and state["rounds"] % self.depth_probe_every == 0:
            best += 1
        return best

    copy_rate = 0.94                   # a copied token lands this often (8+ matching tokens behind it)
    draft_slack = 2                    # drafts a stream offers past what its last shared round granted it

    def _allocate(self, plans: list[list[Any]]) -> None:
        """Trim draft prefixes by landing probability to the shared width with the most expected tokens per unit cost."""

        import mlx.core as mx

        from tensorfold.engine.allocate import allocate, chain_probabilities
        from tensorfold.engine.lane_engine import sanitize_tree

        fixed, probs = [], []
        for stream, cache, _, kind, drafts, forced, parents in plans:
            n = int(drafts.shape[0]) if isinstance(drafts, mx.array) else len(drafts)
            if kind == "forced" or not n:
                fixed.append(1 + (n if kind == "forced" else 0))
                probs.append([])
            elif kind == "copy":
                fixed.append(1)
                probs.append(chain_probabilities([self.copy_rate], n))
            else:
                fixed.append(1)
                probs.append(self._node_chances(stream, cache, n))
        counts = allocate(fixed, probs, self.shared_costs, self._overhead(len(plans)),
                          max(len(plans), int(self.batch_rows)))
        for plan, chances, count in zip(plans, probs, counts):
            if plan[3] == "head":
                self._granted[plan[0].stream_id] = count
            if not chances or count == len(chances):
                continue
            if count == 0:
                plan[3], plan[4], plan[6] = "none", [], None
            elif plan[6] is not None:
                plan[4], plan[6] = sanitize_tree(plan[4], plan[6], count)
            else:
                plan[4] = plan[4][:count]

    def _node_chances(self, stream: Any, cache: list[Any], count: int) -> list[float]:
        """Use model node probabilities or derive chain probabilities from per-depth acceptance."""

        from tensorfold.engine.allocate import chain_probabilities

        if self.node_probabilities:
            chances = self.model.draft_probabilities(cache)
            if chances is not None and len(chances) >= count:
                return [float(p) for p in chances[:count]]
        return chain_probabilities(self._depth_rates(stream), count)

    def _draft_budgets(self, streams: list[Any]) -> list[int]:
        """Budget all nodes for heads providing probabilities; otherwise allocate from per-depth acceptance."""

        from tensorfold.engine.allocate import allocate, chain_probabilities

        if self.node_probabilities:
            # a stream drafts a little past what its last shared round granted it (the lattice's block follows)
            return [self._head_depth(s, min(self._head_depth(s), self._granted.get(s.stream_id, self.most_drafts)
                                            + self.draft_slack)) for s in streams]
        most = [min(self.most_drafts, max(1, s.draft_room - 1)) for s in streams]
        probs = [chain_probabilities(self._depth_rates(s), m) for s, m in zip(streams, most)]
        counts = allocate([1] * len(streams), probs, self.shared_costs, self._overhead(len(streams)),
                          max(len(streams), int(self.batch_rows)))
        return [self._head_depth(s, min(m, max(1, c))) for s, m, c in zip(streams, most, counts)]

    def _overhead(self, streams: int) -> float:
        """A shared round's milliseconds beyond its forward at this many streams (nearest measured count)."""

        if not self._overhead_ms:
            return 8.0
        nearest = min(self._overhead_ms, key=lambda n: abs(n - streams))
        return self._overhead_ms[nearest]

    def _observe_overhead(self, streams: int, rows: int, ms: float) -> None:
        forward = self.shared_costs.get(rows)
        if forward is None:
            return
        extra = max(0.0, ms - forward)
        before = self._overhead_ms.get(streams)
        self._overhead_ms[streams] = extra if before is None else before + 0.2 * (extra - before)
