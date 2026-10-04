"""CPU-only vectors for the Zig lane core, recorded from the Python engine's own functions (no MLX)."""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path
from typing import Any

import numpy as np


def bits(x: float) -> int:
    """A double's exact bits: the replay compares numbers without decimal round trips."""

    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def fbits(x: Any) -> int:
    return int(np.float32(x).view(np.uint32))


def line(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(line(row) + "\n")


class StubStream:
    """What the depth rule reads of a stream."""

    def __init__(self, sid: str) -> None:
        self.stream_id, self.finished, self.force, self.copy, self.draft_room = sid, False, [], False, 12


def depth_vectors(rng: random.Random) -> list[dict[str, Any]]:
    from tensorfold.engine.allocate import allocate, chain_probabilities
    from tensorfold.engine.family_depth import DraftDepth, extend_costs
    from tensorfold.engine.lane_family import FamilyRounds

    class Stub(DraftDepth):
        _head_depth = FamilyRounds._head_depth

        def _copy_proposal(self, stream: Any, min_match: Any = None) -> list[int]:
            return [1, 2] if stream.copy else []

    out = []
    for case in range(48):
        stub = Stub()
        stub.most_drafts = rng.choice([0, 1, 2, 3, 4, 4, 6, 8, 15])
        width = stub.most_drafts + 1 + rng.randint(0, 3)
        costs = {} if rng.random() < 0.1 else {
            w: round(rng.uniform(4, 6) + w * rng.uniform(0.2, 1.8) + rng.uniform(-0.3, 0.3), 3)
            for w in range(1, width + 1) if w == 1 or rng.random() > 0.15}
        stub.family_costs = costs
        stub.mtp_step_ms = 0.0 if rng.random() < 0.2 else rng.uniform(0.1, 1.5)
        stub.plain_guard = rng.random() < 0.4
        if rng.random() < 0.5:
            stub.depth_prior = tuple(round(rng.uniform(0.3, 0.95), 3) for _ in range(rng.randint(1, 9)))
        stub.batch_rows = rng.choice([4, 8, 16, 32])
        timed = {w: round(3 + w * rng.uniform(0.3, 1.2), 3) for w in sorted(rng.sample(range(1, stub.batch_rows + 1),
                                                                                         rng.randint(1, 4)))}
        stub.shared_costs = {} if rng.random() < 0.15 else extend_costs(timed, stub.batch_rows)
        stub.node_probabilities = False
        stub._depth_state, stub._round_ms, stub._overhead_ms, stub._granted = {}, {}, {}, {}
        streams = [StubStream(f"s{i}") for i in range(rng.randint(1, 3))]
        config = {"most_drafts": stub.most_drafts, "family_costs": [[w, bits(c)] for w, c in sorted(costs.items())],
                  "mtp_step_ms": bits(stub.mtp_step_ms), "plain_guard": stub.plain_guard,
                  "depth_prior": [bits(p) for p in stub.depth_prior], "batch_rows": stub.batch_rows,
                  "shared_costs": [[w, bits(c)] for w, c in sorted(stub.shared_costs.items())],
                  "streams": [s.stream_id for s in streams]}
        ops = []
        for _ in range(160):
            s = rng.choice(streams)
            pick = rng.random()
            if pick < 0.12:
                s.draft_room = rng.choice([-1, 0, 1, 2, 3, 5, 9, 40])
                s.force = [7] if rng.random() < 0.08 else []
                s.copy = rng.random() < 0.1
                s.finished = rng.random() < 0.03
                op = {"op": "set", "stream": s.stream_id, "draft_room": s.draft_room, "force": len(s.force),
                      "copy": s.copy, "finished": s.finished}
            elif pick < 0.32:
                op = {"op": "depth", "stream": s.stream_id, "result": stub._depth(s)}
            elif pick < 0.42:
                budget = None if rng.random() < 0.5 else rng.randint(0, 6)
                op = {"op": "head", "stream": s.stream_id, "budget": budget, "result": stub._head_depth(s, budget)}
            elif pick < 0.6:
                proposed, = rng.choices(range(0, 10), k=1)
                accepted = rng.randint(0, proposed)
                stub._observe_depth(s, proposed, accepted)
                op = {"op": "observe_depth", "stream": s.stream_id, "proposed": proposed, "accepted": accepted}
            elif pick < 0.76:
                drafts, ms, init = rng.randint(-1, 6), rng.uniform(3, 30), rng.random() < 0.1
                own = rng.random() < 0.7
                stub._observe_cost(drafts, ms, initializing=init, stream=s if own else None)
                op = {"op": "observe_cost", "stream": s.stream_id if own else None, "drafts": drafts, "ms": bits(ms),
                      "init": init}
            elif pick < 0.82:
                d = rng.randint(0, 8)
                cost = stub._round_cost(d, s)
                op = {"op": "round_cost", "stream": s.stream_id, "drafts": d, "result": None if cost is None else bits(cost)}
            elif pick < 0.9:
                chosen = rng.sample(streams, rng.randint(1, len(streams)))
                op = {"op": "budgets", "streams": [c.stream_id for c in chosen],
                      "result": [int(x) for x in stub._draft_budgets(chosen)]}
            elif pick < 0.95:
                n, rows, ms = rng.randint(1, 4), rng.randint(1, stub.batch_rows + 2), rng.uniform(2, 40)
                stub._observe_overhead(n, rows, ms)
                op = {"op": "observe_overhead", "streams": n, "rows": rows, "ms": bits(ms)}
            else:
                n = rng.randint(1, 5)
                op = {"op": "overhead", "streams": n, "result": bits(stub._overhead(n))}
            op["state"] = depth_state(stub, streams)
            ops.append(op)
        out.append({"kind": "depth", "case": case, "config": config, "ops": ops})
    for case in range(200):
        streams = rng.randint(1, 5)
        fixed = [rng.choice([1, 1, 1, 2, 4]) for _ in range(streams)]
        probs = [chain_probabilities([rng.uniform(0.2, 0.98) for _ in range(rng.randint(1, 4))], rng.randint(0, 12))
                 for _ in range(streams)]
        if rng.random() < 0.2:                       # ties between streams resolve by stream order
            probs = [list(probs[0]) for _ in range(streams)]
        rows = rng.choice([8, 16, 32, 64])
        timed = {w: round(2 + w * rng.uniform(0.2, 1.5), 3) for w in rng.sample(range(1, rows + 1), rng.randint(1, 5))}
        costs = {} if rng.random() < 0.15 else extend_costs(timed, rows)
        overhead = rng.uniform(0, 10)
        most = max(streams, rng.choice([rows, rows // 2, 4]))
        out.append({"kind": "allocate", "case": case, "fixed": fixed, "probs": [[bits(p) for p in ps] for ps in probs],
                    "costs": [[w, bits(c)] for w, c in sorted(costs.items())], "timed": [[w, bits(c)] for w, c in sorted(timed.items())],
                    "rows": rows, "overhead": bits(overhead), "max_rows": most,
                    "result": allocate(fixed, probs, costs, overhead, most)})
    return out


def depth_json(st: dict[str, Any] | None) -> dict[str, Any] | None:
    if st is None:
        return None
    return {"p": [bits(x) for x in st["p"]], "rounds": st["rounds"], "plain": st.get("plain"), "wait": st.get("wait"),
            "ms": [[d, bits(v)] for d, v in sorted(st["ms"].items())] if "ms" in st else None}


def depth_state(stub: Any, streams: list[Any]) -> dict[str, Any]:
    per = [depth_json(stub._depth_state.get(s.stream_id)) for s in streams]
    return {"round_ms": [[d, bits(v)] for d, v in sorted(stub._round_ms.items())],
            "overhead_ms": [[n, bits(v)] for n, v in stub._overhead_ms.items()], "streams": per}


def proposer_vectors(rng: random.Random) -> list[dict[str, Any]]:
    from tensorfold.engine.lane_engine import SuffixLookupProposer

    out = []
    for case in range(40):
        ngram = rng.choice([1, 2, 3, 3, 4])
        params = {"ngram": ngram, "min_match": ngram + rng.choice([0, 1, 3, 5]),
                  "max_extension": rng.choice([8, 64]), "silence_rounds": rng.choice([2, 16]),
                  "window": rng.choice([1, 4])}
        p = SuffixLookupProposer(**params)
        vocab = rng.choice([6, 20, 300])
        context = [rng.randrange(vocab) for _ in range(rng.choice([2, 30, 300, 5000]))]
        first = list(context)
        ops = []
        for _ in range(120):
            pick = rng.random()
            if pick < 0.5:                            # the reply grows: fresh tokens or a copied span
                if rng.random() < 0.5 and len(context) > 8:
                    start = rng.randrange(len(context) - 4)
                    span = context[start:start + rng.randint(2, 40)]
                else:
                    span = [rng.randrange(vocab) for _ in range(rng.randint(1, 6))]
                context.extend(span)
                ops.append({"op": "grow", "tokens": span})
            elif pick < 0.85:
                max_draft = rng.choice([-1, 0, 1, 2, 7, 15, 63])
                result = [int(t) for t in p.propose(context, max_draft)]
                ops.append({"op": "propose", "max_draft": max_draft, "result": result,
                            "last_match": int(p.last_match), "confident": bool(getattr(p, "last_confident", False)),
                            "state": proposer_state(p)})
            elif pick < 0.97:
                proposed = rng.randint(0, 8)
                accepted = 0 if rng.random() < 0.5 else rng.randint(0, proposed)
                p.observe(proposed, accepted)
                ops.append({"op": "observe", "proposed": proposed, "accepted": accepted, "state": proposer_state(p)})
            else:                                      # a new request on the same proposer: the index rebuilds
                context = [rng.randrange(vocab) for _ in range(rng.randint(5, 60))]
                ops.append({"op": "reset", "tokens": list(context)})
        out.append({"kind": "proposer", "case": case, "params": params, "context": first, "ops": ops})
    return out


def proposer_state(p: Any) -> dict[str, Any]:
    return {"silent_for": p._silent_for, "recent": list(p._recent), "proposals": p.proposals,
            "proposed_tokens": p.proposed_tokens, "judged_tokens": p.judged_tokens,
            "accepted_tokens": p.accepted_tokens, "silenced_rounds": p.silenced_rounds}


def sampling_vectors(rng: random.Random) -> list[dict[str, Any]]:
    from tensorfold.engine.exact_sampling import Sampling, choose, choose_rows, seed_for, uniform

    out = []
    for case in range(60):
        tokens = [rng.randrange(200_000) for _ in range(rng.choice([0, 1, 5, 300]))]
        salt = rng.choice([0, 0, 7, -3, 123456789])
        out.append({"kind": "seed", "tokens": tokens, "salt": salt, "result": seed_for(tokens, salt)})
    for case in range(60):
        seed = rng.getrandbits(64)
        position = rng.randrange(1 << 20)
        ids = np.array([rng.randrange(1 << 31) for _ in range(8)], dtype=np.int64)
        out.append({"kind": "uniform", "seed": seed, "position": position, "ids": ids.tolist(),
                    "result": [bits(u) for u in uniform(seed, position, ids)]})

    def settings() -> Any:
        return Sampling(seed=rng.getrandbits(63), temperature=rng.choice([0.05, 0.3, 0.7, 1.0, 1.5, 2.0]),
                        top_k=rng.choice([0, 1, 5, 20, 64]), top_p=rng.choice([0.5, 0.9, 0.95, 1.0, 0.0]),
                        min_p=rng.choice([0.0, 0.0, 0.05, 0.2]))

    def candidates(width: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.array([rng.gauss(0, 3) for _ in range(width)], dtype=np.float32)
        if rng.random() < 0.4:                         # bf16-like ties
            values = values.astype(np.float16).astype(np.float32)
        ids = np.array(rng.sample(range(1 << 17), width), dtype=np.int64)
        return values, ids

    for case in range(300):
        s = settings()
        values, ids = candidates(rng.choice([1, 2, 8, 28, 72, 264]))
        position = rng.randrange(1 << 16)
        out.append({"kind": "choose", "values": [fbits(v) for v in values], "ids": ids.tolist(), "position": position,
                    "sampling": sampling_json(s), "result": choose(values, ids, position, s)})
    for case in range(80):
        s = settings()
        rows, width = rng.randint(1, 6), rng.choice([1, 4, 28, 72, 264])
        pairs = [candidates(width) for _ in range(rows)]
        values, ids = np.stack([v for v, _ in pairs]), np.stack([i for _, i in pairs])
        positions = [rng.randrange(1 << 16) for _ in range(rows)]
        out.append({"kind": "choose_rows", "values": [[fbits(v) for v in row] for row in values],
                    "ids": ids.tolist(), "positions": positions, "sampling": sampling_json(s),
                    "result": choose_rows(values, ids, positions, s)})
    return out


def sampling_json(s: Any) -> dict[str, Any]:
    return {"seed": int(s.seed), "temperature": bits(s.temperature), "top_k": int(s.top_k), "top_p": bits(s.top_p),
            "min_p": bits(s.min_p)}


def stream_vectors(rng: random.Random) -> list[dict[str, Any]]:
    from tensorfold.engine.lane_engine import LaneStream

    out = []
    for case in range(80):
        think = rng.random() < 0.6
        close = (11, 12, 13) if think else ()
        stream = LaneStream(stream_id="t", prompt_ids=[1, 2, 3], max_new_tokens=rng.choice([1, 5, 12, 40]),
                            eos_ids=frozenset({2, 9}), think_budget=rng.choice([0, 3, 8]) if think else 0,
                            think_close=close, think_end=12 if think else -1, think_open=think)
        spec = {"max_new": stream.max_new_tokens, "eos": [2, 9], "think_budget": stream.think_budget,
                "think_close": list(close), "think_end": stream.think_end, "think_open": stream.think_open}
        ops = []
        for _ in range(30):
            pick = rng.random()
            if pick < 0.5:
                tokens = [rng.choice([2, 9, 12, 5, 6, 7, 8, 30, 31]) if rng.random() < 0.2 else rng.randrange(20, 40)
                          for _ in range(rng.randint(1, 5))]
                landed = stream.commit(tokens)
                ops.append({"op": "commit", "tokens": tokens, "result": landed})
            elif pick < 0.75:
                tokens = [rng.choice([12, 20, 21, -1]) for _ in range(rng.randint(1, 5))]
                ops.append({"op": "cut", "tokens": tokens, "result": stream.think_cut(tokens)})
            elif pick < 0.85 and stream._budget_active():
                ops.append({"op": "close", "result": stream.start_close()})
            else:
                ops.append({"op": "room", "result": stream.draft_room})
            ops[-1]["state"] = {"emitted": list(stream.emitted), "finished": stream.finished,
                                "reason": stream.finish_reason, "think_open": stream.think_open,
                                "force": list(stream.force)}
        out.append({"kind": "stream", "case": case, "spec": spec, "ops": ops})
    return out


def run_units(out: Path, seed: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for name, make in (("depth", depth_vectors), ("proposer", proposer_vectors), ("sampling", sampling_vectors),
                       ("stream", stream_vectors)):
        rows = make(random.Random(f"{seed}-{name}"))
        write(out / f"{name}.jsonl", rows)
        print(f"[units] {name}: {len(rows)} cases", flush=True)
    return 0
