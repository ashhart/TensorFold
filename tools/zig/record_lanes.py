"""Record the Python lane engine's decisions and inputs as JSONL for the Zig replay: keys sorted, floats as bits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lanes_units import bits, depth_json, line, proposer_state, run_units, sampling_json  # noqa: E402


def held_count(held: Any) -> int | None:
    """Drafts a stream holds for its next round (None: no entry, -1: a prefill draft not settled yet)."""

    if held is None:
        return None
    if callable(held):
        return -1
    if isinstance(held, tuple):
        held = held[0]
    shape = getattr(held, "shape", None)
    return int(shape[0]) if shape is not None else len(held)


def recording_engine(base: type) -> type:
    """``base`` (LaneEngine) with every round decision logged in the order Python makes it."""

    from tensorfold.engine.lane_engine import SuffixLookupProposer

    class RecordingEngine(base):
        def __init__(self, model: Any, events: list[dict[str, Any]], capture: Any = None, **kw: Any) -> None:
            self.events, self.capture = events, capture
            self._verifying, self._positions, self._paths, self._unread = False, [], {}, []
            self._head_seen = self._depth_seen = self._first_draw = None
            self._in_first, self._steps = False, 0
            super().__init__(model, **kw)

        def emit(self, **event: Any) -> dict[str, Any]:
            self.events.append(event)
            return event

        def _resolve(self) -> None:
            for event, token in self._unread:          # queued draws: read where the engine already synced
                event["token"] = int(token.reshape(-1)[0].item())
            self._unread = []

        def _finish(self, s: Any) -> None:
            self.emit(ev="finish", stream=s.stream_id, reason=s.finish_reason, emitted=list(s.emitted),
                      rounds=s.rounds, drafted=s.drafted, accepted=s.accepted, cache_len=s.cache_len)

        def add_stream(self, stream: Any, **kw: Any) -> None:
            self.emit(ev="add", stream=stream.stream_id)
            super().add_stream(stream, **kw)
            self._resolve()
            if stream.finished:
                self._finish(stream)

        def step(self) -> dict[str, list[int]]:
            self.emit(ev="step", index=self._steps)
            self._steps += 1
            before = [s for s, _ in self._live]
            landed = super().step()
            self._resolve()
            for s in before:
                if s.finished:
                    self._finish(s)
            self.events.append(self.state())
            return landed

        def state(self) -> dict[str, Any]:
            streams = []
            for s, _ in self._live:
                sid, p = s.stream_id, s.proposer
                depth = depth_json(self._depth_state.get(sid))
                proposer = None if not isinstance(p, SuffixLookupProposer) else {
                    **proposer_state(p), "last_match": int(p.last_match)}
                streams.append({"id": sid, "cache_len": s.cache_len, "emitted": len(s.emitted), "pending": list(s.pending),
                                "rounds": s.rounds, "drafted": s.drafted, "accepted": s.accepted, "force": list(s.force),
                                "depth": depth, "mode": self._mode.get(sid), "copy_width": self._copy_width.get(sid),
                                "served": self._served.get(sid), "granted": self._granted.get(sid),
                                "next": held_count(self._next.get(sid)), "inflight": sid in self._inflight,
                                "proposer": proposer})
            return {"ev": "state", "alone": self._alone, "drafted": self.drafted, "accepted": self.accepted,
                    "round_ms": [[d, bits(v)] for d, v in sorted(self._round_ms.items())],
                    "overhead_ms": [[n, bits(v)] for n, v in self._overhead_ms.items()],
                    "shared_rounds": self._shared_rounds, "streams": streams}

        def _draw(self, logits: Any, sampling: Any, positions: Any) -> Any:
            out = super()._draw(logits, sampling, positions)
            if self._in_first and self._first_draw is None:
                self._first_draw = out
            if self._verifying:
                self._positions.append([int(p) for p in positions])
                if self.capture is not None:
                    self.capture.add(logits, sampling, positions, out)
            return out

        def _draw_streams(self, logits: Any, streams: Any) -> Any:
            out = super()._draw_streams(logits, streams)
            self._positions.extend([int(p) for p in positions] for _, positions in streams)
            return out

        def _family_first(self, stream: Any, work: Any, hidden: Any, cached_tokens: int, row: int) -> Any:
            self._depth_seen, self._in_first, self._first_draw = [], True, None
            try:
                token = super()._family_first(stream, work, hidden, cached_tokens, row)
            finally:
                seen, self._depth_seen, self._in_first = self._depth_seen, None, False
            drawn = int(self._first_draw.reshape(-1)[0].item())
            value = int(token.reshape(-1)[0].item()) if hasattr(token, "reshape") else int(token)
            prompt = len(stream.prompt_ids)
            if self.family_mtp and stream.drafts:
                self.emit(ev="draft", stream=stream.stream_id, depth=seen[0], position=prompt + 1, follow=[value],
                          rows=None)
            self.emit(ev="first", stream=stream.stream_id, position=prompt, drawn=drawn, token=value)
            return token

        def _depth(self, stream: Any) -> int:
            depth = super()._depth(stream)
            if self._depth_seen is not None:
                self._depth_seen.append(depth)
            return depth

        def _head_depth(self, stream: Any, budget: int | None = None) -> int:
            depth = super()._head_depth(stream, budget)
            if self._head_seen is not None:
                self._head_seen.append(depth)
            return depth

        def _queue_next(self, stream: Any, cache: Any, token: Any) -> None:
            super()._queue_next(stream, cache, token)
            event = self.emit(ev="queue", stream=stream.stream_id, position=stream.cache_len, token=None)
            self._unread.append((event, self._inflight[stream.stream_id]))

        def _pipelined_round(self, stream: Any, cache: Any) -> Any:
            mode = self._mode.get(stream.stream_id)
            got, rows, keep = super()._pipelined_round(stream, cache)
            self.emit(ev="pipe", stream=stream.stream_id, mode=mode, after=self._mode.get(stream.stream_id),
                      got=list(got))
            return got, rows, keep

        def _land_inflight(self, stream: Any) -> list[int]:
            got = super()._land_inflight(stream)
            self.emit(ev="land", stream=stream.stream_id, got=list(got))
            return got

        def _family_round(self, stream: Any, cache: Any, copied: Any = None) -> Any:
            self._verifying, self._positions = True, []
            try:
                return super()._family_round(stream, cache, copied)
            finally:
                self._verifying = False

        def _family_round_streams(self, entries: Any) -> Any:
            self._positions = []
            return super()._family_round_streams(entries)

        def _take_turns(self, live: Any) -> Any:
            chosen = super()._take_turns(live)
            self.emit(ev="turns", streams=[s.stream_id for s, _ in chosen])
            return chosen

        def _plan_window(self, stream: Any, copied: Any = None) -> Any:
            kind, drafts, forced, parents = super()._plan_window(stream, copied)
            self.emit(ev="plan", stream=stream.stream_id, kind=kind, n=held_count(drafts), tree=parents is not None)
            return kind, drafts, forced, parents

        def _allocate(self, plans: list[list[Any]]) -> None:
            super()._allocate(plans)
            self.emit(ev="alloc", plans=[[p[3], held_count(p[4])] for p in plans])

        def _conclude(self, stream: Any, kind: str, forced: Any, sampled: Any, window: Any, rows_parents: Any) -> Any:
            forced_in = [int(t) for t in forced]
            got, path, cut, follow = super()._conclude(stream, kind, forced, sampled, window, rows_parents)
            chain = list(rows_parents) == [-1, *range(len(rows_parents) - 1)]
            self.emit(ev="round", stream=stream.stream_id, kind=kind, window=[int(t) for t in window],
                      parents=None if chain else [int(q) for q in rows_parents], positions=self._positions.pop(0),
                      sampled=[int(t) for t in sampled], forced=forced_in, path=[int(r) for r in path],
                      got=[int(t) for t in got], cut=cut)
            self._paths[stream.stream_id] = [int(r) for r in path]
            return got, path, cut, follow

        def _draft_late(self, stream: Any, cache: Any, position: int, follow: Any, rows: Any,
                        budget: int | None = None) -> None:
            self._head_seen = []
            try:
                super()._draft_late(stream, cache, position, follow, rows, budget=budget)
            finally:
                seen, self._head_seen = self._head_seen, None
            self.emit(ev="draft", stream=stream.stream_id, depth=seen[0], position=stream.cache_len + 1,
                      follow=[int(t) for t in follow], rows=self._paths.get(stream.stream_id))

        def _draft_budgets(self, streams: Any) -> list[int]:
            depths = super()._draft_budgets(streams)
            self.emit(ev="budgets", streams=[s.stream_id for s in streams], depths=[int(d) for d in depths])
            return depths

        def _observe_cost(self, drafts: int, ms: float, *, initializing: bool = False, stream: Any = None) -> None:
            self.emit(ev="cost", stream=None if stream is None else stream.stream_id, drafts=int(drafts),
                      init=bool(initializing), ms=bits(ms))
            super()._observe_cost(drafts, ms, initializing=initializing, stream=stream)

        def _observe_overhead(self, streams: int, rows: int, ms: float) -> None:
            self.emit(ev="overhead", streams=int(streams), rows=int(rows), ms=bits(ms))
            super()._observe_overhead(streams, rows, ms)

    return RecordingEngine


def wrap_draft_streams(model: Any, engine: Any, original: Any) -> None:
    """Log each stream's draft request where a shared round asks the model's batched head for it."""

    if not callable(original):
        return

    def draft_streams(caches, follows, rows, positions, samplings, depths):  # noqa: ANN001, ANN202
        ids = {id(c): s.stream_id for s, c in engine._live}
        for cache, follow, position, depth in zip(caches, follows, positions, depths):
            sid = ids[id(cache)]
            engine.emit(ev="draft", stream=sid, depth=int(depth), position=int(position),
                        follow=[int(t) for t in follow], rows=engine._paths.get(sid))
        return original(caches, follows, rows, positions, samplings, depths)

    model.draft_streams = draft_streams


def header(engine: Any, model: Any, name: str, specs: list[dict[str, Any]], kw: dict[str, Any]) -> dict[str, Any]:
    """The model attributes and engine arguments the Zig setup reads, and what Python derived from them."""

    def table(costs: Any) -> list[list[int]]:
        return [[int(w), bits(ms)] for w, ms in sorted((costs or {}).items())]

    hook = getattr(model, "draft_streams", None) or getattr(model, "speculate_streams", None)
    attrs = {"exact_width": int(getattr(model, "exact_width", 1) or 1),
             "first_copy_rows": int(getattr(model, "first_copy_rows", 0) or 0),
             "gpu_tokens": bool(getattr(model, "gpu_tokens", False)), "mtp": getattr(model, "mtp", None) is not None,
             "speculate": callable(getattr(model, "speculate", None)),
             "speculate_early": bool(getattr(model, "speculate_early", True)),
             "draft_prior": [bits(p) for p in (getattr(model, "draft_prior", None) or ())],
             "plain_guard": bool(getattr(model, "plain_guard", False)),
             "drafts": int(getattr(model, "drafts", 1) or 0), "window_costs": table(getattr(model, "window_costs", None)),
             "mtp_step_ms": bits(float(getattr(model, "mtp_step_ms", 0.0) or 0.0)),
             "streams_exact": getattr(model, "streams_exact", True) is True,
             "hidden_rows": callable(getattr(model, "hidden_rows", None)),
             "batch_rows": int(getattr(model, "batch_rows", 0) or 0), "max_streams": int(getattr(model, "max_streams", 0) or 0),
             "shared_costs": table(getattr(model, "shared_costs", None)),
             "draft_probabilities": callable(getattr(model, "draft_probabilities", None)),
             "draft_streams": callable(hook)}
    derived = {"family_width": engine.family_width, "max_copy": engine.max_copy, "first_copy": engine.first_copy,
               "base_width": engine.base_width, "family_mtp": engine.family_mtp,
               "speculate_early": engine.speculate_early, "pipelined": engine.pipelined,
               "plain_guard": engine.plain_guard, "most_drafts": engine.most_drafts,
               "family_streams": engine.family_streams, "batch_rows": engine.batch_rows,
               "batch_streams": engine.batch_streams, "node_probabilities": engine.node_probabilities,
               "depth_prior": [bits(p) for p in engine.depth_prior], "family_costs": table(engine.family_costs),
               "shared_costs": table(engine.shared_costs), "mtp_step_ms": bits(engine.mtp_step_ms)}
    constants = {"enter_match": engine.enter_match, "depth_rate": bits(engine.depth_rate),
                 "depth_probe_every": engine.depth_probe_every, "cost_rate": bits(engine.cost_rate),
                 "plain_margin": bits(engine.plain_margin), "plain_wait_most": engine.plain_wait_most,
                 "copy_rate": bits(engine.copy_rate), "draft_slack": engine.draft_slack,
                 "default_prior": [bits(p) for p in type(engine).depth_prior]}
    return {"ev": "header", "version": 1, "name": name, "model": attrs, "engine": kw, "derived": derived,
            "constants": constants, "streams": specs}


class Capture:
    """Verify rows' logits (as float32, exact for bf16) and the GPU sampler's picks, to check a host reference."""

    def __init__(self, limit: int) -> None:
        self.limit, self.rows, self.meta, self.dtype = int(limit), [], [], ""

    def add(self, logits: Any, sampling: Any, positions: Any, out: Any) -> None:
        import mlx.core as mx
        import numpy as np

        if len(self.rows) >= self.limit:
            return
        self.dtype = str(logits.dtype)
        values = np.array(logits.astype(mx.float32)).reshape(len(positions), -1)
        picks = np.array(out).reshape(-1)
        for r, position in enumerate(positions):
            if len(self.rows) >= self.limit:
                break
            self.rows.append(values[r].astype(np.float32))
            s = None if sampling is None else sampling_json(sampling)
            self.meta.append({"position": int(position), "token": int(picks[r]), "sampling": s})

    def save(self, out: Path, stem: str) -> None:
        import numpy as np

        if not self.rows:
            return
        np.stack(self.rows).tofile(out / f"{stem}.bin")
        (out / f"{stem}.json").write_text(json.dumps({"vocab": int(self.rows[0].shape[0]), "dtype": self.dtype,
                                                      "rows": self.meta}))


def spec(stream: Any) -> dict[str, Any]:
    s, p = stream.sampling, stream.proposer
    return {"id": stream.stream_id, "prompt": list(stream.prompt_ids), "max_new": int(stream.max_new_tokens),
            "eos": sorted(int(t) for t in stream.eos_ids), "drafts": bool(stream.drafts),
            "sampling": None if s is None else sampling_json(s),
            "proposer": None if p is None else {"ngram": p.ngram, "min_match": p.min_match,
                                                "max_extension": p.max_extension, "silence_rounds": p.silence_rounds,
                                                "window": p.window, "confident_match": p.confident_match}}


def run_model(args: argparse.Namespace) -> int:
    from tensorfold.families import nemotron_h

    for key, value in nemotron_h.MLX_ENV.items():
        os.environ.setdefault(key, value)               # before MLX starts, as the server does
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream, SuffixLookupProposer
    from tensorfold.server.text import eos_ids_of, render_prompt_ids

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = nemotron_h.load(Path(args.model))
    settings = nemotron_h.engine_settings(model)
    kw = {"max_rows": int(settings["max_rows"]), "max_draft": int(settings["max_draft"])}
    eos = eos_ids_of(tokenizer)
    code = ("def merge(intervals):\n    intervals = sorted(intervals)\n    out = [intervals[0]]\n"
            "    for start, end in intervals[1:]:\n        if start <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], end)\n        else:\n            out.append([start, end])\n"
            "    return out\n")
    prompts = {
        "copy": f"Here is a Python function:\n\n```python\n{code}```\n\nRepeat the function exactly, then add a "
                "one-line docstring.",
        "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    }
    ids = {k: render_prompt_ids(tokenizer, [{"role": "user", "content": v}], enable_thinking=False)
           for k, v in prompts.items()}
    sampled = Sampling(seed=args.seed, temperature=0.7, top_k=0, top_p=0.95, min_p=0.0)
    Recording = recording_engine(LaneEngine)
    original = getattr(model, "draft_streams", None)

    def stream(sid: str, prompt: str, max_new: int, sampling: Any, drafts: bool = True) -> Any:
        return LaneStream(stream_id=sid, prompt_ids=list(ids[prompt]), max_new_tokens=max_new, eos_ids=eos,
                          proposer=SuffixLookupProposer(min_match=4) if drafts else None, drafts=drafts,
                          sampling=sampling)

    scenarios = {
        "greedy": [stream("g", "copy", args.max_new, None)],
        "sampled": [stream("s", "story", args.max_new, sampled)],
        "concurrent": [stream("g", "copy", args.max_new, None), stream("s", "story", args.max_new - 32, sampled)],
        "drafts-off": [stream("g", "copy", args.max_new, None, drafts=False)],
    }
    tokens: dict[str, dict[str, list[int]]] = {}
    for name, streams in scenarios.items():
        events: list[dict[str, Any]] = []
        capture = Capture(args.capture_rows) if name in ("sampled", "greedy") else None
        if original is not None:
            model.draft_streams = original
        engine = Recording(model, events, capture=capture, **kw)
        wrap_draft_streams(model, engine, original)
        head = header(engine, model, name, [spec(s) for s in streams], kw)
        for s in streams:
            engine.add_stream(s)
        while engine.active_count:
            engine.step()
        engine.release_rounds()
        tokens[name] = {s.stream_id: list(s.emitted) for s in streams}
        with open(out / f"{name}.jsonl", "w") as handle:
            handle.write(line(head) + "\n")
            for event in events:
                handle.write(line(event) + "\n")
        if capture is not None:
            capture.save(out, f"gpu_rows_{name}")
        print(f"[record] {name}: {len(events)} events, "
              + ", ".join(f"{sid} {len(t)} tokens" for sid, t in tokens[name].items()), flush=True)
    if original is not None:
        model.draft_streams = original
    checks = {
        "concurrent g == greedy": tokens["concurrent"]["g"] == tokens["greedy"]["g"],
        "concurrent s == sampled prefix": tokens["concurrent"]["s"] == tokens["sampled"]["s"][:len(tokens["concurrent"]["s"])],
        "drafts-off == greedy": tokens["drafts-off"]["g"] == tokens["greedy"]["g"],
    }
    for check, ok in checks.items():
        print(f"[record] {check}: {'yes' if ok else 'NO'}", flush=True)
    (out / "contract.json").write_text(json.dumps(checks))
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    units = sub.add_parser("units", help="CPU-only vectors: depth rule, allocation, proposer, sampler, streams")
    units.add_argument("--out", required=True)
    units.add_argument("--seed", type=int, default=20261003)
    model = sub.add_parser("model", help="the four model traces (loads the model: run it under the GPU lock)")
    model.add_argument("--model", required=True)
    model.add_argument("--out", required=True)
    model.add_argument("--max-new", type=int, default=128)
    model.add_argument("--seed", type=int, default=1234)
    model.add_argument("--capture-rows", type=int, default=48)
    args = parser.parse_args()
    if args.cmd == "units":
        return run_units(Path(args.out), args.seed)
    return run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
