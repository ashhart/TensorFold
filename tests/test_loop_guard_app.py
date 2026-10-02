"""--loop-guard at the app level: a fake family whose replies (marked by a sentinel
prompt token) fall into an exact one-token cycle 64 tokens into their think block.

The guard must fire, close the think block through the family's forced windows, and
report the event in runtime.loop — and change nothing when the flag is off or the
reply is healthy. Budget precedence runs both ways: a 256 budget cuts before the
guard's earliest fire (the budget wins), a 512 budget cuts after it (the guard wins)."""

from __future__ import annotations

import threading
from typing import Any

from tensorfold.server.app import ChatApp
from tests.lane_fakes import VOCAB, FakeBatchItem, FakeEngine, FakeFamily
from tests.test_lane_server import EOS, FakeTokenizer

SENTINEL = 150      # prompt token marking a looper: the content char U+4E96 encodes to this CJK-band id
MARK = 160          # </think> under ThinkTokenizer's lookup; also the close tokens' end id
CYCLE = (90,)       # the cycle: one constant token (parity-immune under windowed commits)
T0 = 64             # reply position where the cycle starts (== the guard's warm-in floor)


class ThinkTokenizer(FakeTokenizer):
    """Adds the </think> lookup so the server can arm close tokens."""

    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        return MARK if token == "</think>" else -1

class CycleFamily(FakeFamily):
    """Healthy prompts behave like the house fake (EOS after ~40 reply tokens);
    sentinel prompts emit a deterministic preamble, then an exact one-token cycle
    while their think block is open — and healthy tokens once it closes."""

    def __init__(self) -> None:
        super().__init__()
        self.plen: dict[int, int] = {}

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
        import mlx.core as mx
        import numpy as np

        history = cache[0].rows[0]
        out = []
        plen = getattr(cache[0], "plen", None)
        looper = (plen is not None and SENTINEL in history[:plen] and MARK not in history)
        for token in np.array(inputs).reshape(-1).tolist():
            history.append(int(token))
            out.append(self._pick(len(history) - plen if plen is not None else 0, looper))
        return mx.array(out, dtype=mx.float32).reshape(1, -1, 1)

    @staticmethod
    def _pick(r: int, looper: bool) -> int:
        if not looper:
            return EOS if r > 40 else _healthy(r)
        if r <= T0:
            token = _healthy(r)
            return (token + 1) % VOCAB if token == EOS else token    # the preamble never ends the reply
        return CYCLE[(r - T0) % len(CYCLE)]


def _healthy(r: int) -> int:
    return (r * 11 + 7) % VOCAB or 1        # deterministic, never 0; == EOS only at r == 92


class CycleEngine(FakeEngine):
    """Records each stream's prompt length so the family can tell reply tokens from prompt tokens;
    its cache copier carries the marker across the copies prefill makes."""

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(model if model is not None else CycleFamily(), **kwargs)

    @staticmethod
    def copy_single_cache(cache: list[Any]) -> list[Any]:
        out = []
        for item in cache:
            copy = FakeBatchItem([list(r) for r in item.rows])
            if getattr(item, "plen", None) is not None:
                setattr(copy, "plen", item.plen)
            out.append(copy)
        return out

    def _family_prefill_steps(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                              checkpoints_at: Any) -> Any:
        work = yield from super()._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                        checkpoints_at=checkpoints_at)
        if work:
            setattr(work[0], "plen", len(stream.prompt_ids))   # rides the item: decode's hidden() sees it
        return work


def make_loopy_app(**kwargs: Any) -> ChatApp:
    settings: dict[str, Any] = {
        "served_name": "fake-loopy",
        "lanes": 2,
        "max_rows": 16,
        "max_draft": 4,
        "default_max_tokens": 900,
        "checkpoint_slots": 2,
        "use_proposer": False,
        "enable_thinking": True,
        "engine_factory": CycleEngine,
    }
    settings.update(kwargs)
    return ChatApp(None, ThinkTokenizer(), **settings)


LOOPY = [{"role": "user", "content": "\u4e96"}]  # encodes the SENTINEL prompt token
HEALTHY = [{"role": "user", "content": "hi"}]


def test_flag_on_fires_labels_and_ends_early() -> None:
    app = make_loopy_app(loop_guard=True)
    try:
        reply = app.chat(LOOPY, max_tokens=900)
        # API surface stays SDK-parseable ("stop"); the event lives in runtime.loop
        assert reply["finish_reason"] == "stop"
        assert reply["runtime"]["loop"] == {"period": 1}
        assert reply["completion_tokens"] < 900                  # ended well before the cap
    finally:
        app.close()


def test_flag_off_runs_the_same_reply_to_the_cap() -> None:
    app = make_loopy_app()
    try:
        reply = app.chat(LOOPY, max_tokens=900)
        assert reply["finish_reason"] == "length"
        assert "loop" not in reply["runtime"]
        assert reply["completion_tokens"] == 900
    finally:
        app.close()


def test_budget_256_cuts_before_the_guard_and_wins() -> None:
    app = make_loopy_app(loop_guard=True)
    try:
        reply = app.chat(LOOPY, max_tokens=900, sampling={"thinking_budget": 256})
        assert reply["finish_reason"] == "stop"
        assert "loop" not in reply["runtime"]
        assert reply["completion_tokens"] < 256 + 40             # cut early, answered, ended
    finally:
        app.close()


def test_budget_512_arrives_after_the_fire_so_the_guard_wins() -> None:
    app = make_loopy_app(loop_guard=True)
    try:
        reply = app.chat(LOOPY, max_tokens=900, sampling={"thinking_budget": 512})
        assert reply["finish_reason"] == "stop"
        assert reply["runtime"]["loop"] == {"period": 1}
    finally:
        app.close()


def test_a_healthy_reply_with_the_guard_on_is_untouched() -> None:
    app = make_loopy_app(loop_guard=True)
    try:
        reply = app.chat(HEALTHY, max_tokens=200)
        assert reply["finish_reason"] == "stop"
        assert "loop" not in reply["runtime"]
    finally:
        app.close()


def test_deltas_before_the_cut_stream_normally_and_nothing_is_retracted() -> None:
    app = make_loopy_app(loop_guard=True)
    try:
        deltas: list[Any] = []
        reply = app.chat(LOOPY, max_tokens=900, on_delta=deltas.append)
        assert reply["finish_reason"] == "stop" and reply["runtime"]["loop"]
        assert deltas                                              # reasoning streamed before the cut
        strings = [d if isinstance(d, str) else d.get("content", "") for d in deltas]
        thought = "".join(d.get("reasoning_content", "") for d in deltas if isinstance(d, dict))
        assert thought                                             # the preamble reached the client
        assert "".join(strings) == reply["content"]                # nothing retracted, no edits
    finally:
        app.close()


def test_a_looper_never_disturbs_its_neighbour() -> None:
    app = make_loopy_app(loop_guard=True, lanes=2)
    try:
        results: dict[str, dict[str, Any]] = {}

        def run(name: str, messages: list[dict[str, Any]]) -> None:
            results[name] = app.chat(messages, max_tokens=900 if name == "looper" else 200)

        threads = [threading.Thread(target=run, args=("looper", LOOPY)),
                   threading.Thread(target=run, args=("healthy", HEALTHY))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert all(not thread.is_alive() for thread in threads), "a request hung"
        assert results["looper"]["finish_reason"] == "stop"
        assert results["looper"]["runtime"]["loop"] == {"period": 1}
        assert results["healthy"]["finish_reason"] == "stop"
        assert "loop" not in results["healthy"]["runtime"]
    finally:
        app.close()


def test_cache_rows_match_committed_tokens_after_a_fire() -> None:
    # the fire's row-retention claim: the family's absorbed history (its cache row) tracks
    # prompt + committed tokens with no double-count and no loss, on both paths — the last
    # committed token has no absorbed row by design (one-token rounds run ahead), so the
    # row count sits in a one-token window under prompt + committed, identical for the
    # fired path and the cap path
    seen: dict[str, int] = {}   # rows/plen: the decode item's last seen state (prefill copies carry no plen)

    class RowCountingFamily(CycleFamily):
        def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
            out = super().hidden(inputs, cache, parents)
            plen = getattr(cache[0], "plen", None)
            if plen is not None:                   # decode's item (prefill copies carry no plen)
                seen["rows"], seen["plen"] = len(cache[0].rows[0]), plen
            return out

    class RowCountingEngine(CycleEngine):
        def __init__(self, model: Any = None, **kwargs: Any) -> None:
            super(CycleEngine, self).__init__(RowCountingFamily(), **kwargs)

    for arm, committed in ((True, None), (False, 900)):
        seen.clear()
        app = make_loopy_app(loop_guard=arm, engine_factory=RowCountingEngine)
        try:
            reply = app.chat(LOOPY, max_tokens=900)
            expected = reply["completion_tokens"] if arm else committed
            assert reply["runtime"].get("loop") == ({"period": 1} if arm else None)
            assert seen["plen"] + expected - 1 <= seen["rows"] <= seen["plen"] + expected, \
                (arm, seen["rows"], seen["plen"], expected)
        finally:
            app.close()
