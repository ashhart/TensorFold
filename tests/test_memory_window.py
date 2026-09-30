"""The context window a server affords: sized at startup from a probe prompt, before any request."""

from __future__ import annotations

from typing import Any
import weakref

import pytest

from tensorfold.server.app import ChatApp
from tests.lane_fakes import FakeEngine, FakeFamily
from tests.test_lane_server import FakeTokenizer
from tests.test_memory_budget import Array

LIVE: "weakref.WeakSet[SizedKV]" = weakref.WeakSet()


class SizedKV:
    """A fake layer: absorbed history (``rows``) plus KV arrays of 1,024 bytes a position, in 256-position steps."""

    def __init__(self) -> None:
        self.rows: list[list[int]] = [[]]
        LIVE.add(self)

    @property
    def offset(self) -> int:
        return len(self.rows[0])

    @property
    def keys(self) -> Array:
        return Array((1, 1, max(256, -(-self.offset // 256) * 256), 256))

    @property
    def values(self) -> Array:
        return self.keys

    @property
    def state(self) -> tuple[Array, Array]:
        return self.keys, self.values


class SizedFamily(FakeFamily):
    def make_cache(self) -> list[Any]:
        return [SizedKV()]


class Runtime:
    """MLX's counters over the fake: weights plus every live fake cache; a chunk peaks 1 MiB above it."""

    def __init__(self, weights: int) -> None:
        self.weights, self.cache, self.peak = weights, 0, weights

    def get_active_memory(self) -> int:
        return self.weights + sum(item.keys.nbytes + item.values.nbytes for item in list(LIVE))

    def get_cache_memory(self) -> int:
        return self.cache

    def get_peak_memory(self) -> int:
        return max(self.peak, self.get_active_memory() + 2**20)

    def reset_peak_memory(self) -> None:
        self.peak = self.get_active_memory()

    def clear_cache(self) -> None:
        self.cache = 0


def app(budget: int, **kwargs: Any) -> ChatApp:
    settings: dict[str, Any] = dict(served_name="fake", lanes=1, default_max_tokens=16, checkpoint_slots=2,
                                    use_proposer=False, engine_factory=FakeEngine, memory_budget_bytes=budget,
                                    memory_overhead_bytes=0, memory_runtime=Runtime(2**30))
    settings.update(kwargs)
    model = SizedFamily()
    model.args = type("Args", (), {"num_attention_heads": 1, "head_dim": 128})()
    return ChatApp(model, FakeTokenizer(), **settings)


def test_the_default_window_is_what_the_budget_affords_after_a_startup_probe():
    # 1 GiB of weights, a 256 MiB workspace floor; 1 KiB of KV a position, its growth and the retained prompt: 128 MiB
    served = app(int(1.375 * 2**30), context_window=262144, fit_context=True)
    try:
        memory = served.prompt_memory
        assert memory.workspace_profiled and memory.profile.bytes_per_token == 1024
        window = memory.largest_window(262144, resumable=True)
        assert served.context_window == window // 1024 * 1024 and 42_000 < served.context_window < 44_000
        assert memory.window == served.context_window
        assert memory.affordable == memory.largest_window(262144) > window
    finally:
        served.close()


def test_a_window_already_inside_the_budget_is_kept():
    served = app(int(1.5 * 2**30), context_window=8192, fit_context=True)
    try:
        assert served.context_window == 8192
    finally:
        served.close()


def test_an_explicit_window_past_the_budget_is_refused_at_startup_naming_one_that_fits():
    with pytest.raises(ValueError, match=r"262,144-token context window.*most one request can use is [\d,]+ tokens"):
        app(int(1.5 * 2**30), context_window=262144)


def test_weights_that_leave_no_room_refuse_to_start():
    with pytest.raises(ValueError, match="no room") as error:
        app(2**30 + 2**20, context_window=0, fit_context=True)
    assert "--drafter none" in str(error.value) and "max_tokens" not in str(error.value)


@pytest.mark.parametrize("ceiling_gib, raise_it", [(64, True), (1, False)])
def test_the_refusal_names_the_budget_that_would_fit_and_this_macs_ceiling(ceiling_gib, raise_it, monkeypatch):
    from tensorfold.server import memory_budget

    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 128 * 2**30)   # the host's RAM plays no part
    runtime = Runtime(2**30)
    runtime.device_info = lambda: {"max_recommended_working_set_size": ceiling_gib * 2**30}
    with pytest.raises(ValueError, match="no room") as error:
        app(2**30 + 2**20, context_window=0, fit_context=True, memory_runtime=runtime)
    message = str(error.value)
    hint = "Raise the budget past 1.3 GiB with TENSORFOLD_MEMORY_LIMIT_GB (this Mac takes up to 64.0"
    assert (hint in message) is raise_it
    assert ("Serve it on a Mac with more memory" in message) is not raise_it


def test_without_prompt_retention_the_default_window_reserves_no_retained_copy():
    served = app(int(1.375 * 2**30), context_window=262144, fit_context=True, checkpoint_slots=0)
    try:
        memory = served.prompt_memory
        assert memory.store is None
        assert served.context_window == memory.affordable // 1024 * 1024 > 60_000
    finally:
        served.close()


class Rows:
    """The last round's rollback rows (a wide window's recurrent states), kept until ``release_rounds``."""

    def __init__(self, positions: int) -> None:
        self.keys = self.values = Array((1, 1, positions, 256))
        LIVE.add(self)


def test_a_finished_rounds_rollback_rows_go_before_the_next_prompt_is_sized():
    """#95: what the engine keeps from its last round belongs to no live stream once the reply ends. The next prompt
    near the promised window is admitted and kept (the startup promise) only if those rows go first."""

    served = app(int(1.375 * 2**30), context_window=262144, fit_context=True)
    try:
        engine, model = served.scheduler.engine, served.scheduler.engine.model
        step = engine.step

        def round_keeping_rows():
            got = step()
            model.rows = Rows(64 * 1024)               # 64 MiB, as a wide window's DeltaNet states would be
            return got

        engine.step = round_keeping_rows
        model.release_rounds = lambda: setattr(model, "rows", None)
        served.chat([{"role": "user", "content": "hello there"}], max_tokens=4)
        long = "notes " * int(served.context_window * 0.9 / 6)
        served.chat([{"role": "user", "content": long}], max_tokens=4)
        kept = max((len(entry.tokens) for entry in served.checkpoints._entries), default=0)
        assert kept >= len(long) - 8                    # its prompt is kept for the next turn
    finally:
        served.close()


def test_every_prompt_inside_the_advertised_window_is_admitted_and_a_refusal_counts_only_what_stays():
    served = app(int(1.375 * 2**30), context_window=262144, fit_context=True)
    try:
        memory, reply = served.prompt_memory, 16
        window = served.context_window
        # the window's fit and a request use one need(): the edge of the window is admitted at idle
        memory.begin(window - reply, reply)
        memory.end()
        held = memory.runtime.get_active_memory()
        assert memory.need(window, held) <= memory.budget
        # past what one copy of the cache affords, the refusal names a prompt length that is admitted
        top = memory.largest_window(1 << 22)
        with pytest.raises(Exception) as refused:
            memory.begin(top + 1024, reply)
        memory.end()
        fits = int(str(refused.value).split("fits up to ")[1].split(" tokens")[0].replace(",", ""))
        assert window - reply <= fits <= top
        memory.begin(fits, reply)
        memory.end()
    finally:
        served.close()


def test_the_default_prompt_cache_takes_what_a_whole_window_request_leaves_idle():
    served = app(2**31, context_window=65536, checkpoint_budget_bytes=2**20, grow_checkpoints=True)   # 0.625 GiB idle
    try:
        memory, store = served.prompt_memory, served.checkpoints
        spare = memory.spare(served.context_window)
        assert store.budget_bytes == spare == 2**31 - memory.need(served.context_window, memory.held()) > 2**20
    finally:
        served.close()
    fixed = app(2**31, context_window=65536, checkpoint_budget_bytes=2**20)
    try:
        assert fixed.checkpoints.budget_bytes == 2**20                   # an explicit --prompt-cache-gib stays
    finally:
        fixed.close()
