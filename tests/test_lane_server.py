"""Tests for the lane server: rendering, checkpoints, scheduling, and chat.

The engine is the fake target from ``tests/lane_fakes.py``; nothing here
loads a model. Concurrency is exercised with real threads through the same
``chat`` entry point the HTTP handler calls.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tensorfold.server.app import ChatApp
from tensorfold.server.checkpoints import CheckpointStore, choose_checkpoints
from tensorfold.server.text import eos_ids_of, render_prompt_ids
from tests.lane_fakes import FakeBatchItem, FakeEngine, FakeFamily, fake_serial

EOS = 49


TOKEN_CHAR_BASE = 0x4E00  # decoded tokens round-trip through the template


class FakeTokenizer:
    eos_token_ids = {EOS}

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []

    @staticmethod
    def _char_ids(text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            code = ord(ch)
            if TOKEN_CHAR_BASE <= code < TOKEN_CHAR_BASE + 200:
                ids.append(code - TOKEN_CHAR_BASE)  # a decoded token, verbatim
            else:
                ids.append(2 + code % 40)
        return ids

    GENERATION_MARKER = [3, 4]  # plays the assistant header + empty think block

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.template_calls.append(dict(kwargs))
        ids: list[int] = []
        for message in messages:
            ids.extend(self._char_ids(str(message.get("content", ""))))
            ids.append(1)
        if kwargs.get("add_generation_prompt", True):
            ids.extend(self.GENERATION_MARKER)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(TOKEN_CHAR_BASE + int(t)) for t in ids)

    def encode(self, text: str, **_: Any) -> list[int]:
        return self._char_ids(text)


class SlowFakeEngine(FakeEngine):
    """Rounds take real time so concurrent submissions overlap deterministically."""

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(FakeFamily(delay=0.004), **kwargs)


def make_app(**kwargs: Any) -> ChatApp:
    settings: dict[str, Any] = {
        "served_name": "fake-27b",
        "lanes": 2,
        "max_rows": 16,
        "max_draft": 4,
        "default_max_tokens": 12,
        "checkpoint_slots": 2,
        "model_aliases": ["alias-a"],
        "use_proposer": False,
        "engine_factory": FakeEngine,
    }
    settings.update(kwargs)
    return ChatApp(None, FakeTokenizer(), **settings)


def expected_reply(app: ChatApp, messages: list[dict[str, Any]], max_new: int) -> tuple[list[int], str]:
    prompt = render_prompt_ids(app.tokenizer, messages)
    tokens = fake_serial(prompt, max_new, {EOS})
    visible = tokens[:-1] if tokens and tokens[-1] == EOS else tokens
    return tokens, app.tokenizer.decode(visible)


# --------------------------------------------------------------------------


def test_render_prompt_ids_passes_tools_and_thinking_and_normalizes_arguments() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "user", "content": "hi"},
    ]
    ids = render_prompt_ids(tokenizer, messages, tools=[{"type": "function"}], enable_thinking=True)
    assert ids and tokenizer.template_calls[-1]["tools"] == [{"type": "function"}]
    assert tokenizer.template_calls[-1]["enable_thinking"] is True
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # caller's copy untouched


def test_eos_ids_prefers_the_set() -> None:
    assert eos_ids_of(FakeTokenizer()) == frozenset({EOS})

    class Single:
        eos_token_id = 7

    assert eos_ids_of(Single()) == frozenset({7})


def test_checkpoint_store_matches_longest_strict_prefix_and_copies() -> None:
    copies: list[list[Any]] = []

    def copier(cache: list[Any]) -> list[Any]:
        copies.append(cache)
        return ["copy-of", *cache]

    store = CheckpointStore(3, copier=copier)
    store.insert([1, 2, 3], ["c1"], last_prompt=[1, 2, 3, 8])
    store.insert([9, 9], ["c2"], last_prompt=[9, 9, 9])
    assert store.match([1, 2, 3]) is None  # equal is not a strict prefix
    assert store.match([1, 2, 4]) is None
    hit = store.match([1, 2, 3, 4, 5])
    assert hit == (3, ["copy-of", "c1"], [1, 2, 3, 8])
    assert copies == [["c1"]]
    assert store.hits == 1 and store.misses == 2
    # the previous prompt is updated on a hit, so the clamp sees this turn next time
    assert store.match([1, 2, 3, 4, 5, 6])[2] == [1, 2, 3, 4, 5]


def test_checkpoint_store_keeps_history_and_finish_entries_and_trims_lru() -> None:
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([1, 2], ["history"], last_prompt=[1, 2, 3])
    store.insert([1, 2, 3, 4], ["finish"], last_prompt=[1, 2, 3])  # same conversation, both kept
    assert len(store) == 2
    assert store.match([1, 2, 3, 4, 5])[:2] == (4, ["finish"])
    assert store.match([1, 2, 9])[:2] == (2, ["history"])
    store.insert([1, 2], ["history-again"], last_prompt=[1, 2, 9])  # exact duplicate replaces
    assert len(store) == 2
    store.insert([7], ["c"], last_prompt=[7, 7])
    store.insert([8], ["d"], last_prompt=[8, 8])
    assert len(store) == 3
    # the least recently used entry (the finish) was evicted; history still matches
    assert store.match([1, 2, 3, 4, 5])[:2] == (2, ["history-again"])
    assert store.match([1, 2, 5])[:2] == (2, ["history-again"])


def test_checkpoint_store_byte_budget_evicts_least_recently_used() -> None:
    sizes = {"big": 700, "mid": 300, "small": 100}
    store = CheckpointStore(10, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: sizes[c[0]])
    store.insert([1], ["big"], last_prompt=[1, 1])
    store.insert([2], ["mid"], last_prompt=[2, 2])
    assert store.nbytes == 1000 and len(store) == 2
    store.insert([3], ["small"], last_prompt=[3, 3])  # 1100 > budget: evict the oldest
    assert [e.cache[0] for e in store._entries] == ["small", "mid"]
    assert store.evictions == 1
    store.insert([4], ["big"], last_prompt=[4, 4])  # a big one keeps only itself plus what fits
    assert [e.cache[0] for e in store._entries] == ["big", "small"]
    assert store.nbytes == 800


def test_checkpoint_store_pins_system_blocks_outside_the_slots() -> None:
    sizes = {"block": 400, "conv": 100}
    store = CheckpointStore(2, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: sizes[c[0]],
                            pinned_slots=2)
    store.insert([1, 2, 3], ["block", "long"], last_prompt=[1, 2, 3], pinned=True)
    store.insert([1, 2], ["block", "short"], last_prompt=[1, 2], pinned=True)
    # a title request and a turn: conversation entries do not push the blocks out
    store.insert([7], ["conv", "title"], last_prompt=[7, 7])
    store.insert([8], ["conv", "title-end"], last_prompt=[8, 8])
    store.insert([9], ["conv", "turn"], last_prompt=[9, 9])
    assert [e.cache[1] for e in store._entries] == ["turn", "title-end", "short", "long"]
    assert store.match([1, 2, 3, 4])[:2] == (3, ["block", "long"])
    # a third block past pinned_slots=2 makes the oldest ("short") ordinary; over the budget
    # the least recently used ordinary entries go first
    store.insert([5], ["block", "third"], last_prompt=[5], pinned=True)
    assert [e.cache[1] for e in store._entries] == ["third", "long", "turn", "title-end"]
    assert [e.pinned for e in store._entries] == [True, True, False, False]
    store.insert([6], ["block", "fourth"], last_prompt=[6], pinned=True)   # 1400 > 1000
    assert [e.cache[1] for e in store._entries] == ["fourth", "third"]
    sizes["tiny"] = 1
    store.insert([4], ["tiny", "b3"], last_prompt=[4], pinned=True)
    assert [(e.cache[1], e.pinned) for e in store._entries] == [("b3", True), ("fourth", True), ("third", False)]
    # re-inserting a pinned prefix keeps it pinned
    store.insert([6], ["block", "fourth-again"], last_prompt=[6, 6])
    assert store._entries[0].pinned


def test_app_sizes_the_store_to_the_lanes() -> None:
    app = make_app(lanes=3, checkpoint_slots=None)
    try:
        assert app.checkpoints is not None and app.checkpoints.slots == 9
    finally:
        app.close()


def test_choose_checkpoints_keeps_history_and_the_stable_same_conversation_prefix() -> None:
    prompt = list(range(100))
    assert choose_checkpoints(80, 0, None, prompt) == [80]
    assert choose_checkpoints(80, 80, None, prompt) == []  # nothing new to snapshot
    assert choose_checkpoints(100, 0, None, prompt) == []  # no generation prompt tail
    assert choose_checkpoints(0, 0, None, prompt) == []
    # previous prompt of the same conversation diverges at 60 (per-turn block)
    previous = list(range(60)) + [999] * 30
    assert choose_checkpoints(80, 0, previous, prompt) == [60, 80]
    # the stable prefix inside the reused part (think-block template) drops out
    assert choose_checkpoints(80, 60, previous, prompt) == [80]
    # an unrelated previous prompt shares almost nothing: history only
    assert choose_checkpoints(80, 0, [5, 6, 7], prompt) == [80]


# --------------------------------------------------------------------------


def test_chat_returns_the_serial_decode_and_telemetry() -> None:
    app = make_app()
    try:
        messages = [{"role": "user", "content": "explain the ring buffer"}]
        tokens, text = expected_reply(app, messages, 12)
        reply = app.chat(messages, max_tokens=12)
        assert reply["content"] == text
        assert reply["completion_tokens"] == len(tokens)
        assert reply["prompt_tokens"] == len(render_prompt_ids(app.tokenizer, messages))
        assert reply["cached_tokens"] == 0
        assert reply["finish_reason"] in {"length", "stop"}
        assert reply["runtime"]["engine"] == "lanes"
        assert reply["runtime"]["sampling"] == "greedy"
        assert reply["speculative"]["rounds"] >= 1
    finally:
        app.close()


def test_chat_streams_visible_text_in_order() -> None:
    app = make_app()
    try:
        messages = [{"role": "user", "content": "stream this please"}]
        _, text = expected_reply(app, messages, 12)
        deltas: list[str] = []
        reply = app.chat(messages, max_tokens=12, temperature=0.7, on_delta=deltas.append)
        assert "".join(deltas) == text == reply["content"]
        assert all(deltas)
        assert reply["runtime"]["sampling"] == "greedy"
    finally:
        app.close()


def test_concurrent_chats_share_rounds_and_each_matches_serial() -> None:
    app = make_app(lanes=2, engine_factory=SlowFakeEngine)
    try:
        prompts = [f"request number {i} about {'x' * (i + 1)}" for i in range(5)]
        results: dict[int, dict[str, Any]] = {}
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                results[index] = app.chat(
                    [{"role": "user", "content": prompts[index]}], max_tokens=30 + index
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors
        for index in range(5):
            _, text = expected_reply(app, [{"role": "user", "content": prompts[index]}], 30 + index)
            assert results[index]["content"] == text, index
        assert max(r.streams for r in app.engine.round_stats) == 2
        assert app.scheduler.completed == 5
        assert app.scheduler.active == 0
    finally:
        app.close()


def test_a_title_request_steps_aside_for_the_turn_and_reruns_exactly() -> None:
    app = make_app(lanes=1, engine_factory=SlowFakeEngine)
    try:
        title = [{"role": "system", "content": "Write a ~5 word title for the next user message."},
                 {"role": "user", "content": "<user>how are you today</user>"}]
        turn = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "how are you today"}]
        finished: dict[str, float] = {}
        results: dict[str, dict[str, Any]] = {}
        deltas: list[str] = []

        def run(name: str, messages: list[dict[str, Any]], limit: int, sink: Any = None) -> None:
            results[name] = app.chat(messages, max_tokens=limit, on_delta=sink)
            finished[name] = time.perf_counter()

        background = threading.Thread(target=run, args=("title", title, 60, deltas.append))
        background.start()
        deadline = time.perf_counter() + 5
        while not deltas and time.perf_counter() < deadline:     # the title is decoding
            time.sleep(0.002)
        assert deltas
        run("turn", turn, 12)
        background.join(timeout=30)
        assert app.scheduler.preemptions >= 1
        assert finished["turn"] < finished["title"]
        # the rerun's tokens are the ones already streamed: nothing streamed twice or lost
        _, title_text = expected_reply(app, title, 60)
        assert results["title"]["content"] == title_text == "".join(deltas)
        _, turn_text = expected_reply(app, turn, 12)
        assert results["turn"]["content"] == turn_text
        assert app.scheduler.active == 0
    finally:
        app.close()


def test_is_title_request_matches_only_short_tool_free_title_prompts() -> None:
    from tensorfold.server.text import is_title_request

    title = [{"role": "system", "content": "Write a ~5 word title using only the task."},
             {"role": "user", "content": "<user>hi</user>"}]
    assert is_title_request(title, None)
    assert not is_title_request(title, [{"type": "function"}])
    assert not is_title_request([title[0], {"role": "user", "content": "x" * 5000}], None)
    assert not is_title_request([{"role": "user", "content": "a title for my essay?"}], None)


def test_a_title_request_sent_just_before_the_turn_waits_for_it() -> None:
    app = make_app(lanes=1, engine_factory=SlowFakeEngine)
    try:
        title = [{"role": "system", "content": "Write a ~5 word title for the next user message."},
                 {"role": "user", "content": "<user>hello there</user>"}]
        turn = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "hello there"}]
        started: dict[str, float] = {}
        results: dict[str, dict[str, Any]] = {}

        def run(name: str, messages: list[dict[str, Any]]) -> None:
            def first(_: Any) -> None:
                started.setdefault(name, time.perf_counter())
            results[name] = app.chat(messages, max_tokens=20, on_delta=first)

        threads = [threading.Thread(target=run, args=("title", title))]
        threads[0].start()
        time.sleep(0.01)                                    # an agent client: the title goes out first
        threads.append(threading.Thread(target=run, args=("turn", turn)))
        threads[1].start()
        for thread in threads:
            thread.join(timeout=30)
        assert started["turn"] < started["title"]
        assert app.scheduler.preemptions == 0               # it never had to step aside
        assert results["title"]["content"] == expected_reply(app, title, 20)[1]
        assert results["turn"]["content"] == expected_reply(app, turn, 20)[1]
    finally:
        app.close()


def test_a_background_priority_request_steps_aside_like_a_title() -> None:
    app = make_app(lanes=1, engine_factory=SlowFakeEngine)
    try:
        batch = [{"role": "user", "content": "a long batch generation for training data"}]
        turn = [{"role": "user", "content": "the user's own question"}]
        finished: dict[str, float] = {}
        results: dict[str, dict[str, Any]] = {}
        deltas: list[str] = []

        def run(name: str, messages: list[dict[str, Any]], limit: int, sink: Any, sampling: Any) -> None:
            results[name] = app.chat(messages, max_tokens=limit, on_delta=sink, sampling=sampling)
            finished[name] = time.perf_counter()

        worker = threading.Thread(target=run, args=("batch", batch, 60, deltas.append, {"priority": "background"}))
        worker.start()
        deadline = time.perf_counter() + 5
        while not deltas and time.perf_counter() < deadline:
            time.sleep(0.002)
        run("turn", turn, 12, None, None)
        worker.join(timeout=30)
        assert app.scheduler.preemptions >= 1 and finished["turn"] < finished["batch"]
        assert results["batch"]["content"] == expected_reply(app, batch, 60)[1] == "".join(deltas)
    finally:
        app.close()


def test_second_turn_reuses_the_history_checkpoint() -> None:
    app = make_app(lanes=1, checkpoint_slots=4)
    try:
        first = [{"role": "user", "content": "first turn of a conversation"}]
        history_len = len(render_prompt_ids(app.tokenizer, first, add_generation_prompt=False))
        reply = app.chat(first, max_tokens=9)
        assert reply["cached_tokens"] == 0
        # one entry at the history boundary, one at the absorbed end of the reply
        assert app.checkpoints is not None and len(app.checkpoints) == 2
        second = [*first, {"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "more"}]
        _, text = expected_reply(app, second, 7)
        reply2 = app.chat(second, max_tokens=7)
        # the generation marker is not re-rendered, so only the history entry matches
        assert reply2["cached_tokens"] == history_len
        assert reply2["content"] == text
        assert app.engine.prefill_calls[-1][1] == history_len
        # the second turn snapshots its own history boundary for a third turn
        third_history = len(render_prompt_ids(app.tokenizer, second, add_generation_prompt=False))
        third = [*second, {"role": "assistant", "content": reply2["content"]}, {"role": "user", "content": "again"}]
        reply3 = app.chat(third, max_tokens=5)
        assert reply3["cached_tokens"] == third_history
        assert reply3["content"] == expected_reply(app, third, 5)[1]
    finally:
        app.close()


def test_finish_checkpoint_serves_a_verbatim_continuation() -> None:
    app = make_app(lanes=1, checkpoint_slots=4)
    try:
        first = [{"role": "user", "content": "verbatim"}]
        reply = app.chat(first, max_tokens=8)
        stream = app.engine.streams[-1] if app.engine.streams else None
        prompt1 = render_prompt_ids(app.tokenizer, first)
        # a raw continuation that literally extends prompt + reply tokens
        absorbed = None
        for entry in app.checkpoints._entries:  # type: ignore[union-attr]
            if len(entry.tokens) > len(prompt1):
                absorbed = entry.tokens
        assert absorbed is not None and absorbed[: len(prompt1)] == prompt1
        del stream, reply
        continuation_ids = absorbed + [5, 6, 7]
        job_prompt = [{"role": "user", "content": app.tokenizer.decode(continuation_ids)}]
        rendered = render_prompt_ids(app.tokenizer, job_prompt)
        assert rendered[: len(absorbed)] == absorbed
        reply2 = app.chat(job_prompt, max_tokens=4)
        assert reply2["cached_tokens"] == len(absorbed)
        assert reply2["content"] == expected_reply(app, job_prompt, 4)[1]
    finally:
        app.close()


def test_scheduler_survives_a_failed_round() -> None:
    app = make_app(lanes=1)
    try:
        original = app.engine._family_round
        state = {"failed": False}

        def flaky(*args: Any, **kwargs: Any) -> Any:
            if not state["failed"]:
                state["failed"] = True
                raise RuntimeError("metal hiccup")
            return original(*args, **kwargs)

        app.engine._family_round = flaky  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="metal hiccup"):
            app.chat([{"role": "user", "content": "boom"}], max_tokens=6)
        assert app.scheduler.failed_rounds == 1
        messages = [{"role": "user", "content": "after the failure"}]
        _, text = expected_reply(app, messages, 6)
        assert app.chat(messages, max_tokens=6)["content"] == text
    finally:
        app.close()


def test_close_stops_the_scheduler_thread() -> None:
    app = make_app()
    app.close()
    deadline = time.perf_counter() + 5
    while app.scheduler._thread.is_alive() and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert not app.scheduler._thread.is_alive()


def test_checkpoint_store_longest_does_not_count_a_hit() -> None:
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([1, 2], ["a"], last_prompt=[1, 2, 3])
    store.insert([1, 2, 3, 4], ["b"], last_prompt=[1, 2, 3, 4, 5])
    assert store.longest([1, 2, 3, 4, 5]) == 4
    assert store.longest([1, 2, 3, 4]) == 2          # strict prefixes only
    assert store.longest([9]) == 0
    assert store.hits == 0 and store.misses == 0


def test_scheduler_reads_a_stored_block_the_store_lacks(tmp_path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache

    from tensorfold.server.scheduler import Scheduler
    from tensorfold.engine.prefix_snapshots import save_snapshot

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 40, 4)), mx.ones((1, 2, 40, 4)))
    block = list(range(40))
    save_snapshot(tmp_path, "model-a", block, [kv])
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert(list(range(10)), ["short"], last_prompt=list(range(12)), pinned=True)
    scheduler = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=store,
                              snapshot_dir=tmp_path, model_id="model-a")
    scheduler._read_disk_block(list(range(50)))
    hit = store.match(list(range(50)))
    assert hit is not None and hit[0] == 40 and hit[1][0].offset == 40
    before = len(store)
    scheduler._read_disk_block(list(range(50)))      # already there: nothing read again
    assert len(store) == before


def test_conversations_saved_at_shutdown_are_read_back_on_demand(tmp_path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache

    from tensorfold.server.checkpoints import save_conversations
    from tensorfold.server.scheduler import Scheduler

    def cache(n: int) -> list[Any]:
        kv = KVCache()
        kv.update_and_fetch(mx.ones((1, 2, n, 4)), mx.ones((1, 2, n, 4)))
        return [kv]

    store = CheckpointStore(4, copier=lambda c: c, sizer=lambda c: 1)
    store.insert(list(range(20)), cache(20), last_prompt=list(range(22)), pinned=True)   # a system block
    store.insert(list(range(50)), cache(50), last_prompt=list(range(52)))
    store.insert(list(range(60)), cache(60), last_prompt=list(range(62)))
    store.insert([5, 5, 5], cache(3), last_prompt=[5, 5, 5, 5])
    assert save_conversations(store, tmp_path / "sessions", "model-a", keep=2) == 2
    assert len(list((tmp_path / "sessions").glob("*.safetensors"))) == 2    # the newest two only

    fresh = CheckpointStore(4, copier=lambda c: c)
    scheduler = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=fresh,
                              model_id="model-a", session_dir=tmp_path / "sessions")
    scheduler._read_disk_block(list(range(70)))
    hit = fresh.match(list(range(70)))
    assert hit is not None and hit[0] == 60 and hit[1][0].offset == 60
    assert not fresh._entries[0].pinned
    other = CheckpointStore(4, copier=lambda c: c)
    Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=other, model_id="model-b",
                  session_dir=tmp_path / "sessions")._read_disk_block(list(range(70)))
    assert len(other) == 0                                               # other kernels: never read


def test_scheduler_runs_its_stop_hook_in_its_own_thread() -> None:
    from tensorfold.server.scheduler import Scheduler

    scheduler = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset())
    seen: list[str] = []
    scheduler.on_stop = lambda: seen.append(threading.current_thread().name)
    scheduler.start()
    scheduler.stop(timeout=5.0)
    assert seen == ["tensorfold-engine"]


def test_checkpoints_and_shared_prefixes_are_taken_on_the_prefill_grid() -> None:
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.server.scheduler import ChatJob, Scheduler

    class GridEngine(FakeEngine):
        prefill_plan = PrefillPlan(4)

    store = CheckpointStore(4, copier=lambda c: c)
    scheduler = Scheduler(GridEngine(), lanes=1, eos_ids=frozenset(), checkpoints=store)
    job = ChatJob(job_id="j", prompt_ids=list(range(1, 11)), max_tokens=2, temperature=0.0, history_len=9,
                  shared_prefix_lens=(7,))
    scheduler._start_job(job)
    # the history boundary 9 is kept at 8, the system block's 7 at 4 (pinned)
    assert sorted((len(e.tokens), e.pinned) for e in store._entries) == [(4, True), (8, False)]


def test_a_saved_block_is_warmed_in_background_jobs_one_grid_chunk_each(tmp_path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache

    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import save_snapshot
    from tensorfold.server.scheduler import Scheduler

    class GridEngine(FakeEngine):
        prefill_plan = PrefillPlan(4)

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 10, 4)), mx.ones((1, 2, 10, 4)))
    save_snapshot(tmp_path, "/models/fake|kernels=old", list(range(5, 15)), [kv])   # computed by other kernels
    seen: list[tuple[int, bool]] = []
    real = Scheduler.submit

    def submit(self: Any, job: Any) -> None:
        seen.append((len(job.prompt_ids), job.background))
        real(self, job)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Scheduler, "submit", submit)
        app = make_app(engine_factory=GridEngine, snapshot_dir=tmp_path, model_id="/models/fake|kernels=new")
        try:
            deadline = time.time() + 10
            while time.time() < deadline and not any(e.pinned for e in app.checkpoints._entries):
                time.sleep(0.02)
        finally:
            app.scheduler.stop(timeout=5.0)
    # grid points 4 and 8 of the 10-token block, each with one padding token, in background jobs; 8 is pinned
    assert seen == [(5, True), (9, True)]
    assert [len(e.tokens) for e in app.checkpoints._entries if e.pinned] == [8]


def test_a_stored_state_off_the_grid_is_never_matched() -> None:
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.server.scheduler import ChatJob, Scheduler

    class GridEngine(FakeEngine):
        prefill_plan = PrefillPlan(4)

    store = CheckpointStore(4, copier=lambda c: c)
    engine = GridEngine()
    scheduler = Scheduler(engine, lanes=1, eos_ids=frozenset(), checkpoints=store)
    prompt = list(range(1, 12))
    for n in (4, 6):                               # a grid state and a longer one off the grid (from disk, say)
        store.insert(prompt[:n], [FakeBatchItem([prompt[:n]])], last_prompt=prompt[:n])
    scheduler._start_job(ChatJob(job_id="j", prompt_ids=prompt, max_tokens=2, temperature=0.0))
    assert engine.prefill_calls[-1] == ("j", 4)


def test_a_job_that_would_not_fit_waits_for_a_live_stream_to_finish() -> None:
    from tensorfold.engine.memory import Admission, StreamMemory
    from tensorfold.server.scheduler import ChatJob, Scheduler

    # 100 bytes up to 1,000 tokens, then 1 a token (a stream's footprint by length); rounds take 50
    memory = StreamMemory(short_tokens=1000, short=100, long_tokens=2000, long=1100, per_token=1.0, prefill_a=0.0,
                          prefill_b=0.0, round_bytes=50)
    jobs = [ChatJob(job_id=f"j{i}", prompt_ids=list(range(1, 11)), max_tokens=40, temperature=0.0) for i in range(2)]

    def started(budget: int) -> tuple[list[str], Any]:
        engine = FakeEngine()
        admission = Admission(budget, memory, used=lambda: 1000)
        scheduler = Scheduler(engine, lanes=4, eos_ids=frozenset(), admission=admission)
        for job in jobs:
            job.stream, job.error = None, None
            scheduler.submit(job)
        scheduler._admit()
        return [s.stream_id for s in engine.streams], scheduler

    # the second: 1,000 in use, the first grown from 11 to its longest 50 tokens (39), itself at its longest (50 +
    # 256 tokens of slack: 100) and a round (50); the first always starts
    need = 1000 + 39 + 100 + 50
    ids, scheduler = started(budget=need - 1)
    assert ids == ["j0"] and scheduler._held is jobs[1] and scheduler.admission.refused == 1
    ids, _ = started(budget=need)
    assert ids == ["j0", "j1"]


def test_admission_projects_growth_prefill_and_rounds() -> None:
    from tensorfold.engine.memory import Admission, StreamMemory

    memory = StreamMemory(short_tokens=64, short=1_000, long_tokens=264, long=3_000, per_token=10.0, prefill_a=2.0,
                          prefill_b=0.001, round_bytes=5_000)
    # 256 tokens of slack: 0 tokens priced at 256, between the two lengths measured; 8 at the longer one
    assert [memory.stream_bytes(t) for t in (0, 8)] == [1_000 + 2_000 * 192 // 200, 3_000]
    admission = Admission(10**6, memory, used=lambda: 20_000)
    # new stream of 100 + 400 tokens; one live stream at 300 of its 1,000; prefill of 100 tokens under the round
    expected = 20_000 + 700 * 10 + (3_000 + 10 * (500 + 256 - 264)) + max(5_000, int(2.0 * 100 + 0.001 * 100 * 100))
    assert admission.projected(100, 500, [(300, 1_000)]) == expected
    assert memory.prefill_bytes(10_000) == int(2.0 * 2048 + 0.001 * 2048 * 10_000)
    assert Admission(expected, memory, used=lambda: 20_000).admits(100, 500, [(300, 1_000)])
    assert not Admission(expected - 1, memory, used=lambda: 20_000).admits(100, 500, [(300, 1_000)])
    assert Admission(40_000, memory, used=lambda: 20_000).fitting(500) == 1          # 7,920 a stream in 15,000


def test_the_app_measures_a_streams_memory_and_prints_its_admission(capsys) -> None:
    pytest.importorskip("mlx.core")
    app = make_app(lanes=2, memory_fraction=0.7)
    try:
        admission = app.scheduler.admission
        assert admission is not None and admission.budget > 0
        memory = admission.memory
        assert memory.short_tokens < memory.long_tokens and memory.per_token >= 0
        assert "memory budget" in capsys.readouterr().out
    finally:
        app.close()


def test_kv_caches_set_the_floor_of_a_streams_per_token_memory() -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache

    from tensorfold.engine.alternating_kv import AlternatingKVCache
    from tensorfold.engine.memory import _kv_bytes

    plain, alternating = KVCache(), AlternatingKVCache()
    for cache in (plain, alternating):
        cache.update_and_fetch(mx.ones((1, 2, 40, 8), dtype=mx.bfloat16), mx.ones((1, 2, 40, 8), dtype=mx.bfloat16))
    each = 2 * (2 * 8 * 2)                 # keys and values: heads x head_dim x bf16 bytes, a position
    assert _kv_bytes([plain]) == (each, 0)
    assert _kv_bytes([plain, alternating, object()]) == (2 * each, each)   # decoding adds a spare buffer
