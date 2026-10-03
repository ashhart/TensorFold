"""Issues #72 and #90: open prompts fill a chunk a step beside each other and the rounds, each stream its solo run."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tensorfold.engine.prefill_plan import PrefillPlan
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.lane_fakes import FakeBatchItem, FakeEngine, fake_serial


class GridEngine(FakeEngine):
    prefill_plan = PrefillPlan(4)          # a chunk every 4 prompt tokens


def _prompt(seed: int, n: int = 22) -> list[int]:
    return [(seed * 13 + 7 * j) % 90 + 5 for j in range(n)]       # 22 tokens: chunks at 0, 4, ..., 20 (six)


def _run(scheduler: Scheduler, jobs: list[ChatJob], on_fill: Any = None) -> list[tuple[str, Any]]:
    """Submit ``jobs``, run the scheduler's own loop on this thread until all end; the fills and rounds in order."""

    events: list[tuple[str, Any]] = []
    fill, step, finish = scheduler._fill, scheduler.engine.step, scheduler._finish

    def fill_step(filling: Any = None, abort: Any = None) -> None:
        filling = filling or scheduler._next_fill()
        events.append(("fill", filling.job.job_id))
        fill(filling, abort)
        if on_fill is not None:
            on_fill(filling.job, events)

    def round_step() -> Any:
        events.append(("round", tuple(sorted(s.stream_id for s, _ in scheduler.engine._live if not s.finished))))
        return step()

    def finished(job: ChatJob) -> None:
        finish(job)
        if all(j.done.is_set() for j in jobs):
            scheduler._stop.set()

    scheduler._fill, scheduler.engine.step, scheduler._finish = fill_step, round_step, finished
    for job in list(jobs):
        scheduler.submit(job)
    scheduler._loop()
    return events


def _tokens(job: ChatJob) -> list[int]:
    out: list[int] = []
    while True:
        chunk = job.chunks.get_nowait()
        if chunk is None:
            return out
        out += chunk


def _fills(events: list[tuple[str, Any]]) -> list[str]:
    return [who for kind, who in events if kind == "fill"]


def _solo(job: ChatJob) -> bool:
    return job.error is None and _tokens(job) == fake_serial(job.prompt_ids, job.max_tokens, {-1})


def test_rounds_run_between_prompt_chunks_and_every_stream_equals_its_solo_run():
    scheduler = Scheduler(GridEngine(), lanes=4, eos_ids=frozenset({-1}))
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 2         # exactly two rounds between two chunks
    live = ChatJob("live", [5, 6], 60, 0.0)
    longs = [ChatJob(f"long{i}", _prompt(i), 6, 0.0) for i in range(3)]
    events = _run(scheduler, [live, *longs])
    assert all(_solo(job) for job in [live, *longs])
    # all three open at once; the fewest tokens left go first, and long2, passed over 8 chunks, takes the next one
    assert _fills(events) == ["live", *["long0"] * 6, "long1", "long1", "long2", *["long1"] * 4, *["long2"] * 5]
    marks = [i for i, (kind, who) in enumerate(events) if kind == "fill"]
    between = [[e for e in events[a + 1:b] if e[0] == "round"] for a, b in zip(marks[1:], marks[2:])]
    assert [len(r) for r in between] == [2] * (len(marks) - 2) and all("live" in r[1] for rs in between for r in rs)


def test_the_rounds_take_their_share_of_the_chunks_time():
    clock = {"t": 0.0}
    engine = GridEngine()
    hidden = engine.model.hidden

    def timed(rows: Any, cache: Any, parents: Any = None) -> Any:
        clock["t"] += float(rows.size)                             # a row a unit: a 4-row chunk takes 4, a round 1
        return hidden(rows, cache, parents)

    engine.model.hidden = timed
    scheduler = Scheduler(engine, lanes=2, eos_ids=frozenset({-1}))
    scheduler.clock = lambda: clock["t"]
    scheduler.decode_share, scheduler.fill_rounds = 0.5, 32
    live, long = ChatJob("live", [5, 6], 80, 0.0), ChatJob("long", _prompt(1, 42), 4, 0.0)
    events = _run(scheduler, [live, long])
    marks = [i for i, (kind, who) in enumerate(events) if kind == "fill" and who == "long"]
    rounds = [sum(1 for e in events[a + 1:b] if e[0] == "round") for a, b in zip(marks, marks[1:])]
    assert rounds == [2] * 10                                      # half of a 4-unit chunk: two 1-unit rounds
    assert _tokens(live) == fake_serial([5, 6], 80, {-1}) and _tokens(long) == fake_serial(long.prompt_ids, 4, {-1})


def test_a_short_prompt_arriving_mid_fill_starts_at_the_next_chunk_and_answers_first():
    scheduler = Scheduler(GridEngine(), lanes=4, eos_ids=frozenset({-1}))
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 1
    live = ChatJob("live", [5, 6], 80, 0.0)
    long = ChatJob("long", _prompt(5, 60), 4, 0.0)                 # 15 chunks
    short = ChatJob("short", _prompt(6, 6), 4, 0.0)                # 2 chunks
    jobs = [live, long]

    def arrive(job: ChatJob, events: list[Any]) -> None:
        if job is long and _fills(events).count("long") == 3 and short not in jobs:
            jobs.append(short)
            scheduler.submit(short)

    events = _run(scheduler, jobs, on_fill=arrive)
    assert _fills(events) == ["live", "long", "long", "long", "short", "short", *["long"] * 12]
    assert short.prefilled_at < long.prefilled_at
    assert all(_solo(job) for job in jobs)


def test_the_age_guard_bounds_a_long_prompts_wait_under_endless_short_prompts():
    """No starvation: between two of a prompt's chunks at most fill_guard + open prompts - 1 others run."""

    scheduler = Scheduler(GridEngine(), lanes=2, eos_ids=frozenset({-1}))
    scheduler.fill_guard = 3
    long = ChatJob("long", _prompt(7, 40), 4, 0.0)                 # 10 chunks
    jobs = [long]

    def arrive(job: ChatJob, events: list[Any]) -> None:
        if not long.done.is_set() and all(j.done.is_set() for j in jobs[1:]):
            short = ChatJob(f"short{len(jobs)}", _prompt(len(jobs), 3), 1, 0.0)     # one chunk; its reply ends there
            jobs.append(short)
            scheduler.submit(short)

    events = _run(scheduler, jobs, on_fill=arrive)
    fills = _fills(events)
    marks = [i for i, who in enumerate(fills) if who == "long"]
    assert len(marks) == 10 and long.done.is_set() and _solo(long)
    assert [b - a - 1 for a, b in zip(marks, marks[1:])] == [3] * 9          # exactly fill_guard shorts between
    assert all(_solo(job) for job in jobs[1:])


def test_a_cancelled_prompt_keeps_its_own_progress_while_another_fills_between_its_chunks():
    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    engine = GridEngine()
    scheduler = Scheduler(engine, lanes=3, eos_ids=frozenset({-1}), checkpoints=store)
    scheduler.fill_guard = 1                                       # the two prompts alternate chunk by chunk
    first, other = ChatJob("first", _prompt(8), 4, 0.0), ChatJob("other", _prompt(9), 4, 0.0)
    store.insert(other.prompt_ids[:8], [FakeBatchItem([other.prompt_ids[:8]])], last_prompt=other.prompt_ids[:8])

    def cancel(job: ChatJob, events: list[Any]) -> None:
        if job is other and _fills(events).count("other") == 2:
            first.cancellation.cancel()                            # first holds 8 rows, other 16

    events = _run(scheduler, [first, other], on_fill=cancel)
    assert _fills(events)[:5] == ["first", "other", "first", "other", "first"]     # the last: the stop between chunks
    assert isinstance(first.error, RequestCancelled) and _solo(other)
    kept = {len(e.tokens): e for e in store._entries if e.tokens == first.prompt_ids[:len(e.tokens)]}
    assert list(kept) == [8] and kept[8].cache[0].rows[0] == first.prompt_ids[:8]      # its own rows, not other's 16
    retry = ChatJob("retry", first.prompt_ids, 4, 0.0)
    _run(Scheduler(engine, lanes=3, eos_ids=frozenset({-1}), checkpoints=store), [retry])
    assert engine.prefill_calls[-1] == ("retry", 8) and _solo(retry)          # resumed there, equal to fresh


def test_a_prompt_waiting_for_its_first_chunk_resumes_from_what_an_earlier_prompt_stored_meanwhile():
    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    engine = GridEngine()
    scheduler = Scheduler(engine, lanes=3, eos_ids=frozenset({-1}), checkpoints=store)
    shared = _prompt(10, 12)                                       # a system block both prompts start with
    first = ChatJob("first", shared + _prompt(11, 10), 4, 0.0, shared_prefix_lens=(12,))
    second = ChatJob("second", shared + _prompt(12, 14), 4, 0.0, shared_prefix_lens=(12,))
    jobs = [first]

    def arrive(job: ChatJob, events: list[Any]) -> None:
        if job is first and second not in jobs:
            jobs.append(second)
            scheduler.submit(second)                               # opens before the block is stored

    events = _run(scheduler, jobs, on_fill=arrive)
    assert _fills(events) == ["first"] * 6 + ["second"] * 4        # 12 of its 26 tokens came from first's block
    assert engine.prefill_calls == [("first", 0), ("second", 12)] and _solo(first) and _solo(second)


def test_a_prompt_cancelled_before_its_first_chunk_ends_without_one():
    engine = GridEngine()
    scheduler = Scheduler(engine, lanes=3, eos_ids=frozenset({-1}))
    short, long = ChatJob("short", _prompt(14, 8), 4, 0.0), ChatJob("long", _prompt(15, 30), 4, 0.0)

    def cancel(job: ChatJob, events: list[Any]) -> None:
        if job is short and _fills(events).count("short") == 2:
            long.cancellation.cancel()                             # open, waiting behind short: not started

    events = _run(scheduler, [short, long], on_fill=cancel)
    assert _fills(events) == ["short", "short", "long"]            # the last: its stop, before any chunk
    assert isinstance(long.error, RequestCancelled) and _solo(short)
    assert [who for who, _ in engine.prefill_calls] == ["short"]


def test_a_prompt_cancelled_between_chunks_stops_there_and_keeps_its_progress():
    store = CheckpointStore(4, copier=lambda c: c)
    scheduler = Scheduler(GridEngine(), lanes=2, eos_ids=frozenset({-1}), checkpoints=store)
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 1
    live, long = ChatJob("live", [5, 6], 30, 0.0), ChatJob("long", _prompt(2), 6, 0.0)

    def cancel_after_two(job: ChatJob, events: list[Any]) -> None:
        if job is long and sum(1 for e in events if e == ("fill", "long")) == 2:
            long.cancellation.cancel()

    events = _run(scheduler, [live, long], on_fill=cancel_after_two)
    assert isinstance(long.error, RequestCancelled) and long.done.is_set()
    assert sum(1 for e in events if e == ("fill", "long")) == 3             # two chunks, then the stop between chunks
    assert [e.tokens for e in store._entries if e.tokens == long.prompt_ids[:8]]   # a retry resumes at 8
    assert _tokens(live) == fake_serial([5, 6], 30, {-1})
    assert not scheduler._fills and all(s.stream_id != "long" for s, _ in scheduler.engine._live)


def test_a_foreground_prompt_fills_before_a_background_one_beside_it_without_stopping_it():
    scheduler = Scheduler(GridEngine(), lanes=3, eos_ids=frozenset({-1}))
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 1
    live = ChatJob("live", [5, 6], 40, 0.0)
    background = ChatJob("background", _prompt(3), 4, 0.0, background=True)
    foreground = ChatJob("foreground", _prompt(4, 6), 4, 0.0)
    jobs = [live, background]

    def arrive(job: ChatJob, events: list[Any]) -> None:
        if job is background and _fills(events).count("background") == 2 and foreground not in jobs:
            jobs.append(foreground)
            scheduler.submit(foreground)

    events = _run(scheduler, jobs, on_fill=arrive)
    assert _fills(events) == ["live", "background", "background", "foreground", "foreground", *["background"] * 4]
    assert not background.preempted and scheduler.preemptions == 0
    assert all(_solo(job) for job in jobs)


def test_a_waiting_foreground_prompt_stops_a_background_prefill_between_chunks_when_no_lane_is_free():
    scheduler = Scheduler(GridEngine(), lanes=2, eos_ids=frozenset({-1}))
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 1
    live = ChatJob("live", [5, 6], 40, 0.0)
    background = ChatJob("background", _prompt(3), 4, 0.0, background=True)
    foreground = ChatJob("foreground", _prompt(4, 6), 4, 0.0)

    jobs = [live, background]

    def arrive(job: ChatJob, events: list[Any]) -> None:
        if job is background and sum(1 for e in events if e == ("fill", "background")) == 2:
            jobs.append(foreground)
            scheduler.submit(foreground)

    events = _run(scheduler, jobs, on_fill=arrive)
    assert background.preempted and background.error is None and background.done.is_set()
    assert foreground.error is None and _tokens(foreground) == fake_serial(foreground.prompt_ids, 4, {-1})
    assert _fills(events) == ["live", "background", "background", "foreground", "foreground"]
    assert _tokens(live) == fake_serial([5, 6], 40, {-1})


def test_a_job_that_fits_only_without_an_open_prompt_waits_for_it_and_is_not_refused():
    from tensorfold.engine.memory import Admission, StreamMemory

    # 100 bytes up to 1,000 tokens, then 1 a token; rounds take 50; 1,000 always in use
    memory = StreamMemory(short_tokens=1000, short=100, long_tokens=2000, long=1100, per_token=1.0, prefill_a=0.0,
                          prefill_b=0.0, round_bytes=50)
    engine = FakeEngine()                                          # each prompt is one chunk
    live, opened, waiting = (ChatJob(n, _prompt(i, 10), 40, 0.0) for i, n in enumerate(["live", "opened", "waiting"]))
    # beside the live stream (11 -> 50 tokens) and the open prompt (0 -> 50): 1,000 + 39 + 50 + 100 + 50
    scheduler = Scheduler(engine, lanes=4, eos_ids=frozenset(), admission=Admission(1238, memory, used=lambda: 1000))
    scheduler.decode_share = 1e9
    scheduler.submit(live)
    scheduler._admit()                                             # alone: it fills at once and joins the rounds
    for job in (opened, waiting):
        scheduler.submit(job)
    scheduler._admit()
    assert [f.job for f in scheduler._fills] == [opened] and scheduler._held is waiting and waiting.error is None
    while scheduler._fills:
        scheduler._fill()
    scheduler._admit()                                             # the prompt became a stream: 1,000 + 39 + 39 + 150
    assert [f.job for f in scheduler._fills] == [waiting] and scheduler._held is None


def test_an_open_prompt_waits_while_streams_hold_its_memory_and_finishes_after_them_instead_of_failing():
    from tensorfold.server.prompt_memory import PromptMemory
    from tests.test_prompt_memory import Runtime

    runtime = Runtime(resident=1000)
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(2000, model, runtime=runtime, overhead_bytes=0, bootstrap_bytes=0)
    engine = GridEngine()
    scheduler = Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), prompt_memory=memory)
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 1
    live, long = ChatJob("live", [5, 6], 12, 0.0), ChatJob("long", _prompt(13), 4, 0.0)
    step = engine.step

    def rounds() -> Any:
        got = step()
        if live.stream is not None and live.stream.finished:
            runtime.resident = 1000                                # the stream ended: its memory is back
        return got

    def squeeze(job: ChatJob, events: list[Any]) -> None:
        if job is long and _fills(events).count("long") == 2:
            runtime.resident = 2500                                # streams now hold more than its admission saw

    engine.step = rounds
    events = _run(scheduler, [live, long], on_fill=squeeze)
    marks = [i for i, (kind, who) in enumerate(events) if kind == "fill" and who == "long"]
    assert _fills(events) == ["live", *["long"] * 6] and long.error is None
    assert all("live" not in e[1] for e in events[marks[2]:] if e[0] == "round")     # it waited for the stream
    assert _solo(live) and _solo(long)


def test_a_prompt_takes_a_pass_only_while_no_other_prompt_is_open():
    from tests.test_prefill_pass_hook import PassFamily

    def filled(jobs: list[ChatJob]) -> list[tuple[int, ...]]:
        family = PassFamily()
        engine = FakeEngine(family, prefill_pass=4)
        engine.prefill_plan = PrefillPlan(32)
        scheduler = Scheduler(engine, lanes=3, eos_ids=frozenset({-1}))
        for job in jobs:
            scheduler._open_job(job)
        while scheduler._fills:
            scheduler._fill()
        return family.passes

    prompt = [(5 * i + 3) % 90 + 1 for i in range(150)]
    assert filled([ChatJob("alone", prompt, 4, 0.0)]) == [(32, 32, 32, 32)]          # alone: a pass, then the tail
    other = [(7 * i + 2) % 90 + 1 for i in range(150)]
    assert filled([ChatJob("a", prompt, 4, 0.0), ChatJob("b", other, 4, 0.0)]) == []   # beside another: a chunk a forward


def test_decode_share_zero_prefills_each_prompt_whole_before_any_round_as_before():
    scheduler = Scheduler(GridEngine(), lanes=4, eos_ids=frozenset({-1}), decode_share=0)
    live = ChatJob("live", [5, 6], 30, 0.0)
    longs = [ChatJob(f"long{i}", _prompt(i), 6, 0.0) for i in range(2)]
    events = _run(scheduler, [live, *longs])
    fills = [i for i, (kind, _) in enumerate(events) if kind == "fill"]
    assert [events[i][1] for i in fills] == ["live"] + ["long0"] * 6 + ["long1"] * 6
    assert fills == list(range(len(fills)))                        # no round before the last prompt's last chunk
    for job in [live, *longs]:
        assert _tokens(job) == fake_serial(job.prompt_ids, job.max_tokens, {-1})


def test_the_serve_option_is_refused_where_it_cannot_apply_before_any_download():
    from tensorfold import cli, serve_options

    family = SimpleNamespace(title="Qwen3.8 dense", package=SimpleNamespace(), model_type="qwen3_5")
    parse = lambda *flags: cli.build_parser().parse_args(["serve", "owner/model", *flags])      # noqa: E731
    assert parse().decode_share is None and parse("--decode-share", "0").decode_share == 0.0
    serve_options.check(parse("--decode-share", "0"), family, "mlx")
    with pytest.raises(ValueError, match="0 .whole prompts first. or more"):
        serve_options.check(parse("--decode-share", "-0.5"), family, "mlx")
    with pytest.raises(ValueError, match="this CUDA engine runs a round after each 1,024 prompt rows"):
        serve_options.check(parse("--decode-share", "0.25"), family, "cuda")
    flash = SimpleNamespace(title="Flash Next", package=SimpleNamespace(CUDA_DECODE_SHARE=True), model_type="qwen4_exp")
    serve_options.check(parse("--decode-share", "0.25"), flash, "cuda")             # its passes take the share
