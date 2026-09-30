"""Issue #72: rounds run between a long prompt's chunks, bounded, each stream still its solo run; stops between."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tensorfold.engine.prefill_plan import PrefillPlan
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.lane_fakes import FakeEngine, fake_serial


class GridEngine(FakeEngine):
    prefill_plan = PrefillPlan(4)          # a chunk every 4 prompt tokens


def _prompt(seed: int, n: int = 22) -> list[int]:
    return [(seed * 13 + 7 * j) % 90 + 5 for j in range(n)]       # 22 tokens: chunks at 0, 4, ..., 20 (six)


def _run(scheduler: Scheduler, jobs: list[ChatJob], on_fill: Any = None) -> list[tuple[str, Any]]:
    """Submit ``jobs``, run the scheduler's own loop on this thread until all end; the fills and rounds in order."""

    events: list[tuple[str, Any]] = []
    fill, step, finish = scheduler._fill, scheduler.engine.step, scheduler._finish

    def fill_step(abort: Any = None) -> None:
        job = scheduler._filling.job
        events.append(("fill", job.job_id))
        fill(abort)
        if on_fill is not None:
            on_fill(job, events)

    def round_step() -> Any:
        events.append(("round", tuple(sorted(s.stream_id for s, _ in scheduler.engine._live if not s.finished))))
        return step()

    def finished(job: ChatJob) -> None:
        finish(job)
        if all(j.done.is_set() for j in jobs):
            scheduler._stop.set()

    scheduler._fill, scheduler.engine.step, scheduler._finish = fill_step, round_step, finished
    for job in jobs:
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


def test_rounds_run_between_a_long_prompts_chunks_and_every_stream_equals_its_solo_run():
    scheduler = Scheduler(GridEngine(), lanes=4, eos_ids=frozenset({-1}))
    scheduler.decode_share, scheduler.fill_rounds = 1e9, 2         # exactly two rounds between two chunks
    live = ChatJob("live", [5, 6], 60, 0.0)
    longs = [ChatJob(f"long{i}", _prompt(i), 6, 0.0) for i in range(3)]
    events = _run(scheduler, [live, *longs])
    for job in [live, *longs]:
        assert job.error is None and _tokens(job) == fake_serial(job.prompt_ids, job.max_tokens, {-1})
    fills = [who for kind, who in events if kind == "fill"]
    assert fills == ["live"] + [f"long{i}" for i in range(3) for _ in range(6)]      # one prompt at a time, in order
    first = [i for i, (kind, who) in enumerate(events) if kind == "fill" and who == "long0"]
    between = [[e for e in events[a + 1:b] if e[0] == "round"] for a, b in zip(first, first[1:])]
    assert [len(r) for r in between] == [2] * 5 and all("live" in r[1] for rs in between for r in rs)


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
    assert scheduler._filling is None and all(s.stream_id != "long" for s, _ in scheduler.engine._live)


def test_a_waiting_foreground_prompt_stops_a_background_prefill_between_chunks():
    scheduler = Scheduler(GridEngine(), lanes=3, eos_ids=frozenset({-1}))
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
    fills = [who for kind, who in events if kind == "fill"]
    assert fills == ["live", "background", "background", "foreground", "foreground"]
    assert _tokens(live) == fake_serial([5, 6], 40, {-1})


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
