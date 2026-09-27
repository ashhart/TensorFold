from types import SimpleNamespace

from tensorfold.server import app


def run_watch(monkeypatch, frames):
    engine = SimpleNamespace(active_count=0, prefill_chunks=0)
    scheduler = app.Scheduler(engine, lanes=1, eos_ids=frozenset())
    scheduler._starting = object()
    scheduler.stall_prefill_s = 10
    clock = [0.0]
    ticks = iter(frames)

    class Stop:
        def wait(self, _timeout):
            try:
                clock[0], engine.prefill_chunks = next(ticks)
            except StopIteration:
                return True
            return False

    scheduler._stop = Stop()
    dumps = []
    monkeypatch.setattr(app.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("faulthandler.dump_traceback", lambda **kw: dumps.append(kw))
    scheduler._watch()
    return dumps


def test_prompt_progress_prevents_a_false_stall_dump(monkeypatch, capsys):
    assert run_watch(monkeypatch, [(1, 0), (20, 1), (40, 2), (60, 3)]) == []
    assert "stalled" not in capsys.readouterr().out


def test_a_genuine_prefill_stall_is_dumped_once(monkeypatch, capsys):
    assert len(run_watch(monkeypatch, [(1, 0), (20, 0), (40, 0)])) == 1
    assert capsys.readouterr().out.count("stalled") == 1


def test_progress_resets_the_stall_warning(monkeypatch, capsys):
    assert len(run_watch(monkeypatch, [(1, 0), (20, 0), (21, 1), (40, 1)])) == 2
    assert capsys.readouterr().out.count("stalled") == 2


def run_decode_watch(monkeypatch, frames):
    engine = SimpleNamespace(active_count=1, prefill_chunks=1)
    scheduler = app.Scheduler(engine, lanes=1, eos_ids=frozenset())
    scheduler._jobs["job"] = object()
    scheduler.stall_s = 120
    scheduler.stall_prefill_s = 900
    clock = [0.0]
    ticks = iter(frames)

    class Stop:
        def wait(self, _timeout):
            try:
                clock[0], starting, scheduler.rounds = next(ticks)
            except StopIteration:
                return True
            scheduler._starting = object() if starting else None
            return False

    scheduler._stop = Stop()
    dumps = []
    monkeypatch.setattr(app.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("faulthandler.dump_traceback", lambda **kw: dumps.append(kw))
    scheduler._watch()
    return dumps


def test_decode_does_not_inherit_a_long_prefill_stall_timer(monkeypatch, capsys):
    frames = [(1, True, 0), (129, False, 0), (130, False, 1)]
    assert run_decode_watch(monkeypatch, frames) == []
    assert "stalled" not in capsys.readouterr().out


def test_genuine_decode_stall_after_prefill_is_dumped_once(monkeypatch, capsys):
    frames = [(1, True, 0), (129, False, 0), (250, False, 0), (300, False, 0)]
    assert len(run_decode_watch(monkeypatch, frames)) == 1
    output = capsys.readouterr().out
    assert "stalled 121s" in output
    assert "starting=False" in output


def test_decode_rounds_reset_the_stall_timer(monkeypatch, capsys):
    frames = [(1, False, 1), (119, False, 2), (237, False, 3), (355, False, 4)]
    assert run_decode_watch(monkeypatch, frames) == []
    assert "stalled" not in capsys.readouterr().out
