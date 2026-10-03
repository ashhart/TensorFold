"""A failed or cut-short snapshot write leaves no ``.partial.safetensors``: errors and startup both remove them."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402
from tensorfold.server.checkpoints import CheckpointEntry, spill_conversation  # noqa: E402

MODEL = "/models/qwen|mlx=1"


class Layer:
    def __init__(self) -> None:
        self.state = mx.arange(64).reshape(8, 8)
        self.offset = 8


def _disk_full(monkeypatch):
    """``mx.save_safetensors`` that writes some bytes, then fails as a full disk does."""

    def fail(path, arrays, metadata=None):
        with open(path, "wb") as handle:
            handle.write(b"\0" * 4096)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ps.mx, "save_safetensors", fail)


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_failed_write_removes_its_partial_and_raises(tmp_path, monkeypatch):
    _disk_full(monkeypatch)
    with pytest.raises(OSError):
        ps.save_snapshot(tmp_path, MODEL, [1, 2, 3], [Layer()])
    assert list(tmp_path.iterdir()) == []


def test_a_failed_spill_leaves_the_directory_as_it_was(tmp_path, monkeypatch):
    ps.save_snapshot(tmp_path, MODEL, [1, 2], [Layer()])
    before = sorted(p.name for p in tmp_path.iterdir())
    _disk_full(monkeypatch)
    entry = CheckpointEntry(tokens=[1, 2, 3, 4], cache=[Layer()], last_prompt=[1, 2, 3, 4], nbytes=512)
    assert spill_conversation(entry, tmp_path, MODEL, limit_bytes=1 << 30) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_each_process_writes_its_own_partial(tmp_path, monkeypatch):
    written = []
    save = ps.mx.save_safetensors

    def record(path, arrays, metadata=None):
        written.append(os.path.basename(path))
        save(path, arrays, metadata=metadata)

    monkeypatch.setattr(ps.mx, "save_safetensors", record)
    target = ps.save_snapshot(tmp_path, MODEL, [4, 5], [Layer()])
    assert written == [f"{target.stem}.{os.getpid()}.partial.safetensors"]
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_startup_removes_partials_of_processes_that_are_gone(tmp_path):
    kept = ps.save_snapshot(tmp_path, MODEL, [7, 8], [Layer()])
    dead = tmp_path / f"{'a' * 32}.{_dead_pid()}.partial.safetensors"
    live = tmp_path / f"{'b' * 32}.{os.getpid()}.partial.safetensors"
    old = tmp_path / f"{'c' * 32}.partial.safetensors"           # 0.6.0's name, no pid: removed once it is stale
    fresh = tmp_path / f"{'d' * 32}.partial.safetensors"
    for path in (dead, live, old, fresh):
        path.write_bytes(b"\0" * 1000)
    stale = time.time() - ps.UNNAMED_PARTIAL_SECONDS - 60
    os.utime(old, (stale, stale))
    assert ps.remove_stale_partials(tmp_path) == 2000
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([kept.name, live.name, fresh.name])


def test_removing_partials_tolerates_a_missing_directory(tmp_path):
    assert ps.remove_stale_partials(tmp_path / "absent") == 0


def test_a_starting_server_clears_both_snapshot_directories(tmp_path, capsys):
    from tensorfold.server.checkpoints import CheckpointStore
    from tensorfold.server.scheduler import Scheduler
    from tests.lane_fakes import FakeEngine

    blocks, sessions = tmp_path / "prefix-snapshots", tmp_path / "session-snapshots"
    pid = _dead_pid()
    for directory in (blocks, sessions):
        directory.mkdir()
        (directory / f"{'e' * 32}.{pid}.partial.safetensors").write_bytes(b"\0" * 2048)
    Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=CheckpointStore(2, copier=lambda c: c),
              snapshot_dir=blocks, session_dir=sessions, model_id=MODEL)
    assert list(blocks.iterdir()) == [] and list(sessions.iterdir()) == []
    assert capsys.readouterr().out.count("unfinished snapshot writes") == 2
