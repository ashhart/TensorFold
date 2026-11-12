import sys
import time

import pytest

from tensorfold_web import pull


def test_bytes_on_disk_counts_partial_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    d = tmp_path / "models--org--name" / "blobs"
    d.mkdir(parents=True)
    (d / "abc").write_bytes(b"x" * 1000)
    (d / "def.incomplete").write_bytes(b"y" * 500)
    assert pull.bytes_on_disk("org/name") == 1500
    assert pull.bytes_on_disk("org/other") == 0


def test_progress_fraction_is_capped():
    p = pull.Progress("org/name", done_bytes=1200, total_bytes=1000)
    assert p.to_dict()["fraction"] == 1.0


def _wait(job, timeout=10):
    end = time.time() + timeout
    while job.state == "running" and time.time() < end:
        time.sleep(0.05)
    return job


def test_a_download_runs_as_its_own_process_and_can_be_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(pull, "command", lambda repo: [sys.executable, "-c", "import time; time.sleep(30)"])
    dl = pull.Downloads()
    job = dl.start("org/name", total_bytes=10)
    assert job.state == "running" and dl.start("org/name") is job and dl.get("org/name").state == "running"
    assert dl.cancel("org/name") and job.state == "cancelled" and job.error is None
    assert not dl.cancel("org/name")
    monkeypatch.setattr(pull, "command", lambda repo: [sys.executable, "-c", "print('warming'); print('/cache/x')"])
    job = _wait(dl.start("org/name", total_bytes=10))
    assert job.state == "done" and job.path == "/cache/x"
    monkeypatch.setattr(pull, "command", lambda repo: [sys.executable, "-c",
                                                       "import sys; sys.stderr.write('trace\\nno space left\\n'); sys.exit(2)"])
    job = _wait(dl.start("org/name", total_bytes=10))
    assert job.state == "failed" and job.error == "no space left"


def test_a_download_the_disk_cannot_hold_is_refused_before_it_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(pull, "free_bytes", lambda: 5 * 1024**3)
    dl = pull.Downloads()
    with pytest.raises(pull.NoRoom, match="needs 12.9 GB more .* and 5.4 GB are free"):
        dl.start("org/name", total_bytes=10 * 1024**3)
    assert dl.get("org/name").state == "idle"
    monkeypatch.setattr(pull, "command", lambda repo: [sys.executable, "-c", "print('ok')"])
    assert _wait(dl.start("org/name", total_bytes=1024**3)).state == "done"     # 1 GiB + the margin fits in 5 GiB
