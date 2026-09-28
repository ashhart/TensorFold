"""A real extension built with nvcc through ``tensorfold.cuda.build``: the lines a start prints, and a stale lock."""

import os
import select
import subprocess
import sys

NAME = "tensorfold_qwen4_exp_gdn_io"
START = "from tensorfold.families.qwen4_exp.cuda import gdn_io; gdn_io._ext()"


def _start(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", START], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=1800)


def test_the_first_start_says_it_builds_and_the_next_is_quiet(tmp_path):
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path))
    first = _start(env)
    assert first.returncode == 0, first.stderr
    assert f"[tensorfold] building CUDA extension {NAME}" in first.stdout
    again = _start(env)
    assert again.returncode == 0, again.stderr
    assert NAME not in again.stdout


def test_a_lock_left_by_a_killed_build_is_named_and_the_steps_it_names_clear_it(tmp_path):
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path))
    assert _start(env).returncode == 0                   # built once
    lock = tmp_path / NAME / "lock"
    lock.write_text("")                                  # what a start killed during the build leaves
    start = subprocess.Popen([sys.executable, "-u", "-c", START], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        ready, _, _ = select.select([start.stdout], [], [], 600)
        assert ready, "no line within 600 s"
        line = start.stdout.readline()
        assert str(lock) in line and "stop this start, delete the lock and start again" in line
        assert start.poll() is None                      # waiting on the lock
    finally:
        start.kill()
        start.wait()
    lock.unlink()
    again = _start(env)
    assert again.returncode == 0, again.stderr
    assert NAME not in again.stdout
