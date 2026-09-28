"""Under an emulated M1/M2 thread limit every launch fits and every output keeps the unconstrained run's bits."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels import threads  # noqa: E402

HERE = Path(__file__).parent


def _run(scenario: str, limit: int) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent / "src"), str(HERE)]))
    done = subprocess.run([sys.executable, str(HERE / "limit_scenarios.py"), scenario, str(limit)],
                          capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scenario", ["simd_qmm", "norm", "row_forward", "sampling", "nemotron", "row_attention",
                                      "flash_next"])
def test_every_launch_fits_a_lower_limit_with_the_same_bits(scenario):
    free = _run(scenario, 0)
    assert free["guessed"] == 0
    for limit in (448, 256):
        got = _run(scenario, limit)
        assert got["out"] == free["out"], f"limit {limit}: outputs differ"
        over = {k: v for k, v in got["largest"].items() if v > got["reserved"].get(k, limit)}
        assert not over, f"launches past the limit {limit}: {over}"
        assert got["guessed"] == 0


def test_the_limit_is_read_from_mlx_s_own_error():
    kernel = mx.fast.metal_kernel(name="tf_limit_probe", input_names=["X"], output_names=["Y"],
                                  source="Y[thread_position_in_grid.x] = X[thread_position_in_grid.x];")
    x = mx.zeros((4096,))
    with pytest.raises(ValueError) as err:
        mx.eval(kernel(inputs=[x], grid=(4096, 1, 1), threadgroup=(2048, 1, 1), output_shapes=[(4096,)],
                       output_dtypes=[mx.float32])[0])
    assert 32 <= (threads.limit_in(err.value) or 0) <= 1024


def test_a_launch_that_cannot_fit_names_the_chip(monkeypatch):
    monkeypatch.setattr(threads, "probing", True)

    def launch(size):
        raise ValueError(f"Thread group size ({size}) is greater than  the maximum allowed threads per threadgroup "
                         "(8).")

    with pytest.raises(RuntimeError) as err:
        threads.fit(("tf test kernel", 1), [512, 64], launch)
    assert threads.chip() in str(err.value) and "allows 8 threads" in str(err.value)


def test_a_fitted_launch_is_remembered(monkeypatch):
    monkeypatch.setattr(threads, "probing", True)
    calls = []

    def launch(size):
        calls.append(size)
        if size > 256:
            raise ValueError(f"Thread group size ({size}) is greater than  the maximum allowed threads per threadgroup "
                             "(300).")
        return mx.zeros((1,))

    threads.fit(("tf test kernel", 2), [512, 256, 128], launch)
    threads.fit(("tf test kernel", 2), [512, 256, 128], launch)
    assert calls == [512, 256, 256]


def test_no_probe_where_every_pipeline_takes_1024(monkeypatch):
    monkeypatch.setattr(threads, "probing", False)
    calls = []

    def launch(size):
        calls.append(size)
        return mx.zeros((1,))

    threads.fit(("tf test kernel", 3), [512, 256], launch, [mx.zeros((4,))])
    threads.fit(("tf test kernel", 3), [512, 256], launch)
    assert calls == [512, 512]
