"""A verify row samples independently of other rows in the window."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.family import Qwen35Family  # noqa: E402

V = 512


@pytest.mark.parametrize("reverse", [False, True])
def test_a_verify_row_draws_what_the_serial_step_draws(reverse):
    wide = mx.zeros((1, V), dtype=mx.bfloat16)                       # flat: its nucleus passes 256 candidates
    narrow = mx.full((1, V), -100.0, dtype=mx.bfloat16)
    narrow[0, 3] = 0.0
    narrow[0, 7] = 0.0                                                # two equal tokens, half the mass each
    window = mx.concatenate([narrow, wide] if reverse else [wide, narrow])
    differing = []
    for seed in range(64):
        s = Sampling(seed=seed, temperature=1.0, top_k=0, top_p=0.5)
        serial = int(Qwen35Family.sample(None, narrow, s, [101])[0])
        drafted = int(Qwen35Family.sample(None, window, s, [101, 100] if reverse else [100, 101])[0 if reverse else 1])
        wide_alone = Qwen35Family.sample(None, wide, s, [100])[0]
        assert Qwen35Family.sample(None, window, s, [101, 100] if reverse else [100, 101])[1 if reverse else 0] == wide_alone
        streams = Qwen35Family.sample_streams(None, [window, narrow], [s, s],
                                               [[101, 100] if reverse else [100, 101], [101]])
        assert streams[0][0 if reverse else 1] == drafted
        assert streams[1][0] == serial
        if serial != drafted:
            differing.append((seed, serial, drafted))
    assert not differing, differing
