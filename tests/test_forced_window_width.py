"""The thinking budget's forced close is verified in one window, whatever the model's exact width."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_family_streams import END, NL, NLNL, StreamsModel  # noqa: E402

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402


class NarrowModel(StreamsModel):
    """Exact up to 2 rows a window (as checked at load); a row past that gets other bits."""

    exact_width = 2
    prefill = StreamsModel.hidden

    def hidden(self, inputs, cache, parents=None):
        self.widths.append(int(inputs.shape[-1]))
        tokens = self._feed([int(t) for t in np.array(inputs).reshape(-1)], cache)
        bits = [t if r < self.exact_width else (t + 1) % 97 for r, t in enumerate(tokens)]
        return mx.array(bits, dtype=mx.float32).reshape(1, -1, 1)


    def hidden_rows(self, windows, caches, parents=None):
        return mx.concatenate([self.hidden(mx.array(w).reshape(1, -1), c) for w, c in zip(windows, caches)], axis=1)


def _run(drafts, width=2, gpu=True, count=1, limit=12, batch_rows=32):
    model = NarrowModel(drafts=1, gpu_tokens=gpu)
    model.exact_width, model.widths = width, []
    engine = LaneEngine(model)
    engine.batch_rows = batch_rows
    streams = [LaneStream(stream_id=str(i), prompt_ids=[8, 2], max_new_tokens=limit, think_budget=4,
                          think_close=(NL, END, NLNL), think_end=END, think_open=True, drafts=drafts)
               for i in range(count)]
    for stream in streams:
        engine.add_stream(stream)
    model.widths.clear()
    while engine.active_count:
        engine.step()
    assert all(r.rows <= batch_rows for r in engine.round_stats)
    return [s.emitted for s in streams], max(model.widths, default=0)


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("limit", [5, 12])
def test_forced_windows_stay_exact(width, gpu, count, limit):
    drafted, widest = _run(True, width, gpu, count, limit)
    serial, serial_width = _run(False, width, gpu, 1, limit)
    assert widest <= width
    assert serial_width <= 1
    assert drafted == serial * count


def test_forced_windows_fit_the_shared_row_budget():
    drafted, widest = _run(True, 3, False, 3, batch_rows=2)
    assert widest <= 2
    assert drafted == _run(False, 3, False)[0] * 3
