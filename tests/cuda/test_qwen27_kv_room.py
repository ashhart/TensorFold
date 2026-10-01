"""A serial run evicts a kept conversation only when its grow needs the room; either way the next prompt is exact."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.streams import KVRoom, PrefixCache  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import prefill  # noqa: E402

from test_qwen27_prefill import _model, _prompt, _same_state  # noqa: E402

BUFFER = 1024 * 128 * 2 * 2                              # the tiny model's 1,024-row attention keys and values


@pytest.mark.parametrize("budget,kept", [(4 * BUFFER, True), (BUFFER + BUFFER // 2, False)])
def test_a_serial_run_evicts_the_drafted_conversation_only_for_room(budget, kept):
    w = _model()
    cache = PrefixCache(4)
    room = KVRoom(cache, budget)
    drafted, serial = _prompt(600, seed=31), _prompt(600, seed=32)
    st, _ = prefill(w, drafted[:500], None, room=room)
    cache.add(drafted[:500], st, None)                    # the drafted conversation's kept entry
    prefill(w, serial, None, room=room)                   # a serial run on buffers of its own
    assert bool(cache.entries) == kept
    hit = cache.longest(drafted)
    assert (hit is not None) == kept
    resumed, first = prefill(w, drafted, None, state=hit[1] if hit else None, room=room)
    fresh, first_fresh = prefill(w, drafted, None)
    _same_state(fresh, resumed)
    assert first == first_fresh
