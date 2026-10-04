"""A lone stream under --parallel keeps the graph slot's captured graphs once kept prompt ends fill every slot."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.graphs import Graphs  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, MultiDecoder  # noqa: E402


@pytest.fixture
def counted(monkeypatch):
    """Graph sets made and graphs captured, as a wrapper around the engine's own ``Graphs``."""

    count = {"sets": 0, "captures": 0}
    init, capture = Graphs.__init__, Graphs._capture

    def made(self, *args, **kwargs):
        count["sets"] += 1
        init(self, *args, **kwargs)

    def captured(self, fn):
        count["captures"] += 1
        return capture(self, fn)
    monkeypatch.setattr(Graphs, "__init__", made)
    monkeypatch.setattr(Graphs, "_capture", captured)
    return count


def _ref(w, prompt, count, sampling, kv_dtype="bf16"):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
    return serial_decode(e, prefill(e, list(prompt), sampling), count, sampling).tokens


def _run(dec, prompt, count, sampling):
    s = Stream(list(prompt), count, sampling)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    return s


def _prompts(n):
    return [[(13 * i + 7 * j + 1) % (V - 1) + 1 for i in range(5 + j)] for j in range(n)]


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_eight_different_prompts_on_five_slots_capture_what_one_slot_does(counted, kv_dtype):
    """#317: from the sixth different prompt every slot held a kept end, and each request moved and recaptured the
    graph slot's graphs. Now each request captures exactly what it does on one slot, with the same tokens."""

    w = _model()
    out, captures = {}, {}
    for slots in (1, 5):
        dec = MultiDecoder(w, slots=slots, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)
        graphs, out[slots], captures[slots] = dec.solo.graphs, [], []
        for prompt in _prompts(8):
            before = counted["captures"]
            s = _run(dec, prompt, 24, None)
            out[slots].append(s.out)
            captures[slots].append(counted["captures"] - before)
            assert s.st is dec.solo.st and dec.solo.graphs is graphs
        assert len(dec.kept) == min(slots, 8)
    assert out[5] == out[1] == [_ref(w, p, 24, None, kv_dtype) for p in _prompts(8)]
    assert captures[5] == captures[1] and sum(captures[5][5:]) == 0, captures


def test_a_warmed_graph_slot_captures_nothing_for_new_prompts_on_a_full_pool(counted):
    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, prefill_rows=16)
    dec.warm()
    sets, captures = counted["sets"], counted["captures"]
    smp = Sampling(seed=3, top_k=20, top_p=0.95)
    for prompt in _prompts(8):
        assert _run(dec, prompt, 16, smp).out == _ref(w, prompt, 16, smp)
    assert (counted["sets"], counted["captures"]) == (sets, captures)


@pytest.mark.parametrize("slots", [3, 4])
def test_two_conversations_taking_turns_on_a_full_pool_capture_once_per_slot(counted, slots):
    """Each turn resumes its own kept end in its own slot; the graphs go to that slot and stay there for the next
    turn, so after each conversation's slot has held the graphs once, no turn makes a new graph set."""

    w = _model()
    dec = MultiDecoder(w, slots=slots, capacity=1024, depth=3, confidence=0.3)
    smp = Sampling(seed=3, top_k=20, top_p=0.95)
    prompts = _prompts(slots + 3)
    for p in prompts[:slots + 1]:
        _run(dec, p, 12, smp)
    talks, kept = list(prompts[-2:]), [0, 0]
    for turn in range(4):
        if turn == 2:
            sets = counted["sets"]
        for i, p in enumerate(talks):
            s = _run(dec, p, 10, smp)
            assert s.out == _ref(w, p, 10, smp) and s.cached == kept[i]    # resumed from its own kept end
            talks[i], kept[i] = p + s.out[:-1] + [40 + turn], len(p) - 1
    assert counted["sets"] == sets


def test_a_graph_slot_without_room_for_the_copy_gives_its_graphs_to_the_stream():
    """When the graph slot cannot grow to hold a lone stream's caches, the stream decodes through graphs in its own
    slot instead of failing the round, with its serial tokens."""

    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3)
    grow = dec._grow
    dec._grow = lambda st, rows, **kw: False if dec._is_solo(st) and rows > st.capacity else grow(st, rows, **kw)
    alone = dec._solo_round
    rounds = []
    dec._solo_round = lambda s: rounds.append(s.st) or alone(s)
    smp = Sampling(seed=1234, top_k=20, top_p=0.95)
    a, b = Stream([5, 17, 99, 250], 40, None), Stream([1023, 7, 64, 300, 11, 12], 6, smp)
    dec.admit(b)                                              # b takes the graph slot
    dec.admit(a)
    b.st.resize(FIRST)                                        # the graph slot is smaller than a's caches
    a.st.resize(512)
    slot = dec.solo.st
    while dec.live():
        dec.finish(dec.round())
    assert a.error is None and [a.out, b.out] == [_ref(w, a.prompt, 40, None), _ref(w, b.prompt, 6, smp)]
    assert dec.solo.st is a.st is not slot and a.st in rounds
