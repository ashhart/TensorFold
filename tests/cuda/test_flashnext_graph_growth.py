"""Graph captures across a conversation whose context grows turn by turn (issue #336): a turn whose window
crosses a context bucket or grows the graph slot captures mid-reply, turns inside a bucket capture nothing, and
every reply matches serial decoding's tokens — including while an earlier engine's graphs are still alive."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

CAP, DEPTH, TURNS = 40000, 10, 5  # a 9000-token start and 3000-token turns cross 16384 and 32768
SAMPLING = Sampling(seed=23, top_k=20, top_p=0.95)


def _turns(outs):
    """The conversation's prompts: a 9000-token start, then each reply plus 3000 fresh tokens."""
    gen = torch.Generator().manual_seed(7)
    prompt = torch.randint(5, 4000, (9000,), generator=gen).tolist()
    for i in range(TURNS):
        yield prompt
        prompt = prompt + outs[i] + torch.randint(5, 4000, (3000,), generator=gen).tolist()


def _run_graphs(w, keep=None):
    """Each turn as its own request through one shared decoder; returns (replies, captures per turn)."""
    dec = MultiDecoder(w, slots=2, capacity=CAP, depth=DEPTH, confidence=0.3, prefill_rows=128)
    assert dec.solo is not None
    graphs, last, per_turn = dec.solo.graphs, 0, []
    outs = []
    for prompt in _turns(outs):                                # the chain grows as replies land
        s = Stream(list(prompt), 40, SAMPLING, draft=True, stop_eos=False)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        if dec.solo.graphs is not graphs:                      # the slot was reallocated: a fresh counter
            graphs, last = dec.solo.graphs, 0
        per_turn.append(dec.solo.graphs.captures - last)
        last = dec.solo.graphs.captures
        outs.append(list(s.out))
        if keep is not None:
            keep.append(dec)
    return outs, per_turn


def _serial(w, outs):
    """Serial references for every turn's reply over the same growing conversation."""
    refs = []
    for prompt in _turns(outs):
        e = Engine(w, capacity=CAP, max_rows=DEPTH + 1, prefill_rows=128)
        refs.append(serial_decode(e, prefill(e, prompt, SAMPLING), 40, SAMPLING).tokens)
    return refs


def test_growing_conversation_captures_only_where_its_window_crosses():
    w = _model()
    outs, per_turn = _run_graphs(w)
    assert per_turn[0] > 0 and per_turn[3] > 0, per_turn       # the 16K and 32K crossings captured a set
    assert per_turn[1] == per_turn[2] == 0, per_turn           # turns inside a bucket: nothing
    assert per_turn[4] <= 1, per_turn        # at most a straggler key the crossing turn's rounds never hit
    assert outs == _serial(w, outs)                            # and every reply is serial's bits


def test_captures_stay_exact_while_an_earlier_engines_graphs_are_alive():
    """The pre-0.3.0 failure (a second engine's captures while old graphs live) and repeated slot resizes."""
    w = _model()
    engines, rounds = [], []
    for round_number in range(3):
        outs, _ = _run_graphs(w, keep=engines)
        refs = _serial(w, outs)
        for turn, (out, ref) in enumerate(zip(outs, refs)):
            assert out == ref, (round_number, turn)
        rounds.append(outs)
    assert rounds[0] == rounds[1] == rounds[2]                 # the same conversation every round
