"""Flash Next concurrent rounds: each row gets its own stream's bits, and streams together emit what each does alone."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, forward, stage  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]


def test_segments_give_each_stream_its_own_rows():
    w = _model()
    b = Buffers(w, 32, 1024)
    chains = [[401, 33, 2048], [5], [77, 1500, 9, 10, 11], [3, 4]]
    states = []
    for prompt in PROMPTS:
        st = State(w, 1024, 16)
        forward(w, st, b, prompt)
        commit(w, st, b, len(prompt), len(prompt))
        states.append(st)
    alone = [st.clone() for st in states]
    ref = []
    for st, chain in zip(alone, chains):
        lg = forward(w, st, b, chain)
        ref.append((lg[:len(chain)].clone(), b.streams[:len(chain)].clone()))
        keep = max(1, len(chain) // 2)
        commit(w, st, b, len(chain), keep)
        ref[-1] += (forward(w, st, b, [chain[keep] if keep < len(chain) else 1])[0].clone(),)
    segs = stage(w, b, list(zip(states, chains)))
    lg = compute(w, segs, b)
    for (st, a0, a1), (rl, rs, _) in zip(segs, ref):
        assert torch.equal(lg[a0:a1], rl) and torch.equal(b.streams[a0:a1], rs), (a0, a1)
    for (st, a0, a1), chain in zip(segs, chains):
        commit(w, st, b, a1 - a0, max(1, (a1 - a0) // 2), at=a0)
    for st, chain, (_, _, nxt) in zip(states, chains, ref):
        keep = max(1, len(chain) // 2)
        assert torch.equal(forward(w, st, b, [chain[keep] if keep < len(chain) else 1])[0], nxt)


@pytest.mark.parametrize("confidence,vocab,kv_dtype", [(0.0, False, "bf16"), (0.3, False, "bf16"), (0.3, True, "bf16"),
                                                       (0.3, True, "int8"), (0.0, False, "int4")])
def test_streams_decoded_together_equal_each_alone(confidence, vocab, kv_dtype):
    w = _model()
    if vocab:                                            # drafts over a token subset (the real model's draft head)
        words, scales, biases = qmm.to_mlx(w.head)
        ids = torch.arange(1, V, 3, device="cuda")
        w.draft_ids = ids
        w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(PROMPTS, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        first = prefill(e, prompt, sampling)
        refs.append(serial_decode(e, first, 20, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=confidence, kv_dtype=kv_dtype)
    assert all(st.kv_dtype == kv_dtype and st.kc[0].dtype == kv_dtype for st in dec.free)
    streams = []
    for i, (prompt, sampling) in enumerate(zip(PROMPTS, samplings)):
        got: list[int] = []
        s = Stream(prompt, 20, sampling, draft=i != 3, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        streams.append((s, got))
    while dec.live():
        dec.finish(dec.round())
    for i, (s, got) in enumerate(streams):
        assert got == refs[i] and s.out == refs[i], i
        assert s.min_rows >= (2 if s.draft else 1), (i, s.min_rows)
    assert len(dec.free) + len({id(k[1]) for k in dec.kept}) == 4 and not dec.live()     # every slot back or kept


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_prompts_that_extend_a_finished_stream_resume_from_its_slot(sampling, kv_dtype):
    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)

    def run(prompt, count, draft=True):
        s = Stream(list(prompt), count, sampling, draft=draft)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return s

    def fresh(prompt, count):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens

    first = run(PROMPTS[1], 12)
    longer = PROMPTS[1] + first.out[:-1] + [42, 43]          # the reply's committed tokens, then new ones
    warm = run(longer, 10)
    assert warm.cached == len(PROMPTS[1]) and warm.out == fresh(longer, 10)       # the reply prefills again
    ext = PROMPTS[0] + [7, 8]                                 # a prompt kept at admission, extended
    run(PROMPTS[0], 6)
    other = run(ext, 8)
    assert other.cached > 0 and other.out == fresh(ext, 8)
    serial = run(longer, 10, draft=False)
    assert serial.cached == 0 and serial.out == warm.out
