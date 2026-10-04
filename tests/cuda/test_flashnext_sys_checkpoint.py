"""System-block checkpoints on the GPU: a prompt whose system block differs from a kept one only after the checkpoint
resumes there, and its reply is token-for-token a fresh prefill's (greedy and sampled, bf16 and int8 caches)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda.markers import snapshot_points  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

OPEN, ASSISTANT = V - 2, V - 1


def _chat(block, user):
    return [OPEN] + block + [OPEN] + user + [OPEN, ASSISTANT]


@pytest.fixture(scope="module")
def w():
    return _model()


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_a_block_with_a_new_tail_resumes_from_the_checkpoint_with_a_fresh_prefills_reply(w, sampling, kv_dtype):
    g = torch.Generator().manual_seed(5)
    fixed = torch.randint(1, V - 2, (800,), generator=g).tolist()
    tail_a = torch.randint(1, V - 2, (60,), generator=g).tolist()
    tail_b = torch.randint(1, V - 2, (75,), generator=g).tolist()            # a different "date line"
    user = torch.randint(1, V - 2, (20,), generator=g).tolist()
    a, b = _chat(fixed + tail_a, user), _chat(fixed + tail_b, user[:12])
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT), checkpoint=256)
    cp = points.checkpoint(a)
    assert cp == points.checkpoint(b) == 512 and a[:cp] == b[:cp]

    def fresh(prompt, count):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling, stop_eos=False).tokens

    def run(dec, prompt, count):
        s = Stream(list(prompt), count, sampling, stop_eos=False)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return s

    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, points=points)
    first = run(dec, a, 10)
    assert first.cached == 0 and first.out == fresh(a, 10)
    assert any(len(k[0]) == cp for k in dec.kept)
    second = run(dec, b, 10)
    assert second.cached == cp and second.out == fresh(b, 10)
    again = run(dec, a, 10)                                    # the first block still resumes at its own end or later
    assert again.cached >= len(fixed + tail_a) + 1 and again.out == first.out

    off = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype,
                       points=snapshot_points((OPEN,), (OPEN, ASSISTANT)))    # no granule: the same replies
    assert run(off, a, 10).out == first.out and not any(len(k[0]) == cp for k in off.kept)
    b_off = run(off, b, 10)
    assert b_off.cached < cp and b_off.out == second.out
