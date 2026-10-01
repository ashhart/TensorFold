"""Clones share one attention list: a grow moves every clone and frees the old buffers; a resume stays exact."""

import weakref

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.decode import clone_state, draft_decode, prefill  # noqa: E402

from test_qwen27_prefill import _model, _prompt, _same_state  # noqa: E402

ATT = 1                                                  # the tiny model's attention layer


@pytest.mark.parametrize("grow_by", ["prompt", "decode"])
def test_a_grow_moves_every_clone_and_frees_the_old_buffers(grow_by):
    w = _model()
    prompt = _prompt(1300, seed=9)
    st, first = prefill(w, prompt[:1000], None)
    kept = clone_state(st)
    old = weakref.ref(st.kv[ATT][0])
    assert st.kv[ATT][0].shape[0] == 1024 and kept.kv is st.kv
    if grow_by == "prompt":
        prefill(w, prompt, None, state=st)
    else:
        draft_decode(w, st, prompt[:1000], first, 40, None, None, stop_eos=False)
    assert kept.kv[ATT][0].shape[0] == 2048 and old() is None           # one buffer for every clone
    resumed, first_resumed = prefill(w, prompt[:1200], None, state=kept)
    fresh, first_fresh = prefill(w, prompt[:1200], None)
    _same_state(fresh, resumed)
    assert first_fresh == first_resumed
