from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.drafters import dflash_drafter
from tensorfold.families.glm5_next.linear import Q
from tensorfold.families.glm5_next.model import GLM5
from tensorfold.families.glm5_next.runtime import GLMDFlash


def test_glm_target_taps_are_the_stream_mean_after_selected_layers():
    model = GLM5.__new__(GLM5)
    model.layers = [object(), object(), object()]
    model.capture_layers((0, 2))
    rows = mx.arange(24, dtype=mx.float32).reshape(2, 4, 3).astype(mx.bfloat16)

    model._capture_layer(0, rows)
    model._capture_layer(1, rows + 100)
    model._capture_layer(2, rows + 12)
    mx.eval(*model._hidden_states)

    assert tuple(model._hidden_states[0].shape) == (1, 2, 3)
    assert bool(mx.array_equal(model._hidden_states[0][0], mx.mean(rows.astype(mx.float32), axis=1).astype(mx.bfloat16)))
    assert bool(mx.array_equal(model._hidden_states[1][0],
                               mx.mean((rows + 12).astype(mx.float32), axis=1).astype(mx.bfloat16)))


def test_glm_target_rejects_dflash_taps_outside_its_layers():
    model = GLM5.__new__(GLM5)
    model.layers = [object(), object()]
    with pytest.raises(ValueError, match="do not fit"):
        model.capture_layers((2,))


def test_glm_dflash_embedding_preserves_batch_and_row_axes():
    model = GLM5.__new__(GLM5)
    model.args = SimpleNamespace(hidden_size=3)
    model.embed_tokens = lambda tokens: mx.arange(int(tokens.size) * 3).reshape(-1, 3)
    tokens = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.uint32)

    embedded = GLM5.draft_embed_tokens(model, tokens)

    assert tuple(embedded.shape) == (2, 3, 3)


def test_dflash_uses_a_custom_target_capture_interface(tmp_path, monkeypatch):
    config = SimpleNamespace(target_layer_ids=(1, 3), block_size=8, mask_token_id=9,
                             sliding_window=None, vocab_size=154880)

    class Draft:
        def __init__(self):
            self.config = config
            self.bound = None
            self.embed_tokens = None

        def bind(self, target):
            self.bound = target

    draft = Draft()
    vendor = SimpleNamespace(load_draft=lambda _path: draft, snapshot_download=None,
                             _trim_recent_cache=lambda _cache, _tokens: None)
    monkeypatch.setattr(dflash_drafter, "_vendor", lambda: vendor)

    class Target:
        def __init__(self):
            self.layers = [object()] * 4
            self.lm_head = object()
            self.captured = None

        def embed_tokens(self, tokens):
            return tokens

        def draft_embed_tokens(self, tokens):
            return tokens

        def capture_layers(self, layers):
            self.captured = tuple(layers)

    target = Target()
    loaded = dflash_drafter.DFlashDrafter(target, str(tmp_path), bits=0)

    assert target.captured == (1, 3)
    assert draft.bound is target
    assert draft.embed_tokens == target.draft_embed_tokens
    assert loaded.draft_vocab == ((0, 154880),)


def test_dflash_reduced_head_accepts_glm_affine_weights():
    weight = mx.zeros((10, 8), dtype=mx.uint32)
    scales = mx.ones((10, 1), dtype=mx.bfloat16)
    biases = mx.zeros((10, 1), dtype=mx.bfloat16)
    head = Q(weight, scales, biases, bits=4, group=64)
    draft = dflash_drafter.DFlashDrafter.__new__(dflash_drafter.DFlashDrafter)
    draft.model = SimpleNamespace(lm_head=head)
    draft.draft_vocab = ((0, 4), (6, 10))
    draft._plain_sub = False

    parts, ids, group, bits = draft._plain_sub_head()
    mx.eval(ids)

    assert [int(part[0].shape[0]) for part in parts] == [4, 4]
    assert ids.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    assert (group, bits) == (64, 4)


def test_glm_dflash_primes_the_first_rows_after_an_imported_cache():
    proposer = SimpleNamespace(ready=False)

    class Slot:
        def get(self, _sampling):
            return proposer

    cache = [object(), Slot()]
    calls = []
    runtime = GLMDFlash.__new__(GLMDFlash)
    runtime._last = {id(cache): (33, 2, 5)}
    runtime.head_drafts = SimpleNamespace(
        absorb=lambda actual, position, rows, first: calls.append((actual, position, rows, first))
    )

    assert runtime._prime_imported(cache, [5, 6], None) == []
    assert calls == [(cache, 33, 2, 5)]

    proposer.ready = True
    assert runtime._prime_imported(cache, [5, 6], None) == [5, 6]


def test_glm_dflash_ignores_spark_hidden_without_local_target_taps():
    runtime = GLMDFlash.__new__(GLMDFlash)
    runtime._last = {}
    runtime.head_drafts = SimpleNamespace(absorb=lambda *_args: pytest.fail("stale taps were absorbed"))

    runtime.absorb_draft_context(mx.zeros((1, 1, 4)), [7], [object()])
