"""Committed Nemotron prefixes restore into fresh full-capacity runtime buffers."""

import pytest
import torch
from nemotron_fakes import tiny_weights

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
from tensorfold.families.nemotron_h.cuda.engine import Engine
from tensorfold.families.nemotron_h.cuda.mtp import MTPHead


@pytest.mark.parametrize("drafts", [False, True])
def test_compact_prefix_restores_exact_continuation(drafts):
    w = tiny_weights(5, mtp=drafts)
    app = NemotronEngine.__new__(NemotronEngine)
    app._make = lambda: Engine(w, max_len=1024, graphs=False, prefill_rows=16)
    app.e = app._make()
    app.mtp = MTPHead(app.e) if drafts else None
    app.tp, app.rank, app.drafts, app.confidence = 1, 0, 3 if drafts else 0, .3
    app.max_len, app.cache, app.serial, app.eos = 1024, [], None, ()
    sampling = Sampling(23, 1.0, 20, .95)
    prompt = list(range(11, 30))

    def ask(tokens, draft=True):
        out = []
        stats = app.generate(tokens, 8, sampling, lambda new: out.extend(new), draft=draft, stop_eos=False)
        return out, stats

    ask(prompt)
    ids, state = app.cache[0]
    pos = len(ids)
    compact = {**state, "engine": {**state["engine"],
        "k_cache": state["engine"]["k_cache"][:, :pos].clone(),
        "v_cache": state["engine"]["v_cache"][:, :pos].clone()}}
    if state["mtp"] is not None:
        compact["mtp"] = {**state["mtp"], "k": state["mtp"]["k"][:pos - 1].clone(),
                          "v": state["mtp"]["v"][:pos - 1].clone()}
    for tokens in (prompt, prompt[:-1] + [271, 77, 78]):
        app.e = app._make()
        app.mtp = MTPHead(app.e) if drafts else None
        app.cache = [(ids, compact)]
        actual, stats = ask(tokens)
        assert stats["cached"] == pos
        assert actual == ask(tokens, False)[0]
    torch.cuda.synchronize()
