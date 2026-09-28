"""A family's engine_settings names its prompt chunk: the prefill plan and the stream memory probe take it."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.engine import memory
from tensorfold.engine.prefill_plan import PrefillPlan


def parse(*extra):
    return cli.build_parser().parse_args(["serve", "some/model", *extra])


def test_the_plan_takes_the_chosen_step_and_cuts_replies_256_apart(monkeypatch):
    seen = {}

    class Built(Exception):
        pass

    def app(*args, **kwargs):
        seen["plan"] = kwargs["engine_factory"].keywords["prefill_plan"]
        seen["rows"] = kwargs["max_rows"]
        raise Built

    monkeypatch.setattr("tensorfold.server.app.ChatApp", app)
    monkeypatch.setattr("tensorfold.engine.prefill_step.choose", lambda make, steps, *a, **k: max(steps))
    for settings, step in (({"prefill_steps": (4096, 2048), "max_rows": 4}, 4096), ({"max_rows": 4}, 2048)):
        package = SimpleNamespace(load=lambda model_dir, **options: (SimpleNamespace(), None),
                                  engine_settings=lambda model, s=settings: dict(s), kernel_version=lambda m: "k")
        family = SimpleNamespace(title="fake", model_type="fake", package=package)
        with pytest.raises(Built):
            cli._serve_mlx(parse("--no-drafts"), family, Path("some/model"), 0, [], 1 << 30)
        assert seen["plan"].step == step and seen["plan"].min_chunk == 256 and seen["rows"] == 4


def test_the_memory_probe_ends_on_full_chunks_of_the_plan_step(monkeypatch):
    import mlx.core as mx

    lengths = []

    class Engine:
        prefill_plan = PrefillPlan(8)

        def prefill_prefix(self, tokens, cache=None, cached_tokens=0):
            lengths.append(len(tokens))
            return [SimpleNamespace(keys=mx.zeros((1, 1, len(tokens), 4)), values=mx.zeros((1, 1, len(tokens), 4)))]

    monkeypatch.setattr("tensorfold.engine.family_common.cache_arrays",
                        lambda cache: [a for c in cache for a in (c.keys, c.values)])
    measured = memory.measure(Engine())
    assert lengths[1:] == [64, 8 + 64, 2 * 8 + 64] and measured.chunk == 8
    assert measured.prefill_bytes(1000) == int(measured.prefill_a * 8 + measured.prefill_b * 8 * 1000)


def test_the_largest_step_that_leaves_the_context_floor_is_chosen(monkeypatch):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from tensorfold.engine import prefill_step

    made = []

    class Model:
        tightened = 0

        def tighten_prefill(self):
            self.tightened += 1
            return self.tightened == 1

    class Engine:
        model = Model()

        def prefill_prefix(self, tokens, cache=None, cached_tokens=0):
            kv = KVCache()
            kv.update_and_fetch(mx.zeros((1, 1, len(tokens), 8)), mx.zeros((1, 1, len(tokens), 8)))
            return [kv]

    def make(grid):
        made.append(grid)
        return Engine()

    monkeypatch.setattr(prefill_step, "CONTEXT_FLOOR", 1000)
    assert prefill_step.choose(make, (2048,), 1 << 40, []) == 2048 and made == []
    assert prefill_step.choose(make, (8192, 4096, 2048), 1 << 40, list(range(50))) == 8192 and made == [2048]
    assert prefill_step.choose(make, (8192, 4096, 2048), 0, []) == 2048 and Engine.model.tightened == 2
