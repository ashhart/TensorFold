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


@pytest.mark.parametrize("noise", [(0, 0, 2), (2, 0, 0), (0, 2, 0)])
def test_a_probe_whose_peak_varies_run_to_run_gives_the_same_step(monkeypatch, noise):
    """#95: a streamed-expert probe's peak moves between runs; the worst of three decides, wherever the high one falls."""

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from tensorfold.engine import prefill_step

    class Engine:
        model = SimpleNamespace()

        def prefill_prefix(self, tokens, cache=None, cached_tokens=0):
            kv = KVCache()
            kv.update_and_fetch(mx.zeros((1, 1, len(tokens), 8)), mx.zeros((1, 1, len(tokens), 8)))
            return [kv]

    runs = []
    mib = 1 << 20
    monkeypatch.setattr(prefill_step, "CONTEXT_FLOOR", 1000)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 1 << 30)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: runs.append(len(runs)))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: (1 << 30) + (1 + noise[runs[-1]]) * mib)
    # 8,192 rows take 4x the 2,048-row probe's work: 12 MiB at the high peak, 4 at the low; 4,096 take 6 or 2
    budget = (1 << 30) + 7 * mib + 1000 * 32
    assert prefill_step.choose(lambda grid: Engine(), (8192, 4096, 2048), budget, list(range(50))) == 4096
    assert len(runs) == 3


def test_nemotron_offers_8192_token_prompt_chunks_with_tensor_units(monkeypatch):
    from tensorfold.families import nemotron_h, qwen3_5

    model = SimpleNamespace(exact_width=5)
    for units, steps in ((True, (8192, 4096, 2048)), (False, None)):
        monkeypatch.setattr(qwen3_5, "tensor_units", lambda units=units: units)
        settings = nemotron_h.engine_settings(model)
        assert settings.get("prefill_steps") == steps and settings["max_rows"] == 5 and settings["max_draft"] == 4
