"""Memory admission for GLM-5.3-Flash: its sparse-attention cache grows with the context and its prefill copies keys."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import TEXT, write_checkpoint  # noqa: E402
from tensorfold.engine.memory import _kv_bytes  # noqa: E402
from tensorfold.families.glm5_next import caches, weights  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402
from tensorfold.server.memory_budget import CacheMemory  # noqa: E402
from tensorfold.server.prompt_memory import PromptMemory  # noqa: E402


def _mla_cache(tokens: int) -> caches.MLACache:
    cache = caches.MLACache()
    ape = mx.zeros((4, 128), dtype=mx.bfloat16)
    cache.append(mx.zeros((tokens, 512)), mx.zeros((tokens, 128)), mx.zeros((tokens, 128)), ape, 4)
    return cache


def test_mla_cache_growth_is_counted_a_token():
    # latent keys 512, indexer keys 128 and gates 128 a position, pooled keys 128 a block of 4, all bf16
    assert _mla_cache(300).memory_growth() == (0, 2 * (512 + 128 + 128 + 128 // 4))
    memory = CacheMemory.from_cache([_mla_cache(300)])
    assert memory.bytes_per_token == 1600 and memory.fixed_bytes == 0
    assert memory.cache_bytes(100_000) - memory.cache_bytes(1_000) == 1600 * (100_096 - 1_024)


def test_concurrency_admission_counts_declared_growth():
    assert _kv_bytes([_mla_cache(300)]) == (1600.0, 0.0)


class _Runtime:
    def __init__(self, used: int) -> None:
        self.used = used

    def get_active_memory(self) -> int:
        return self.used

    def get_cache_memory(self) -> int:
        return 0

    def get_peak_memory(self) -> int:
        return self.used

    def reset_peak_memory(self) -> None:
        pass

    def clear_cache(self) -> None:
        pass


def test_prompt_admission_takes_the_models_prefill_workspace():
    generic = PromptMemory(10**12, SimpleNamespace(num_attention_heads=8), runtime=_Runtime(0))
    own = PromptMemory(10**12, SimpleNamespace(prefill_workspace_per_token=1000), runtime=_Runtime(0))
    for memory in (generic, own):
        memory.observe_cache([_mla_cache(300)], workspace=False)
    assert own._work(10_000) - own._work(0) == 1600 * 10_240 + 1000 * 10_000
    assert generic._work(10_000) - generic._work(0) == 1600 * 10_240 + 2 * 144 * 8 * 10_000 * 2


def test_glm_states_its_prefill_workspace(tmp_path):
    runtime = GLMFlash(weights.load_backbone(write_checkpoint(tmp_path / "glm5")), check=False)
    # a 512-query chunk's indexer scores per head (bf16, two live copies) and its selection arrays, a block of 4
    assert runtime.prefill_workspace_per_token == 512 * (2 * TEXT["index_n_heads"] * 2 + 10) // TEXT["index_kpool"]


def test_glm_states_its_allowance_for_a_256_gb_mac():
    from tensorfold.families import glm5_next
    from tensorfold.server.memory_budget import model_fraction

    gib = 1024**3
    assert model_fraction(glm5_next, 256 * gib) == 0.85
    assert model_fraction(glm5_next, 512 * gib) == 0.70
