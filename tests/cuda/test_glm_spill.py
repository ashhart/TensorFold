"""The spill tier end to end on the tiny synthetic checkpoint: an evicted kept state is written to the
per-rank directory off the request path, a matching prompt resumes from it with a fresh prefill's exact reply,
an engine restart reads it back, and another configuration's model id finds nothing."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402  (pytest puts tests/cuda on sys.path)
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

SAMPLING = Sampling(4321, 1.0, 20, 0.95)


def _spill_engine(path, directory, *, entries: int = 1, mtp: str | None = None) -> object:
    """The engine with the spill tier on and room for one kept state, so the next conversation evicts.
    ``mtp='0'`` builds a serial-only engine: another configuration, whose model id matches no spilled file."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    saved = {name: os.environ.get(name) for name in ("TF_GLM_CACHE_ENTRIES", "TENSORFOLD_MEMORY_RESERVE_GIB", "TF_GLM_MTP")}
    os.environ["TF_GLM_CACHE_ENTRIES"] = str(entries)
    os.environ.setdefault("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    if mtp is not None:
        os.environ["TF_GLM_MTP"] = mtp
    try:
        return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(),
                         snapshot_dir=str(directory), spill_gib=1.0, serial_only=mtp is not None)
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)


def _retire(engine) -> None:
    """Free one engine before the next builds: the GPU here has a serving rank resident beside the tests."""

    import gc

    engine._spill = None
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def test_spilled_conversations_resume(tmp_path):
    path = tmp_path / "model"
    _checkpoint(path)
    directory = tmp_path / "spill"
    rng = np.random.default_rng(9)
    a = list(rng.integers(0, 1000, size=70))
    b = list(rng.integers(1000, 2000, size=70))      # no shared prefix with a

    engine = _spill_engine(path, directory)
    reply_a, _ = _generate(engine, a, SAMPLING)
    after_a = a + reply_a + [5, 6, 7]

    _, _ = _generate(engine, b, SAMPLING)            # b evicts a's kept state
    engine._spill.flush()                            # the writer is off the request path: wait for the write
    files = list(directory.glob("*.rank0.safetensors"))
    assert files, "the evicted conversation was not spilled"

    warm, stats = _generate(engine, after_a, SAMPLING)
    assert stats["cached"] == len(a) - 1             # resumed from the spilled snapshot, not memory
    _forget(engine)
    cold_a, _ = _generate(engine, after_a, SAMPLING, draft=False)   # the fresh reply the spill must reproduce
    assert warm == cold_a
    _retire(engine)

    # a restarted engine reads the spill back
    restarted = _spill_engine(path, directory)
    warm2, stats2 = _generate(restarted, after_a, SAMPLING)
    assert stats2["cached"] == len(a) - 1 and warm2 == cold_a
    _retire(restarted)

    # another configuration's model id finds nothing of this one's
    other = _spill_engine(path, directory, mtp="0")
    cold, stats = _generate(other, after_a, SAMPLING)
    assert stats["cached"] == 0 and cold == cold_a
    _retire(other)

    # a clean shutdown spills what eviction has not (the live kept state's rows copied before exit)
    from tensorfold.engine.prefix_snapshots import snapshot_key
    from tensorfold.families.glm5_next.cuda.spill import rank_path

    shutdown = _spill_engine(path, directory)
    _generate(shutdown, a, SAMPLING)
    shutdown.save_sessions()
    key = snapshot_key(shutdown.snapshot_model_id, a[:-1])
    assert rank_path(directory, key, 0).exists(), "the shutdown save wrote nothing"
    _retire(shutdown)
