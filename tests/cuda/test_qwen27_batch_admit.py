"""Prompts admitted together on the real 27B with DFlash2: every stream's reply and every kept prompt entry (states,
attention rows, drafter context) equal one-prompt-a-round admission's, resumed and long prompts included. Needs
``TENSORFOLD_QWEN27_DRAFTER`` and ``TENSORFOLD_QWEN27_NVFP4`` and/or ``TENSORFOLD_MLX_MODEL``; skipped otherwise."""

import gc
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.streams import PrefixCache, Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda import multi  # noqa: E402

MODELS = {"nvfp4": os.environ.get("TENSORFOLD_QWEN27_NVFP4", ""), "mlx": os.environ.get("TENSORFOLD_MLX_MODEL", "")}
DRAFTER = os.environ.get("TENSORFOLD_QWEN27_DRAFTER", "")


@pytest.fixture(scope="module", params=sorted(MODELS))
def engine(request):
    path = MODELS[request.param]
    if not (path and Path(path).is_dir() and DRAFTER and Path(DRAFTER).is_dir()):
        pytest.skip(f"needs TENSORFOLD_QWEN27_DRAFTER and the {request.param} checkpoint's variable")
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    gc.collect()                                              # the checkpoint before is gone now: its blocks go back,
    torch.cuda.empty_cache()                                  # so this one admits as a fresh process would
    eng = Qwen27Engine(Path(path), Path(DRAFTER), streams=6, context=4096, context_explicit=True)
    yield eng
    eng.close()


def _prompts(eng):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(eng.model_dir / "tokenizer.json"))
    enc = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731
    base = enc("Write a short Python function that computes the Fibonacci sequence and explain it.")
    story = enc(" ".join(f"Line {i}: the quick brown fox jumps over the lazy dog near river {i % 7}."
                         for i in range(90)))                            # past STEP rows
    return base, [base + enc(" Then make it iterative."), enc("Explain how a GPU multiplies matrices."),
                  enc("List three prime numbers."), story + enc(" Summarize the lines above."),
                  enc("What is 17 times 23?"), base + enc(" Use memoization.")]


def _entry(e):
    ids, st, snap = e
    rec = [None if r is None else r.clone() for r in st.rec]
    conv = [None if c is None else c.clone() for c in st.conv]
    kv = [None if kv is None else (kv[0][:st.pos].clone(), kv[1][:st.pos].clone()) for kv in st.kv]
    ctx = None if snap is None else ([None if t is None else t.clone() for t in snap[0]],
                                     [None if t is None else t.clone() for t in snap[1]], snap[2], snap[3])
    return list(ids), st.pos, rec, conv, kv, ctx


def _equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _run(eng, batch, monkeypatch):
    monkeypatch.setattr(multi, "BATCH", batch)
    dec = eng.multi
    dec.cache = PrefixCache(dec.cache.keep)
    base, later = _prompts(eng)

    def stream(prompt, count, seed=None):
        s = Stream(list(prompt), count, None if seed is None else Sampling(seed, 1.0, 20, 0.95), draft=True,
                   stop_eos=False)
        s.emit = lambda new: False
        dec.admit(s)
        return s

    first = stream(base, 6)                                   # alone: the later ones resume from its entry
    while dec.live():
        dec.finish(dec.round())
    wave = [stream(p, 40, seed) for p, seed in zip(later[:3], (None, 1234, None))]
    for _ in range(2):                                        # the first wave decodes while the second fills
        dec.finish(dec.round())
    wave += [stream(p, 40, seed) for p, seed in zip(later[3:], (None, None, 1237))]
    while dec.live():
        dec.finish(dec.round())
    return [first.out] + [s.out for s in wave], [s.cached for s in wave], [_entry(e) for e in dec.cache.entries]


def test_prompts_admitted_together_equal_one_a_round(engine, monkeypatch):
    alone, cached_alone, entries_alone = _run(engine, False, monkeypatch)
    together, cached, entries = _run(engine, True, monkeypatch)
    assert together == alone and all(len(out) == 40 for out in together[1:])
    assert cached == cached_alone and cached[0] > 0                    # the first later prompt resumed
    assert len(entries) == len(entries_alone) and all(_equal(a, b) for a, b in zip(entries, entries_alone))
