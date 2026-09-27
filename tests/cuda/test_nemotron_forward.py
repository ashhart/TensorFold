"""Nemotron-H CUDA forward on a tiny model and TF_NEMOTRON_MODEL: windows, graphs, drafts and resumes stay serial."""

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from nemotron_fakes import tiny_weights  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.nemotron_h.cuda import reference as ref  # noqa: E402
from tensorfold.families.nemotron_h.cuda.decode import CopyIndex, draft_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.nemotron_h.cuda import glue as G  # noqa: E402
from tensorfold.families.nemotron_h.cuda.engine import Engine  # noqa: E402
from tensorfold.families.nemotron_h.cuda.mtp import MTPHead  # noqa: E402

MODEL = os.environ.get("TF_NEMOTRON_MODEL", "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")


@pytest.fixture(scope="module")
def tiny():
    return tiny_weights(5)


def _tokens(n, seed=0, vocab=512):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab, (n,), generator=g).tolist()


def test_windows_with_partial_keeps_equal_serial(tiny):
    toks = _tokens(40, 1)
    eng = Engine(tiny, max_len=1024, graphs=False)
    for t in toks[:20]:
        eng.forward([t])
        eng.commit(1)
    base = eng.snapshot()
    serial = []
    for t in toks[20:]:
        serial.append(eng.forward([t]).clone())
        eng.commit(1)
    serial = torch.cat(serial)
    eng.restore(base)
    at = 0
    for rows, keep in ((4, 2), (2, 2), (7, 3), (16, 9), (3, 1), (5, 3)):
        logits = eng.forward(toks[20 + at:20 + at + rows])
        assert torch.equal(logits[:keep], serial[at:at + keep]), (at, rows, keep)
        eng.commit(keep)
        at += keep
    assert at == 20


def test_prefill_windows_equal_single_rows(tiny):
    toks = _tokens(37, 2)
    eng = Engine(tiny, max_len=1024, graphs=False)
    for t in toks:
        one = eng.forward([t]).clone()
        eng.commit(1)
    eng.reset()
    for s in range(0, len(toks), 16):
        chunk = toks[s:s + 16]
        many = eng.forward(chunk)
        eng.commit(len(chunk))
    assert torch.equal(many[-1:], one)


def test_graphs_equal_eager(tiny):
    toks = _tokens(30, 3)
    eager = Engine(tiny, max_len=1024, graphs=False)
    graphed = Engine(tiny, max_len=1024, graphs=True)
    graphed.capture([1, 3, 16])
    for eng in (eager, graphed):
        eng.forward(toks[:16])
        eng.commit(16)
    for rows, keep in ((3, 2), (1, 1), (3, 3)):
        at = eager.pos
        a = eager.forward(toks[at:at + rows]).clone()
        b = graphed.forward(toks[at:at + rows]).clone()
        assert torch.equal(a, b), rows
        eager.commit(keep)
        graphed.commit(keep)


def test_fast_forward_tracks_fp32_reference(tiny):
    toks = _tokens(48, 4)
    eng = Engine(tiny, max_len=1024, graphs=False)
    fast = []
    for s in range(0, 48, 16):
        fast.append(eng.forward(toks[s:s + 16]).float().clone())
        eng.commit(16)
    fast = torch.cat(fast)
    want = ref.forward(tiny, torch.tensor(toks, device="cuda"))
    agree = (fast.argmax(-1) == want.argmax(-1)).float().mean().item()
    rel = ((fast - want).norm() / want.norm()).item()
    assert agree >= 0.9 and rel < 0.05, (agree, rel)


@pytest.mark.parametrize("greedy", [False, True])
def test_drafted_decode_equals_serial(tiny, greedy):
    prompt = _tokens(21, 6)
    sampling = None if greedy else Sampling(1234, 1.0, 20, 0.95)
    eng = Engine(tiny, max_len=1024, graphs=True)
    eng.capture(range(1, 17), sampling)
    mtp = MTPHead(eng)
    mtp.capture(range(1, 17), (0, 1, 3))
    pre = prefill(eng, mtp, prompt, sampling)
    serial = serial_decode(eng, pre, 48, sampling)
    for drafts, confidence in ((1, 0.0), (3, 0.0), (3, 0.2)):
        drafted = draft_decode(eng, mtp, pre, 48, sampling, drafts=drafts, confidence=confidence)
        assert drafted.tokens == serial.tokens, (drafts, confidence)
        assert drafted.rounds <= 47 and min(drafted.widths) >= 2


def test_copy_chains_equal_serial(tiny):
    prompt = _tokens(12, 7) * 3                 # a repeated prompt makes copies fire; their windows reach 16 rows
    eng = Engine(tiny, max_len=1024, graphs=False)
    mtp = MTPHead(eng)
    pre = prefill(eng, mtp, prompt, None)
    serial = serial_decode(eng, pre, 40, None)
    drafted = draft_decode(eng, mtp, pre, 40, None, drafts=2, copy=True)
    assert drafted.tokens == serial.tokens


def _copy_chain(context, max_nodes, min_match):
    """The definition: the longest continuation of an earlier occurrence of the last ``min_match`` tokens."""

    if len(context) < 2 * min_match or max_nodes < 1:
        return []
    needle = list(context[-min_match:])
    best = []
    for start in range(len(context) - min_match - 1, -1, -1):
        if list(context[start:start + min_match]) == needle:
            cont = list(context[start + min_match:start + min_match + max_nodes])
            if len(cont) > len(best):
                best = cont
                if len(best) == max_nodes:
                    break
    return best if len(best) >= min_match else []


def test_copy_index_matches_the_definition():
    import random

    rng = random.Random(1)
    for _ in range(60):
        vocab = rng.choice([3, 50, 5000])
        ctx = [rng.randrange(vocab) for _ in range(rng.randrange(60))]
        ctx += ctx[:rng.randrange(len(ctx) + 1)]                 # a repeated stretch
        m = rng.choice([2, 4, 8])
        index = CopyIndex(ctx, m)
        for _ in range(20):
            for most in (1, 15):
                assert index.chain(most) == _copy_chain(ctx, most, m)
            new = [rng.randrange(vocab) for _ in range(rng.randrange(1, 5))]
            ctx += new
            index.extend(new)


def _state(eng, mtp) -> list[torch.Tensor]:
    """What a prompt leaves: every cache row below the position, the SSM and conv states, the head's cache rows."""

    return [eng.k_cache[:, :eng.pos], eng.v_cache[:, :eng.pos], eng.ssm, eng.conv_base, mtp.k_cache[:mtp.pos],
            mtp.v_cache[:mtp.pos]]


def test_prompt_chunks_track_the_decode_path(tiny):
    """Prompt chunks and decode windows agree to bf16 rounding: SSM states, attention keys, the last logits."""

    prompt = _tokens(60, 14)
    pre = Engine(tiny, max_len=1024, graphs=False, prefill_rows=32)
    last = prefill(pre, None, prompt, None).last_hidden
    dec = Engine(tiny, max_len=1024, graphs=False)
    for s0 in range(0, len(prompt), 16):
        chunk = prompt[s0:s0 + 16]
        S_last = dec.forward(chunk)[len(chunk) - 1:len(chunk)].float()
        dec.commit(len(chunk))
    dec.forward([prompt[-1]])                       # flushes the last window's rows into the committed state
    a, b = pre.ssm, dec.ssm
    assert float((a - b).abs().max()) <= 2e-2 * float(b.abs().max())
    ka, kb = pre.k_cache[:, :len(prompt)].float(), dec.k_cache[:, :len(prompt)].float()
    assert float((ka - kb).abs().max()) <= 2e-2 * float(kb.abs().max())
    logits = G.prefill_dense(last, tiny.head).float()
    assert float(torch.nn.functional.cosine_similarity(logits, S_last, dim=1)) > 0.999


@pytest.mark.parametrize("greedy", [False, True])
def test_prompt_chunks_leave_the_same_state(tiny, greedy):
    """Chunks of 7, 16, 64 and 300 rows leave bit-identical states and first tokens; drafts after them stay serial."""

    sampling = None if greedy else Sampling(3, 1.0, 20, 0.95)
    prompt = _tokens(300, 13)
    eng = Engine(tiny, max_len=1024, graphs=False, prefill_rows=300)
    mtp = MTPHead(eng)
    pre = prefill(eng, mtp, prompt, sampling)
    want = [t.clone() for t in _state(eng, mtp)]
    serial = serial_decode(eng, pre, 24, sampling).tokens
    for rows in (7, 16, 64):
        e2 = Engine(tiny, max_len=1024, graphs=False, prefill_rows=rows)
        m2 = MTPHead(e2)
        p2 = prefill(e2, m2, prompt, sampling)
        assert p2.pending == pre.pending and torch.equal(p2.last_hidden, pre.last_hidden), rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e2, m2), want)), rows
        assert draft_decode(e2, m2, p2, 24, sampling, drafts=3).tokens == serial, rows


@pytest.mark.parametrize("greedy", [False, True])
def test_resumed_prompts_equal_fresh(tiny, greedy):
    """A resume from a kept prompt end, with or without the reply between, equals a fresh prefill."""

    sampling = None if greedy else Sampling(99, 1.0, 20, 0.95)
    eng = Engine(tiny, max_len=1024, graphs=False, prefill_rows=16)
    mtp = MTPHead(eng)
    first = _tokens(19, 10)
    pre = prefill(eng, mtp, first, sampling)
    kept = (pre.engine, pre.mtp, len(first), pre.last_hidden)
    reply = draft_decode(eng, mtp, pre, 20, sampling, drafts=3)
    for prompt in (first + _tokens(9, 11), first + reply.tokens + _tokens(9, 12)):
        fresh = prefill(eng, mtp, prompt, sampling)
        want = draft_decode(eng, mtp, fresh, 24, sampling, drafts=3).tokens
        resumed = prefill(eng, mtp, prompt, sampling, resume=kept)
        assert resumed.pending == fresh.pending and torch.equal(resumed.last_hidden, fresh.last_hidden)
        assert draft_decode(eng, mtp, resumed, 24, sampling, drafts=3).tokens == want


@pytest.mark.parametrize("greedy", [False, True])
def test_draft_id_subset_equals_serial(tiny, greedy):
    prompt = _tokens(21, 8)
    sampling = None if greedy else Sampling(77, 1.0, 20, 0.95)
    eng = Engine(tiny, max_len=1024, graphs=False)
    mtp = MTPHead(eng, draft_ids=list(range(0, 512, 2)))       # the head scores even ids only
    pre = prefill(eng, mtp, prompt, sampling)
    serial = serial_decode(eng, pre, 40, sampling)
    drafted = draft_decode(eng, mtp, pre, 40, sampling, drafts=2, copy=False)
    assert drafted.tokens == serial.tokens
    assert all(d % 2 == 0 for d in mtp.drafts())


def test_gpu_sampler_matches_host_rule():
    from tensorfold.engine.exact_sampling import choose_rows
    from tensorfold.families.nemotron_h.cuda import sampler as S

    g = torch.Generator(device="cuda").manual_seed(11)
    rows, vocab = 64, 131072
    logits = (torch.randn(rows, vocab, generator=g, device="cuda") * 3).bfloat16()
    logits[5, :40] = 7.0                                    # a tie across the top-k boundary
    logits[6, 100:130] = logits[6].max()
    meta = torch.tensor([1000, 0, 0, 0], dtype=torch.int32, device="cuda")
    out = torch.zeros(rows, dtype=torch.int32, device="cuda")
    agree = 0
    for seed in (1, 2, 3):
        params = S.Params("cuda")
        params.set(Sampling(seed * 7919, 1.0, 20, 0.95))
        S.sample(logits, meta, params, out)
        vals, ids = torch.topk(logits.float(), 28, dim=-1)
        host = choose_rows(vals.cpu().numpy(), ids.cpu().numpy(), [1001 + r for r in range(rows)], params.sampling)
        agree += sum(int(a == b) for a, b in zip(out.tolist(), host))
    assert agree == 3 * rows, agree


@pytest.mark.skipif(not Path(MODEL, "config.json").is_file(), reason="real checkpoint not mounted")
def test_real_checkpoint_drafted_equals_serial():
    from tensorfold.families.nemotron_h.cuda.weights import load

    w = load(MODEL)
    eng = Engine(w, max_len=2048)
    sampling = Sampling(99, 1.0, 20, 0.95)
    eng.capture(range(1, 17), sampling)
    mtp = MTPHead(eng)
    mtp.capture(range(1, 17), (0, 3))
    prompt = [18746, 1261, 4958, 17616, 2254, 1455, 1115, 1261, 1766]
    pre = prefill(eng, mtp, prompt, sampling)
    serial = serial_decode(eng, pre, 32, sampling)
    drafted = draft_decode(eng, mtp, pre, 32, sampling, drafts=3, confidence=0.2)
    assert drafted.tokens == serial.tokens
    assert drafted.accepted > 0
