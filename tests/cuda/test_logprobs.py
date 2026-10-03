"""Probability rows stay exact across batching and never alter target logits."""

import pytest
import torch

from tensorfold.cuda.logprobs import capture
from tensorfold.engine.probabilities import Probabilities


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [31, 1024, 1025, 131073])
def test_probability_rows_match_reference_and_batching(dtype, vocab):
    g = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.randn((7, vocab), generator=g, device="cuda", dtype=torch.float32).to(dtype)
    saved = logits.clone()
    tokens = [i % vocab for i in (0, 1, 17, 30, 21, 14, 5)]
    together = Probabilities(20, 100, 7)
    capture(logits, tokens, list(range(100, 107)), together)
    reference = logits.double().log_softmax(-1)
    for i, token in enumerate(tokens):
        alone = Probabilities(20, 100 + i, 1)
        capture(logits[i:i + 1], [token], [100 + i], alone)
        assert alone.rows[100 + i] == together.rows[100 + i]
        assert together.rows[100 + i]["logprob"] == pytest.approx(float(reference[i, token]), abs=3e-6)
    assert torch.equal(logits, saved)


def test_ties_signed_zero_and_acceptance_path():
    logits = torch.zeros((6, 41), device="cuda")
    logits[:, ::2] = -0.0
    logits[2, 40] = 3.0
    record = Probabilities(20, 50, 3)
    capture(logits, [8, 40, 3], [50, 51, 52], record, rows=[0, 2, 5])
    assert [x[0] for x in record.rows[50]["top"]] == list(range(20))
    assert [x[0] for x in record.rows[51]["top"]] == [40, *range(19)]
    assert [row["id"] for row in record.emitted([8, 40, 3])] == [8, 40, 3]
    assert len(record.rows) == 3
    capture(logits, [8, 40, 3], [50, 51, 52], record, rows=[0, 2, 5])
    assert len(record.rows) == 3


def test_zero_alternatives_and_rows_outside_the_reply_limit():
    logits = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    record = Probabilities(0, 10, 1)
    capture(logits, [2], [10], record)
    capture(logits, [2], [11], record)
    assert len(record.rows) == 1 and record.rows[10]["top"] == []


@pytest.mark.parametrize("sampled", [False, True])
def test_flashnext_probabilities_match_drafted_and_serial_tokens(sampled):
    from test_flashnext_forward import _model
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    weights = _model()
    prompt, count = [5, 17, 99, 250], 12
    sampling = Sampling(seed=13, top_k=20, top_p=0.95) if sampled else None
    results = []
    for drafted in (False, True):
        engine = Engine(weights, capacity=512, max_rows=8, prefill_rows=64)
        record = Probabilities(20, len(prompt), count)
        first = prefill(engine, prompt, sampling, mtp=drafted, probabilities=record)
        decode = mtp_decode if drafted else serial_decode
        result = decode(engine, first, count, sampling, probabilities=record,
                        **({"depth": 4, "confidence": 0.0} if drafted else {}))
        results.append((result.tokens, record.emitted(result.tokens)))
    assert results[0] == results[1]


def test_packed_prompts_and_shared_rounds_match_solo_probabilities():
    from test_flashnext_forward import _model
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    weights = _model()
    prompts = [[5, 17, 99, 250], [13, 7, 11, 99, 20]]

    def run(together):
        decoder = MultiDecoder(weights, slots=2, capacity=512, depth=4, confidence=0.0, prefill_rows=64)
        streams = [Stream(p, 12, probabilities=Probabilities(5, len(p), 12)) for p in prompts]
        for stream in streams:
            decoder.admit(stream)
            if not together:
                while not stream.done:
                    decoder.finish(decoder.round())
        if together:
            while decoder.live():
                decoder.finish(decoder.round())
        return [(s.out, s.probabilities.emitted(s.out)) for s in streams]

    assert run(False) == run(True)


def test_resumed_target_probabilities_match_a_fresh_prompt():
    from test_flashnext_forward import _model
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill

    weights = _model()
    prompt = [(37 * i + 11) % 4096 for i in range(96)]
    sampling = Sampling(seed=17, top_k=20, top_p=0.95)
    resumed = Engine(weights, capacity=512, max_rows=8, prefill_rows=64)
    prefill(resumed, prompt[:67], sampling)
    snapshot = {"state": resumed.st.snapshot(), "tail": resumed.last_streams.clone()}
    results = []
    fresh = Engine(weights, capacity=512, max_rows=8, prefill_rows=64)
    for engine, state in ((resumed, snapshot), (fresh, None)):
        record = Probabilities(5, len(prompt), 12)
        first = prefill(engine, prompt, sampling, resume=state, probabilities=record)
        result = mtp_decode(engine, first, 12, sampling, depth=4, confidence=0.0, probabilities=record)
        results.append((result.tokens, record.emitted(result.tokens)))
    assert results[0] == results[1]


def test_widest_probability_rows_bound_sorting_memory():
    rows, vocab = 128, 248320
    logits = torch.zeros((rows, vocab), device="cuda", dtype=torch.bfloat16)
    tokens, positions = [0] * rows, list(range(rows))
    probabilities = Probabilities(20, 0, rows)
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    capture(logits, tokens, positions, probabilities, rows=list(reversed(positions)))
    transient = torch.cuda.max_memory_allocated() - live
    print(f"top20 rows={rows} vocab={vocab} transient_bytes={transient}")
    assert transient < 256 * 1024**2
    assert len(probabilities.rows) == rows
    assert all(row == probabilities.rows[0] for row in probabilities.rows.values())
