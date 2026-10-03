"""min_p in the keyed rule: after top_k and top_p, a draw keeps only the tokens at least min_p times as likely as the
likeliest (after temperature). Every rule (numpy host, MLX nucleus, Metal kernel, the CUDA ranks' headers) cuts the
same prefix, and both servers read and refuse the field alike."""

import json
import math

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling, choose, choose_rows
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import parse_numbers


def _rows(seed, rows=6, width=48):
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=(rows, width)) * 2).astype(np.float32)
    values[0, :3] = values[0, 3]                                  # a tie at the top
    ids = np.stack([rng.permutation(5000)[:width] for _ in range(rows)]).astype(np.int64)
    return values, ids


@pytest.mark.parametrize("min_p", [0.02, 0.3, 0.9, 1.0])
@pytest.mark.parametrize("top_k, top_p", [(20, 0.95), (0, 1.0), (5, 1.0), (40, 0.7)])
def test_every_row_rule_cuts_the_same_prefix(min_p, top_k, top_p):
    values, ids = _rows(int(min_p * 100) + top_k)
    s = Sampling(seed=321, temperature=0.7, top_k=top_k, top_p=top_p, min_p=min_p)
    positions = list(range(40, 40 + len(values)))
    alone = [choose(values[r], ids[r], positions[r], s) for r in range(len(values))]
    assert choose_rows(values, ids, positions, s) == alone
    for r, token in enumerate(alone):
        scaled = values[r].astype(np.float64) / 0.7
        assert scaled[list(ids[r]).index(token)] >= scaled.max() + math.log(min_p)


def test_min_p_drops_the_unlikely_tail_and_renormalizes_the_rest():
    values = np.log(np.array([0.5, 0.3, 0.15, 0.05])).astype(np.float32)
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for min_p, kept in ((0.25, {1, 2, 3}), (0.35, {1, 2}), (1.0, {1})):
        s = Sampling(seed=11, temperature=1.0, top_k=4, top_p=1.0, min_p=min_p)
        assert {choose(values, ids, p, s) for p in range(3000)} == kept
    s = Sampling(seed=11, temperature=1.0, top_k=4, top_p=1.0, min_p=0.35)
    counts = np.bincount([choose(values, ids, p, s) for p in range(20000)], minlength=5)[1:3]
    assert np.allclose(counts / counts.sum(), [0.625, 0.375], atol=0.015)


def test_off_is_the_rule_without_it():
    values, ids = _rows(3)
    positions = list(range(len(values)))
    plain = Sampling(seed=5, temperature=1.1, top_k=20, top_p=0.9)
    assert choose_rows(values, ids, positions, plain) == choose_rows(
        values, ids, positions, Sampling(seed=5, temperature=1.1, top_k=20, top_p=0.9, min_p=0.0))
    assert plain.min_log == -math.inf and Sampling(1, min_p=0.5).min_log == math.log(0.5)


@pytest.mark.parametrize("value, words", [(-0.1, "between 0 and 1"), (1.5, "between 0 and 1"),
                                          ("x", "a finite number"), (float("nan"), "a finite number"),
                                          (True, "a finite number")])
def test_a_bad_min_p_is_refused(value, words):
    with pytest.raises(RequestError, match=f"min_p must be {words}"):
        parse_numbers({"min_p": value})


def test_a_good_min_p_is_read():
    assert parse_numbers({"min_p": "0.05"})["min_p"] == 0.05
    assert parse_numbers({"min_p": 0})["min_p"] == 0.0 and parse_numbers({"min_p": None})["min_p"] is None


def test_the_cuda_server_takes_min_p_and_refuses_a_bad_one(tmp_path):
    from tests.test_cuda_admission import http_server, post
    from tests.test_cuda_request_policy import Engine, app_for

    class Recording(Engine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            self.sampling = sampling
            return super().generate(prompt, max_tokens, sampling, on_tokens, draft)

    app = app_for(tmp_path)
    app.engine = Recording("Hi")
    app.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.02}
    body = {"messages": [{"role": "user", "content": "x"}], "seed": 3}
    with http_server(app) as port:
        assert post(port, body, True)[0] == 200
        assert app.engine.sampling == Sampling(3, 1.0, 20, 0.95, 0.02)          # the server's default
        assert post(port, {**body, "min_p": 0.1}, True)[0] == 200
        assert app.engine.sampling.min_p == 0.1
        for stream in (False, True):
            status, text = post(port, {**body, "min_p": 2, "stream": stream}, True)
            assert status == 400 and "min_p must be between 0 and 1" in json.loads(text)["error"]["message"]


def test_the_mac_server_reads_min_p():
    from tests.test_request_reasoning import SamplingApp
    from tests.test_server_openai_compat import post_json, serve_fake

    app = SamplingApp()
    http = serve_fake(app)
    try:
        payload = {"messages": [{"role": "user", "content": "hi"}], "min_p": 0.2, "temperature": 0.8}
        assert post_json(http, "/v1/chat/completions", payload)[0] == 200 and app.sampling["min_p"] == 0.2
        assert post_json(http, "/v1/chat/completions", {**payload, "min_p": -1})[0] == 400
    finally:
        http.shutdown()
        http.server_close()


def test_the_mac_app_resolves_min_p_into_the_draw():
    from tensorfold.server.request_options import RequestOptions

    options = RequestOptions()
    options.default_sampling = {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.01}
    assert options._resolve_sampling({"seed": 4}, 0.7, [1, 2]) == Sampling(4, 0.7, 20, 0.95, 0.01)
    assert options._resolve_sampling({"seed": 4, "min_p": 0.3}, 0.7, [1, 2]).min_p == 0.3


def test_two_ranks_read_the_sampling_rank_zero_sent():
    pytest.importorskip("triton")                            # the 27B's CUDA modules
    from tensorfold.families.qwen3_5.cuda.decode_tp import SAMPLING_WORDS, pack_sampling, unpack_sampling

    for s in (Sampling(2**63 - 5, 0.7, 20, 0.95, 0.05), Sampling(7, 1.0, 0, 1.0), None):
        words = pack_sampling(s)
        assert len(words) == SAMPLING_WORDS and all(0 <= w < 1 << 16 for w in words)
        assert unpack_sampling(words) == s


# -- MLX: the nucleus path and the Metal kernel ---------------------------------------------------------------------
def test_the_mlx_nucleus_path_cuts_what_the_full_rule_cuts():
    """top_k 0: the fast nucleus (GPU candidates + normalizer) draws what sorting every logit draws, with min_p."""

    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine import exact_sampling as es

    rng = np.random.default_rng(6)
    drawn = 0
    for trial in range(60):
        row = (mx.array(rng.normal(size=(1, 50_000)).astype(np.float32)) * 8.0).astype(mx.bfloat16)
        s = es.Sampling(seed=int(rng.integers(1 << 62)), temperature=float(rng.choice([0.7, 1.0])), top_k=0,
                        top_p=0.95, min_p=float(rng.choice([0.001, 0.05, 0.3])))
        fast = es._nucleus_rows(row, [trial], s)
        full = es.choose_rows(np.array(row.astype(mx.float32)), np.arange(50_000, dtype=np.int64)[None], [trial], s)
        if fast is not None:
            drawn += 1
            assert fast == full
    assert drawn > 40


def test_the_metal_kernel_draws_its_reference_with_min_p():
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine import gpu_sampling

    logits = (mx.random.normal((6, 700), key=mx.random.key(8)) * 3).astype(mx.bfloat16)
    rows = np.array(logits.astype(mx.float32))
    for top_k in (20, 0, 64):                   # top_p off: the reference sums a nucleus in another order
        for min_p in (0.0, 0.03, 0.5, 1.0):
            s = Sampling(seed=77, temperature=0.8, top_k=top_k, top_p=1.0, min_p=min_p)
            got = gpu_sampling.sample(logits, s, list(range(10, 16))).tolist()
            assert got == [gpu_sampling.reference(rows[r], s, 10 + r) for r in range(6)], (top_k, min_p)
    mixed = [Sampling(1, 1.0, 20, 0.95, 0.1), None, Sampling(2, 0.7, 0, 0.8, 0.3), Sampling(3, 1.3)] * 2
    positions = [40 + r for r in range(6)]
    together = gpu_sampling.sample_rows(logits, mixed[:6], positions)
    alone = [gpu_sampling.sample(logits[r:r + 1], mixed[r], [positions[r]]) for r in range(6)]
    assert together.tolist() == mx.concatenate(alone).tolist()
