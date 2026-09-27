"""top_k -1 is how vLLM-style clients say "no top-k"; it must sample, not fail the request with std::bad_cast."""

import json
from types import SimpleNamespace

import pytest

from tensorfold.server.app import ChatApp
from tests.http_fakes import post
from tests.test_server_openai_compat import FakeApp
from tests.test_lane_server import make_app
from tests.test_server_openai_compat import post_json, serve_fake


def test_http_top_k_minus_one_samples():
    app = make_app()
    server = serve_fake(app)
    try:
        status, body = post_json(server, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                                  "temperature": 1.0, "top_k": -1, "seed": 7})
        assert status == 200, body
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_bad_sampling_field_is_a_400_not_a_500():
    app = make_app()
    server = serve_fake(app)
    try:
        status, body = post_json(server, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                                  "temperature": 1.0, "seed": "abc"})
        assert status == 400, (status, body)
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_numpy_sampler_top_k_minus_one_is_not_greedy():
    import numpy as np

    from tensorfold.engine.exact_sampling import Sampling, choose_rows

    values = np.array([[2.0, 1.9, 1.8, 1.7]] * 64, dtype=np.float32)
    ids = np.tile(np.arange(4, dtype=np.int64), (64, 1))
    positions = list(range(64))
    off = choose_rows(values, ids, positions, Sampling(seed=1, temperature=1.0, top_k=0, top_p=1.0))
    minus_one = choose_rows(values, ids, positions, Sampling(seed=1, temperature=1.0, top_k=-1, top_p=1.0))
    assert len(set(off)) > 1                      # top_k 0: a real draw over the four tokens
    assert minus_one == off, f"top_k -1 drew {set(minus_one)}: greedy"


class SamplingApp(FakeApp):
    accepts_sampling = True
    accepts_raw_prompt = True
    default_sampling = {"temperature": 0.8, "top_p": 0.9, "top_k": 20}

    def chat(self, messages, *, sampling=None, prompt=None, **kwargs):
        self.resolved = ChatApp._resolve_sampling(self, sampling, kwargs.get("temperature", 0), [1, 2])
        return super().chat(messages, **kwargs)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("field", ["seed", "temperature", "top_p", "top_k", "thinking_budget",
                                   "max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", ["abc", [], {}, float("nan")])
def test_bad_numbers_are_refused_before_the_app(stream, route, field, value):
    app = SamplingApp()
    body = {"messages": [{"role": "user", "content": "hi"}], "prompt": "hi", "stream": stream, field: value}
    status, response = post(app, body, route)
    assert status == 400, response
    assert field in json.loads(response)["error"]["message"]
    assert app.messages is None


@pytest.mark.parametrize("field", ["temperature", "top_p", "top_k"])
def test_null_sampling_uses_model_defaults(field):
    app = SamplingApp()
    status, response = post(app, {"messages": [{"role": "user", "content": "hi"}], field: None})
    assert status == 200, response
    assert getattr(app.resolved, field) == app.default_sampling[field]


@pytest.mark.parametrize("top_k", [-100, -1, 0])
def test_nonpositive_top_k_resolves_to_no_filter(top_k):
    app = SimpleNamespace(default_sampling={"temperature": 1.0})
    assert ChatApp._resolve_sampling(app, {"top_k": top_k}, 1.0, [1]).top_k == 0


@pytest.mark.parametrize("backend", ["gpu", "qwen"])
def test_both_family_samplers_treat_minus_one_as_zero(backend):
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.gpu_sampling import sample
    from tensorfold.families.qwen3_5.family import Qwen35Family

    logits = mx.array([[2.0, 1.9, 1.8, 1.7]] * 64, dtype=mx.bfloat16)
    positions = list(range(64))
    draw = sample if backend == "gpu" else lambda x, s, p: Qwen35Family.sample(None, x, s, p)
    off = list(draw(logits, Sampling(seed=1, top_k=0, top_p=1.0), positions))
    minus_one = list(draw(logits, Sampling(seed=1, top_k=-1, top_p=1.0), positions))
    assert [int(t) for t in minus_one] == [int(t) for t in off]
    assert len({int(t) for t in off}) > 1


@pytest.mark.parametrize("field", ["seed", "top_k", "thinking_budget", "max_tokens", "max_completion_tokens"])
def test_fractional_integer_controls_are_refused(field):
    status, response = post(SamplingApp(), {"messages": [{"role": "user", "content": "hi"}], field: 1.5})
    assert status == 400 and field in response


def test_numeric_strings_are_parsed_and_explicit_zero_is_preserved():
    app = SamplingApp()
    status, response = post(app, {"messages": [{"role": "user", "content": "hi"}], "temperature": "1",
                                 "top_k": "-1", "top_p": "0", "seed": "7", "max_tokens": "4"})
    assert status == 200, response
    assert (app.resolved.temperature, app.resolved.top_k, app.resolved.top_p, app.resolved.seed) == (1.0, 0, 0, 7)


def test_invalid_sampling_is_refused_even_when_greedy():
    status, response = post(SamplingApp(), {"messages": [{"role": "user", "content": "hi"}],
                                          "temperature": 0, "seed": "abc"})
    assert status == 400 and "seed" in response
