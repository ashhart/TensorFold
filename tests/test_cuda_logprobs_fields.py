"""Unsupported probability requests and multiple choices fail before generation."""

import json

import pytest

from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, app_for, request


@pytest.mark.parametrize("fields, name", [({"logprobs": True, "top_logprobs": 5}, "logprobs"), ({"n": 2}, "n")])
def test_unsupported_choices_are_refused_before_generation(tmp_path, fields, name):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "max_tokens": 1, "temperature": 0, **fields})
    assert status == 400, text
    assert name in json.loads(text)["error"]["message"]
    assert app.engine.calls == []


@pytest.mark.parametrize("fields, name", [({"n": True}, "n"), ({"n": 0}, "n"), ({"n": 1.5}, "n"),
                                        ({"logprobs": "true"}, "logprobs"),
                                        ({"top_logprobs": 21}, "top_logprobs"),
                                        ({"top_logprobs": -1}, "top_logprobs"),
                                        ({"top_logprobs": True}, "top_logprobs"),
                                        ({"top_logprobs": 5}, "top_logprobs")])
def test_malformed_probability_options_are_refused(tmp_path, fields, name):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, **fields})
    assert status == 400 and name in json.loads(text)["error"]["message"]
    assert app.engine.calls == []


def test_single_choice_without_probabilities_still_generates(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "n": 1, "logprobs": False})
    assert status == 200 and len(json.loads(text)["choices"]) == 1
    assert len(app.engine.calls) == 1


@pytest.mark.parametrize("fields", [{"n": 2}, {"logprobs": True, "top_logprobs": 5}])
def test_mac_handler_also_refuses_unsupported_options(fields):
    from tests.test_server_openai_compat import FakeApp, post_json, serve_fake

    app = FakeApp()
    server = serve_fake(app)
    try:
        status, text = post_json(server, "/v1/chat/completions", {"messages": HI, **fields})
        assert status == 400 and "error" in json.loads(text)
        assert app.messages is None
    finally:
        server.shutdown()
        server.server_close()


def probability_app(tmp_path):
    from tokenizers import Tokenizer, decoders, models
    from tests.test_cuda_server_errors import Engine

    class Target(Engine):
        supports_logprobs = True

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, probabilities=None):
            self.calls.append(list(prompt))
            if probabilities is not None:
                probabilities.add([len(prompt)], [1], [-0.25], [[1, 2]], [[-0.25, -1.5]])
            on_tokens([1])
            return {}

    app = app_for(tmp_path)
    app.engine = Target()
    app.tok = Tokenizer(models.WordLevel({"[UNK]": 0, "A": 1, "ĠB": 2}, unk_token="[UNK]"))
    app.tok.decoder = decoders.ByteLevel()
    return app


def test_one_token_chat_exposes_the_target_probabilities(tmp_path):
    app = probability_app(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "max_tokens": 1, "temperature": 0,
                                        "logprobs": True, "top_logprobs": 2})
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert choice["message"]["content"] == "A"
    assert choice["logprobs"]["content"] == [{"token": "A", "bytes": [65], "logprob": -0.25,
        "top_logprobs": [{"token": "A", "bytes": [65], "logprob": -0.25},
                        {"token": " B", "bytes": [32, 66], "logprob": -1.5}]}]


@pytest.mark.parametrize("fields", [{"stream": True}, {"stop": ["A"]},
                                   {"chat_template_kwargs": {"enable_thinking": True}}])
def test_unaligned_probability_modes_fail_before_generation(tmp_path, fields):
    app = probability_app(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "logprobs": True, **fields})
    assert status == 400 and "logprobs" in json.loads(text)["error"]["message"]
    assert app.engine.calls == []


def test_split_utf8_probabilities_preserve_the_original_bytes():
    from tokenizers import Tokenizer, decoders, models
    from tensorfold.server.probabilities import TokenBytes

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "Ã": 1, "©": 2}, unk_token="[UNK]"))
    tok.decoder = decoders.ByteLevel()
    decoder = TokenBytes(tok)
    assert decoder.token(1) == {"token": "\ufffd", "bytes": [195]}
    assert decoder.token(2) == {"token": "\ufffd", "bytes": [169]}
    assert tok.decode([1, 2]) == "é"


def test_added_tokens_follow_the_bytelevel_decoder():
    from tokenizers import AddedToken, Tokenizer, decoders, models
    from tensorfold.server.probabilities import TokenBytes

    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.decoder = decoders.ByteLevel()
    tok.add_tokens([AddedToken("ĠB"), AddedToken("你好")])
    decoder = TokenBytes(tok)
    for piece in ("ĠB", "你好"):
        token = tok.token_to_id(piece)
        expected = tok.decode([token], skip_special_tokens=False)
        assert decoder.token(token) == {"token": expected, "bytes": list(expected.encode("utf-8"))}
