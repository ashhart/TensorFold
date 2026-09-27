"""Raw completion ids outside the vocabulary are refused before the engine reads them."""

import json
from types import SimpleNamespace

import pytest

from tests.http_fakes import post
from tests.test_raw_completions import RawApp

from tests.test_lane_server import make_app
from tests.test_server_openai_compat import post_json, serve_fake


@pytest.mark.parametrize("ids", [[5, -1], [5, 1_000_000_000]])
def test_out_of_vocabulary_token_ids_are_refused(ids):
    app = make_app()
    app.tokenizer.vocab_size = 97
    seen = []
    prefill = app.engine._family_prefill

    def spy(stream, **kwargs):
        seen.append(list(stream.prompt_ids))
        return prefill(stream, **kwargs)

    app.engine._family_prefill = spy
    server = serve_fake(app)
    try:
        status, body = post_json(server, "/v1/completions", {"prompt": ids, "max_tokens": 2})
        assert status == 400, (status, body, "ids given to the engine:", seen)
        assert not seen
    finally:
        server.shutdown()
        server.server_close()
        app.close()


class SizedTokenizer:
    vocab_size = 95
    _tokenizer = object()

    def __len__(self):
        return 97


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("token", [-1, 97, 1_000_000_000])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_raw_ids_are_rejected_before_chat(wrapped, token, stream):
    app = RawApp()
    tokenizer = SizedTokenizer()
    app.tokenizer = SimpleNamespace(_tokenizer=tokenizer) if wrapped else tokenizer
    status, body = post(app, {"prompt": [5, token], "stream": stream}, "/v1/completions")
    assert status == 400
    assert "0" in json.loads(body)["error"]["message"] and "96" in body
    assert app.messages is None


@pytest.mark.parametrize("token", [0, 96])
def test_raw_ids_include_added_tokens_and_keep_the_boundaries(token):
    app = RawApp()
    app.tokenizer = SizedTokenizer()
    status, body = post(app, {"prompt": [token]}, "/v1/completions")
    assert status == 200, body
    assert app.prompt == [token]


def test_tokenizer_vocab_size_is_used_when_length_is_unavailable():
    app = RawApp()
    status, body = post(app, {"prompt": [app.tokenizer.vocab_size]}, "/v1/completions")
    assert status == 400 and "255" in body
    assert post(app, {"prompt": [255]}, "/v1/completions")[0] == 200
