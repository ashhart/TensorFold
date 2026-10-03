"""A reply names the model id its request asked for when the server answers to it, else the served name, on both backends and /v1/responses."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")                  # the Mac server's fakes

from tensorfold.server.http import reply_model
from tests.http_fakes import post
from tests.test_lane_server import make_app

HI = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("asked, named", [("alias-a", "alias-a"), ("fake-27b", "fake-27b"), ("other", "fake-27b"),
                                          (None, "fake-27b"), (7, "fake-27b")])
def test_reply_model_picks_the_asked_id_only_when_served(asked, named):
    mlx = SimpleNamespace(served_name="fake-27b", model_ids=["fake-27b", "alias-a"])
    cuda = SimpleNamespace(served="fake-27b", model_ids=["fake-27b", "alias-a"])
    body = {} if asked is None else {"model": asked}
    assert reply_model(mlx, body) == reply_model(cuda, body) == named


def test_mac_chat_completion_and_stream_name_the_alias():
    app = make_app()
    try:
        status, text = post(app, {"model": "alias-a", "messages": HI, "max_tokens": 2})
        assert status == 200 and json.loads(text)["model"] == "alias-a"
        status, text = post(app, {"model": "alias-a", "messages": HI, "max_tokens": 2, "stream": True})
        models = {json.loads(line[6:])["model"] for line in text.splitlines()
                  if line.startswith("data: {") and '"model"' in line}
        assert status == 200 and models == {"alias-a"}
        status, text = post(app, {"model": "someone-else", "messages": HI, "max_tokens": 2})
        assert status == 200 and json.loads(text)["model"] == "fake-27b"
    finally:
        app.close()


def test_mac_completion_and_responses_name_the_alias():
    app = make_app()
    try:
        status, text = post(app, {"model": "alias-a", "prompt": "hi", "max_tokens": 2}, "/v1/completions")
        assert status == 200 and json.loads(text)["model"] == "alias-a"
        status, text = post(app, {"model": "alias-a", "input": "hi", "max_output_tokens": 2}, "/v1/responses")
        assert status == 200 and json.loads(text)["model"] == "alias-a"
    finally:
        app.close()
