"""Every CUDA reply reports usage as the Mac server does: on the finish chunk of a stream that names no
stream_options, in the OpenAI spec's own usage chunk before [DONE] when it asks for one, with the prompt tokens its
first run found cached and the reply's thinking tokens."""

import json

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_cuda_thinking_controls import ChainEngine, app_for
from tests.test_cuda_tool_choice import events


class CachedEngine(ChainEngine):
    """Reports 7 cached prompt tokens on its first run, 30 on a later one (a restart after a cut)."""

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        super().generate(prompt, max_tokens, sampling, on_tokens, draft)
        return {"rounds": 1, "cached": 7 if len(self.prompts) == 1 else 30}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", [0, 3])
def test_usage_counts_the_reply_and_the_first_runs_cache(tmp_path, stream, budget):
    engine = CachedEngine()
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": stream, "thinking_budget": budget}
    with http_server(app_for(tmp_path, engine)) as port:
        status, text = post(port, body, True)
    assert status == 200 and len(engine.prompts) == (2 if budget else 1)
    usage = [c for c in events(text) if c.get("choices")][-1]["usage"] if stream else json.loads(text)["usage"]
    prompt = len(engine.prompts[0])
    assert usage == {"prompt_tokens": prompt, "completion_tokens": 24, "total_tokens": prompt + 24,
                     "prompt_tokens_details": {"cached_tokens": 7},
                     # thinking through the budget's close (its third token becomes a newline, then </think>)
                     "completion_tokens_details": {"reasoning_tokens": 4 if budget else 24}}


def streamed(port, body, chat=True):
    """(this stream's data events, its raw text) for one request."""

    status, text = post(port, {**body, "stream": True}, chat)
    assert status == 200, text
    return events(text), text


@pytest.mark.parametrize("chat", [False, True])
def test_include_usage_moves_usage_into_its_own_chunk_before_done(tmp_path, chat):
    """litellm reads usage only from the spec's chunk: no choices, the reply's id and model, after the finish chunk."""

    engine = CachedEngine()
    prompt = {"messages": [{"role": "user", "content": "Hi"}]} if chat else {"prompt": "Hi"}
    body = {**prompt, "thinking_budget": 0, "stream_options": {"include_usage": True}}
    with http_server(app_for(tmp_path, engine)) as port:
        chunks, text = streamed(port, body, chat)
    assert text.endswith("data: [DONE]\n\n") and text.count("data: [DONE]") == 1
    silent = [c for c in chunks if not c["choices"]]
    assert len(silent) == 1 and chunks[-1] is silent[0]
    end = [c for c in chunks if c["choices"]][-1]
    assert end["choices"][0]["finish_reason"] and end["tensorfold"] and "usage" not in end
    assert all(silent[0][key] == end[key] for key in ("id", "object", "created", "model"))
    prompt_tokens = len(engine.prompts[0])
    # a plain completion thinks of its own accord only in the chat lane, and its usage chunk says so
    assert silent[0]["usage"] == {"prompt_tokens": prompt_tokens, "completion_tokens": 24,
                                  "total_tokens": prompt_tokens + 24,
                                  "prompt_tokens_details": {"cached_tokens": 7},
                                  "completion_tokens_details": {"reasoning_tokens": 24 if chat else 0}}


@pytest.mark.parametrize("chat", [False, True])
def test_a_stream_without_the_option_keeps_usage_on_the_finish_chunk(tmp_path, chat):
    """The clients that never send stream_options keep counting tokens from the finish chunk, as they always did."""

    engine = CachedEngine()
    prompt = {"messages": [{"role": "user", "content": "Hi"}]} if chat else {"prompt": "Hi"}
    with http_server(app_for(tmp_path, engine)) as port:
        chunks, text = streamed(port, {**prompt, "thinking_budget": 0}, chat)
    assert text.endswith("data: [DONE]\n\n") and all(c["choices"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] and chunks[-1]["usage"]["completion_tokens"] == 24
