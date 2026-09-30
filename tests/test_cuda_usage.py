"""Every CUDA reply reports usage as the Mac server does, streamed ones too (without stream_options), with the prompt
tokens its first run found cached and the reply's thinking tokens."""

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
