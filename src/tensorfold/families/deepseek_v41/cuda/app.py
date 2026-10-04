"""DeepSeek-V4.1 requests on the shared CUDA App: DeepSeek's prompt encoder, DSML tool calls, integer efforts."""

from __future__ import annotations

from typing import Any

from tensorfold.cuda.reply_text import parse_tool_calls
from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.families.deepseek_v41.prompts import Template, numeric_effort
from tensorfold.server.errors import RequestError
from tensorfold.server.text import hide_tool_calls
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.server.tools import parse_tool_calls_from_content

DSML_CALLS = "<｜DSML｜ calls>"
# both ranks decode to max_tokens or an end token, so a budget's cut would decode a whole reply and throw it away
NO_BUDGET = "thinking_budget is not supported by DeepSeek-V4.1-Flash's two-rank engine"


def dsml_calls(text: str, tools: list[dict[str, Any]], *,
               max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
    """The reply's content and OpenAI tool calls, from V4.1's DSML block when it wrote one."""

    if DSML_CALLS in text:
        return parse_tool_calls_from_content(text, tools, max_calls=max_calls)
    return parse_tool_calls(text, tools, max_calls=max_calls)


class DeepSeekV41App(App):
    template_class = Template
    parse_calls = staticmethod(dsml_calls)

    def __init__(self, engine, model_dir, served: str, *, sampling: dict[str, Any] | None = None,
                 **kwargs: Any) -> None:
        # DeepSeek recommends no top-k; a CLI or request top_k still wins
        super().__init__(engine, model_dir, served, sampling={"top_k": 0, **(sampling or {})}, **kwargs)

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        prepared = super()._prepare(numeric_effort(body), chat)
        if prepared.think_budget > 0:
            raise RequestError(NO_BUDGET)
        return prepared

    def _visible_answer(self, answer: str, policy: ToolCallPolicy, finished: bool) -> str:
        answer = hide_tool_calls(answer, finished=finished)     # the policy's one-call filter knows no DSML
        return policy.content(answer, finished=finished) if policy.single else answer
