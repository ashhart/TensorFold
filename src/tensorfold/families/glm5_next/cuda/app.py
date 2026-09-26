"""The OpenAI routes for GLM-5.3-Flash (``CUDA_APP``): ``tensorfold.cuda.server.App`` plus two request fields.

- A request picks its draft policy with ``"model": "<served name>@<spec>"`` or ``"tf_policy": "<spec>"`` (specs in
  ``engine.py``); without one the engine's default applies.
- ``"ignore_eos": true`` decodes ``max_tokens`` tokens past an end-of-sequence token (the benchmarks use it).

A request whose prompt plus ``max_tokens`` needs more context than the engine was started for gets an HTTP 400
before anything streams, naming the ``--context`` to restart both ranks with. Without ``max_tokens`` the reply
stops where the context ends.

With thinking off, a chat renders the way GLM-5.3's thinking-off template does: the checkpoint's template always
opens a think block, so it gets an empty ``<think></think>`` and no reasoning-effort line.
"""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda.server import App


class ThinkingOffTemplate:
    """The checkpoint's chat template, rendered as GLM-5.3's thinking-off template renders it when thinking is off."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def render(self, messages, *, tools, enable_thinking, extra=None) -> str:
        text = self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra)
        if not enable_thinking:
            text = text.replace("<|system|>Reasoning Effort: Max", "", 1)
            if text.endswith("<|assistant|><think>"):
                text += "</think>"
        return text


class GlmApp(App):
    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        self.template = ThinkingOffTemplate(self.template)

    def check(self, body: dict[str, Any]) -> str | None:
        """``App.check``, then the context: the prompt as ``run`` renders it plus ``max_tokens`` must fit the
        engine's limit (2,051 tokens unless the server was started with a larger ``--context``)."""

        problem = super().check(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kwargs.pop("enable_thinking", self.default_thinking))
        if "messages" in body:
            text = self.template.render(body["messages"], tools=body.get("tools") or [], enable_thinking=thinking,
                                        extra=kwargs)
        else:
            text = body.get("prompt") or ""
        prompt = len(self.tok.encode(text, add_special_tokens=False).ids)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return None
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        return (f"this request needs a {need}-token context ({detail}), and this server was started for {limit}: "
                f"restart both ranks with --context {need} or more")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit)
