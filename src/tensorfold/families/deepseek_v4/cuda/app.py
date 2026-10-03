"""DeepSeek prompt and DSML adapters on TensorFold shared CUDA serving."""

from __future__ import annotations

import copy
import json

from tensorfold.cuda.server import App
from tensorfold.engine.call_gate import CallGate, call_format
from tensorfold.families.deepseek_v4.prompts import EFFORTS, DeepSeekTokenizer
from tensorfold.server.errors import RequestError
from tensorfold.server.messages import normalize_messages


class DeepSeekTemplate:
    efforts = frozenset(EFFORTS)

    def __init__(self):
        # Reuse the existing encoder's generation-prefix handling without invoking
        # any external tokenizer (tokenize=False below).
        self.encoder = DeepSeekTokenizer(None)

    def render(self, messages, *, tools, enable_thinking, extra=None, allow_images=False):
        try:
            messages = copy.deepcopy(normalize_messages(messages, allow_images=allow_images))
            # The DeepSeek encoder expects JSON strings, unlike the Jinja adapters.
            for message in messages:
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    if isinstance(function.get("arguments"), dict):
                        function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False, allow_nan=False)
            return self.encoder.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                enable_thinking=enable_thinking,
                reasoning_effort=(extra or {}).get("reasoning_effort"),
            )
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            raise RequestError(f"DeepSeek prompt encoder rejected the request: {exc}") from exc


class DeepSeekApp(App):
    def __init__(self, engine, model_dir, served, **kwargs):
        super().__init__(engine, model_dir, served, **kwargs)
        self.template = DeepSeekTemplate()

    def close(self):
        self.engine.close()

    def _call_gate(self, prompt, tools):
        """Reuse CallGate with the first token of DSML's mult-token envelope.

        Its remaining envelope and invoke prefix are the gate's forced lead,
        rather than incorrectly assuming the whole DSML opener is one token.
        """
        opener = "<｜DSML｜tool_calls>"
        probe = [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}
                ],
            },
        ]
        rendered = self.template.render(probe, tools=None, enable_thinking=False)
        form = call_format(rendered, "tfprobe_fn", [opener])
        ids = self.tok.encode(opener).ids
        first = self.tok.decode(ids[:1], skip_special_tokens=False)
        if form is None or not ids or not first or not opener.startswith(first):
            raise RequestError("native tokenizer cannot represent the DeepSeek DSML tool-call prefix")
        _, lead, tail = form
        names = [str((t.get("function") or t).get("name") or "") for t in tools]
        eos = set(self.engine.eos)
        return CallGate.after_prompt(
            prompt,
            ids[0],
            lambda t: t not in eos and not self.tok.decode([t], skip_special_tokens=False).strip(),
            think_open=self.tok.token_to_id("<think>"),
            think_end=self.tok.token_to_id("</think>"),
            text=lambda t: self.tok.decode([t], skip_special_tokens=False),
            lead=opener[len(first) :] + lead,
            names=names,
            tail=tail,
            encode=lambda text: self.tok.encode(text).ids,
        )
