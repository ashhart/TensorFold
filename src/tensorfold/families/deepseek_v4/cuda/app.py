"""Embedded ds4 tokenizer and existing DeepSeek prompt/DSML adapters on shared HTTP."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from types import SimpleNamespace

from tensorfold.cuda.server import App
from tensorfold.engine.call_gate import CallGate, call_format
from tensorfold.families.deepseek_v4.prompts import EFFORTS, DeepSeekTokenizer
from tensorfold.server.errors import RequestError
from tensorfold.server.messages import normalize_messages
from tensorfold.server.text import hide_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content


class NativeTokenizer:
    """Only the tokenizer API shared CUDA serving uses; bytes come from native ds4.

    Join token bytes before UTF8 decoding, so split multibyte characters stream
    without replacement characters. Caching bounds RPCs and host memory.
    """

    def __init__(self, session):
        self.session = session
        self._piece = lru_cache(maxsize=8192)(self._piece)
        self.token_to_id = lru_cache(maxsize=128)(self.token_to_id)

    def encode(self, text, *, add_special_tokens=False):
        if not isinstance(text, str):
            raise TypeError("native tokenizer input must be text")
        if add_special_tokens:
            raise ValueError("rendered DeepSeek prompts already contain BOS; automatic insertion is unsupported")
        return SimpleNamespace(ids=self.session.encode(text, rendered=True))

    def _piece(self, token):
        return self.session.token_text(int(token))

    def token_to_id(self, text):
        ids = self.encode(text).ids
        return ids[0] if len(ids) == 1 and self._piece(ids[0]) == text.encode("utf-8") else None

    def decode(self, ids, *, skip_special_tokens=False):
        if skip_special_tokens:
            # The serving path explicitly keeps special tokens for think/tool parsing.
            special = {self.session.eos, self.token_to_id("<｜begin▁of▁sentence｜>")}
            ids = [t for t in ids if t not in special]
        return b"".join(self._piece(int(token)) for token in ids).decode("utf-8", errors="replace")


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
    parse_tool_calls = staticmethod(parse_tool_calls_from_content)

    def _tokenizer_template(self, model_dir):
        return NativeTokenizer(self.engine.session), DeepSeekTemplate()

    def close(self):
        self.engine.close()

    def _tool_content(self, policy, answer, *, finished):
        # The shared single-call policy filters generic envelopes; DeepSeek's
        # existing text filter also understands partial DSML envelopes.
        if policy.single:
            answer = policy.content(answer, finished=finished)
        return hide_tool_calls(answer, finished=finished)

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
        first = self.tok.decode(ids[:1])
        if form is None or not ids or not first or not opener.startswith(first):
            raise RequestError("native tokenizer cannot represent the DeepSeek DSML tool-call prefix")
        _, lead, tail = form
        names = [str((t.get("function") or t).get("name") or "") for t in tools]
        eos = set(self.engine.eos)
        return CallGate.after_prompt(
            prompt,
            ids[0],
            lambda t: t not in eos and not self.tok.decode([t]).strip(),
            think_open=self.tok.token_to_id("<think>"),
            think_end=self.tok.token_to_id("</think>"),
            text=lambda t: self.tok.decode([t]),
            lead=opener[len(first) :] + lead,
            names=names,
            tail=tail,
            encode=lambda text: self.tok.encode(text).ids,
        )
