"""The Mac and CUDA servers render the same prompt ids for the same request from each checkpoint's own chat template
and tokenizer (Qwen3.8-27B, Flash Next, GLM-5.3 and Nemotron): thinking on and off, every effort or none, a server
default effort or none, tools and a tool-call history. Checkpoints come from the environment or the Hugging Face
cache; a missing one is skipped."""

import json
import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tensorfold import hub
from tensorfold.cuda import server
from tensorfold.server.request_options import RequestOptions, thinking_fields
from tensorfold.server.text import render_prompt_ids, template_late_system

CHECKPOINTS = {                       # name: (environment variable, Hugging Face repo)
    "qwen27": ("TENSORFOLD_MLX_MODEL", "Vontra/Qwen3.8-27B-MLX-4bit"),
    "flashnext": ("TF_FLASHNEXT_MODEL", "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"),
    "glm": ("TF_GLM5_MODEL", "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"),
    "nemotron": ("TF_NEMOTRON_MODEL", "Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit"),
}
WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
                                             "parameters": {"type": "object", "required": ["city"],
                                                            "properties": {"city": {"type": "string"}}}}}]
CONVERSATIONS = {
    "plain": ([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 2+2?"}], None),
    "tools": ([{"role": "user", "content": "Weather in Oslo?"},
               {"role": "assistant", "content": "", "tool_calls": [
                   {"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "{\"celsius\": 12}"},
               {"role": "user", "content": "And Bergen?"}], WEATHER),
    "late system": ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
                     {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Bye"}], None),
}
REQUESTS = [{}, {"reasoning_effort": "none"}, {"reasoning_effort": "minimal"}, {"reasoning_effort": "low"},
            {"reasoning_effort": "medium"}, {"reasoning_effort": "high"}, {"reasoning_effort": "xhigh"},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}]


def _folder(name):
    variable, repo = CHECKPOINTS[name]
    found = os.environ.get(variable) or hub.cached(repo)
    folder = Path(found) if found else None
    if folder is None or not (folder / "tokenizer.json").is_file():
        pytest.skip(f"needs the {name} tokenizer and chat template ({variable} or {repo})")
    return folder


class ReturnIds:
    """mlx_lm's tokenizer wrapper where mlx_lm is missing: its chat call returns the ids, not a dict."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.inner.apply_chat_template(*args, return_dict=False, **kwargs)


def mac_tokenizer(name, folder):
    """The tokenizer the Mac family loads: mlx_lm's (GLM's wrapped to render its thinking-off prompt)."""

    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError:
        tokenizer = ReturnIds(transformers.AutoTokenizer.from_pretrained(str(folder), local_files_only=True))
    else:
        tokenizer = load_tokenizer(folder)
    if name == "glm":
        from tensorfold.families.glm5_next.prompts import GlmTokenizer

        tokenizer = GlmTokenizer(tokenizer)
    return tokenizer


class Mac(RequestOptions):
    """The Mac server's request path to prompt ids: its HTTP fields, then ``ChatApp.render``'s call."""

    def __init__(self, name, folder, thinking, effort):
        self.tokenizer = mac_tokenizer(name, folder)
        self.late_system = template_late_system(self.tokenizer)
        self.enable_thinking, self.reasoning_effort = thinking, effort

    def prompt(self, body, messages, tools):
        fields = thinking_fields(body, self.effort_levels)
        requested = fields.get("enable_thinking")
        thinking = self.enable_thinking if requested is None else bool(requested)
        return render_prompt_ids(self.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                 reasoning_effort=fields.get("reasoning_effort", self.reasoning_effort),
                                 late_system=self.late_system)


def cuda_app(name, folder, thinking, effort):
    from tokenizers import Tokenizer

    cls = server.App
    if name == "glm":
        from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate
        cls = GlmApp
    app = cls.__new__(cls)
    app.engine, app.served, app.model_dir = None, "parity", folder
    app.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    app.template = server.ChatTemplate(folder)
    if name == "glm":
        app.template = ThinkingOffTemplate(app.template)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = thinking, effort, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 16
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


@pytest.mark.parametrize("default_effort", [None, "low"])
@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_the_mac_and_cuda_render_the_same_prompt(name, default_effort):
    folder = _folder(name)
    mac, cuda = Mac(name, folder, True, default_effort), cuda_app(name, folder, True, default_effort)
    for label, (messages, tools) in CONVERSATIONS.items():
        for request in REQUESTS:
            body = {"messages": messages, **({"tools": tools} if tools else {}), **request}
            mac_ids = mac.prompt(body, messages, tools)
            cuda_ids = cuda._prepare(body, True).prompt
            assert mac_ids == cuda_ids, (label, request, mac.tokenizer.decode(mac_ids)[-400:],
                                         mac.tokenizer.decode(cuda_ids)[-400:])


@pytest.mark.parametrize("name, request_fields, words", [
    ("qwen27", {}, "Reasoning effort is set to xhigh"),                 # the template's own default, as vLLM's
    ("qwen27", {"reasoning_effort": "high"}, "Reasoning effort is set to xhigh"),
    ("qwen27", {"reasoning_effort": "low"}, "Reasoning effort is set to low"),
    ("qwen27", {"reasoning_effort": "medium"}, "<|im_start|>system\nYou are terse."),     # medium adds no line
    ("flashnext", {}, "Reasoning effort is set to xhigh"),
    ("glm", {}, "Reasoning Effort: Max"),
    ("glm", {"reasoning_effort": "high"}, "Reasoning Effort: High"),   # GLM-5.3 names high: it stays high
    ("glm", {"reasoning_effort": "minimal"}, "Reasoning Effort: Low"),
])
def test_the_effort_each_template_writes(name, request_fields, words):
    folder = _folder(name)
    messages, _ = CONVERSATIONS["plain"]
    mac, cuda = Mac(name, folder, True, None), cuda_app(name, folder, True, None)
    body = {"messages": messages, **request_fields}
    ids = mac.prompt(body, messages, None)
    assert ids == cuda._prepare(body, True).prompt and words in mac.tokenizer.decode(ids)
