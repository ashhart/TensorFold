"""A tool result's images through the vision checkpoints' own chat templates (no GPU, no weights).

Agents return screenshots as tool results, and every later turn of the session sends them again. Each vision family's
template renders a tool message's image parts in place, inside its tool response, so the server passes them through as
it does a user message's. The prompt then holds exactly one image marker per image, the count the image frontends
check. The test needs each checkpoint's template files in the Hugging Face cache; repos that are absent are skipped:

    hf download Vontra/GLM-5.3-Flash-MLX-4bit-MTP tokenizer_config.json chat_template.jinja
"""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("jinja2")
Image = pytest.importorskip("PIL.Image")

from tensorfold.vision.images import split_images

FILES = ("tokenizer_config.json", "chat_template.jinja")
CHECKPOINTS = [                    # each repo and its image frontend's marker
    ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP", "<|image|>"),
    ("Vontra/Qwen3.8-27B-MLX-4bit", "<|image_pad|>"),
    ("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", "<|image_pad|>"),
]


def _checkpoint(repo):
    from tensorfold import hub

    try:
        found = hub.cached(repo)
    except ImportError:
        return None
    return found if found is not None and all((found / name).is_file() for name in FILES) else None


def _url(color):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _session():
    call = {"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}
    return [{"role": "user", "content": [{"type": "text", "text": "Compare the page with this mockup: "}, _url("red")]},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "PAGE"}, _url("blue")]},
            {"role": "user", "content": "Does it match?"}]


TOOLS = [{"type": "function", "function": {"name": "screenshot", "description": "Capture the page.",
                                           "parameters": {"type": "object", "properties": {}}}}]


@pytest.mark.parametrize("repo, marker", CHECKPOINTS, ids=[c[0].split("/")[1] for c in CHECKPOINTS])
@pytest.mark.parametrize("thinking", [False, True])
def test_a_tool_result_s_image_renders_inside_its_tool_response(repo, marker, thinking):
    checkpoint = _checkpoint(repo)
    if checkpoint is None:
        pytest.skip(f"needs {repo}'s template files in the Hugging Face cache (see this file's docstring)")
    from tensorfold.cuda.chat_template import ChatTemplate

    template, sources = split_images(_session())
    assert len(sources) == 2
    text = ChatTemplate(checkpoint).render(template, tools=TOOLS, enable_thinking=thinking, allow_images=True)
    assert text.count(marker) == len(sources)            # what the frontends' prepare() checks
    start = text.index("<tool_response>")
    tool = text[start:text.index("</tool_response>", start)]
    assert "PAGE" in tool and tool.count(marker) == 1     # the screenshot sits in its tool response, after its text
    assert tool.index("PAGE") < tool.index(marker)
    assert text.index(marker) < start                     # the user's image comes first, as sent
