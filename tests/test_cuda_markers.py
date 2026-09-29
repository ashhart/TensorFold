"""CUDA snapshot points: the second message's start and the last assistant start, spaced by MIN_GAP."""

import pytest

from tensorfold.cuda.markers import MIN_GAP, snapshot_points

OPEN, ASSISTANT = 900, 901


def _prompt(system: int, user: int, history: int = 0) -> list[int]:
    ids = [OPEN] + [5] * system + [OPEN] + [6] * user
    if history:
        ids += [OPEN, ASSISTANT] + [7] * history + [OPEN] + [6] * user
    return ids + [OPEN, ASSISTANT, 8]


def test_system_block_end_and_last_reply_start():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    ids = _prompt(3 * MIN_GAP, 2 * MIN_GAP)
    assert points(ids) == [3 * MIN_GAP + 1, len(ids) - 3]
    chat = _prompt(3 * MIN_GAP, MIN_GAP, history=2 * MIN_GAP)
    assert points(chat) == [3 * MIN_GAP + 1, len(chat) - 3]


def test_points_too_close_are_dropped():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    assert points(_prompt(10, 3 * MIN_GAP)) == [len(_prompt(10, 3 * MIN_GAP)) - 3]      # system block under MIN_GAP
    ids = _prompt(3 * MIN_GAP, 10)
    assert points(ids) == [3 * MIN_GAP + 1]                                              # reply start too close
    assert points([5] * 4 * MIN_GAP) == []


# Qwen3.6's chat template, cut to its roles: a reply after the last user message keeps a think block that history
# drops; Qwen3.8's keeps it everywhere
TEMPLATE = """
{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- set ns = namespace(last_query_index=0) %}
{%- for message in messages %}
    {%- if message.role == 'user' %}{%- set ns.last_query_index = loop.index0 %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.role == 'user' %}
        {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role == 'assistant' %}
        {%- if KEPT %}
            {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message.content + '<|im_end|>\\n' }}
        {%- else %}
            {{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}"""


@pytest.mark.parametrize("kept", ["loop.index0 > ns.last_query_index", "true"], ids=["qwen3.6", "qwen3.8"])
def test_a_template_that_drops_history_think_blocks_keeps_states_at_message_starts(tmp_path, kept: str):
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    from tensorfold.cuda.markers import TemplateTokens, resume_points
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.engine.prefill_plan import message_markers

    chars = [chr(c) for c in range(32, 127)] + ["\n"]                    # a token per character
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({c: i for i, c in enumerate(chars)}, unk_token=" "))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split(tokenizers.Regex(r"[\s\S]"), behavior="isolated")
    tok.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tok.add_tokens(["<think>", "</think>"])
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE.replace("KEPT", kept))
    opener = tok.token_to_id("<|im_start|>")
    assert message_markers(TemplateTokens(tok, ChatTemplate(tmp_path))) == ((opener,), (opener, chars.index("a")))

    points, template = resume_points(tmp_path), ChatTemplate(tmp_path)
    assert points is not None

    def ids(messages):
        return tok.encode(template.render(messages, tools=None, enable_thinking=False), add_special_tokens=False).ids

    def starts(prompt):
        return [i for i, t in enumerate(prompt) if t == opener]

    talk = [{"role": "system", "content": "s" * 3 * MIN_GAP}, {"role": "user", "content": "u" * 2 * MIN_GAP}]
    first = ids(talk)
    assert points(first) == starts(first)[1:3]                   # the system block's end and the reply's start
    follow = ids([*talk, {"role": "assistant", "content": "r" * MIN_GAP}, {"role": "user", "content": "v" * MIN_GAP}])
    reply = starts(first)[2]
    assert follow[:reply] == first[:reply]                       # a follow-up resumes where the last reply began
    assert points(follow) == [starts(follow)[1], starts(follow)[-1]]
