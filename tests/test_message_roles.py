"""Instructions keep their place in the conversation, so a resumed prompt's earlier tokens never change."""

import jinja2
import pytest

from tensorfold.server.messages import late_system_role, normalize_messages

# Qwen3.8's rule: one system message, and only first
_FIRST_ONLY = jinja2.Environment().from_string(
    "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('late system') }}{% endif %}"
    "<{{ m.role }}>{{ m.content }}{% endfor %}")
# Nemotron's rule: a system message renders wherever it is
_ANYWHERE = jinja2.Environment().from_string("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}")


def render(template, messages):
    def fail(message):
        raise jinja2.TemplateError(message)
    return template.render(messages=messages, raise_exception=fail)


def test_the_template_decides_how_a_later_instruction_renders():
    assert late_system_role(lambda m: render(_FIRST_ONLY, m)) == "user"
    assert late_system_role(lambda m: render(_ANYWHERE, m)) == "system"
    assert late_system_role(lambda m: "a template that drops it") == "user"


@pytest.mark.parametrize("template", [_FIRST_ONLY, _ANYWHERE])
def test_a_later_instruction_leaves_the_earlier_conversation_unchanged(template):
    role = late_system_role(lambda m: render(template, m))
    history = [{"role": "developer", "content": "be terse"}, {"role": "system", "content": "use tools"},
               {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    later = history + [{"role": "developer", "content": "cwd changed"}, {"role": "user", "content": "go on"}]
    before = render(template, normalize_messages(history, late_system=role))
    after = render(template, normalize_messages(later, late_system=role))
    assert after.startswith(before)
    assert before.startswith("<system>be terse\n\nuse tools<user>hi")
    assert f"<{role}>cwd changed<user>go on" in after


def test_normalizing_twice_changes_nothing():
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"},
                {"role": "system", "content": "c"}]
    once = normalize_messages(messages, late_system="user")
    assert normalize_messages(once, late_system="user") == once
    assert normalize_messages(normalize_messages(messages), late_system="user") == once
