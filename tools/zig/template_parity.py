"""Byte parity of zig/src/core/template with transformers' render_jinja_template on every cached chat template."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WEATHER = {"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
COMPLEX = {"type": "function", "function": {"name": "book_trip", "description": "  Book a trip; æøå 世界 \"quoted\"\n", "parameters": {
    "type": "object",
    "properties": {
        "destination": {"type": "string", "description": "City name", "enum": ["Copenhagen", "Tokyo", "Zürich"]},
        "nights": {"type": "integer", "description": "How many nights", "minimum": 1, "default": 2},
        "budget": {"type": ["number", "null"], "nullable": True},
        "Travellers": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}},
        "flags": {"type": "array", "items": {"type": "string"}},
        "options": {"type": "object", "properties": {"window": {"type": "boolean"}}, "required": ["window"]},
        "Ärger": {"type": "string"},
        "äpfel": {"type": "number", "description": "fruit"},
    },
    "required": ["destination", "nights"]}}}
BARE = {"name": "lookup", "description": "Bare function schema", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
TOOL_SETS = [[WEATHER], [WEATHER, COMPLEX], [BARE]]

CALL = {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Copenhagen"}}}
NESTED_ARGS = {"city": "æøå 世界\u0000\n\"", "values": [True, False, None, 1, -7, 1.0, -0.0, 1e-05, 0.0001, 1e15, 1e16, 12345678901234567890],
               "object": {"a": 2, "b": [1, {"c": "d"}]}, "empty": {}, "list": [], "text": "line1\nline2\ttab"}


def user(text, **extra):
    return {"role": "user", "content": text, **extra}


def assistant(text, **extra):
    return {"role": "assistant", "content": text, **extra}


CONVERSATIONS = [
    [user("Hello, æøå 世界 👋\n123456789")],
    [{"role": "system", "content": "Be brief."}, user("Hi")],
    [{"role": "system", "content": "Be brief.\n\nUse Danish."}, user("Hi")],
    [user("Hi"), assistant("Hello"), user("Next?")],
    [{"role": "system", "content": "s"}, user("u"), assistant("a"), {"role": "system", "content": "tensorfold-late-system-probe"}, user("v")],
    [{"role": "system", "content": "s"}, user("u"), assistant("a"), user("tensorfold-late-system-probe"), user("v")],
    [user("Alpha"), assistant("Beta"), user("Gamma"), assistant("Delta")],
    [user("one two"), assistant("three four"), user("five six")],
    [user("x"), assistant("", tool_calls=[{"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}])],
    [user("Weather in Copenhagen?"), assistant("", tool_calls=[CALL]), {"role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "Sunny"}, user("Summarize.")],
    [user("Weather?"), assistant("Let me check.", reasoning_content="Need the tool.", tool_calls=[
        {"id": "call_a", "type": "function", "function": {"name": "weather", "arguments": NESTED_ARGS}},
        {"id": "call_b", "type": "function", "function": {"name": "weather", "arguments": {"city": "Tokyo", "days": 3}}}]),
     {"role": "tool", "tool_call_id": "call_a", "content": "Rain"}, {"role": "tool", "tool_call_id": "call_b", "content": "{\"temp\": 21.5}"},
     assistant("Rain and 21.5 degrees.", reasoning_content="Both answered.")],
    [user("Call it"), assistant("", tool_calls=[{"id": "c", "type": "function", "function": {"name": "weather", "arguments": "not JSON"}}])],
    [user("Call it"), assistant(None, tool_calls=[{"type": "function", "function": {"name": "weather"}}])],
    [user("Think"), assistant("The answer is 4.", reasoning_content="2+2=4")],
    [user("Think"), assistant("<think>\nhidden steps\n</think>\n\nVisible answer")],
    [user("Q1"), assistant("A1", reasoning_content="R1"), user("Q2"), assistant("A2", reasoning_content="R2"), user("Q3")],
    [user(""), assistant(""), user("")],
    [user([{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}])],
    [user([{"type": "image"}, {"type": "text", "text": "What is this?"}])],
    [user([{"type": "text", "text": "Two:"}, {"type": "image"}, {"type": "image"}]), assistant("Cats."), user("Thanks")],
    [user("Use tools"), assistant("", tool_calls=[CALL]), {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "Part one"}, {"type": "text", "text": " and two"}]}],
    [user("nul \u0000 nbsp\u00a0zwj 👨\u200d👩\u200d👧 rtl \u05e9\u05dc\u05d5\u05dd combining e\u0301 cjk 中文 tab\t cr\r")],
    [{"role": "developer", "content": "Developer rules"}, user("Hi")],
    [user("Continue this"), assistant("Partial answer")],
    [user("  padded  \n"), assistant("\n  trailing spaces  \n")],
    [user("{{ not a tag }} {% raw %} {# no comment #}")],
    [user("Run it"), assistant("", tool_calls=[CALL]), user("<tool_response>\nmanual\n</tool_response>")],
    [{"role": "system", "content": "Only a system message"}],
    [],
    [user("Hi", name="alice"), assistant("Hello", name="bot")],
    [user("Reason"), assistant("Done", reasoning="Gemma-style reasoning field", tool_calls=[CALL]), {"role": "tool", "tool_call_id": "zzz", "name": "weather", "content": "Cloudy"}],
    [user("Unknown tool id"), assistant("", tool_calls=[CALL]), {"role": "tool", "tool_call_id": "nope", "content": "x"}],
    [user("Greek ΟΔΥΣΣΕΥΣ and ß and İstanbul"), assistant("", tool_calls=[CALL]), {"role": "tool", "tool_call_id": "call_1", "content": "ERROR: Traceback ΣΙΓΜΑ"}],
    [user("think_off"), {"role": "system", "content": "late <|think_off|> system"}, user("again <|think_on|>")],
    [user("Long tool"), assistant("", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": {"city": "x" * 40, "n": 12}}}]),
     {"role": "tool", "tool_call_id": "call_1", "content": "y" * 50 + " error: failed to — done"}],
]
CONTEXTS = [
    {},
    {"enable_thinking": False, "thinking_mode": "chat"},
    {"enable_thinking": True, "thinking_mode": "thinking"},
    {"enable_thinking": True, "thinking_mode": "thinking", "reasoning_effort": "low"},
    {"enable_thinking": True, "thinking_mode": "thinking", "reasoning_effort": "medium"},
    {"enable_thinking": True, "thinking_mode": "thinking", "reasoning_effort": "xhigh"},
    {"enable_thinking": True, "reasoning_effort": "bogus"},
    {"enable_thinking": False, "preserve_thinking": False, "clear_thinking": False, "truncate_history_thinking": False, "add_vision_id": True},
    {"enable_thinking": True, "preserve_thinking": True, "max_tool_arg_chars": 10, "max_tool_response_chars": 20, "auto_disable_thinking_with_tools": True},
    {"enable_thinking": None, "thinking_mode": None},
]
TOOL_CONTEXTS = (0, 1, 2, 7, 8)

# Feature probes rendered against SYNTHETIC_VARS: scoping, whitespace control, filters, tests, Undefined and errors.
SYNTHETIC = [
    '{% for m in messages %}{{ loop.index }}/{{ loop.length }} {{ loop.revindex0 }} {{ loop.first }} {{ loop.last }} {{ loop.previtem.role if loop.previtem }}|{{ loop.nextitem.role if loop.nextitem is defined }}\n{% endfor %}',
    '{% set x = 1 %}{% for i in [1, 2] %}{{ x }}{% set x = x + i %}{{ x }}{% endfor %}{{ x }}',
    "{% set ns = namespace(n=0, s='') %}{% for i in range(5) if i is odd %}{% set ns.n = ns.n + i %}{% set ns.s = ns.s ~ i %}{% endfor %}{{ ns.n }}:{{ ns.s }}:{{ ns }}",
    "{% macro m(a, b=2, c='x') %}[{{ a }}|{{ b }}|{{ c }}|{{ d }}]{% endmacro %}{{ m(1) }}{{ m(1, 3) }}{{ m(1, c='z') }}{{ m() }}{% set d = 'late' %}{{ m(4) }}",
    '{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}',
    '{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, b=2) }}',
    '{% for i in [3, 1, 2] %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% if i == 2 %}{% break %}{% endif %}{% else %}empty{% endfor %}',
    '{% for i in [] %}x{% else %}empty{% endfor %}|{% for i in [1] %}{% continue %}{% else %}else-after-continue{% endfor %}',
    "{%- if true -%}   trimmed   {%- endif -%}  |  {% if true %}\n  kept\n{% endif %}\n    {% if true %}lstrip{% endif %}  {%+ if true %}plus{% endif %}  {{- ' dash' }}  {#- comment -#}  end",
    '{{ \'a\\tb\\n\\x41é\\101\\\\q\' if false else \'esc:\\t\\x41é\\101\\q\' }}|{{ "it\'s" }}|{{ \'say "hi"\' }}',
    "{{ [1, 'a', none, true, 1.5, [2], {'k': 'v'}, ('t',), (1, 2)] }}|{{ {'a': 1, 'b': [1.0, 1e16, -0.0]} }}|{{ (1,) | string }}",
    "{{ messages | tojson }}|{{ tools | tojson(indent=2) }}|{{ {'b': 1, 'a': 'é'} | tojson(sort_keys=true, ensure_ascii=true) }}|{{ [1, [2]] | tojson(separators=(',', ':')) }}",
    "{{ 'Hello World'[::-1] }}|{{ 'héllo'[1:4] }}|{{ [1, 2, 3, 4][-2:] }}|{{ 'abc'[5:] }}|{{ (1, 2, 3)[::2] }}|{{ 'abc'[-1] }}|{{ 'abc'[10] is defined }}",
    "{{ '  x y  '.strip() }}|{{ 'xxhixx'.strip('x') }}|{{ 'a,b,,c'.split(',') }}|{{ ' a  b '.split() }}|{{ 'a b c'.split(None, 1) }}|{{ 'a b c'.rsplit(' ', 1) }}|{{ 'aaa'.replace('a', 'b', 2) }}|{{ 'abc'.replace('', '-') }}",
    "{{ 'hello'.find('l') }}|{{ 'hello'.rfind('l') }}|{{ 'héllo'.find('l') }}|{{ 'hello'.startswith(('x', 'he')) }}|{{ 'hello'.endswith('lo', 0, 5) }}|{{ 'a-b'.count('-') }}|{{ 'x'.join(['a', 'b']) }}|{{ 'pre_x'.removeprefix('pre_') }}",
    "{{ 'ÆØÅ straße'.lower() }}|{{ 'ÆØÅ straße'.upper() }}|{{ 'abc' | upper }}|{{ none | upper }}|{{ 'ABC' | lower }}|{{ 42 | string }}|{{ 1.0 | string }}|{{ [1] | string }}",
    "{{ 'ΣΑΣ ΟΔΥΣΣΕΥΣ AΣ\\'b' | lower }}|{{ ['Σa', 'σb', 'ΣA'] | sort }}|{{ {'Ärger': 1, 'äpfel': 2, 'Zed': 3} | dictsort }}",
    "{{ undefined_name }}|{{ undefined_name is defined }}|{{ undefined_name | default('dflt') }}|{{ '' | default('empty', true) }}|{{ none | default('n') }}|{{ undefined_name | length }}|{{ undefined_name ~ 'x' }}",
    '{{ x.y }}',
    "{{ 'a' + undefined_name }}",
    '{{ messages[0].nope.deeper }}',
    "{{ messages[0]['role'] }}|{{ messages[0].role }}|{{ messages.0.role }}|{{ messages[0].items is defined }}|{{ messages[0].get('role') }}|{{ messages[0].get('nope', 'd') }}|{{ messages[0].pop is defined }}",
    "{{ [3, 1, 2] | sort }}|{{ ['b', 'A', 'a'] | sort }}|{{ ['b', 'A', 'a'] | sort(case_sensitive=true) }}|{{ ['b', 'a'] | sort(reverse=true) }}|{{ ['a', 'A', 'b'] | unique | list }}|{{ [1, 2] | reverse | list }}|{{ 'abc' | reverse }}",
    "{{ {'b': 1, 'A': 2, 'a': 3} | dictsort }}|{{ {'b': 1, 'a': 2} | dictsort(by='value', reverse=true) }}|{{ {'x': 1} | items | list }}|{{ [1, 2, 3] | first }}|{{ [1, 2, 3] | last }}|{{ [] | first is defined }}",
    "{{ ['a', 'b'] | map('upper') | list }}|{{ messages | map(attribute='role') | join(',') }}|{{ messages | selectattr('role', 'equalto', 'user') | list | length }}|{{ [0, 1, 2] | select | list }}|{{ [0, 1, 2] | reject('odd') | list }}|{{ messages | rejectattr('content') | list | length }}",
    '{{ \'a\\nb\\n\\nc\' | indent(2) }}|{{ \'a\\nb\' | indent(2, true) }}|{{ \'a\\n\\nb\' | indent(\'> \', blank=true) }}|{{ \'x&<>"\\\'\' | e }}|{{ (\'<b>\' | safe) + \'<i>\' }}|{{ \'<i>\' + (\'<b>\' | safe) }}|{{ \'x\' | replace(\'x\', \'y\') }}',
    "{{ 1 + 2 }}|{{ 7 // 2 }}|{{ -7 // 2 }}|{{ 7 % 3 }}|{{ -7 % 3 }}|{{ 7 / 2 }}|{{ 2 ** 10 }}|{{ 2 ** -1 }}|{{ 1.5 * 2 }}|{{ -7.5 // 2 }}|{{ -7.5 % 2 }}|{{ true + 1 }}|{{ 'ab' * 2 }}|{{ [1] * 2 }}|{{ 10 / 4 }}|{{ 0.1 + 0.2 }}|{{ 1e300 * 1e10 }}",
    "{{ 1 == 1.0 }}|{{ 'a' < 'b' }}|{{ [1, 2] < [1, 3] }}|{{ 1 < 2 < 3 }}|{{ 3 > 2 > 2 }}|{{ 'a' in 'cat' }}|{{ 'x' not in ['x'] }}|{{ 'k' in {'k': 1} }}|{{ none == none }}|{{ undefined_name == undefined_other }}|{{ (1, 2) == [1, 2] }}",
    "{{ 1 < 'a' }}",
    "{{ 'a' in 1 }}",
    "{{ none is none }}|{{ 1 is number }}|{{ true is number }}|{{ true is integer }}|{{ 1.0 is float }}|{{ 'x' is string }}|{{ {} is mapping }}|{{ {} is sequence }}|{{ 'x' is sequence }}|{{ undefined_name is sequence }}|{{ undefined_name is iterable }}|{{ none is iterable }}|{{ 3 is odd }}|{{ 4 is even }}|{{ 9 is divisibleby 3 }}|{{ 'a' is in 'abc' }}|{{ 1 is eq 1 }}|{{ 2 is gt 1 }}|{{ false is sameas false }}|{{ 'tojson' is filter }}|{{ 'odd' is test }}|{{ x is callable }}|{{ range is callable }}",
    '{{ range(3) | list }}|{{ range(1, 10, 3) | list }}|{{ range(5, 0, -2) | list }}|{{ range(3) }}|{{ range(0, 10, 2) }}|{{ range(3) | length }}',
    '{% set captured %}  inner {{ 1 + 1 }}  {% endset %}[{{ captured }}]|{% set up | upper %}shout{% endset %}{{ up }}',
    "{% set a, b = [1, 2] %}{{ a }}{{ b }}|{% for k, v in {'x': 1, 'y': 2} | items %}{{ k }}={{ v }};{% endfor %}|{% for a, (b, c) in [(1, (2, 3))] %}{{ a }}{{ b }}{{ c }}{% endfor %}",
    '{% set a, b = [1] %}',
    "{{ raise_exception('custom failure: ' ~ 42) }}",
    "{{ messages[0].content if messages else 'none' }}|{{ 'yes' if '' else 'no' }}|{{ ('a' if false) is defined }}|{{ none or 'fallback' }}|{{ 0 and 'never' }}|{{ not none }}",
    "{% for i in range(3) %}{{ loop.cycle('odd', 'even') }}{{ loop.changed(i // 2) }}{% endfor %}",
    '{% generation %}gen {{ 1 }}{% endgeneration %}|{% raw %}{{ raw }} {% not a tag %}{% endraw %}',
    '{% macro outer() %}{{ inner() }}{% endmacro %}{% macro inner() %}inner-ok{% endmacro %}{{ outer() }}',
    "{% for m in messages %}{{ late_var }}{% endfor %}{% set late_var = 'root' %}{{ late_var }}",
    '{% if false %}{% set y = 1 %}{% endif %}{{ y }}|{% if true %}{% set y = 2 %}{% endif %}{{ y }}',
    "{{ messages | length }}|{{ messages[-1].role }}|{{ messages[99] is defined }}|{{ messages[0]['nope'] is defined }}|{{ (messages | first).role }}|{{ tools | length if tools else 0 }}|{{ documents }}|{{ bos_token }}{{ eos_token }}",
    "{{ messages[0].content.type }}|{{ 'abc'.nope is defined }}|{{ none.x is defined }}|{{ (1).real is defined }}|{{ {}.items() | list }}|{{ {'a': 1}.items() }}|{{ {'a': 1}.keys() | list }}|{{ {'a': 1}.values() | list }}",
    "{{ [1, 2, 3] | join('-') }}|{{ [1, none, 'x'] | join }}|{{ 'abc' | list }}|{{ {'a': 1, 'b': 2} | list }}|{{ 'héllo' | length }}|{{ {'a': 1} | length }}|{{ 'a b' | trim }}|{{ ' x ' | trim('x ') }}",
    "{{ strftime_now('%Y-%m-%d') | length }}",
    '{{ x | nofilter }}',
    "{{ 'a' ~ 1 ~ none ~ 1.5 ~ [1] ~ true }}|{{ -1 }}|{{ -(1) }}|{{ +1 }}|{{ - 1.5 }}|{{ not true }}|{{ 'x' * 0 }}|{{ 1.0 == 1 }}|{{ 1e2 }}|{{ 1_000 }}|{{ 0x1F }}|{{ 1.5e-7 }}|{{ 12.5E3 }}",
    "{{ dict(a=1, b='x') }}|{{ namespace(a=1).a }}|{{ namespace({'q': 2}).q }}|{% set n = namespace() %}{% set n.v = [1] %}{{ n.v }}",
    "{% set t = 'x' %}{% set t.attr = 1 %}",
    "{% macro m() %}{{ v }}|{{ d }}{% endmacro %}{{ m() }}{% set v = 'root' %}{{ m() }}{{ v }}",
    "{% for i in [1] %}{{ late_var }}{% endfor %}{% set late_var = 'root' %}{{ late_var }}",
    "{% macro m() %}{{ y }}{% endmacro %}{% if false %}{% set y = 'branch' %}{% endif %}{{ m() }}{% set y = 'set' %}{{ m() }}",
    "{% set x = x ~ '!' if x is defined else 'none' %}{{ x }}|{% for x in [1, 2] %}{{ x }}{% for x in 'ab' %}{{ x }}{{ loop.index }}{% endfor %}{{ loop.index }}{% endfor %}{{ x }}",
    "{% for i in range(3) %}{% macro sq(n) %}{{ n * n }}{% endmacro %}{{ sq(i) }}{% endfor %}|{% macro f(n) %}{{ n }}{% if n > 0 %}{{ f(n - 1) }}{% endif %}{% endmacro %}{{ f(3) }}",
    "{% set range = 'shadow' %}{{ range }}|{% set ns = namespace(a=1) %}{% macro bump() %}{% set ns.a = ns.a + 1 %}{% endmacro %}{{ bump() }}{{ bump() }}{{ ns.a }}",
    "a  \t{% if true %}b{% endif %}\n\t  {%- if true %}c{% endif %}\n  {#- c -#}  d\n   {# keep #} e\n{%+ if true %}f{% endif %}\n  {{ 'g' }}\n  {%- if true -%}\n  h\n{%- endif %}",
    "line1\r\nline2\r{% if true %}\r\nx{% endif %}\n\n",
    "\u00a0\u00a0{% if true %}nbsp-indent{% endif %}|\u3000{% if true %}ideo{% endif %}",
    "{%- raw -%}  {{ raw }}  {%- endraw -%}  |{% raw %}\nkeep{% endraw %}|  {%+ raw %}x{% endraw %}",
    "{{ '\\x41\\101\\u00e9\\U0001F44B\\n\\t\\q\\'' }}|{{ \"\\\"\" }}|{{ 0x1f }}|{{ 0o17 }}|{{ 0b101 }}|{{ 1_000_000 }}|{{ 1e3 }}|{{ 2.5E-3 }}|{{ 007 if false else 0 }}|{{ messages.0.content }}",
    "{{ ['a\u00a0b', '\u00e9', '\\x00\\x1f\\x7f', \"it's\", 'say \"x\"', 'both \\'\"', '\u200d', '\U0001F44B', '\\u2028', '\\\\'] | string }}",
    "{{ [0.1 + 0.2, 1 / 3, 2 / 3, 1e-07, 123456789.123, -1e-05, 1234567890123456.0, 12345678901234567.0, 5e-324, 1.7976931348623157e308, 1e21, 1e22, 0.5, -0.0] }}",
    "{{ 1 == true }}|{{ 0 == false }}|{{ '1' == 1 }}|{{ none == false }}|{{ [1, 2] == [1, 2] }}|{{ {'a': 1} == {'a': 1} }}|{{ (1,) == [1] }}|{{ 1 < 2 > 1 }}|{{ 2 >= 2.0 }}|{{ 'b' > 'a' }}|{{ 'B' < 'a' }}",
    "{{ '' in 'abc' }}|{{ 1 in [1.0] }}|{{ none in [none] }}|{{ 'a' in {'a': 1} }}|{{ [1] in [[1]] }}|{{ 'x' in undefined_name }}|{{ 2 in range(3) }}",
    "{{ [1] in {'a': 1} }}",
    "{{ ('<b>' | safe) is string }}|{{ namespace() is mapping }}|{{ namespace() is iterable }}|{% for i in [1] %}{{ loop is sequence }}{{ loop is iterable }}{% endfor %}|{{ {}.items() is sequence }}|{{ 'a' is lower is defined if false else 1 }}",
    "{{ 'abcdef'[::-2] }}|{{ 'abcdef'[1:-1] }}|{{ 'abc'[-10:10] }}|{{ [1, 2, 3][5:1:-1] }}|{{ [1, 2, 3][::-1] }}|{{ 'h\u00e9\U0001F44Bx'[1:3] }}|{{ 'abc'[1:1] }}|{{ (1, 2)[0:1] }}",
    "{{ 'abc'[::0] }}",
    "{{ 'a b c'.split(' ', 0) }}|{{ ' a b '.rsplit() }}|{{ ' a b '.rsplit(None, 1) }}|{{ '\u00e9xx\u00e9'.strip('\u00e9') }}|{{ 'aaa'.replace('a', 'b', 0) }}|{{ 'hello'.find('l', -2) }}|{{ 'hello'.startswith('l', 2, 3) }}|{{ 'abc'.removesuffix('') }}|{{ 'a\u2028b\u3000c'.split() }}",
    "{{ 'abc'.split('') }}",
    "{{ ','.join([1, 2]) }}",
    "{% for i in [1, 2, 3, 4] if i is even %}{{ loop.index }}/{{ loop.length }}{{ loop.last }} {% endfor %}|{% for i in [1, 2] %}{{ loop.revindex }}{% endfor %}",
    "{% for i in [1] %}{{ loop.cycle() }}{% endfor %}",
    "{{ undefined_name | string }}|{{ undefined_name | list }}|{{ undefined_name | join }}|{{ undefined_name | items | list }}|{{ undefined_name is callable }}|{{ undefined_name == undefined_other }}|{{ undefined_name | first is defined }}|{{ undefined_name | reverse | list }}",
    "{{ undefined_name[0] }}",
    "{{ undefined_name() }}",
    "{{ undefined_name | tojson }}",
    "{{ -undefined_name }}",
    "{{ undefined_name < 1 }}",
    "{{ {'a': 1}.get(['x']) }}",
    "{{ {'a': 1}.keys() }}|{{ {'a': 1}.values() }}|{{ [('<b>' | safe)] | string }}|{{ {'a': 1}.items() | list }}|{{ {'a': 1}.copy() }}",
    "{{ [1, 2, 3, 2].count(2) }}|{{ [1, 2].index(2) }}|{{ (1, 2).count(1) }}|{{ range(5).index(3) }}",
    "{{ range() }}",
    "{{ range(1, 2, 0) }}",
    "{{ namespace(1) }}",
    "{{ raise_exception(42) }}",
    "{{ [3, 1, 2] | sort(reverse=true) }}|{{ [{'k': 'b', 'n': 1}, {'k': 'a', 'n': 2}, {'k': 'B', 'n': 3}] | sort(attribute='k') | map(attribute='n') | list }}|{{ [1, true, 1.0, 2] | unique | list }}|{{ ['b', 'B', 'a'] | unique(case_sensitive=true) | list }}|{{ {'b': 1, 'B': 2, 'a': 3} | dictsort(true) }}",
    "{{ [{'a': {'b': [10, 20]}}] | map(attribute='a.b.1') | list }}|{{ [{'x': 1}, {}] | map(attribute='x', default='-') | list }}|{{ [{'n': 'a'}, {'n': 'b'}] | join('+', attribute='n') }}|{{ 'abc' | first }}|{{ 'abc' | last }}|{{ {'p': 1, 'q': 2} | first }}|{{ {'p': 1, 'q': 2} | last }}",
    "{{ 'a\r\nb\x0bc\x0cd\x1ce\u2028f' | indent(1) }}|{{ '' | indent }}|{{ 'x\n' | indent(first=true) }}|{{ 'a b' | replace(' ', '_') }}|{{ 5 | replace('5', 'five') }}|{{ '  x  ' | trim(' x') }}|{{ 'x' | trim('') }}",
    "{{ {'s': 'é😀\n\t\x00\x1f\"\\\\/', 'n': [1.5, -0.0, 12345678901234567890, true, none]} | tojson }}|{{ {'s': 'é😀\x7f'} | tojson(ensure_ascii=true) }}|{{ {'b': {'d': 1, 'c': [2, {}]}, 'a': []} | tojson(indent=1, sort_keys=true) }}|{{ [1] | tojson(indent='--') }}|{{ [] | tojson(indent=2) }}",
    "{{ none | tojson }}|{{ 'x' | tojson }}|{{ 1.0 | tojson }}|{{ (1, 2) | tojson }}|{{ messages[0] | tojson(separators=(',', ':')) }}",
    "{{ range(3) | tojson }}",
    "{{ namespace(a=1) | tojson }}",
    "{% if true %}{% break %}{% endif %}after",
    "{% for i in [1, 2] %}{% macro m() %}{% break %}{% endmacro %}{{ i }}{% endfor %}",
    "{% macro m(a, a) %}{% endmacro %}x",
    "{{ cycler is defined }}|{{ joiner is defined }}|{{ lipsum is defined }}",
    "{{ messages[0].content[0] }}|{{ 'abc'[0:2] }}|{{ 'abc'[:-1] }}|{{ 'abc'['upper']() }}",
    "{{ ['b', 'a'] | sort | join }}{{ [{'n': 'b'}, {'n': 'a'}] | sort(attribute='n') | map(attribute='n') | join }}{{ [{'n': 1}] | map(attribute='m', default='d') | join }}",
    "{% for x in 'ab' %}{{ x }}{{ loop.index0 }}{% endfor %}|{% for k in {'p': 1, 'q': 2} %}{{ k }}{% endfor %}|{% for x in none %}{% endfor %}",
    "{%- for m in messages -%}\n    {%- set content = m.content | trim -%}\n    {{- '<' + m.role + '>' + content -}}\n{%- endfor %}\nx",
]
# Jinja2 features the native engine refuses on purpose; parity here means the native render fails loudly.
UNSUPPORTED = [
    "{{ '%s' % 1 }}",
    "{{ '\\N{BULLET}' }}",
]
SYNTHETIC_VARS = [
    {"messages": [user("Hi"), assistant(" Hello "), user("Bye")], "tools": [WEATHER], "bos_token": "<s>", "eos_token": "</s>"},
    {"messages": [user("Solo")], "tools": None, "bos_token": "<s>"},
    {"messages": [user("Ctx")], "tools": None, "v": "ctx-v", "d": "ctx-d", "late_var": "ctx-late", "y": "ctx-y", "x": "ctx-x"},
]


ALPHABET = list("abcXYZ019 _-.,:;!?'\"\\/<>{}[]()|\n\t") + ["\u0000", "\u001f", "\u007f", "\u00a0", "\u00e9", "\u00df", "\u0130", "\u03a3", "\u03c3", "\u200d", "\u2028", "\u3000", "\u4e16", "\U0001f44b", "<think>", "</think>", "<tool_call>", "<|im_end|>", "{{", "{%", "\r\n"]


def fuzz_text(rng, limit=24):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randrange(limit)))


def fuzz_json(rng, depth=0):
    kind = rng.randrange(9 if depth < 4 else 6)
    if kind == 0:
        return fuzz_text(rng)
    if kind == 1:
        return rng.choice([0, 1, -7, 2**31, 2**63 - 1, -(2**63), 2**64 + 5])
    if kind == 2:
        return rng.choice([0.0, -0.0, 1.0, 0.1, 1e-5, 1e-4, 1e15, 1e16, 1e22, 123.456, -2.5e-300, 1.7976931348623157e308, 5e-324])
    if kind == 3:
        return rng.choice([True, False, None])
    if kind in (4, 5):
        return fuzz_text(rng, 8)
    if kind in (6, 7):
        return {fuzz_text(rng, 6) or "k": fuzz_json(rng, depth + 1) for _ in range(rng.randrange(4))}
    return [fuzz_json(rng, depth + 1) for _ in range(rng.randrange(4))]


def fuzz_turns(rng):
    """A well-formed order: optional system, then user turns, assistant replies and tool results after calls."""
    roles = ["system"] if rng.random() < 0.5 else []
    for _ in range(rng.randrange(1, 4)):
        roles.append("user")
        if rng.random() < 0.8:
            roles.append("assistant")
            roles.extend(["tool"] * rng.randrange(3))
    return roles


def fuzz_conversation(rng):
    messages = []
    wellformed = rng.random() < 0.5
    roles = fuzz_turns(rng) if wellformed else [rng.choice(["user", "user", "assistant", "assistant", "tool", "system", "developer"]) for _ in range(rng.randrange(1, 7))]
    for role in roles:
        content = rng.choice([fuzz_text(rng, 40), fuzz_text(rng, 40), "", None, [{"type": "text", "text": fuzz_text(rng)}], [{"type": "image"}, {"type": "text", "text": fuzz_text(rng)}]])
        if wellformed and role in ("system", "user"):
            content = fuzz_text(rng, 40)
        message = {"role": role, "content": content}
        if role == "assistant" and rng.random() < 0.5:
            message["reasoning_content"] = fuzz_text(rng)
        if role == "assistant" and rng.random() < 0.5:
            message["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": rng.choice(["weather", "book_trip", fuzz_text(rng, 6) or "f"]),
                                     "arguments": rng.choice([{fuzz_text(rng, 6) or "a": fuzz_json(rng) for _ in range(rng.randrange(3))}, fuzz_text(rng)])}}
                                     for i in range(rng.randrange(1, 3))]
        if role == "tool":
            message["tool_call_id"] = rng.choice(["call_0", "call_1", "nope"])
            if rng.random() < 0.5:
                message["name"] = "weather"
        messages.append(message)
    return messages


def fuzz_tools(rng):
    schema = {"type": "object", "properties": {fuzz_text(rng, 6) or "p": {"type": rng.choice(["string", "number", "array", "object", ["string", "null"]]),
                                                                          "description": fuzz_text(rng), "enum": [fuzz_json(rng, 3)]} for _ in range(rng.randrange(3))}}
    random_tool = {"type": "function", "function": {"name": fuzz_text(rng, 8) or "t", "description": fuzz_text(rng), "parameters": schema}}
    return rng.choice([None, [WEATHER], [random_tool], [WEATHER, COMPLEX, random_tool]])


def hub_dirs():
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface")), "hub")
    return sorted(Path(hub).glob("*/snapshots/*"))


def label_of(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        repo = parts[parts.index("snapshots") - 1]
        return repo.removeprefix("models--").replace("--", "/")
    return path.name


def specials_of(config: dict) -> dict:
    """The engine's context: every *_token string (or {"content": string}) in tokenizer_config.json."""
    out = {}
    for key, value in config.items():
        if not key.endswith("_token"):
            continue
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            out[key] = value["content"]
    return out


def templates_in(path: Path):
    if path.is_file():
        yield label_of(path.parent) + "/" + path.name, path.read_text(encoding="utf-8"), {}
        return
    config_path = path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    specials = specials_of(config)
    name = label_of(path)
    if (path / "chat_template.jinja").exists():
        yield name, (path / "chat_template.jinja").read_text(encoding="utf-8"), specials
    if (path / "chat_template.json").exists():
        source = json.loads((path / "chat_template.json").read_text(encoding="utf-8")).get("chat_template")
        if isinstance(source, str):
            yield name + " (chat_template.json)", source, specials
    source = config.get("chat_template")
    if isinstance(source, str):
        yield name + " (tokenizer_config)", source, specials
    elif isinstance(source, list):
        for entry in source:
            yield f"{name} (tokenizer_config:{entry['name']})", entry["template"], specials
    elif isinstance(source, dict):
        for key, entry in source.items():
            yield f"{name} (tokenizer_config:{key})", entry, specials


def collect(extra):
    found = {}
    for path in [*hub_dirs(), *map(Path, extra)]:
        for label, source, specials in templates_in(path):
            key = hashlib.sha256(source.encode()).hexdigest()[:12]
            entry = found.setdefault(key, {"source": source, "labels": [], "specials": specials})
            entry["labels"].append(label)
            if not entry["specials"]:
                entry["specials"] = specials
    return found


def oracle(source, messages, tools, context, generation):
    import jinja2
    from transformers.utils.chat_template_utils import render_jinja_template
    try:
        out, _ = render_jinja_template(conversations=[copy.deepcopy(messages)], tools=copy.deepcopy(tools), documents=None,
                                       chat_template=source, add_generation_prompt=generation, **copy.deepcopy(context))
        return {"ok": True, "text": out[0]}
    except Exception as err:  # every failure is compared: the native engine must fail too
        return {"ok": False, "error": str(err), "raised": type(err) is jinja2.exceptions.TemplateError}


def build(found, fuzz, seed):
    """Corpus tables plus one case per template, conversation, tool set, context and generation flag."""
    templates, contexts, cases = {}, [], []
    conversations = json.loads(json.dumps(CONVERSATIONS + [v["messages"] for v in SYNTHETIC_VARS]))
    tools = json.loads(json.dumps(TOOL_SETS))
    rng = random.Random(seed)
    fuzzed = []
    for _ in range(fuzz):
        conversations.append(json.loads(json.dumps(fuzz_conversation(rng))))
        tool_set, tl = fuzz_tools(rng), None
        if tool_set is not None:
            tools.append(json.loads(json.dumps(tool_set)))
            tl = len(tools) - 1
        fuzzed.append((len(conversations) - 1, tl, rng.randrange(len(CONTEXTS)), rng.random() < 0.5))
    for key, entry in found.items():
        templates[key] = entry["source"]
        base = len(contexts)
        contexts.extend({**entry["specials"], **ctx} for ctx in CONTEXTS)
        for m, convo in enumerate(CONVERSATIONS):
            has_calls = any(msg.get("tool_calls") or msg.get("role") == "tool" for msg in convo)
            for c, g in itertools.product(range(len(CONTEXTS)), (False, True)):
                cases.append({"t": key, "m": m, "tl": None, "c": base + c, "g": g})
            for tl, c, g in itertools.product(range(len(TOOL_SETS)), TOOL_CONTEXTS if has_calls or m < 2 else (1,), (False, True)):
                cases.append({"t": key, "m": m, "tl": tl, "c": base + c, "g": g})
        for m, tl, c, g in fuzzed:
            cases.append({"t": key, "m": m, "tl": tl, "c": base + c, "g": g})
    for j, variables in enumerate(SYNTHETIC_VARS):
        tool_index = None
        if variables["tools"] is not None:
            tools.append(json.loads(json.dumps(variables["tools"])))
            tool_index = len(tools) - 1
        contexts.append({k: val for k, val in variables.items() if k not in ("messages", "tools")})
        for i, probe in enumerate(SYNTHETIC + UNSUPPORTED):
            key = f"synthetic-{i:02d}" if i < len(SYNTHETIC) else f"unsupported-{i - len(SYNTHETIC):02d}"
            templates[key] = probe
            cases.append({"t": key, "m": len(CONVERSATIONS) + j, "tl": tool_index, "c": len(contexts) - 1, "g": True})
    assert all(conversations[len(CONVERSATIONS) + j] == json.loads(json.dumps(v["messages"])) for j, v in enumerate(SYNTHETIC_VARS))
    for case in cases:
        tl = case["tl"]
        case.update(oracle(templates[case["t"]], conversations[case["m"]], tools[tl] if tl is not None else None, contexts[case["c"]], case["g"]))
        case.setdefault("text", "")
        case.setdefault("error", "")
        case.setdefault("raised", False)
    return {"templates": templates, "conversations": conversations, "tools": tools, "contexts": contexts, "cases": cases}


def first_difference(want: str, got: str) -> str:
    at = next((i for i, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
    return f"at char {at}: python {want[max(0, at - 40):at + 40]!r} / native {got[max(0, at - 40):at + 40]!r}"


def compare(corpus, results, labels):
    report, mismatches = {}, []
    for case, got in zip(corpus["cases"], results):
        row = report.setdefault(case["t"], {"labels": labels.get(case["t"], [case["t"]]), "cases": 0, "same_text": 0, "both_failed": 0, "mismatches": 0})
        row["cases"] += 1
        if case["t"].startswith("unsupported-") and not got["ok"]:
            row["both_failed"] += 1
            continue
        if case["ok"] and got["ok"] and got["text"] == case["text"]:
            row["same_text"] += 1
            continue
        if not case["ok"] and not got["ok"] and (not case["raised"] or (got["raised"] and got["error"] == case["error"])):
            row["both_failed"] += 1
            continue
        row["mismatches"] += 1
        detail = {"template": row["labels"][0], "conversation": case["m"], "tools": case["tl"], "context": corpus["contexts"][case["c"]], "generation": case["g"]}
        if case["ok"] and got["ok"]:
            detail["diff"] = first_difference(case["text"], got["text"])
        else:
            detail["python"] = case["text"][:200] if case["ok"] else "error: " + case["error"]
            detail["native"] = got["text"][:200] if got["ok"] else "error: " + got["error"]
        mismatches.append(detail)
    return report, mismatches


def unicode_tables():
    """Python's own str.isprintable, str.lower and str.upper as compact Zig tables."""
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]

    def ranges(pred):
        bounds, inside = [], False
        for cp in range(0x110001):
            hit = cp <= 0x10FFFF and pred(cp)
            if hit != inside:
                bounds.append(cp if hit else cp - 1)
                inside = hit
        return bounds

    sigma = "\u03a3"
    # Final_Sigma's two properties, read from how str.lower() itself treats a neighbour of capital sigma.
    surrogate = range(0xD800, 0xE000)
    cased = ranges(lambda cp: cp not in surrogate and (chr(cp) + sigma).lower()[-1] == "\u03c2")
    ignorable = ranges(lambda cp: cp not in surrogate and ("A" + sigma + chr(cp)).lower()[1] == "\u03c2" and ("A" + sigma + chr(cp) + "a").lower()[1] != "\u03c2")
    bounds = ranges(lambda cp: cp in surrogate or not chr(cp).isprintable())

    def runs(method):
        single, special = {}, []
        for cp in cps:
            mapped = getattr(chr(cp), method)()
            if mapped == chr(cp):
                continue
            if len(mapped) == 1:
                single[cp] = ord(mapped) - cp
            else:
                special.append([cp] + [ord(c) for c in mapped] + [0] * (3 - len(mapped)))
        out, keys, i = [], sorted(single), 0
        while i < len(keys):
            first, delta, best = keys[i], single[keys[i]], (keys[i], 1)
            for stride in (1, 2):
                last = first
                while single.get(last + stride) == delta and (stride == 1 or last + 1 not in single):
                    last += stride
                if last > best[0]:
                    best = (last, stride)
            out.append((first, best[0], delta, best[1]))
            covered = set(range(first, best[0] + 1, best[1]))
            while i < len(keys) and keys[i] in covered:
                i += 1
        return out, special

    lower, lower_special = runs("lower")
    upper, upper_special = runs("upper")

    def rows(items, per):
        return "\n".join("    " + " ".join(items[j:j + per]) for j in range(0, len(items), per))

    def run_rows(table):
        return rows([f".{{ 0x{a:x}, 0x{b:x}, {d}, {s} }}," for a, b, d, s in table], 4)

    def special_rows(table):
        return rows(["." + "{ " + ", ".join(f"0x{c:x}" for c in row) + " }," for row in table], 4)

    return "\n".join([
        f"//! Generated by tools/zig/template_parity.py --unicode-data from Python {sys.version.split()[0]} (Unicode {unicodedata.unidata_version}); do not edit.",
        f'pub const version = "{unicodedata.unidata_version}";',
        "/// Inclusive [first, last] pairs of code points str.isprintable() rejects.",
        "pub const nonprintable = [_]u21{", rows([f"0x{b:x}," for b in bounds], 12), "};",
        "/// Code points Python's Final_Sigma rule treats as cased (outside case-ignorable ones), as [first, last] pairs.",
        "pub const cased = [_]u21{", rows([f"0x{b:x}," for b in cased], 12), "};",
        "/// Code points Python's Final_Sigma rule skips as case-ignorable, as [first, last] pairs.",
        "pub const case_ignorable = [_]u21{", rows([f"0x{b:x}," for b in ignorable], 12), "};",
        "/// Lowercase runs [first, last, delta, stride] for single code point mappings.",
        "pub const lower = [_][4]i32{", run_rows(lower), "};",
        "/// Code points whose lowercase is several code points, zero padded.",
        "pub const lower_special = [_][4]u21{", special_rows(lower_special), "};",
        "/// Uppercase runs [first, last, delta, stride] for single code point mappings.",
        "pub const upper = [_][4]i32{", run_rows(upper), "};",
        "/// Code points whose uppercase is several code points, zero padded.",
        "pub const upper_special = [_][4]u21{", special_rows(upper_special), "};",
        ""])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extra", action="append", default=[], help="checkpoint directory or .jinja file to add")
    p.add_argument("--out", type=Path, required=False, help="directory for corpus.json, results.json and report.json")
    p.add_argument("--zig", default=str(ROOT / ".zig-toolchain" / "zig"), help="pinned Zig compiler")
    p.add_argument("--unicode-data", type=Path, help="write zig/src/core/template/unicode_data.zig and exit")
    p.add_argument("--fuzz", type=int, default=0, help="random conversations per template")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.unicode_data:
        args.unicode_data.write_text(unicode_tables())
        subprocess.run([args.zig, "fmt", str(args.unicode_data)], check=True)
        return 0
    if args.out is None:
        p.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    found = collect(args.extra)
    corpus = build(found, args.fuzz, args.seed)
    corpus_path, results_path = args.out / "corpus.json", args.out / "results.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False))
    subprocess.run([args.zig, "run", "-lc", "-OReleaseSafe", str(ROOT / "zig/src/core/template/parity.zig"), "--", str(corpus_path), str(results_path)], check=True)
    labels = {key: entry["labels"] for key, entry in found.items()}
    report, mismatches = compare(corpus, json.loads(results_path.read_text(encoding="utf-8")), labels)
    (args.out / "report.json").write_text(json.dumps({"python": sys.version.split()[0], "templates": report, "mismatches": mismatches}, ensure_ascii=False, indent=1))
    models = {k: r for k, r in report.items() if not k.startswith(("synthetic-", "unsupported-"))}
    synthetic = [r for k, r in report.items() if k.startswith(("synthetic-", "unsupported-"))]
    for key, row in sorted(models.items(), key=lambda kv: kv[1]["labels"][0]):
        print(f"{row['labels'][0]:<60} {key}  cases {row['cases']:5d}  same {row['same_text']:5d}  both-failed {row['both_failed']:4d}  mismatches {row['mismatches']}")
    print(f"synthetic feature templates: {len(synthetic)}, cases {sum(r['cases'] for r in synthetic)}, mismatches {sum(r['mismatches'] for r in synthetic)}")
    for detail in mismatches[:40]:
        print(json.dumps(detail, ensure_ascii=False))
    total = sum(r["mismatches"] for r in report.values())
    print(f"{sum(r['cases'] for r in report.values())} cases, {total} mismatches")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
