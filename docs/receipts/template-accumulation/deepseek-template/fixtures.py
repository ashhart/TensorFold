"""Generate public DeepSeek template cases and a Hugging Face-compatible Jinja2 byte oracle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_SHA256 = '7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e'
TEMPLATE_BYTES = 11410
COUNTS = (9, 129, 257, 513, 1001, 2001)
WORDS = tuple('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november '
              'oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu'.split())
TOJSON_CONVENTION = ('Hugging Face json.dumps: ensure_ascii=False, sort_keys=False, '
                     'standard separators, no HTML escaping')


def hf_tojson(
    value: Any, ensure_ascii: bool = False, indent: int | str | None = None,
    separators: tuple[str, str] | None = None, sort_keys: bool = False,
) -> str:
    """Match Hugging Face's filter rather than Jinja2's HTML-safe, sorted-key default."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def reject(message: str) -> None:
    """Implement the template's explicit error function."""
    raise jinja2.TemplateError(message)


def environment() -> ImmutableSandboxedEnvironment:
    """Create the pinned sandbox with Hugging Face's JSON serialization convention."""
    if jinja2.__version__ != '3.1.6':
        raise RuntimeError('Jinja2 3.1.6 is required')
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'],
    )
    env.filters['tojson'] = hf_tojson
    env.globals['raise_exception'] = reject
    return env


def issue_messages() -> list[dict[str, str]]:
    """Reproduce the issue's seeded public message generator without its unavailable tool schemas."""
    rng = random.Random(424242)
    messages = [{'role': 'system', 'content': 'You are a test agent. Reply with the single word OK.'}]
    for index in range(2000):
        words = [rng.choice(WORDS) for _ in range(234)]
        messages.append({
            'role': 'user' if index % 2 == 0 else 'assistant',
            'content': 'turn %04d: ' % index + ' '.join(words) + '.',
        })
    return messages


def synthetic_tools(count: int = 157) -> list[dict[str, Any]]:
    """Create clearly synthetic schemas, independent of the issue's unavailable original tools."""
    return [
        {'type': 'function', 'function': {
            'name': f'synthetic_tool_{index:03d}',
            'description': f'Synthetic schema {index:03d} for deterministic host rendering only.',
            'parameters': {'type': 'object', 'properties': {
                'query': {'type': 'string', 'description': 'Synthetic public query.'},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 10},
                'enabled': {'type': 'boolean'},
            }, 'required': ['query'], 'additionalProperties': False},
        }}
        for index in range(count)
    ]


def feature_cases() -> list[dict[str, Any]]:
    """Exercise generation, thinking, tools, ordering, reminders and JSON serialization."""
    cases: list[dict[str, Any]] = []
    dialogue = [
        {'role': 'system', 'content': 'Synthetic rules: preserve Unicode é雪 and literal <&>.'},
        {'role': 'user', 'content': 'Question one.'},
        {'role': 'assistant', 'content': 'Answer one.', 'reasoning_content': 'Old synthetic reasoning.'},
        {'role': 'user', 'content': 'Question two.'},
    ]
    for thinking in (False, True):
        for generation in (False, True):
            cases.append({
                'id': f'features-thinking-{int(thinking)}-generation-{int(generation)}',
                'messages': copy.deepcopy(dialogue), 'tools': None,
                'context': {'enable_thinking': thinking, 'reasoning_effort': 'low'},
                'add_generation_prompt': generation,
            })
    tools = synthetic_tools(2)
    tools[0]['function']['description'] = 'Insertion order é雪 <>&\'" is preserved, without HTML escaping.'
    object_arguments = {
        'z_text': 'é雪 <>&\'"\n', 'a_number': 3, 'enabled': True,
        'items': [False, None, 2], 'nested': {'z': 1, 'a': 'value'},
    }
    for string_arguments in (False, True):
        for thinking in (False, True):
            arguments = hf_tojson(object_arguments) if string_arguments else copy.deepcopy(object_arguments)
            messages: list[dict[str, Any]] = [
                {'role': 'system', 'content': 'Use only the synthetic tools.'},
                {'role': 'user', 'content': 'Call both tools.'},
                {'role': 'assistant', 'content': '', 'reasoning_content': 'Synthetic plan.', 'tool_calls': [
                    {'id': 'synthetic_call_a', 'type': 'function', 'function': {
                        'name': 'synthetic_tool_000', 'arguments': arguments}},
                    {'id': 'synthetic_call_b', 'type': 'function', 'function': {
                        'name': 'synthetic_tool_001', 'arguments': {'query': 'second', 'limit': 2}}},
                ]},
                {'role': 'tool', 'tool_call_id': 'synthetic_call_b', 'content': 'SECOND_RESULT'},
                {'role': 'tool', 'tool_call_id': 'synthetic_call_a', 'content': 'FIRST_RESULT'},
                {'role': 'user', 'content': 'Summarise both results.'},
            ]
            cases.append({
                'id': f'features-tools-{"string" if string_arguments else "object"}-thinking-{int(thinking)}',
                'messages': messages, 'tools': copy.deepcopy(tools),
                'context': {'enable_thinking': thinking, 'reasoning_effort': 'max'},
                'add_generation_prompt': True,
            })
    for thinking in (False, True):
        cases.append({
            'id': f'features-mid-system-reminder-thinking-{int(thinking)}',
            'messages': [
                {'role': 'system', 'content': 'Initial synthetic system.'},
                {'role': 'user', 'content': 'First user.'},
                {'role': 'assistant', 'content': 'First answer.', 'reasoning_content': 'Old reasoning.'},
                {'role': 'system', 'content': 'Mid-conversation synthetic system.'},
                {'role': 'latest_reminder', 'content': 'Synthetic reminder.'},
                {'role': 'assistant', 'content': 'Final answer.', 'reasoning_content': 'Kept reasoning.'},
            ],
            'tools': None, 'context': {'enable_thinking': thinking, 'reasoning_effort': 37},
            'add_generation_prompt': False,
        })
    cases.append({
        'id': 'features-json-values-response-format',
        'messages': [
            {'role': 'system', 'content': 'Synthetic structured response.'},
            {'role': 'user', 'content': {'z_text': 'é雪 <>&\'"', 'a_values': [True, None, 7]}},
        ],
        'tools': None,
        'context': {'enable_thinking': False, 'response_format': {
            'type': 'object', 'properties': {'z': {'type': 'string'}, 'a': {'type': 'integer'}},
        }},
        'add_generation_prompt': True,
    })
    cases.append({
        'id': 'features-tools-without-leading-system',
        'messages': [{'role': 'user', 'content': 'Use a synthetic tool.'}],
        'tools': [tools[0]['function']], 'context': {'enable_thinking': False},
        'add_generation_prompt': True,
    })
    cases.extend([
        {
            'id': 'provenance-user-marker-collision',
            'messages': [{'role': 'user', 'content': '<｜User｜>'}], 'tools': None,
            'context': {'enable_thinking': False}, 'add_generation_prompt': True,
        },
        {
            'id': 'provenance-reasoning-marker-collision',
            'messages': [
                {'role': 'user', 'content': ''},
                {'role': 'assistant', 'content': '', 'reasoning_content': '</think>'},
            ],
            'tools': None, 'context': {}, 'add_generation_prompt': False,
        },
    ])
    return cases


def expected_request_spans(case: dict[str, Any]) -> list[list[int]] | None:
    """Derive exact byte coverage for direct message copies and deliberate marker collisions."""
    output = case['expected'].encode('utf-8')
    if case['id'].startswith('scaled-') and case['tools'] is None:
        spans, cursor = [], 0
        for message in case['messages']:
            value = message['content'].encode('utf-8')
            start = output.index(value, cursor)
            spans.append([start, start + len(value)])
            cursor = start + len(value)
        return spans
    if case['id'] == 'provenance-user-marker-collision':
        prefix = '<｜begin▁of▁sentence｜><｜User｜>'.encode('utf-8')
        value = '<｜User｜>'.encode('utf-8')
        assert output == prefix + value + '<｜Assistant｜></think>'.encode('utf-8')
        return [[len(prefix), len(prefix) + len(value)]]
    if case['id'] == 'provenance-reasoning-marker-collision':
        start = output.index(b'<think>') + len(b'<think>')
        assert output[start:] == '</think></think><｜end▁of▁sentence｜>'.encode('utf-8')
        return [[start, start + len(b'</think>')]]
    return None


def render_cases(source: str, pilot: bool = False) -> list[dict[str, Any]]:
    """Render exact UTF-8 oracle strings while preserving the original template bytes."""
    env = environment()
    template = env.from_string(source)
    messages = issue_messages()
    cases: list[dict[str, Any]] = [{
        'id': f'scaled-{count:04d}-tools-none', 'messages': messages[:count], 'tools': None,
        'context': {}, 'add_generation_prompt': True,
    } for count in ((9,) if pilot else COUNTS)]
    if not pilot:
        cases.append({
            'id': 'scaled-2001-tools-synthetic-157', 'messages': messages,
            'tools': synthetic_tools(), 'context': {}, 'add_generation_prompt': True,
        })
        cases.extend(feature_cases())
    for case in cases:
        case['expected'] = template.render(
            messages=case['messages'], tools=case['tools'],
            add_generation_prompt=case['add_generation_prompt'], **case['context'],
        )
        spans = expected_request_spans(case)
        if spans is not None:
            case['expected_request_spans'] = spans
    return cases


def verify_oracle(cases: list[dict[str, Any]]) -> None:
    """Check controls that distinguish the intended JSON and tool-result conventions."""
    assert hf_tojson({'z': 'é雪 <>&\'"', 'a': 1}) == '{"z": "é雪 <>&\'\\\"", "a": 1}'
    assert hf_tojson({'z': 1, 'a': 2}, sort_keys=True) == '{"a": 2, "z": 1}'
    assert hf_tojson('é', ensure_ascii=True) == '"\\u00e9"'
    assert len({case['id'] for case in cases}) == len(cases)
    for case in cases:
        if case['id'].startswith('features-tools-object-'):
            text = case['expected']
            assert text.index('FIRST_RESULT') < text.index('SECOND_RESULT')
            assert 'parameter name="z_text" string="true">é雪 <>&\'"\n' in text
            assert 'parameter name="enabled" string="false">true' in text
        if case['id'].startswith('features-tools-string-'):
            assert 'parameter name="arguments" string="true">' in case['expected']


def write_corpus(template_path: Path, output: Path, pilot: bool = False) -> dict[str, Any]:
    """Validate template identity and write only public synthetic inputs and Jinja2 expected output."""
    raw = template_path.read_bytes()
    if len(raw) != TEMPLATE_BYTES or hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('DeepSeek template bytes differ from the pinned issue template')
    cases = render_cases(raw.decode('utf-8'), pilot)
    verify_oracle(cases)
    encoded = (json.dumps(cases, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return {
        'template_sha256': TEMPLATE_SHA256, 'template_bytes': len(raw),
        'jinja2': jinja2.__version__, 'tojson_convention': TOJSON_CONVENTION,
        'cases': len(cases), 'corpus_sha256': hashlib.sha256(encoded).hexdigest(),
        'expected_utf8_bytes': sum(len(case['expected'].encode('utf-8')) for case in cases),
        'case_ids': [case['id'] for case in cases],
        'message_generator': 'Issue public generator: seed424242,2000 alternating messages,234 NATO words each.',
        'tools_scope': 'Original157 tool schemas unavailable; synthetic157 case is separately labelled.',
        'exact_provenance_cases': sum('expected_request_spans' in case for case in cases),
        'provenance_scope': ('Exact byte coverage for toolsNone scaled cases and marker probes; '
                             'structural checks elsewhere.'),
    }


def main() -> int:
    """Generate the pilot or complete corpus from the separately supplied pinned template."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    summary = write_corpus(args.template, args.output, args.pilot)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
