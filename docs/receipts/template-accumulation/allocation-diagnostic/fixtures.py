"""Generate a fixed allocation diagnostic grid with public synthetic tool conversations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

TEMPLATE_SHA256 = '7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e'
CATALOG_BYTES = 240 * 1024
CYCLES = (8, 64, 128)
CONTROL_TEMPLATE = (
    "{% set ns=namespace(left='L',right='R') %}{% for m in messages %}"
    "{% set breaker='B' ~ m.content %}{% set ns.left=ns.left+m.content %}"
    "{% set ns.right=ns.right+('X' ~ m.content) %}{% endfor %}{{ ns.left }}|{{ ns.right }}"
)


def helpers() -> Any:
    """Load the existing public deterministic generator and pinned Jinja2 environment."""
    path = Path(__file__).resolve().parents[1] / 'deepseek-template/fixtures.py'
    spec = importlib.util.spec_from_file_location('deepseek_public_fixtures', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('the public DeepSeek fixture helpers are unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact(value: Any) -> bytes:
    """Use compact Unicode-preserving JSON solely for stated request-byte counts."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def tool_schemas(shared: Any, count: int) -> list[dict[str, Any]]:
    """Add a payload field to the existing explicitly synthetic tool definitions."""
    tools = shared.synthetic_tools(count)
    for tool in tools:
        params = tool['function']['parameters']
        params['properties']['payload'] = {'type': 'string', 'description': 'Synthetic diagnostic payload.'}
        params['required'] = ['query', 'payload']
    return tools


def large_catalog(shared: Any) -> list[dict[str, Any]]:
    """Pad synthetic descriptions to a 240 KiB serialized byte proxy without a token claim."""
    tools = tool_schemas(shared, 157)
    remaining = CATALOG_BYTES - len(compact(tools))
    if remaining < 0:
        raise ValueError('base synthetic catalog exceeds its fixed byte target')
    common, extra = divmod(remaining, len(tools))
    filler = 'Public synthetic schema guidance. '
    for index, tool in enumerate(tools):
        length = common + int(index < extra)
        tool['function']['description'] += (filler * (length // len(filler) + 1))[:length]
    if len(compact(tools)) != CATALOG_BYTES:
        raise ValueError('catalog byte target differs')
    return tools


def arguments(shared: Any, cycle: int, slot: int, size: int) -> dict[str, str]:
    """Set the complete HF-serialized object argument to exactly the requested UTF-8 size."""
    value = {'query': f'cycle{cycle:03d} tool{slot}', 'payload': ''}
    remaining = size - len(shared.hf_tojson(value).encode('utf-8'))
    if remaining < 0:
        raise ValueError('argument size is smaller than its fixed keys and query')
    payload = 'synthetic-payload-'
    value['payload'] = (payload * (remaining // len(payload) + 1))[:remaining]
    if len(shared.hf_tojson(value).encode('utf-8')) != size:
        raise ValueError('serialized argument byte count differs')
    return value


def cycle_messages(shared: Any, cycles: int, size: int, representation: str) -> list[dict[str, Any]]:
    """Model ordinary two-tool turns with reversed result arrivals and final assistant summaries."""
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': 'Use only these synthetic diagnostic tools.'}]
    for cycle in range(cycles):
        calls = []
        for slot in range(2):
            args: Any = arguments(shared, cycle, slot, size)
            if representation == 'string':
                args = shared.hf_tojson(args)
            calls.append({'id': f'synthetic_{cycle:03d}_{slot}', 'type': 'function', 'function': {
                'name': f'synthetic_tool_{slot:03d}', 'arguments': args,
            }})
        messages.extend([
            {'role': 'user', 'content': f'Cycle {cycle:03d}: call both synthetic tools and summarise.'},
            {'role': 'assistant', 'content': '', 'reasoning_content': f'Synthetic plan for cycle {cycle:03d}.',
             'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': calls[1]['id'],
             'content': f'Synthetic result for cycle {cycle:03d} tool1.'},
            {'role': 'tool', 'tool_call_id': calls[0]['id'],
             'content': f'Synthetic result for cycle {cycle:03d} tool0.'},
            {'role': 'assistant', 'content': f'Synthetic cycle {cycle:03d} completed.',
             'reasoning_content': 'Both synthetic tool results were received.'},
        ])
    return messages


def string_sizes(value: Any) -> list[int]:
    """Count actual UTF-8 request strings, including object keys, without output or template text."""
    if isinstance(value, str):
        return [len(value.encode('utf-8'))]
    if isinstance(value, list):
        return [size for item in value for size in string_sizes(item)]
    if isinstance(value, dict):
        return [size for key, item in value.items() for size in string_sizes(key) + string_sizes(item)]
    return []


def add_counts(case: dict[str, Any], shared: Any, source: str) -> None:
    """Attach raw input sizes and exact successful-render list-concatenation expectations."""
    messages, tools = case['messages'], case['tools']
    diag = case['diagnostic']
    calls = [call for message in messages for call in message.get('tool_calls', [])]
    serialized = [call['function']['arguments'] for call in calls]
    lengths = [len((arg if isinstance(arg, str) else shared.hf_tojson(arg)).encode('utf-8'))
               for arg in serialized]
    request = {'messages': messages, 'tools': tools, 'context': case['context']}
    sizes = string_sizes(request)
    diag.update({
        'message_count': len(messages), 'tool_call_count': len(calls),
        'tool_result_count': sum(message['role'] == 'tool' for message in messages),
        'catalog_schema_count': len(tools) if tools else 0,
        'catalog_compact_utf8_bytes': len(compact(tools)) if tools else 0,
        'catalog_byte_convention': 'Complete tools array, ensure_ascii=False, compact separators.',
        'argument_serialized_utf8_sizes': sorted(set(lengths)),
        'argument_serialized_utf8_total': sum(lengths),
        'request_compact_utf8_bytes': len(compact(request)),
        'request_string_utf8_total_including_keys': sum(sizes),
        'largest_request_string_utf8_bytes': max(sizes, default=0),
        'expected_utf8_bytes': len(case['expected'].encode('utf-8')),
        'expected_sha256': hashlib.sha256(case['expected'].encode('utf-8')).hexdigest(),
        'template_sha256': hashlib.sha256(case.get('template_source', source).encode('utf-8')).hexdigest(),
        'list_counter_scope': 'Evaluator list+ only; excludes input conversion, list literals and slicing.',
    })


def make_cases(shared: Any, source: str) -> list[dict[str, Any]]:
    """Freeze twelve exact-template cases and one separately identified synthetic control."""
    plain = shared.issue_messages()
    catalog = large_catalog(shared)
    ordinary = tool_schemas(shared, 2)
    cases: list[dict[str, Any]] = [
        {'id': 'reported-2001-no-tools', 'messages': plain, 'tools': None, 'context': {},
         'add_generation_prompt': True, 'diagnostic': {
             'kind': 'reported-template-filler', 'synthetic_control': False,
             'expected_list_concat_calls': 2001, 'expected_list_concat_elements': 2001 * 2002 // 2,
         }},
        {'id': 'catalog-157-240kib-small-dialogue',
         'messages': [{'role': 'system', 'content': 'Use synthetic tools only.'},
                      {'role': 'user', 'content': 'Explain the catalog briefly.'},
                      {'role': 'assistant', 'content': 'The catalog contains synthetic diagnostic functions.'}],
         'tools': catalog, 'context': {}, 'add_generation_prompt': True, 'diagnostic': {
             'kind': 'catalog-byte-proxy', 'synthetic_control': False,
             'catalog_scope': '240 KiB serialized byte proxy; not the original schemas or a token-count equivalent.',
             'expected_list_concat_calls': 3, 'expected_list_concat_elements': 6,
         }},
    ]
    for cycles in CYCLES:
        for representation, size in (('object', 64), ('object', 8192), ('string', 8192)):
            cases.append({
                'id': f'tool-cycles-{cycles:03d}-{representation}-{size}',
                'messages': cycle_messages(shared, cycles, size, representation),
                'tools': ordinary, 'context': {}, 'add_generation_prompt': True,
                'diagnostic': {'kind': 'tool-cycles', 'synthetic_control': False, 'cycles': cycles,
                               'argument_representation': representation, 'argument_size_utf8_bytes': size,
                               'expected_list_concat_calls': 1 + 8 * cycles,
                               'expected_list_concat_elements': 10 * cycles * cycles + 13 * cycles + 1},
            })
    cases.append({
        'id': 'tool-cycles-128-object-8192-catalog-240kib',
        'messages': cycle_messages(shared, 128, 8192, 'object'), 'tools': catalog,
        'context': {}, 'add_generation_prompt': True,
        'diagnostic': {'kind': 'tool-cycles-large-catalog', 'synthetic_control': False, 'cycles': 128,
                       'argument_representation': 'object', 'argument_size_utf8_bytes': 8192,
                       'catalog_scope': '240 KiB serialized byte proxy; no token-count equivalence claimed.',
                       'expected_list_concat_calls': 1025, 'expected_list_concat_elements': 165505},
    })
    cases.append({
        'id': 'synthetic-interleaved-breaker-control',
        'messages': [{'role': 'user', 'content': f'item{index:03d}:' + 'x' * 248} for index in range(32)],
        'tools': None, 'context': {}, 'add_generation_prompt': False, 'template_source': CONTROL_TEMPLATE,
        'diagnostic': {'kind': 'synthetic-string-control', 'synthetic_control': True,
                       'scope': 'Separate interleaved-string diagnostic control; not the reported template.',
                       'expected_list_concat_calls': 0, 'expected_list_concat_elements': 0},
    })
    env = shared.environment()
    exact = env.from_string(source)
    for case in cases:
        template = env.from_string(case['template_source']) if 'template_source' in case else exact
        case['expected'] = template.render(messages=case['messages'], tools=case['tools'],
                                           add_generation_prompt=case['add_generation_prompt'], **case['context'])
        add_counts(case, shared, source)
    return cases


def write_corpus(template: Path, output: Path) -> dict[str, Any]:
    """Validate the exact template and write the complete fixed grid before any renderer runs."""
    source = template.read_bytes()
    if len(source) != 11410 or hashlib.sha256(source).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('the separately supplied DeepSeek template differs from its pinned bytes')
    shared = helpers()
    cases = make_cases(shared, source.decode('utf-8'))
    if len(cases) != 13 or len({case['id'] for case in cases}) != 13:
        raise ValueError('the fixed diagnostic grid differs')
    encoded = compact(cases) + b'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return {
        'template_sha256': TEMPLATE_SHA256, 'template_bytes': len(source), 'jinja2': shared.jinja2.__version__,
        'tojson_convention': shared.TOJSON_CONVENTION,
        'shared_fixture_helpers_sha256': hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        'fixture_generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'corpus_sha256': hashlib.sha256(encoded).hexdigest(), 'corpus_bytes': len(encoded),
        'exact_template_cases': 12, 'separate_synthetic_control_cases': 1,
        'catalog_target_compact_utf8_bytes': CATALOG_BYTES,
        'expected_utf8_bytes': sum(case['diagnostic']['expected_utf8_bytes'] for case in cases),
        'grid': [{'id': case['id'], **case['diagnostic']} for case in cases],
        'limits': 'Synthetic public byte proxies only; no original schemas, private messages, tokenizer or model run.',
    }


def main() -> int:
    """Generate deterministic fixtures and their byte identities for a bounded allocation diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args()
    summary = write_corpus(args.template, args.output)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'grid'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
