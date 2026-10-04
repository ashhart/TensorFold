//! Feature probes from tools/template_parity.py; expected strings come from Python jinja2 in transformers' environment.
const std = @import("std");
const template = @import("template.zig");

const Outcome = struct { text: ?[]const u8, raised: ?[]const u8 };

fn render(source: []const u8, vars: []const u8, diag: *template.Diag) ![]u8 {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, vars, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const generation = if (obj.get("add_generation_prompt")) |g| g.bool else true;
    return template.renderChat(a, source, obj.get("messages") orelse .null, obj.get("tools") orelse .null, parsed.value, generation, diag);
}

fn expect(source: []const u8, vars: []const u8, want: Outcome) !void {
    const a = std.testing.allocator;
    var diag = template.Diag{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    const out = render(source, vars, &diag) catch |err| {
        if (err != error.TemplateFailed or want.text != null) {
            std.debug.print("{s}\nfailed: {s}\n", .{ source, diag.msg });
            return error.TestUnexpectedError;
        }
        if (want.raised) |msg| {
            try std.testing.expect(diag.raised);
            try std.testing.expectEqualStrings(msg, diag.msg);
        }
        return;
    };
    defer a.free(out);
    if (want.text == null) {
        std.debug.print("{s}\nrendered instead of failing: {s}\n", .{ source, out });
        return error.TestExpectedError;
    }
    try std.testing.expectEqualStrings(want.text.?, out);
}

const shared = "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}, {\"role\": \"assistant\", \"content\": \" Hello \"}, {\"role\": \"user\", \"content\": \"Bye\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}], \"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}";

const probes = [_]struct { []const u8, ?[]const u8, ?[]const u8 }{
    .{ "{% for m in messages %}{{ loop.index }}/{{ loop.length }} {{ loop.revindex0 }} {{ loop.first }} {{ loop.last }} {{ loop.previtem.role if loop.previtem }}|{{ loop.nextitem.role if loop.nextitem is defined }}\n{% endfor %}", "1/3 2 True False |assistant\n2/3 1 False False user|user\n3/3 0 False True assistant|\n", null },
    .{ "{% set x = 1 %}{% for i in [1, 2] %}{{ x }}{% set x = x + i %}{{ x }}{% endfor %}{{ x }}", "12131", null },
    .{ "{% set ns = namespace(n=0, s='') %}{% for i in range(5) if i is odd %}{% set ns.n = ns.n + i %}{% set ns.s = ns.s ~ i %}{% endfor %}{{ ns.n }}:{{ ns.s }}:{{ ns }}", "4:13:<Namespace {'n': 4, 's': '13'}>", null },
    .{ "{% macro m(a, b=2, c='x') %}[{{ a }}|{{ b }}|{{ c }}|{{ d }}]{% endmacro %}{{ m(1) }}{{ m(1, 3) }}{{ m(1, c='z') }}{{ m() }}{% set d = 'late' %}{{ m(4) }}", "[1|2|x|][1|3|x|][1|2|z|][|2|x|][4|2|x|late]", null },
    .{ "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}", null, null },
    .{ "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, b=2) }}", null, null },
    .{ "{% for i in [3, 1, 2] %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% if i == 2 %}{% break %}{% endif %}{% else %}empty{% endfor %}", "32", null },
    .{ "{% for i in [] %}x{% else %}empty{% endfor %}|{% for i in [1] %}{% continue %}{% else %}else-after-continue{% endfor %}", "empty|else-after-continue", null },
    .{ "{%- if true -%}   trimmed   {%- endif -%}  |  {% if true %}\n  kept\n{% endif %}\n    {% if true %}lstrip{% endif %}  {%+ if true %}plus{% endif %}  {{- ' dash' }}  {#- comment -#}  end", "trimmed|    kept\nlstrip  plus dashend", null },
    .{ "{{ 'a\\tb\\n\\x41\u{e9}\\101\\\\q' if false else 'esc:\\t\\x41\u{e9}\\101\\q' }}|{{ \"it's\" }}|{{ 'say \"hi\"' }}", "esc:\tA\u{e9}A\\q|it's|say \"hi\"", null },
    .{ "{{ [1, 'a', none, true, 1.5, [2], {'k': 'v'}, ('t',), (1, 2)] }}|{{ {'a': 1, 'b': [1.0, 1e16, -0.0]} }}|{{ (1,) | string }}", "[1, 'a', None, True, 1.5, [2], {'k': 'v'}, ('t',), (1, 2)]|{'a': 1, 'b': [1.0, 1e+16, -0.0]}|(1,)", null },
    .{ "{{ messages | tojson }}|{{ tools | tojson(indent=2) }}|{{ {'b': 1, 'a': '\u{e9}'} | tojson(sort_keys=true, ensure_ascii=true) }}|{{ [1, [2]] | tojson(separators=(',', ':')) }}", "[{\"role\": \"user\", \"content\": \"Hi\"}, {\"role\": \"assistant\", \"content\": \" Hello \"}, {\"role\": \"user\", \"content\": \"Bye\"}]|[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"weather\",\n      \"description\": \"Get weather\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n          \"city\": {\n            \"type\": \"string\"\n          }\n        },\n        \"required\": [\n          \"city\"\n        ]\n      }\n    }\n  }\n]|{\"a\": \"\\u00e9\", \"b\": 1}|[1,[2]]", null },
    .{ "{{ 'Hello World'[::-1] }}|{{ 'h\u{e9}llo'[1:4] }}|{{ [1, 2, 3, 4][-2:] }}|{{ 'abc'[5:] }}|{{ (1, 2, 3)[::2] }}|{{ 'abc'[-1] }}|{{ 'abc'[10] is defined }}", "dlroW olleH|\u{e9}ll|[3, 4]||(1, 3)|c|False", null },
    .{ "{{ '  x y  '.strip() }}|{{ 'xxhixx'.strip('x') }}|{{ 'a,b,,c'.split(',') }}|{{ ' a  b '.split() }}|{{ 'a b c'.split(None, 1) }}|{{ 'a b c'.rsplit(' ', 1) }}|{{ 'aaa'.replace('a', 'b', 2) }}|{{ 'abc'.replace('', '-') }}", "x y|hi|['a', 'b', '', 'c']|['a', 'b']|['a', 'b c']|['a b', 'c']|bba|-a-b-c-", null },
    .{ "{{ 'hello'.find('l') }}|{{ 'hello'.rfind('l') }}|{{ 'h\u{e9}llo'.find('l') }}|{{ 'hello'.startswith(('x', 'he')) }}|{{ 'hello'.endswith('lo', 0, 5) }}|{{ 'a-b'.count('-') }}|{{ 'x'.join(['a', 'b']) }}|{{ 'pre_x'.removeprefix('pre_') }}", "2|3|2|True|True|1|axb|x", null },
    .{ "{{ '\u{c6}\u{d8}\u{c5} stra\u{df}e'.lower() }}|{{ '\u{c6}\u{d8}\u{c5} stra\u{df}e'.upper() }}|{{ 'abc' | upper }}|{{ none | upper }}|{{ 'ABC' | lower }}|{{ 42 | string }}|{{ 1.0 | string }}|{{ [1] | string }}", "\u{e6}\u{f8}\u{e5} stra\u{df}e|\u{c6}\u{d8}\u{c5} STRASSE|ABC|NONE|abc|42|1.0|[1]", null },
    .{ "{{ '\u{3a3}\u{391}\u{3a3} \u{39f}\u{394}\u{3a5}\u{3a3}\u{3a3}\u{395}\u{3a5}\u{3a3} A\u{3a3}\\'b' | lower }}|{{ ['\u{3a3}a', '\u{3c3}b', '\u{3a3}A'] | sort }}|{{ {'\u{c4}rger': 1, '\u{e4}pfel': 2, 'Zed': 3} | dictsort }}", "\u{3c3}\u{3b1}\u{3c2} \u{3bf}\u{3b4}\u{3c5}\u{3c3}\u{3c3}\u{3b5}\u{3c5}\u{3c2} a\u{3c3}'b|['\u{3a3}a', '\u{3a3}A', '\u{3c3}b']|[('Zed', 3), ('\u{e4}pfel', 2), ('\u{c4}rger', 1)]", null },
    .{ "{{ undefined_name }}|{{ undefined_name is defined }}|{{ undefined_name | default('dflt') }}|{{ '' | default('empty', true) }}|{{ none | default('n') }}|{{ undefined_name | length }}|{{ undefined_name ~ 'x' }}", "|False|dflt|empty|None|0|x", null },
    .{ "{{ x.y }}", null, null },
    .{ "{{ 'a' + undefined_name }}", null, null },
    .{ "{{ messages[0].nope.deeper }}", null, null },
    .{ "{{ messages[0]['role'] }}|{{ messages[0].role }}|{{ messages.0.role }}|{{ messages[0].items is defined }}|{{ messages[0].get('role') }}|{{ messages[0].get('nope', 'd') }}|{{ messages[0].pop is defined }}", "user|user|user|True|user|d|False", null },
    .{ "{{ [3, 1, 2] | sort }}|{{ ['b', 'A', 'a'] | sort }}|{{ ['b', 'A', 'a'] | sort(case_sensitive=true) }}|{{ ['b', 'a'] | sort(reverse=true) }}|{{ ['a', 'A', 'b'] | unique | list }}|{{ [1, 2] | reverse | list }}|{{ 'abc' | reverse }}", "[1, 2, 3]|['A', 'a', 'b']|['A', 'a', 'b']|['b', 'a']|['a', 'b']|[2, 1]|cba", null },
    .{ "{{ {'b': 1, 'A': 2, 'a': 3} | dictsort }}|{{ {'b': 1, 'a': 2} | dictsort(by='value', reverse=true) }}|{{ {'x': 1} | items | list }}|{{ [1, 2, 3] | first }}|{{ [1, 2, 3] | last }}|{{ [] | first is defined }}", "[('A', 2), ('a', 3), ('b', 1)]|[('a', 2), ('b', 1)]|[('x', 1)]|1|3|False", null },
    .{ "{{ ['a', 'b'] | map('upper') | list }}|{{ messages | map(attribute='role') | join(',') }}|{{ messages | selectattr('role', 'equalto', 'user') | list | length }}|{{ [0, 1, 2] | select | list }}|{{ [0, 1, 2] | reject('odd') | list }}|{{ messages | rejectattr('content') | list | length }}", "['A', 'B']|user,assistant,user|2|[1, 2]|[0, 2]|0", null },
    .{ "{{ 'a\\nb\\n\\nc' | indent(2) }}|{{ 'a\\nb' | indent(2, true) }}|{{ 'a\\n\\nb' | indent('> ', blank=true) }}|{{ 'x&<>\"\\'' | e }}|{{ ('<b>' | safe) + '<i>' }}|{{ '<i>' + ('<b>' | safe) }}|{{ 'x' | replace('x', 'y') }}", "a\n  b\n\n  c|  a\n  b|a\n> \n> b|x&amp;&lt;&gt;&#34;&#39;|<b>&lt;i&gt;|&lt;i&gt;<b>|y", null },
    .{ "{{ 1 + 2 }}|{{ 7 // 2 }}|{{ -7 // 2 }}|{{ 7 % 3 }}|{{ -7 % 3 }}|{{ 7 / 2 }}|{{ 2 ** 10 }}|{{ 2 ** -1 }}|{{ 1.5 * 2 }}|{{ -7.5 // 2 }}|{{ -7.5 % 2 }}|{{ true + 1 }}|{{ 'ab' * 2 }}|{{ [1] * 2 }}|{{ 10 / 4 }}|{{ 0.1 + 0.2 }}|{{ 1e300 * 1e10 }}", "3|3|-4|1|2|3.5|1024|0.5|3.0|-4.0|0.5|2|abab|[1, 1]|2.5|0.30000000000000004|inf", null },
    .{ "{{ 1 == 1.0 }}|{{ 'a' < 'b' }}|{{ [1, 2] < [1, 3] }}|{{ 1 < 2 < 3 }}|{{ 3 > 2 > 2 }}|{{ 'a' in 'cat' }}|{{ 'x' not in ['x'] }}|{{ 'k' in {'k': 1} }}|{{ none == none }}|{{ undefined_name == undefined_other }}|{{ (1, 2) == [1, 2] }}", "True|True|True|True|False|True|False|True|True|True|False", null },
    .{ "{{ 1 < 'a' }}", null, null },
    .{ "{{ 'a' in 1 }}", null, null },
    .{ "{{ none is none }}|{{ 1 is number }}|{{ true is number }}|{{ true is integer }}|{{ 1.0 is float }}|{{ 'x' is string }}|{{ {} is mapping }}|{{ {} is sequence }}|{{ 'x' is sequence }}|{{ undefined_name is sequence }}|{{ undefined_name is iterable }}|{{ none is iterable }}|{{ 3 is odd }}|{{ 4 is even }}|{{ 9 is divisibleby 3 }}|{{ 'a' is in 'abc' }}|{{ 1 is eq 1 }}|{{ 2 is gt 1 }}|{{ false is sameas false }}|{{ 'tojson' is filter }}|{{ 'odd' is test }}|{{ x is callable }}|{{ range is callable }}", "True|True|True|False|True|True|True|True|True|True|True|False|True|True|True|True|True|True|True|True|True|True|True", null },
    .{ "{{ range(3) | list }}|{{ range(1, 10, 3) | list }}|{{ range(5, 0, -2) | list }}|{{ range(3) }}|{{ range(0, 10, 2) }}|{{ range(3) | length }}", "[0, 1, 2]|[1, 4, 7]|[5, 3, 1]|range(0, 3)|range(0, 10, 2)|3", null },
    .{ "{% set captured %}  inner {{ 1 + 1 }}  {% endset %}[{{ captured }}]|{% set up | upper %}shout{% endset %}{{ up }}", "[  inner 2  ]|SHOUT", null },
    .{ "{% set a, b = [1, 2] %}{{ a }}{{ b }}|{% for k, v in {'x': 1, 'y': 2} | items %}{{ k }}={{ v }};{% endfor %}|{% for a, (b, c) in [(1, (2, 3))] %}{{ a }}{{ b }}{{ c }}{% endfor %}", "12|x=1;y=2;|123", null },
    .{ "{% set a, b = [1] %}", null, null },
    .{ "{{ raise_exception('custom failure: ' ~ 42) }}", null, "custom failure: 42" },
    .{ "{{ messages[0].content if messages else 'none' }}|{{ 'yes' if '' else 'no' }}|{{ ('a' if false) is defined }}|{{ none or 'fallback' }}|{{ 0 and 'never' }}|{{ not none }}", "Hi|no|False|fallback|0|True", null },
    .{ "{% for i in range(3) %}{{ loop.cycle('odd', 'even') }}{{ loop.changed(i // 2) }}{% endfor %}", "oddTrueevenFalseoddTrue", null },
    .{ "{% generation %}gen {{ 1 }}{% endgeneration %}|{% raw %}{{ raw }} {% not a tag %}{% endraw %}", "gen 1|{{ raw }} {% not a tag %}", null },
    .{ "{% macro outer() %}{{ inner() }}{% endmacro %}{% macro inner() %}inner-ok{% endmacro %}{{ outer() }}", "inner-ok", null },
    .{ "{% for m in messages %}{{ late_var }}{% endfor %}{% set late_var = 'root' %}{{ late_var }}", "root", null },
    .{ "{% if false %}{% set y = 1 %}{% endif %}{{ y }}|{% if true %}{% set y = 2 %}{% endif %}{{ y }}", "|2", null },
    .{ "{{ messages | length }}|{{ messages[-1].role }}|{{ messages[99] is defined }}|{{ messages[0]['nope'] is defined }}|{{ (messages | first).role }}|{{ tools | length if tools else 0 }}|{{ documents }}|{{ bos_token }}{{ eos_token }}", "3|user|False|False|user|1|None|<s></s>", null },
    .{ "{{ messages[0].content.type }}|{{ 'abc'.nope is defined }}|{{ none.x is defined }}|{{ (1).real is defined }}|{{ {}.items() | list }}|{{ {'a': 1}.items() }}|{{ {'a': 1}.keys() | list }}|{{ {'a': 1}.values() | list }}", "|False|False|True|[]|dict_items([('a', 1)])|['a']|[1]", null },
    .{ "{{ [1, 2, 3] | join('-') }}|{{ [1, none, 'x'] | join }}|{{ 'abc' | list }}|{{ {'a': 1, 'b': 2} | list }}|{{ 'h\u{e9}llo' | length }}|{{ {'a': 1} | length }}|{{ 'a b' | trim }}|{{ ' x ' | trim('x ') }}", "1-2-3|1Nonex|['a', 'b', 'c']|['a', 'b']|5|1|a b|", null },
    .{ "{{ x | nofilter }}", null, null },
    .{ "{{ 'a' ~ 1 ~ none ~ 1.5 ~ [1] ~ true }}|{{ -1 }}|{{ -(1) }}|{{ +1 }}|{{ - 1.5 }}|{{ not true }}|{{ 'x' * 0 }}|{{ 1.0 == 1 }}|{{ 1e2 }}|{{ 1_000 }}|{{ 0x1F }}|{{ 1.5e-7 }}|{{ 12.5E3 }}", "a1None1.5[1]True|-1|-1|1|-1.5|False||True|100.0|1000|31|1.5e-07|12500.0", null },
    .{ "{{ dict(a=1, b='x') }}|{{ namespace(a=1).a }}|{{ namespace({'q': 2}).q }}|{% set n = namespace() %}{% set n.v = [1] %}{{ n.v }}", "{'a': 1, 'b': 'x'}|1|2|[1]", null },
    .{ "{% set t = 'x' %}{% set t.attr = 1 %}", null, null },
    .{ "{{ messages[0].content[0] }}|{{ 'abc'[0:2] }}|{{ 'abc'[:-1] }}|{{ 'abc'['upper']() }}", "H|ab|ab|ABC", null },
    .{ "{{ ['b', 'a'] | sort | join }}{{ [{'n': 'b'}, {'n': 'a'}] | sort(attribute='n') | map(attribute='n') | join }}{{ [{'n': 1}] | map(attribute='m', default='d') | join }}", "ababd", null },
    .{ "{% for x in 'ab' %}{{ x }}{{ loop.index0 }}{% endfor %}|{% for k in {'p': 1, 'q': 2} %}{{ k }}{% endfor %}|{% for x in none %}{% endfor %}", null, null },
    .{ "{%- for m in messages -%}\n    {%- set content = m.content | trim -%}\n    {{- '<' + m.role + '>' + content -}}\n{%- endfor %}\nx", "<user>Hi<assistant>Hello<user>Byex", null },
};

test "feature probes match Python jinja2" {
    for (probes) |p| try expect(p[0], shared, .{ .text = p[1], .raised = p[2] });
}

test "engine bridge: NUL content, merged system prompt and generation prompt" {
    try expect("{% for m in messages %}{{m.role}}:{{m.content}}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
        \\{"messages":[{"role":"system","content":"first\n\nsecond"},{"role":"user","content":"a\u0000b"}]}
    , .{ .text = "system:first\n\nseconduser:a\x00bassistant:", .raised = null });
}

test "engine context: thinking switches and effort defaults" {
    const source = "{{ 'on' if enable_thinking else 'off' }}:{{ reasoning_effort|default('unset') }}";
    try expect(source, "{\"enable_thinking\":true,\"reasoning_effort\":\"medium\"}", .{ .text = "on:medium", .raised = null });
    try expect(source, "{\"enable_thinking\":false}", .{ .text = "off:unset", .raised = null });
}

test "numeric members, undefined tools and documents" {
    try expect("{{ messages.0.role }}|{{ tools }}|{{ documents }}|{{ tools is defined }}", "{\"messages\":[{\"role\":\"user\"}]}", .{ .text = "user|None|None|True", .raised = null });
}

test "syntax and unknown filters fail at compile time" {
    try expect("{{ x | nofilter }}", shared, .{ .text = null, .raised = null });
    try expect("{% if %}", shared, .{ .text = null, .raised = null });
    try expect("{% for x in y %}", shared, .{ .text = null, .raised = null });
}
