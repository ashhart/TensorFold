//! Exact byte maps and a complete dispatch census prevent derived strings from losing request provenance.
const std = @import("std");
const template = @import("template.zig");
const v = @import("value.zig");
const eval = @import("eval.zig");
const access = @import("access.zig");
const filters = @import("filters.zig");
const t = std.testing;

fn covered(s: v.Str, at: usize) bool {
    for (s.spans) |p| if (p[0] <= at and at < p[1]) return true;
    return false;
}

fn spelling(s: v.Str) !void {
    for ([_][]const u8{ "<SPECIAL>", "<special>", ">LAICEPS<" }) |word| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, s.s, at, word)) |hit| {
            for (hit..hit + word.len) |i| try t.expect(covered(s, i));
            at = hit + word.len;
        }
    }
}

fn checkValue(rt: *v.Rt, value: v.Value) anyerror!void {
    switch (value) {
        .str => |s| try spelling(s),
        .dict => |d| for (d.map.keys(), d.map.values()) |key, item| {
            try checkValue(rt, try v.key(rt, d, key));
            try checkValue(rt, item);
        },
        .list => |l| for (l.items) |item| try checkValue(rt, item),
        .iter => |it| {
            try t.expect(it.err == null);
            for (it.items) |item| try checkValue(rt, item);
        },
        else => {},
    }
}

fn exact(source: []const u8, context: []const u8, want: []const u8, spans: []const v.Span) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ctx = try std.json.parseFromSliceLeaky(std.json.Value, a, context, .{});
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    const got = try template.renderProvenance(a, &compiled, .null, .null, ctx, true, null);
    try t.expectEqualStrings(want, got.s);
    const expected = v.Str{ .s = want, .spans = spans };
    for (0..want.len) |i| try t.expectEqual(covered(expected, i), covered(got, i));
    var previous: usize = 0;
    for (got.spans) |p| {
        try t.expect(previous <= p[0] and p[0] < p[1] and p[1] <= got.s.len);
        previous = p[1];
    }
}

test "provenance exact maps for concatenation repetition indexing slicing and string" {
    const ctx = "{\"x\":\"<SPECIAL>\"}";
    for ([_][]const u8{ "L{{ x }}R", "{{ 'L' + x ~ 'R' }}", "{{ ('L'~x~'R')|string }}" }) |source|
        try exact(source, ctx, "L<SPECIAL>R", &.{.{ 1, 10 }});
    try exact("{{ x * 2 }}", ctx, "<SPECIAL><SPECIAL>", &.{.{ 0, 18 }});
    try exact("{{ ('L'~x~'R')[1:-1] }}", ctx, "<SPECIAL>", &.{.{ 0, 9 }});
    try exact("{{ ('L'~x~'R')[::-1] }}", ctx, "R>LAICEPS<L", &.{.{ 1, 10 }});
    try exact("{{ x[0] }}{{ x[-1] }}", ctx, "<>", &.{.{ 0, 2 }});
    try exact("{{ x|first }}{{ x|last }}", ctx, "<>", &.{.{ 0, 2 }});
}

test "provenance exact maps for split strip affix replace and joins" {
    const ctx = "{\"x\":\"<SPECIAL>\",\"y\":\"SPECIAL\"}";
    try exact("{{ ('#'~x~'!').split('#')[1].rsplit('!')[0] }}", ctx, "<SPECIAL>", &.{.{ 0, 9 }});
    for ([_][]const u8{ "strip()", "lstrip().rstrip()", "removeprefix(' ').removesuffix(' ')" }) |method| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const source = try std.fmt.allocPrint(arena.allocator(), "{{{{ (' '~x~' ').{s} }}}}", .{method});
        try exact(source, ctx, "<SPECIAL>", &.{.{ 0, 9 }});
    }
    try exact("{{ (' '~x~' ')|trim }}", ctx, "<SPECIAL>", &.{.{ 0, 9 }});
    try exact("{{ x.replace('SPECIAL', 'literal') }}", ctx, "<literal>", &.{ .{ 0, 1 }, .{ 8, 9 } });
    try exact("{{ '<X>'.replace('X', y) }}", ctx, "<SPECIAL>", &.{.{ 1, 8 }});
    try exact("{{ '<X>'|replace('X', y) }}", ctx, "<SPECIAL>", &.{.{ 1, 8 }});
    try exact("{{ 'ab'.replace('', x, 1) }}", ctx, "<SPECIAL>ab", &.{.{ 0, 9 }});
    try exact("{{ [x, 'T', x]|join('-') }}", ctx, "<SPECIAL>-T-<SPECIAL>", &.{ .{ 0, 9 }, .{ 12, 21 } });
    try exact("{{ x.join(['A','B']) }}", ctx, "A<SPECIAL>B", &.{.{ 1, 10 }});
    try exact("{{ missing|default(x) }}", ctx, "<SPECIAL>", &.{.{ 0, 9 }});
}

test "provenance exact maps through capture scopes namespaces loops and request keys" {
    const ctx = "{\"x\":\"<SPECIAL>\",\"d\":{\"<SPECIAL>\":\"<SPECIAL>\"}}";
    for ([_][]const u8{
        "{% macro wrap(x) %}L{{ x }}R{% endmacro %}{{ wrap(x) }}",
        "{% set block %}L{{ x }}R{% endset %}{{ block }}",
        "{% generation %}L{{ x }}R{% endgeneration %}",
        "{% set ns=namespace(y=x) %}L{{ ns.y }}R",
        "{% for y in [x] %}L{{ y }}R{% endfor %}",
    }) |source| try exact(source, ctx, "L<SPECIAL>R", &.{.{ 1, 10 }});
    for ([_][]const u8{
        "{% for k in d %}{{ k }}{% endfor %}",            "{{ d.keys()|join }}",
        "{% for k,v in d.items() %}{{ k }}{% endfor %}",  "{% for k,v in d|items %}{{ k }}{% endfor %}",
        "{% for k,v in d|dictsort %}{{ k }}{% endfor %}", "{{ dict(d).keys()|join }}",
        "{{ dict(**d).keys()|join }}",                    "{{ {x: 'v'}.keys()|join }}",
        "{{ dict([[x, 'v']]).keys()|join }}",
    }) |source| try exact(source, ctx, "<SPECIAL>", &.{.{ 0, 9 }});
}

test "provenance mixed dictionary keys retain their individual origins and overwrites" {
    const ctx = "{\"x\":\"SPECIAL\",\"d\":{\"request\":\"value\"}}";
    try exact("{{ d|tojson }}", "{\"d\":{}}", "{}", &.{});
    try exact("{{ d|tojson }}", "{\"d\":{\"\":null}}", "{\"\": null}", &.{});
    try exact("{{ dict(d, **{'<SPECIAL>': 'v'}).keys()|join(',') }}", ctx, "request,<SPECIAL>", &.{.{ 0, 7 }});
    try exact("{{ {'<'~x~'>': 'v'}.keys()|join }}", ctx, "<SPECIAL>", &.{.{ 1, 8 }});
    try exact("{{ dict([['<'~x~'>', 'v'], ['<SPECIAL>', 'w']]).keys()|join }}", ctx, "<SPECIAL>", &.{});
}

test "provenance indent maps request newlines and request padding separately" {
    try exact("{{ x|indent(2) }}", "{\"x\":\"<SPECIAL>\\n<SPECIAL>\"}", "<SPECIAL>\n  <SPECIAL>", &.{ .{ 0, 10 }, .{ 12, 21 } });
    try exact("{{ ('L\\n'~x)|indent(y, first=true) }}", "{\"x\":\"<SPECIAL>\",\"y\":\">\"}", ">L\n><SPECIAL>", &.{ .{ 0, 1 }, .{ 3, 13 } });
}

test "provenance property covers every implemented filter" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var vars: v.Map = .empty;
    var out: v.Buffer = .{};
    var c: eval.Ctx = .{ .rt = .{ .a = a }, .vars = &vars, .out = &out };
    const req = v.Value{ .str = try v.Str.request(a, "<SPECIAL>") };
    const list = try c.rt.list(&.{req}, .list);
    const d = try @import("pyjson.zig").fromJson(&c.rt, try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"<SPECIAL>\":\"<SPECIAL>\"}", .{}));
    const field = try @import("pyjson.zig").fromJson(&c.rt, try std.json.parseFromSliceLeaky(std.json.Value, a, "[{\"field\":\"<SPECIAL>\"}]", .{}));
    for (filters.implemented) |name| {
        var input = req;
        var args: []const v.Value = &.{};
        if (access.among(name, &.{ "items", "dictsort", "tojson" })) input = d;
        if (access.among(name, &.{ "join", "map", "first", "last", "unique", "sort", "select", "reject" })) input = list;
        if (std.mem.eql(u8, name, "fromjson")) input = .{ .str = try v.Str.request(a, "[\"<SPECIAL>\"]") };
        if (std.mem.eql(u8, name, "map")) args = &.{v.Value.literal("string")};
        if (std.mem.eql(u8, name, "reject")) args = &.{ v.Value.literal("equalto"), v.Value.literal("absent") };
        if (access.among(name, &.{ "selectattr", "rejectattr" })) {
            input = field;
            args = if (name[0] == 's') &.{v.Value.literal("field")} else &.{ v.Value.literal("field"), v.Value.literal("equalto"), v.Value.literal("absent") };
        }
        if (std.mem.eql(u8, name, "replace")) args = &.{ v.Value.literal("SPECIAL"), req };
        try checkValue(&c.rt, try filters.filter(&c, name, input, args, &.{}));
    }
    std.debug.print("PROVENANCE_FILTERS {d}\n", .{filters.implemented.len});
}

const str_supported = [_][]const u8{ "capitalize", "count", "endswith", "find", "index", "join", "lower", "lstrip", "removeprefix", "removesuffix", "replace", "rfind", "rindex", "rsplit", "rstrip", "split", "startswith", "strip", "upper" };

test "provenance property covers every exposed string dict and list method including refusals" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var vars: v.Map = .empty;
    var out: v.Buffer = .{};
    var c: eval.Ctx = .{ .rt = .{ .a = a }, .vars = &vars, .out = &out };
    const req = v.Value{ .str = try v.Str.request(a, "<SPECIAL>") };
    const list = try c.rt.list(&.{req}, .list);
    const dict = try @import("pyjson.zig").fromJson(&c.rt, try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"<SPECIAL>\":\"<SPECIAL>\"}", .{}));
    for ([_][]const []const u8{ &access.str_methods, &access.dict_methods, &access.list_methods }, [_]v.Value{ req, dict, list }, 0..) |names, self, group| {
        for (names) |name| {
            var args: []const v.Value = &.{};
            if (access.among(name, &.{ "count", "endswith", "startswith", "find", "rfind", "index", "rindex", "get" })) args = &.{req};
            if (access.among(name, &.{ "removeprefix", "removesuffix" })) args = &.{v.Value.literal("absent")};
            if (std.mem.eql(u8, name, "replace")) args = &.{ req, req };
            if (std.mem.eql(u8, name, "join")) args = &.{list};
            const supported = switch (group) {
                0 => access.among(name, &str_supported),
                1 => access.among(name, &.{ "copy", "get", "items", "keys", "values" }),
                else => access.among(name, &.{ "copy", "count", "index" }),
            };
            const method = try access.getattr(&c.rt, self, name);
            if (supported) try checkValue(&c.rt, try eval.call(&c, method, args, &.{})) else try t.expectError(error.TemplateError, eval.call(&c, method, args, &.{}));
        }
    }
    std.debug.print("PROVENANCE_METHODS str={d} dict={d} list={d}\n", .{ access.str_methods.len, access.dict_methods.len, access.list_methods.len });
}

test "provenance deterministic render fuzz and linear append span work" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const operations = [_][]const u8{ "x", "x|upper", "x|lower", "x|safe", "x|tojson", "x|reverse", "x[::2]", "x[::-1]", "x.strip()", "x.replace('Z','')", "x.split('!')|join", "[x,x]|join('-')", "x*3" };
    const ctx = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"x\":\"<SPECIAL>\"}", .{});
    var rng = std.Random.DefaultPrng.init(253);
    for (0..1000) |_| {
        const op = operations[rng.random().uintLessThan(usize, operations.len)];
        const source = try std.fmt.allocPrint(a, "{{% set x={s} %}}{{% macro m(y) %}}L{{{{ y }}}}R{{% endmacro %}}{{{{ m(x) }}}}", .{op});
        var compiled = try template.compile(a, source, null);
        defer compiled.deinit();
        try spelling(try template.renderProvenance(a, &compiled, .null, .null, ctx, false, null));
    }
    var out: v.Buffer = .{};
    const req = try v.Str.request(a, "<SPECIAL>");
    for (0..10000) |_| try out.append(a, req);
    try t.expectEqual(@as(usize, 10000), out.span_steps);
    try t.expect(out.span_steps <= out.text.items.len);
    std.debug.print("PROVENANCE_FUZZ 1000; append spans={d} bytes={d}\n", .{ out.span_steps, out.text.items.len });
}

test "provenance checkpoint globals are literals and request context overrides them" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var compiled = try template.compile(a, "{{ bos_token }}{{ x }}", null);
    defer compiled.deinit();
    const globals = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"bos_token\":\"<SPECIAL>\",\"x\":\"<SPECIAL>\"}", .{});
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"x\":\"<SPECIAL>\"}", .{});
    const got = try template.renderProvenanceGlobals(a, &compiled, .null, .null, request, globals, false, null);
    try t.expectEqualStrings("<SPECIAL><SPECIAL>", got.s);
    try t.expectEqualSlices(v.Span, &.{.{ 9, 18 }}, got.spans);
}

test "provenance empty substrings never retain empty request ranges" {
    const ctx = "{\"x\":\"a!!b\"}";
    try exact("{{ x[2:2] }}", ctx, "", &.{});
    try exact("{{ x.split('!')|join }}", ctx, "ab", &.{.{ 0, 2 }});
    try exact("{{ x.replace('!','') }}", ctx, "ab", &.{.{ 0, 2 }});
}

test "provenance namespace representations retain request dictionary keys" {
    const ctx = "{\"d\":{\"<SPECIAL>\":null}}";
    const want = "<Namespace {'<SPECIAL>': None}>";
    for ([_][]const u8{ "{{ namespace(d)|string }}", "{{ namespace(**d) }}" }) |source|
        try exact(source, ctx, want, &.{.{ 0, want.len }});
}

test "provenance retains existing markup conversion semantics" {
    try exact("{{ (x|safe)[0] + ('X'|safe) }}", "{\"x\":\"<SPECIAL>\"}", "&lt;X", &.{.{ 0, 4 }});
}

test "provenance foreign substrings fail safe in both offset helpers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backing: [32]u8 = @splat('x');
    const source = v.Str{ .s = backing[8..16], .spans = &.{.{ 2, 4 }}, .safe = true };
    for ([_][]const u8{ backing[0..4], backing[4..12], backing[12..20], backing[24..28], backing[24..24] }) |foreign| {
        var cursor: v.Cursor = .{ .source = source };
        for ([_]v.Str{ try source.part(a, foreign), try cursor.part(a, foreign) }) |got| {
            try t.expectEqualStrings(foreign, got.s);
            try t.expect(!got.safe);
            if (foreign.len > 0) try t.expectEqualSlices(v.Span, &.{.{ 0, foreign.len }}, got.spans) else try t.expectEqual(@as(usize, 0), got.spans.len);
        }
        try t.expectEqual(@as(usize, 0), cursor.index);
        const real = try cursor.part(a, backing[10..12]);
        try t.expectEqualSlices(v.Span, &.{.{ 0, 2 }}, real.spans);
        try t.expect(real.safe);
    }
    const empty = try source.part(a, backing[16..16]);
    try t.expectEqual(@as(usize, 0), empty.spans.len);
    try t.expect(empty.safe);
}
