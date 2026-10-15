//! The tool_parse tests, as their own test binary: whole-reply parses and the close-call helpers.
const std = @import("std");
const json = @import("json.zig");
const tool_parse = @import("tool_parse.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;
const repairGlmKeys = tool_parse.repairGlmKeys;

test "families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", "{\"q\":\"x\"}" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>", "{\"n\":5}" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", "{\"n\":7}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>hi, there<|\"|>,n:3}<tool_call|>", "{\"q\":\"hi, there\",\"n\":3}" },
        .{ "{\"tool\":\"lookup\",\"query\":\"ping\"}", "{\"query\":\"ping\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    const kept = try parse(a, "hi <tool_call>{\"name\":\"other tool\"}</tool_call>", tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings("hi <tool_call>{\"name\":\"other tool\"}</tool_call>", kept.content);
}

test "a call the end token left open closes when it parses whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, the call's arguments, the content left
        .{ "Looking.<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n", "</tool_call>", "{\"q\":\"x\"}", "Looking." },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":[\"x\"", "]}}</tool_call>", "{\"q\":[\"x\"]}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>\n<function=lookup>\n", "</function></tool_call>", "{}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", "{\"n\":7}", "" },
        .{ "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", "</arg_value></tool_call>", "{\"n\":7}", "Trying." }, // a GLM value left open (#285)
        .{ "<tool_call>{\"name\":\"lookup\"}</tool_call>\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}", "}</tool_call>", "{\"n\":2}", "" },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>\"}", "}</tool_call>", "{\"q\":\"<tool_call>\"}", "" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings(c[3], r.content);
    }
}

test "a call left open that would not parse whole stays the reply's text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const replies = [_][]const u8{
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"fast ca", // inside a string
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", // a key without its value
        "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca", // inside a parameter
        "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n", // before its </parameter>
        "Trying.<tool_call>{\"name\":\"launch rocket!\",\"arguments\":{}}", // no function name
        "Trying.<tool_call>", // nothing written
        "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}", // an earlier opener would take the closer
    };
    for (replies) |text| {
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
}

test "replies that end outside a call close nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const replies = [_][]const u8{
        "Done.",
        "<tool_call>{\"name\":\"lookup\"}</tool_call>",
        "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
    };
    for (replies) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

test "a call to a tool the request did not declare stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const whole = [_][]const u8{
        "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings("Launching.", (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}

test "a GLM call's keys without values read in linear time" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    var ns: [2]u64 = undefined;
    for ([_]usize{ 800, 6400 }, &ns) |n, *best| {
        var reply: std.ArrayList(u8) = .empty;
        try reply.appendSlice(a, "<tool_call>lookup");
        for (0..n) |_| try reply.appendSlice(a, "<arg_key>n</arg_key>");
        try reply.appendSlice(a, "</tool_call>");
        best.* = std.math.maxInt(u64);
        for (0..3) |_| {
            var run: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer run.deinit();
            const start = std.Io.Clock.awake.now(std.testing.io).nanoseconds;
            const r = try parse(run.allocator(), reply.items, tools, null);
            best.* = @min(best.*, @as(u64, @intCast(std.Io.Clock.awake.now(std.testing.io).nanoseconds - start)));
            try std.testing.expectEqualStrings("{}", r.calls.?[0].get("function").?.get("arguments").?.string);
        }
    }
    // 8x the keys: near 8x the time, not 64x
    try std.testing.expect(ns[1] <= 20 * std.time.ns_per_ms or ns[1] <= 24 * ns[0]);
}

test "GLM calls keep a declared name with spaces, dots, colons or unicode; an undeclared one stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"Get Classes List\"}},{\"type\":\"function\",\"function\":{\"name\":\"search.web\"}},{\"type\":\"function\",\"function\":{\"name\":\"ns:lookup\"}},{\"type\":\"function\",\"function\":{\"name\":\"Caf\u{e9} Lookup\"}}]")).ok.array;
    const cases = [_][3][]const u8{
        .{ "<tool_call>Get Classes List<arg_key>class_type</arg_key><arg_value>ACTIVE</arg_value></tool_call>", "Get Classes List", "{\"class_type\":\"ACTIVE\"}" },
        .{ "<tool_call>get classes list<arg_key>n</arg_key><arg_value>2</arg_value></tool_call>", "Get Classes List", "{\"n\":\"2\"}" },
        .{ "<tool_call>search.web<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>", "search.web", "{\"q\":\"x\"}" },
        .{ "<tool_call>ns:lookup<arg_key>q</arg_key><arg_value>y</arg_value></tool_call>", "ns:lookup", "{\"q\":\"y\"}" },
        .{ "<tool_call>Caf\u{e9} Lookup<arg_key>q</arg_key><arg_value>z</arg_value></tool_call>", "Caf\u{e9} Lookup", "{\"q\":\"z\"}" },
    };
    for (cases) |c| {
        for ([_]?usize{ null, 1 }) |max_calls| {
            const r = try parse(a, c[0], tools, max_calls);
            const f = r.calls.?[0].get("function").?;
            try std.testing.expectEqualStrings(c[1], f.get("name").?.string);
            try std.testing.expectEqualStrings(c[2], f.get("arguments").?.string);
            try std.testing.expectEqualStrings("", r.content);
        }
    }
    const open = "<tool_call>Get Classes List<arg_key>class_type</arg_key><arg_value>ACTIVE</arg_value>";
    try std.testing.expectEqualStrings("</tool_call>", try closeCall(a, open, tools));
    const other = "<tool_call>Get Other List<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>";
    const kept = try parse(a, other, tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings(other, kept.content);
    const prose = "<tool_call>two\nlines<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>";
    try std.testing.expect((try parse(a, prose, tools, null)).calls == null);
}

const dsml_prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
    "<｜DSML｜parameter name=\"q\" string=\"true\">";
const dsml_suffix = "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
const nested_calls = [_][]const u8{
    "<tool_call><function=lookup></function></tool_call>",
    "<tool_call><function=lookup></function>",
    "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>",
    "<tool_call>{\"name\":\"lookup\",\"arguments\":{}",
};

test "DSML parameters contain complete and unfinished legacy calls without consuming adjacent calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const adjacent = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":7}}</tool_call>";
    for (nested_calls) |literal| {
        const text = try std.mem.concat(a, u8, &.{ "Before.", dsml_prefix, literal, dsml_suffix, adjacent, "After." });
        for ([_]?usize{ null, 1 }) |quota| {
            const parsed = try parse(a, text, tools, quota);
            try std.testing.expectEqual(@as(usize, if (quota == null) 2 else 1), parsed.calls.?.len);
            const outer = parsed.calls.?[0].get("function").?;
            try std.testing.expectEqualStrings("lookup", outer.strField("name").?);
            const arguments = (try json.parse(a, outer.strField("arguments").?)).ok;
            try std.testing.expectEqualStrings(literal, arguments.strField("q").?);
            if (quota == null) {
                try std.testing.expectEqualStrings("{\"n\":7}", parsed.calls.?[1].get("function").?.strField("arguments").?);
            }
            try std.testing.expectEqualStrings("Before.After.", parsed.content);
            try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        }
    }
}

test "an unfinished DSML envelope never parses or autocompletes its nested legacy call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    for (nested_calls) |literal| {
        const text = try std.mem.concat(a, u8, &.{ "Before.", dsml_prefix, literal });
        for ([_]?usize{ null, 1 }) |quota| {
            const parsed = try parse(a, text, tools, quota);
            try std.testing.expect(parsed.calls == null);
            try std.testing.expectEqualStrings(text, parsed.content);
        }
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
    }
}

test "closeCall skips a closed DSML parameter's unfinished legacy opener and completes the adjacent call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const text = dsml_prefix ++ "<tool_call><function=lookup></function>" ++ dsml_suffix ++
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":7}";
    const tail = try closeCall(a, text, tools);
    try std.testing.expectEqualStrings("}</tool_call>", tail);
    const parsed = try parse(a, try std.mem.concat(a, u8, &.{ text, tail }), tools, null);
    try std.testing.expectEqual(@as(usize, 2), parsed.calls.?.len);
    try std.testing.expectEqualStrings("{\"n\":7}", parsed.calls.?[1].get("function").?.strField("arguments").?);
}

test "bare JSON calls keep literal unfinished DSML markers as argument data" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const call = "{\"name\":\"lookup\",\"arguments\":{\"q\":\"<｜DSML｜tool_calls>\"}}";
    const inputs = [_][]const u8{ call, "[" ++ call ++ "]", "```json\n" ++ call ++ "\n```" };
    for (inputs) |text| for ([_]?usize{ null, 1 }) |quota| {
        const parsed = try parse(a, text, tools, quota);
        try std.testing.expectEqual(@as(usize, 1), parsed.calls.?.len);
        const arguments = (try json.parse(a, parsed.calls.?[0].get("function").?.strField("arguments").?)).ok;
        try std.testing.expectEqualStrings("<｜DSML｜tool_calls>", arguments.strField("q").?);
        try std.testing.expectEqualStrings("", parsed.content);
    };
}

test "a GLM call whose <arg_key> the model wrote as whitespace gets it back" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const repaired = [_][2][]const u8{ // the block as written, with its keys put back
        .{ "read_file path</arg_key><arg_value>/x</arg_value>", "read_file <arg_key>path</arg_key><arg_value>/x</arg_value>" },
        .{ "read_file\npath</arg_key><arg_value>/x</arg_value>", "read_file\n<arg_key>path</arg_key><arg_value>/x</arg_value>" },
        .{ "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\nlimit</arg_key><arg_value>3</arg_value>", "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\n<arg_key>limit</arg_key><arg_value>3</arg_value>" },
        .{ "read_file path</arg_key><arg_value>/x</arg_value> limit</arg_key><arg_value>3</arg_value>", "read_file <arg_key>path</arg_key><arg_value>/x</arg_value> <arg_key>limit</arg_key><arg_value>3</arg_value>" },
    };
    for (repaired) |c| {
        try std.testing.expectEqualStrings(c[1], try repairGlmKeys(a, c[0]));
        try std.testing.expectEqualStrings(c[1], try repairGlmKeys(a, c[1]));
    }
    const unchanged = [_][]const u8{
        "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>", // nothing missing
        "read_file",
        "read_filepath</arg_key><arg_value>/x</arg_value>", // no space: no name left before the key
        " path</arg_key><arg_value>/x</arg_value>", // no name at all
        "{\"name\": \"run\", \"arguments\": {\"command\": \"ls\"}}",
    };
    for (unchanged) |b| try std.testing.expectEqualStrings(b, try repairGlmKeys(a, b));
}

test "a repaired GLM call parses, in both modes and once the end token closed it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"properties\":{\"filePath\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}]")).ok.array;
    // before: the first came back as text, the second lost its second argument
    const cases = [_][2][]const u8{
        .{ "<tool_call>read_file filePath</arg_key><arg_value>/home/a/main.go</arg_value></tool_call>", "{\"filePath\":\"/home/a/main.go\"}" },
        .{ "<tool_call>read_file<arg_key>filePath</arg_key><arg_value>/a</arg_value>\nlimit</arg_key><arg_value>3</arg_value></tool_call>", "{\"filePath\":\"/a\",\"limit\":3}" },
    };
    for (cases) |c| for ([_]?usize{ null, 1 }) |max_calls| {
        const r = try parse(a, c[0], tools, max_calls);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings("", r.content);
    };
    const open = "<tool_call>read_file filePath</arg_key><arg_value>/x</arg_value>";
    try std.testing.expectEqualStrings("</tool_call>", try closeCall(a, open, tools));
}

test "a GLM call the end token left open closes as #285 closed it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"read_file","parameters":{"properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}},
        \\ {"type":"function","function":{"name":"run","parameters":{"properties":{"command":{"type":"string"}}}}},
        \\ {"type":"function","function":{"name":"now","parameters":{"properties":{}}}}]
    )).ok.array;
    const closes = [_][3][]const u8{ // the reply, what closes it, the tool the last call names (tests/test_glm_tool_call_end.py)
        .{ "<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>", "</tool_call>", "read_file" },
        .{ "<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", "</arg_value></tool_call>", "run" },
        .{ "Let me look:<tool_call>now", "</tool_call>", "now" },
        .{ "<tool_call>now</tool_call>\n<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>\n", "</tool_call>", "read_file" },
        .{ "<tool_call>read_file path</arg_key><arg_value>x", "</arg_value></tool_call>", "read_file" }, // and repaired
    };
    for (closes) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("name").?.string);
    }
    const left_open = [_][]const u8{
        "no call at all",
        "<tool_call>now</tool_call>", // closed already
        "<tool_call>launch<arg_key>x</arg_key><arg_value>1", // a tool the client did not offer
        "<tool_call>run<arg_key>comm", // a key with no value
        "<tool_call>run<arg_key>command</arg_key>", // ... or no value yet
        "<tool_call>run command=ls", // not GLM's markup
        "<tool_call>", // nothing written
        "<tool_call>now</think>The answer.", // the think block went on past it
        "<tool_call>run<arg_key>command</arg_key><arg_value>ls</arg_value> and then", // text after an argument
    };
    for (left_open) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}
