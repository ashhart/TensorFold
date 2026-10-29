//! tool_choice auto with strict tools: text or calls, strict arguments in their schema, the rest free JSON objects.
const std = @import("std");
const json = @import("json");
const grammar = @import("tool_grammar.zig");
const parse = @import("tool_parse.zig");
const Value = json.Value;

const offered =
    \\[{"type":"function","function":{"name":"lookup","strict":true,"parameters":{"type":"object","properties":{"q":{"type":"string"},"n":{"type":"integer"}},"required":["q","n"],"additionalProperties":false}}},{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}}}]
;

fn value(a: std.mem.Allocator, text: []const u8) !Value {
    return (try json.parse(a, text)).ok;
}

/// The json_schema node of one call sequence: the element whose type is json_schema.
fn schemaOf(call: Value) Value {
    for (call.get("elements").?.array) |part| if (part.typeIs("json_schema")) return part.get("json_schema").?;
    unreachable;
}

test "auto tools: a strict tool asks for a grammar, non-strict tools only with the knob, and an explicit false is not strict" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    try std.testing.expect(grammar.autoWanted(tools, false));
    try std.testing.expect(!grammar.autoWanted(tools[1..], false));
    try std.testing.expect(grammar.autoWanted(tools[1..], true));
    try std.testing.expect(!grammar.autoWanted(&.{}, true));
    const off = try value(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"strict\":false}},{\"name\":\"g\",\"strict\":\"true\"}]");
    try std.testing.expect(!grammar.autoWanted(off.array, false));
    try std.testing.expect(grammar.autoWanted((try value(a, "[{\"name\":\"g\",\"strict\":true}]")).array, false));
}

test "auto tools: DeepSeek's optional call block holds the strict tool to its schema and leaves the other free" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    for ([_]bool{ false, true }) |parallel| {
        const root = try value(a, try grammar.optional(a, .deepseek, tools, false, parallel));
        try std.testing.expectEqualStrings("structural_tag", root.strField("type").?);
        const format = root.get("format").?;
        try std.testing.expectEqualStrings("triggered_tags", format.strField("type").?);
        try std.testing.expectEqualStrings("<｜DSML｜function_calls>", format.get("triggers").?.array[0].string);
        try std.testing.expectEqual(!parallel, format.get("stop_after_first").?.bool);
        try std.testing.expect(format.get("at_least_one") == null);
        const tag = format.get("tags").?.array[0];
        try std.testing.expectEqualStrings("<｜DSML｜function_calls>", tag.strField("begin").?);
        try std.testing.expectEqualStrings("</｜DSML｜function_calls>", tag.strField("end").?);
        const parts = tag.get("content").?.get("elements").?.array;
        try std.testing.expectEqual(@as(usize, if (parallel) 4 else 3), parts.len);
        const choices = parts[1].get("elements").?.array;
        try std.testing.expectEqual(@as(usize, 2), choices.len);
        try std.testing.expectEqualStrings("<｜DSML｜invoke name=\"lookup\">", choices[0].get("elements").?.array[0].strField("value").?);
        const strict = schemaOf(choices[0]);
        try std.testing.expectEqualStrings("q", strict.get("required").?.array[0].string);
        try std.testing.expect(!strict.get("additionalProperties").?.bool);
        const free = schemaOf(choices[1]);
        try std.testing.expectEqualStrings("object", free.strField("type").?);
        try std.testing.expect(free.get("properties") == null and free.get("required") == null);
    }
    const held = try value(a, try grammar.optional(a, .deepseek, tools, true, true));
    const choices = held.get("format").?.get("tags").?.array[0].get("content").?.get("elements").?.array[1].get("elements").?.array;
    try std.testing.expectEqualStrings("tz", schemaOf(choices[1]).get("required").?.array[0].string);
}

test "auto tools: the other styles open one call tag at their shared opener and keep each tool's name after it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    const Case = struct { style: grammar.Style, begin: []const u8, end: []const u8, first: ?[]const u8 };
    for ([_]Case{
        .{ .style = .xml, .begin = "<tool_call>", .end = "</tool_call>", .first = null },
        .{ .style = .json, .begin = "<tool_call>", .end = "</tool_call>", .first = null },
        .{ .style = .gemma, .begin = "<|tool_call>", .end = "<tool_call|>", .first = "call:lookup" },
        .{ .style = .glm, .begin = "<tool_call>", .end = "</tool_call>", .first = "lookup" },
    }) |c| {
        const text = try grammar.optional(a, c.style, tools, false, true);
        const format = (try value(a, text)).get("format").?;
        try std.testing.expectEqualStrings("triggered_tags", format.strField("type").?);
        try std.testing.expectEqualStrings(c.begin, format.get("triggers").?.array[0].string);
        const tag = format.get("tags").?.array[0];
        try std.testing.expectEqualStrings(c.begin, tag.strField("begin").?);
        try std.testing.expectEqualStrings(c.end, tag.strField("end").?);
        const bodies = tag.get("content").?.get("elements").?.array;
        try std.testing.expectEqual(@as(usize, 2), bodies.len);
        if (c.first) |first| try std.testing.expectEqualStrings(first, bodies[0].get("elements").?.array[0].strField("value").?);
        const strict = schemaOf(bodies[0]);
        if (c.style == .json) {
            try std.testing.expectEqualStrings("lookup", strict.get("properties").?.get("name").?.strField("const").?);
            try std.testing.expectEqualStrings("q", strict.get("properties").?.get("arguments").?.get("required").?.array[0].string);
            try std.testing.expect(schemaOf(bodies[1]).get("properties").?.get("arguments").?.get("required") == null);
        } else {
            try std.testing.expectEqualStrings("q", strict.get("required").?.array[0].string);
            try std.testing.expect(schemaOf(bodies[1]).get("required") == null);
        }
        try std.testing.expect(std.mem.indexOf(u8, text, "\"value\":\"" ++ "<tool_call>") == null);
    }
    const xml = try grammar.optional(a, .xml, tools[0..1], false, false);
    try std.testing.expect(std.mem.indexOf(u8, xml, "\"value\":\"<function=lookup>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "\"stop_after_first\":true") != null);
}

test "auto tools: required grammars are unchanged and a reply with text before DSML calls still parses" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    const required = try value(a, try grammar.build(a, .deepseek, tools, false));
    try std.testing.expectEqualStrings("sequence", required.get("format").?.strField("type").?);
    const choices = required.get("format").?.get("elements").?.array[3].get("elements").?.array;
    try std.testing.expectEqualStrings("tz", schemaOf(choices[1]).get("required").?.array[0].string);
    const reply = "Let me check.\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">hi</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>";
    const result = try parse.parse(a, reply, tools, null);
    try std.testing.expect(std.mem.indexOf(u8, result.content, "Let me check.") != null);
    try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
    try std.testing.expectEqualStrings("lookup", result.calls.?[0].get("function").?.strField("name").?);
}
