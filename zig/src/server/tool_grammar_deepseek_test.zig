//! DSML required-call grammar, actual template probing, filtered tools, and the production reply parser.
const std = @import("std");
const json = @import("json");
const grammar = @import("tool_grammar.zig");
const probe = @import("tool_grammar_deepseek.zig");
const specs = @import("tool_specs.zig");
const parse = @import("tool_parse.zig");
const errors = @import("errors.zig");
const chat = @import("dsv41_chat");
const template = if (@hasDecl(chat, "template")) chat.template else chat;
const Value = json.Value;

const offered =
    \\[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string"},"n":{"type":"integer"}},"required":["q","n"]}}},{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}]
;

fn value(a: std.mem.Allocator, text: []const u8) !Value {
    return (try json.parse(a, text)).ok;
}

fn has(text: []const u8, expected: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, text, expected) != null);
}

test "required tools: DeepSeek style comes from the rendered assistant call and tolerates tag blanks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const messages = try value(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function","function":{"name":"tfprobe_fn","arguments":{"tfprobe_k":"v"}}}]}]
    );
    var why: template.Problem = .{};
    const rendered = try template.render(a, messages, .{ .thinking = false, .add_generation_prompt = false }, &why);
    try std.testing.expectEqual(grammar.Style.deepseek, grammar.style(rendered).?);
    try std.testing.expect(probe.isStyle("<｜DSML｜function_calls>\n<｜DSML｜ invoke\tname = \"tfprobe_fn\" >\n<｜DSML｜ parameter name = \"tfprobe_k\" string=\"true\">v</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜function_calls>", "tfprobe_fn", "tfprobe_k"));
    try std.testing.expect(!probe.isStyle("<tool_call>{\"name\":\"tfprobe_fn\",\"arguments\":{\"tfprobe_k\":\"v\"}}</tool_call>", "tfprobe_fn", "tfprobe_k"));
    try std.testing.expect(!probe.isStyle("<｜DSML｜ calls><｜DSML｜ invoke name=\"other\"><｜DSML｜ parameter name=\"tfprobe_k\">v", "tfprobe_fn", "tfprobe_k"));
}

test "required tools: DeepSeek required and named schemas keep one envelope and parallel policy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try value(a, offered);
    for ([_]bool{ false, true }) |parallel| {
        const text = try grammar.build(a, .deepseek, tools.array, parallel);
        const root = try value(a, text);
        try std.testing.expectEqualStrings("structural_tag", root.strField("type").?);
        const parts = root.get("format").?.get("elements").?.array;
        try std.testing.expectEqual(@as(usize, if (parallel) 8 else 7), parts.len);
        try std.testing.expectEqualStrings("<｜DSML｜function_calls>", parts[1].strField("value").?);
        try std.testing.expectEqualStrings("</｜DSML｜function_calls>", parts[parts.len - 2].strField("value").?);
        const choices = parts[3].get("elements").?.array;
        try std.testing.expectEqual(@as(usize, 2), choices.len);
        const args = choices[0].get("elements").?.array[2];
        try std.testing.expectEqualStrings("deepseek_v4_1_xml", args.strField("style").?);
        try std.testing.expectEqualStrings("q", args.get("json_schema").?.get("required").?.array[0].string);
        try has(text, "<｜DSML｜invoke name=\\\"lookup\\\">");
        try has(text, "<｜DSML｜invoke name=\\\"now\\\">");
        try std.testing.expectEqual(parallel, std.mem.indexOf(u8, text, "\"type\":\"star\"") != null);
    }
    var cx: errors.Cx = .{ .a = a };
    const named = try specs.active(&cx, tools, try value(a, "{\"type\":\"function\",\"function\":{\"name\":\"now\"}}"));
    const only = try grammar.build(a, .deepseek, named, false);
    try has(only, "<｜DSML｜invoke name=\\\"now\\\">");
    try std.testing.expect(std.mem.indexOf(u8, only, "lookup") == null);
    try std.testing.expect(std.mem.indexOf(u8, only, "\"type\":\"or\"") == null);
}

const lookup = "<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">hello</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜invoke>";
const now = "<｜DSML｜invoke name=\"now\"></｜DSML｜invoke>";

test "required tools: DeepSeek grammar shaped single and parallel replies reach the real parser" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    for ([_][]const u8{ lookup, lookup ++ "\n" ++ now }) |calls| {
        const text = try std.mem.concat(a, u8, &.{ "\n<｜DSML｜function_calls>\n", calls, "\n</｜DSML｜function_calls>\n" });
        const result = try parse.parse(a, text, tools, null);
        try std.testing.expectEqualStrings("", result.content);
        try std.testing.expectEqual(@as(usize, if (calls.len == lookup.len) 1 else 2), result.calls.?.len);
        const first = result.calls.?[0].get("function").?;
        try std.testing.expectEqualStrings("lookup", first.strField("name").?);
        const args = try value(a, first.strField("arguments").?);
        try std.testing.expectEqualStrings("hello", args.strField("q").?);
        try std.testing.expectEqual(@as(?i64, 3), args.get("n").?.int64());
        if (result.calls.?.len > 1) try std.testing.expectEqualStrings("now", result.calls.?[1].get("function").?.strField("name").?);
        var streamer = try @import("tool_stream.zig").Streamer.init(a, tools);
        var deltas: std.ArrayList(Value) = .empty;
        for (1..text.len + 1) |end| try streamer.feed(text[0..end], &deltas);
        var sent: std.ArrayList(u8) = .empty;
        var expected: std.ArrayList(u8) = .empty;
        for (result.calls.?) |call| try expected.appendSlice(a, call.get("function").?.strField("arguments").?);
        for (deltas.items) |delta| for (delta.get("tool_calls").?.array) |call| {
            try sent.appendSlice(a, call.get("function").?.strField("arguments").?);
        };
        try std.testing.expectEqualStrings(expected.items, sent.items);
    }
    const named = try parse.parse(a, "<｜DSML｜function_calls>" ++ now ++ "</｜DSML｜function_calls>", tools[1..], null);
    try std.testing.expectEqual(@as(usize, 1), named.calls.?.len);
}
