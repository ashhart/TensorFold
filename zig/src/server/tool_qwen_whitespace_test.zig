//! Qwen XML string bytes must survive both full parsing and every streaming split.
const std = @import("std");
const json = @import("json");
const parse = @import("tool_parse.zig");
const stream = @import("tool_stream.zig");
const A = std.mem.Allocator;
const V = json.Value;
const source = "export function caption() {\n  return \"GPU \\ shaders: π 🚀\";\n}\n";

fn tools(a: A, value: []const u8) ![]const V {
    return schemaTools(a, try std.fmt.allocPrint(a, "{{\"type\":\"string\",\"enum\":[{s}]}}", .{try json.quote(a, value, .{ .ascii = false })}), "");
}

fn schemaTools(a: A, schema: []const u8, extra: []const u8) ![]const V {
    const text = try std.fmt.allocPrint(a, "[{{\"type\":\"function\",\"function\":{{\"name\":\"write_file\",\"parameters\":{{\"type\":\"object\",\"properties\":{{\"content\":{s}}}{s}}}}}}}]", .{ schema, extra });
    return (try json.parse(a, text)).ok.array;
}

fn content(a: A, arguments: []const u8) ![]const u8 {
    return (try json.parse(a, arguments)).ok.get("content").?.string;
}

fn full(a: A, text: []const u8, offered: []const V) ![]const u8 {
    const result = try parse.parse(a, text, offered, 1);
    try std.testing.expect(result.calls != null and result.calls.?.len == 1);
    return content(a, result.calls.?[0].get("function").?.get("arguments").?.string);
}

fn argumentText(a: A, out: []const V) ![]const u8 {
    var args: std.ArrayList(u8) = .empty;
    for (out) |delta| for (delta.get("tool_calls").?.array) |call| {
        if (call.get("function").?.get("arguments")) |value| try args.appendSlice(a, value.string);
    };
    return args.items;
}

fn streamed(a: A, text: []const u8, offered: []const V, split: ?usize) ![]const u8 {
    var state = try stream.Streamer.init(a, offered);
    var out: std.ArrayList(V) = .empty;
    if (split) |at| {
        try state.feed(text[0..at], &out);
        try state.feed(text, &out);
    } else for (1..text.len + 1) |end| try state.feed(text[0..end], &out);
    return content(a, try argumentText(a, out.items));
}

fn wire(a: A, value: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "<tool_call>\n<function=write_file>\n<parameter=content>", value, "</parameter>\n</function>\n</tool_call>" });
}

fn expectAllSplits(a: A, text: []const u8, offered: []const V, expected: []const u8) !void {
    try std.testing.expectEqualStrings(expected, try full(a, text, offered));
    try std.testing.expectEqualStrings(expected, try streamed(a, text, offered, null));
    for (0..text.len + 1) |split|
        try std.testing.expectEqualStrings(expected, try streamed(a, text, offered, split));
}

test "Qwen string bytes: captured source enum keeps its final newline in full and stream" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, source);
    const text = try std.fmt.allocPrint(a, "<tool_call>\n<function=write_file>\n<parameter=path>\nsample.js\n</parameter>\n<parameter=content>\n{s}</parameter>\n</function>\n</tool_call>", .{source});
    try expectAllSplits(a, text, offered, source);
}

test "Qwen string bytes: literal leading trailing blank CRLF and code whitespace survive all splits" {
    const Case = struct { value: []const u8, raw: []const u8 };
    const cases = [_]Case{
        .{ .value = "\nfirst", .raw = "\nfirst" },
        .{ .value = "\n\nfirst\n", .raw = "\n\n\nfirst\n\n" },
        .{ .value = "last\n", .raw = "\nlast\n" },
        .{ .value = "last\n\n", .raw = "\nlast\n\n" },
        .{ .value = "\n", .raw = "\n" },
        .{ .value = "\r\n", .raw = "\r\n" },
        .{ .value = "last\r\n", .raw = "\r\nlast\r\n" },
        .{ .value = "\r\n\tcode\r\n", .raw = "\r\n\r\n\tcode\r\n\r\n" },
        .{ .value = " \t return x;\t ", .raw = "\n \t return x;\t \n" },
        .{ .value = "\t ", .raw = "\t " },
        .{ .value = "", .raw = "\n\n" },
    };
    for (cases) |c| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        try expectAllSplits(a, try wire(a, c.raw), try tools(a, c.value), c.value);
    }
}

test "Qwen string bytes: ordinary LF and CRLF framing still strips exactly one boundary" {
    for ([_][2][]const u8{
        .{ "\ntext\n", "text" },           .{ "\r\ntext\r\n", "text" },
        .{ "\n\ntext\n\n", "\ntext\n" },   .{ "\r\n\r\ntext\r\n\r\n", "\r\ntext\r\n" },
        .{ "\n  code\t \n", "  code\t " },
    }) |c| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        try expectAllSplits(a, try wire(a, c[0]), try schemaTools(a, "{\"type\":\"string\"}", ""), c[1]);
    }
}

test "Qwen string bytes: const references and combiners disambiguate without inventing a value" {
    const schemas = [_][2][]const u8{
        .{ "{\"type\":\"string\",\"const\":\"tail\\n\"}", "" },
        .{ "{\"$ref\":\"#/$defs/content\"}", ",\"$defs\":{\"content\":{\"enum\":[\"tail\\n\"]}}" },
        .{ "{\"allOf\":[{\"type\":\"string\"},{\"enum\":[\"tail\\n\",\"other\"]},{\"const\":\"tail\\n\"}]}", "" },
        .{ "{\"oneOf\":[{\"const\":\"tail\\n\"},{\"const\":\"other\"}]}", "" },
    };
    for (schemas) |schema| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const offered = try schemaTools(a, schema[0], schema[1]);
        try expectAllSplits(a, try wire(a, "\ntail\n"), offered, "tail\n");
        try expectAllSplits(a, try wire(a, "\nwrong\n"), offered, "wrong");
    }
}

test "Qwen string bytes: truncated and malformed parameters never become complete calls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, source);
    const first = "<tool_call><function=write_file><parameter=content>\n";
    const text = try std.mem.concat(a, u8, &.{ first, source });
    for (first.len..text.len + 1) |at| {
        const partial = text[0..at];
        try std.testing.expect((try parse.parse(a, partial, offered, 1)).calls == null);
        var state = try stream.Streamer.init(a, offered);
        var out: std.ArrayList(V) = .empty;
        try state.finishCut(partial, &out);
        try std.testing.expect(state.open());
        try std.testing.expect((try @import("tool_params.zig").closedJson(a, try argumentText(a, out.items))) == null);
    }
    const malformed = try std.mem.concat(a, u8, &.{ text, "</function></tool_call>" });
    const result = try parse.parse(a, malformed, offered, 1);
    try std.testing.expect(result.calls == null and result.rejected_envelopes == 1);
}

test "Qwen string bytes: streaming alone preserves a source literal before full parsing is checked" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, source);
    const text = try wire(a, try std.mem.concat(a, u8, &.{ "\n", source }));
    try std.testing.expectEqualStrings(source, try streamed(a, text, offered, null));
    for (0..text.len + 1) |split|
        try std.testing.expectEqualStrings(source, try streamed(a, text, offered, split));
}

test "Qwen string bytes: anyOf and escaped references preserve literals without replacing wrong text" {
    const cases = [_][2][]const u8{
        .{ "{\"anyOf\":[{\"enum\":[\"tail\\n\"]},{\"const\":\"other\"}]}", "" },
        .{ "{\"$ref\":\"#/$defs/a~1b\"}", ",\"$defs\":{\"a/b\":{\"const\":\"tail\\n\"}}" },
        .{ "{\"allOf\":[{\"type\":\"string\"},{\"$ref\":\"#/$defs/a~0b\"}]}", ",\"$defs\":{\"a~b\":{\"enum\":[\"tail\\n\"]}}" },
    };
    for (cases) |schema| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const offered = try schemaTools(a, schema[0], schema[1]);
        try expectAllSplits(a, try wire(a, "\ntail\n"), offered, "tail\n");
        try expectAllSplits(a, try wire(a, "\nwrong\n"), offered, "wrong");
    }
}
