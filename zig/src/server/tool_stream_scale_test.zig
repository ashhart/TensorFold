const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const Streamer = @import("tool_stream.zig").Streamer;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const Call = struct {
    name: []const u8 = "",
    args: std.ArrayList(u8) = .empty,
};

fn offered(a: Allocator) Allocator.Error![]Value {
    return (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"q\":{\"type\":\"string\"}}}}},{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
}

fn assemble(a: Allocator, deltas: []const Value) ![]Call {
    var calls: std.ArrayList(Call) = .empty;
    for (deltas) |delta| for (delta.get("tool_calls").?.array) |one| {
        const index: usize = @intCast(one.get("index").?.int64().?);
        const function = one.get("function").?;
        if (one.get("id")) |_| {
            try std.testing.expectEqual(calls.items.len, index);
            try calls.append(a, .{ .name = function.strField("name").? });
        } else {
            try std.testing.expect(index < calls.items.len);
        }
        try calls.items[index].args.appendSlice(a, function.strField("arguments").?);
    };
    return calls.items;
}

fn expectCalls(expected: []const Value, actual: []const Call) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        const function = want.get("function").?;
        try std.testing.expectEqualStrings(function.strField("name").?, got.name);
        try std.testing.expectEqualStrings(function.strField("arguments").?, got.args.items);
    }
}

fn feed(a: Allocator, text: []const u8, tools: []const Value, boundary: ?usize) ![]Call {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    if (boundary) |at| try s.feed(text[0..at], &out);
    try s.feed(text, &out);
    try s.feed(text, &out);
    return assemble(a, out.items);
}

fn feedBytes(a: Allocator, text: []const u8, tools: []const Value) ![]Call {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (1..text.len + 1) |n| try s.feed(text[0..n], &out);
    return assemble(a, out.items);
}

test "tool stream scaling: every split preserves Unicode, escaped delimiters and multiple calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offered(a);
    const text = "prose<tool_call><function=lookup><parameter=q>\n😀é &lt;/parameter&gt; \"quoted\" \\ path  \n</parameter></function></tool_call>between<tool_call><function=write><parameter=n>\n 5 \n</parameter></function></tool_call>";
    const expected = (try tool_parse.parse(a, text, tools, null)).calls.?;
    for (0..text.len + 1) |boundary| try expectCalls(expected, try feed(a, text, tools, boundary));
}

test "tool stream scaling: byte arrivals preserve large late parameter names" {
    for ([_]usize{ 1024, 16 * 1024, 64 * 1024 }) |size| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try offered(a);
        var key: std.ArrayList(u8) = .empty;
        for (0..size) |_| try key.append(a, 'k');
        const text = try std.mem.concat(a, u8, &.{ "<tool_call><function=lookup><parameter=", key.items, ">\n😀é &lt;/parameter&gt; \"quoted\" \\ path</parameter></function></tool_call>" });
        const expected = (try tool_parse.parse(a, text, tools, null)).calls.?;
        try expectCalls(expected, try feedBytes(a, text, tools));
    }
}

test "tool stream scaling: byte arrivals preserve large trailing whitespace across multiple calls" {
    for ([_]usize{ 1024, 16 * 1024, 64 * 1024 }) |size| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try offered(a);
        var text: std.ArrayList(u8) = .empty;
        for (0..4) |_| {
            try text.appendSlice(a, "<tool_call><function=lookup><parameter=q>\n");
            for (0..size / 4) |_| try text.append(a, ' ');
            try text.appendSlice(a, "\n</parameter></function></tool_call>");
        }
        const expected = (try tool_parse.parse(a, text.items, tools, null)).calls.?;
        try expectCalls(expected, try feedBytes(a, text.items, tools));
    }
}

test "tool stream scaling: a short one-pass turn keeps its exact deltas" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offered(a);
    const text = "<tool_call><function=lookup><parameter=q>\n😀 &lt;/parameter&gt;</parameter></function></tool_call>";
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    try s.feed(text, &out);
    try std.testing.expectEqual(@as(usize, 5), out.items.len);
    const expected = [_][]const u8{ "", "{\"q\":\"", "😀 &lt;/parameter&gt;", "\"", "}" };
    for (out.items, expected) |delta, arguments| {
        const one = delta.get("tool_calls").?.array[0];
        try std.testing.expectEqualStrings(arguments, one.get("function").?.strField("arguments").?);
    }
}
