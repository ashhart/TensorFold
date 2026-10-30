//! LFM2 tool calls while they stream: each call of a list is sent as it ends.
const std = @import("std");
const json = @import("json");
const tool_lfm2 = @import("tool_lfm2.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const open = tool_lfm2.open;
const close = tool_lfm2.close;
const dsml = tool_lfm2.dsml;

const test_tools =
    \\[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"filters":{"type":"object"}}}}},
    \\ {"type":"function","function":{"name":"web-search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
;

/// The names a stream has sent once `reply[0..n]` has arrived, and the arguments sent for each so far.
fn sentBy(a: Allocator, reply: []const u8, n: usize, tools: []const Value) ![]const []const u8 {
    var s = try tool_stream.Streamer.init(a, tools);
    var deltas: std.ArrayList(Value) = .empty;
    try s.feed(try dsml(a, reply[0..n], tools, false), &deltas);
    var out: std.ArrayList([]const u8) = .empty;
    for (deltas.items) |d| for (d.get("tool_calls").?.array) |one| {
        const f = one.get("function").?;
        if (f.get("name")) |name| try out.append(a, name.string) else out.items[out.items.len - 1] = try std.mem.concat(a, u8, &.{ out.items[out.items.len - 1], f.get("arguments").?.string });
    };
    return out.items;
}

test "tool_lfm2: each call of a list is sent as soon as it ends, in either spelling" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const python = "Go. " ++ open ++ "[lookup(query=\"fast\", limit=4), web-search(q=\"a)b\"), lookup()]" ++ close;
    const first = std.mem.indexOf(u8, python, "4)").? + 2;
    const second = std.mem.indexOf(u8, python, "b\")").? + 3;
    try std.testing.expectEqual(@as(usize, 0), (try sentBy(a, python, first - 1, tools)).len);
    try std.testing.expectEqual(@as(usize, 1), (try sentBy(a, python, first, tools)).len);
    try std.testing.expectEqual(@as(usize, 1), (try sentBy(a, python, second - 1, tools)).len);
    const two = try sentBy(a, python, second, tools);
    try std.testing.expectEqual(@as(usize, 2), two.len);
    // a call is sent with its arguments whole
    try std.testing.expectEqualStrings("lookup{\"query\":\"fast\",\"limit\":4}", two[0]);
    try std.testing.expectEqualStrings("web-search{\"q\":\"a)b\"}", two[1]);
    const listed = "[{\"name\":\"lookup\",\"arguments\":{\"limit\":\"4\"}}, {\"name\":\"web-search\",\"arguments\":{\"q\":\"{x}\"}}]";
    const json_reply = open ++ listed ++ close;
    const done = std.mem.indexOf(u8, json_reply, "}}").? + 2;
    try std.testing.expectEqual(@as(usize, 0), (try sentBy(a, json_reply, done - 1, tools)).len);
    const one = try sentBy(a, json_reply, done, tools);
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqualStrings("lookup{\"limit\":4}", one[0]);
    // the closed block streams as the same calls it was while open
    try std.testing.expectEqual(@as(usize, 2), (try sentBy(a, json_reply, json_reply.len, tools)).len);
}

test "tool_lfm2: a block that is not a list of calls shows no markup, and streams no call when its first is bad" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const bodies = [_][]const u8{
        "[other(), lookup(query=\"x\")]", // the first tool was not offered
        "[lookup(limit=4, \"x\"), lookup()]",
        "[{\"name\":\"other\"}, {\"name\":\"lookup\"}]",
        "[\x0b{\"name\":\"lookup\"}]",
        "[lookup(query=\"a\u{ff5c}DSML\u{ff5c}\"), lookup()]", // markup of the envelope's own
    };
    inline for (bodies) |body| {
        const reply = "Hi " ++ open ++ body ++ close;
        for (1..reply.len + 1) |n| try std.testing.expectEqual(@as(usize, 0), (try sentBy(a, reply, n, tools)).len);
        try std.testing.expectEqualStrings(reply, try dsml(a, reply, tools, true));
        const r = try tool_parse.parse(a, try dsml(a, reply, tools, true), tools, null);
        try std.testing.expect(r.calls == null);
    }
}

test "tool_lfm2: what follows a sent call and is not a call is dropped from the stream, and the whole reply stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const bodies = [_][]const u8{ "[lookup(query=\"x\"), other()] ok", "[lookup(query=\"x\")] trailing", "[lookup(query=\"x\"), lookup(1 2)]" };
    inline for (bodies) |body| {
        const reply = "Hi " ++ open ++ body ++ close ++ " after";
        var last: []const u8 = "";
        for (std.mem.indexOf(u8, reply, ")").? + 1..reply.len + 1) |n| {
            const seen = try dsml(a, reply[0..n], tools, false);
            try std.testing.expect(std.mem.indexOf(u8, seen, "other") == null and std.mem.indexOf(u8, seen, "trailing") == null);
            try std.testing.expect(std.mem.startsWith(u8, seen, last)); // the stream's parser only ever meets an extension
            try std.testing.expectEqual(@as(usize, 1), (try sentBy(a, reply, n, tools)).len);
            last = seen;
        }
        try std.testing.expectEqualStrings(reply, try dsml(a, reply, tools, true));
    }
}
