//! Every prefix is compared with the unchanged whole parser, including incomplete headers and UTF-8.
const std = @import("std");
const json = @import("json");
const reply = @import("kimi_reply.zig");
const mt = @import("model_text.zig");
const A = std.mem.Allocator;

fn prefixes(a: A, text: mt.Text, ids: []const u32, thinking: bool, tools: []const json.Value) !void {
    for ([_]usize{ 1, 2, 7 }) |stride| {
        var stream = (try reply.Stream.init(a, text, thinking, tools)).?;
        var n: usize = 0;
        while (true) {
            const want = try reply.parse(a, text, ids[0..n], thinking, tools);
            const got = try stream.extend(ids[0..n]);
            try std.testing.expectEqualStrings(want.content, got.content);
            try std.testing.expectEqualStrings(want.reasoning, got.reasoning);
            try std.testing.expectEqual(want.reasoning_tokens, got.reasoning_tokens);
            try std.testing.expectEqual(want.rejected, got.rejected);
            try std.testing.expectEqualStrings(try json.stringify(a, if (want.calls) |v| .{ .array = v } else .null, .{}), try json.stringify(a, if (got.calls) |v| .{ .array = v } else .null, .{}));
            if (n == ids.len) break;
            n = @min(n + stride, ids.len);
        }
    }
}

test "Server recovery: Kimi incremental: every release reply prefix agrees with whole parsing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try @import("kimi_text_test.zig").synthetic(std.testing.allocator);
    defer t.deinit();
    const fixture = (try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok;
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    for (fixture.get("cases").?.array) |c| {
        const id = c.strField("id").?;
        if (!std.mem.eql(u8, id, "history-reasoning") and !std.mem.eql(u8, id, "tool-roundtrip") and !std.mem.eql(u8, id, "raw-json-tool-argument")) continue;
        const ids = try @import("kimi_reply_test.zig").assistant(a, t.text(), c);
        try prefixes(a, t.text(), ids, true, tools);
        try prefixes(a, t.text(), ids, true, &.{});
    }
}

test "Server recovery: Kimi incremental: split Unicode, invalid subparts, repeated channels and malformed controls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try @import("kimi_text_test.zig").synthetic(std.testing.allocator);
    defer t.deinit();
    const texts = [_][]const u8{
        " \u{3000}\u{1f600}\u{e9}\u{fffd}z<|close|>think<|sep|><|open|>response<|sep|>\u{3000}a\u{1f600}b",
        "a<|close|>think<|sep|><|open|>response<|sep|>b<|close|>response<|sep|><|open|>think<|sep|>c",
        "a<|open|>call tool=\"lookup\" index=\"1\"<|sep|>bad<|close|>call<|sep|>",
        "a<|close|>think<|open|>unknown<|sep|>bad<|reserved_token_300|>tail<|end_of_msg|>ignored",
    };
    for (texts) |text| try prefixes(a, t.text(), try t.text().encode(a, text, false), true, &.{});
    try prefixes(a, t.text(), &.{ 'x', 0xf0, 0x9f, 0x98, 0x80, 0xe0, 0x80, 0x80, 0xed, 0xa0, 0x80, 0xff, 'y', 0xc2 }, false, &.{});
    var rng: std.Random.DefaultPrng = .init(1234);
    var bytes: [256]u8 = undefined;
    rng.random().bytes(&bytes);
    var ids: [256]u32 = undefined;
    for (bytes, &ids) |b, *id| id.* = b;
    try prefixes(a, t.text(), &ids, false, &.{});
}

test "Server recovery: Kimi incremental: an id with no bytes hands the reply to the whole parse instead of failing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try @import("kimi_text_test.zig").synthetic(std.testing.allocator);
    defer t.deinit();
    const ids = [_]u32{ 'a', 'b', 1_000_000, 'c' };
    try std.testing.expect(try t.text().tokenBytes(a, ids[2]) == null);
    var stream = (try reply.Stream.init(a, t.text(), false, &.{})).?;
    _ = try stream.extend(ids[0..2]);
    try std.testing.expect(!stream.fallback);
    _ = try stream.extend(&ids);
    try std.testing.expect(stream.fallback);
    _ = try reply.parse(a, t.text(), &ids, false, &.{});
}
