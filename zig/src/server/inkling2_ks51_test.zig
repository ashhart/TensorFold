//! KS51's four Inkling 2 release-path findings through the route, whole and streamed at strides 1, 2, 3 and 256.
const std = @import("std");
const h = @import("inkling2_chat_test.zig");
const t = std.testing;

/// Byte tokens of the release fixture's ByteLevel alphabet, by the byte each carries.
fn byte(b: u8) u32 {
    return switch (b) {
        0x80 => 222,
        0x8F => 237,
        0x90 => 238,
        0x9F => 253,
        0xA0 => 254,
        0xBF => 123,
        0xC0 => 124,
        0xC1 => 125,
        0xC2 => 126,
        0xCE => 138,
        0xE0 => 156,
        0xED => 169,
        0xEE => 170,
        0xF0 => 172,
        0xF4 => 176,
        0xF5 => 177,
        else => unreachable,
    };
}

fn text(a: std.mem.Allocator, bytes: []const u8) ![]const u32 {
    const ids = try a.alloc(u32, bytes.len);
    for (bytes, ids) |b, *id| id.* = byte(b);
    return h.fixture(a, "<|content_text|>", ids);
}

test "Inkling 2 reply: KS51 1, a replacement-character stop cuts where the generation found it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // CE alone decodes as U+FFFD: the generation's stop, and the cut's
    const alone = try h.probeIds(a, "release", try text(a, &.{0xCE}), ",\"stop\":\"\u{fffd}\"");
    try t.expectEqualStrings("", alone.content);
    try t.expectEqualStrings("stop", alone.finish);
    // U+FFFD before ASCII stop text, and the ASCII stop beside it
    const with_stop = try h.fixture(a, "<|content_text|>", &.{ 138, 50, 51, 46, 47 });
    const before = try h.probeIds(a, "release", with_stop, ",\"stop\":\"\u{fffd}STOP\"");
    try t.expectEqualStrings("", before.content);
    try t.expectEqualStrings("stop", before.finish);
    const beside = try h.probeIds(a, "release", with_stop, ",\"stop\":\"STOP\"");
    try t.expectEqualStrings("\u{fffd}", beside.content);
}

test "Inkling 2 reply: KS51 2, every invalid UTF-8 form reads as the tokenizer's replacements, valid boundaries as text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = "\u{fffd}";
    const Case = struct { bytes: []const u8, want: []const u8 };
    const cases = [_]Case{
        .{ .bytes = &.{ 0xC0, 0x80 }, .want = r ++ r }, // overlong two-byte lead
        .{ .bytes = &.{ 0xC1, 0xBF }, .want = r ++ r },
        .{ .bytes = &.{ 0xE0, 0x80, 0x80 }, .want = r ++ r ++ r }, // overlong three-byte form
        .{ .bytes = &.{ 0xF0, 0x80, 0x80, 0x80 }, .want = r ++ r ++ r ++ r }, // overlong four-byte form
        .{ .bytes = &.{ 0xED, 0xA0, 0x80 }, .want = r ++ r ++ r }, // a surrogate
        .{ .bytes = &.{ 0xF4, 0x90, 0x80, 0x80 }, .want = r ++ r ++ r ++ r }, // past U+10FFFF
        .{ .bytes = &.{ 0xF5, 0x80, 0x80, 0x80 }, .want = r ++ r ++ r ++ r }, // a lead past F4
        .{ .bytes = &.{ 0xF5, 0x80 }, .want = r ++ r }, // the same, cut by the cap
        .{ .bytes = &.{ 0xC2, 0x80 }, .want = "\u{80}" }, // the valid boundaries
        .{ .bytes = &.{ 0xED, 0x9F, 0xBF }, .want = "\u{d7ff}" },
        .{ .bytes = &.{ 0xEE, 0x80, 0x80 }, .want = "\u{e000}" },
        .{ .bytes = &.{ 0xF0, 0x90, 0x80, 0x80 }, .want = "\u{10000}" },
        .{ .bytes = &.{ 0xF4, 0x8F, 0xBF, 0xBF }, .want = "\u{10ffff}" },
    };
    for (cases) |case| try t.expectEqualStrings(case.want, (try h.probeIds(a, "release", try text(a, case.bytes), "")).content);
}

test "Inkling 2 reply: KS51 3, finished bare calls before a cut one all return, and the reply keeps length" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"one\"}}</tool_call>";
    const two = try h.probeIds(a, "release", try h.encodeReply(a, one ++ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"tw"), h.lookup);
    try t.expectEqualStrings("lookuplookup", two.names);
    try t.expectEqualStrings("{\"q\":\"one\"}{\"q\":\"tw", two.arguments);
    try t.expectEqualStrings("", two.content);
    try t.expectEqualStrings("length", two.finish);
    const many = try h.probeIds(a, "release", try h.encodeReply(a, one ++ one ++ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x"), h.lookup);
    try t.expectEqualStrings("lookuplookuplookup", many.names);
    try t.expectEqualStrings("length", many.finish);
    const prose = try h.probeIds(a, "release", try h.encodeReply(a, "Sure.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x"), h.lookup);
    try t.expectEqualStrings("lookup", prose.names);
    try t.expectEqualStrings("{\"q\":\"x", prose.arguments);
    try t.expectEqualStrings("Sure.", prose.content);
    try t.expectEqualStrings("length", prose.finish);
}

test "Inkling 2 reply: KS51 4, under one call a bare call cut by the cap is no call and no text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cut = try h.encodeReply(a, "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x");
    const got = try h.probeIds(a, "release", cut, h.lookup ++ ",\"parallel_tool_calls\":false");
    try t.expectEqualStrings("", got.names);
    try t.expectEqualStrings("", got.arguments);
    try t.expectEqualStrings("", got.content);
    try t.expectEqualStrings("length", got.finish);
}
