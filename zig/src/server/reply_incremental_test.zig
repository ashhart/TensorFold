//! Incremental reply text: a character split across tokens is held whole, and each decode reads only the new window.
const std = @import("std");
const reply_text = @import("reply_text.zig");

/// A byte tokenizer whose decode, like model_text's, spells a trailing partial UTF-8 sequence as U+FFFD.
const Bytes = struct {
    pieces: []const []const u8,
    longest: usize = 0,
    fn decode(t: *Bytes, a: std.mem.Allocator, tokens: []const u32) ![]u8 {
        t.longest = @max(t.longest, tokens.len);
        var out: std.ArrayList(u8) = .empty;
        for (tokens) |id| try out.appendSlice(a, t.pieces[id]);
        var cut = out.items.len;
        while (cut > 0 and out.items[cut - 1] & 0xc0 == 0x80) cut -= 1;
        if (cut > 0 and out.items[cut - 1] >= 0xc0) {
            const need = std.unicode.utf8ByteSequenceLength(out.items[cut - 1]) catch 1;
            if (out.items.len - (cut - 1) < need) {
                out.shrinkRetainingCapacity(cut - 1);
                try out.appendSlice(a, "\u{fffd}");
            }
        }
        return out.items;
    }
};

test "incremental reply text: a character split across tokens streams only once it is whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Bytes = .{ .pieces = &.{ "a", "\xe2\x82", "\xac", "b" } };
    var inc: reply_text.Incremental = .{};
    try std.testing.expectEqualStrings("a", try inc.extend(a, &t, Bytes.decode, &.{0}));
    try std.testing.expect(!inc.pending());
    try std.testing.expectEqualStrings("a", try inc.extend(a, &t, Bytes.decode, &.{1}));
    try std.testing.expect(inc.pending());
    try std.testing.expectEqualStrings("a\u{20ac}", try inc.extend(a, &t, Bytes.decode, &.{2}));
    try std.testing.expect(!inc.pending());
    const all = try inc.extend(a, &t, Bytes.decode, &.{3});
    try std.testing.expectEqualStrings(try t.decode(a, &.{ 0, 1, 2, 3 }), all);
}

test "incremental reply text: a long reply decodes the previous chunk and the new tokens, never the whole reply" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Bytes = .{ .pieces = &.{ "x", "\xc3", "\xa9" } };
    var inc: reply_text.Incremental = .{};
    var want: std.ArrayList(u8) = .empty;
    for (0..300) |i| {
        if (i % 50 == 49) {
            _ = try inc.extend(a, &t, Bytes.decode, &.{ 1, 2 }); // one character in two tokens, in one chunk
            try want.appendSlice(a, "\u{e9}");
        } else {
            _ = try inc.extend(a, &t, Bytes.decode, &.{0});
            try want.append(a, 'x');
        }
    }
    try std.testing.expectEqualStrings(want.items, inc.text.items);
    try std.testing.expect(t.longest <= 4);
}
