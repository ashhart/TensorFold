//! Nemotron's draft vocabulary: the Python package's frequency-ranked token ids the MTP head drafts among.
const std = @import("std");

/// Whitespace-separated ids, each below `vocab`, a multiple of 64 of them (the draft head is tiled by 64 rows).
pub fn parse(gpa: std.mem.Allocator, text: []const u8, vocab: usize) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, " \n\r\t");
    while (it.next()) |tok| {
        const v = try std.fmt.parseInt(u32, tok, 10);
        if (v >= vocab) return error.DraftIdPastVocabulary;
        try ids.append(gpa, v);
    }
    if (ids.items.len == 0 or ids.items.len % 64 != 0) return error.DraftIdsNotTiled;
    return ids.toOwnedSlice(gpa);
}

/// The draft vocabulary (draft_ids.txt beside this file), embedded at build time.
pub fn load(gpa: std.mem.Allocator, vocab: usize) ![]u32 {
    return parse(gpa, @import("nemotron_draft_ids").text, vocab);
}

/// The draft vocabulary as a set: a stream drafts on the subset while its context's last `recent` tokens are all in it.
pub const Coverage = struct {
    pub const recent = 16;
    bits: std.bit_set.Dynamic,

    pub fn init(gpa: std.mem.Allocator, vocab: usize, ids: []const u32) !Coverage {
        var bits: std.bit_set.Dynamic = try .initEmpty(gpa, vocab);
        for (ids) |id| bits.set(id);
        return .{ .bits = bits };
    }

    pub fn deinit(c: *Coverage, gpa: std.mem.Allocator) void {
        c.bits.deinit(gpa);
    }

    /// Whether the context's last `recent` tokens are all draft ids.
    pub fn covers(c: *const Coverage, context: []const u32) bool {
        if (context.len < recent) return false;
        for (context[context.len - recent ..]) |id| if (id >= c.bits.bit_length or !c.bits.isSet(id)) return false;
        return true;
    }
};

test "draft coverage reads only the context's last tokens" {
    var c: Coverage = try .init(std.testing.allocator, 131072, &.{ 1, 131071 });
    defer c.deinit(std.testing.allocator);
    var tokens: [Coverage.recent + 1]u32 = @splat(1);
    try std.testing.expect(!c.covers(&.{}));
    try std.testing.expect(!c.covers(tokens[0 .. Coverage.recent - 1]));
    try std.testing.expect(c.covers(tokens[0..Coverage.recent]));
    tokens[0] = 2;
    try std.testing.expect(!c.covers(tokens[0..Coverage.recent]));
    try std.testing.expect(c.covers(&tokens));
    tokens[Coverage.recent] = 131071;
    try std.testing.expect(c.covers(&tokens));
    tokens[Coverage.recent] = 2;
    try std.testing.expect(!c.covers(&tokens));
    tokens[Coverage.recent] = 131072;
    try std.testing.expect(!c.covers(&tokens));
}

test "draft ids parse and refuse" {
    const a = std.testing.allocator;
    var text: [64 * 3]u8 = undefined;
    for (0..64) |i| _ = try std.fmt.bufPrint(text[i * 3 ..][0..3], "{d:>2} ", .{i});
    const ids = try parse(a, &text, 64);
    defer a.free(ids);
    try std.testing.expectEqual(@as(u32, 63), ids[63]);
    try std.testing.expectError(error.DraftIdPastVocabulary, parse(a, &text, 10));
    try std.testing.expectError(error.DraftIdsNotTiled, parse(a, "1 2 3", 10));
}
