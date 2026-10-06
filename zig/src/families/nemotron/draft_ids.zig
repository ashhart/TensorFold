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
