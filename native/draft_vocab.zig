//! The original fixed draft ID lists are embedded by build.zig without duplicating them.
//! Only proposals use this head; verification always scores the full vocabulary.
const std = @import("std");
const mx = @import("mlx.zig");
const Store = @import("checkpoint.zig").Store;
pub const data = @import("draft_vocab_data");

pub fn parse(a: std.mem.Allocator, text: []const u8, vocab: usize, multiple: usize) ![]u32 {
    if (vocab == 0 or multiple == 0) return error.InvalidDraftVocabulary;
    const seen = try a.alloc(bool, vocab);
    defer a.free(seen);
    @memset(seen, false);
    var count: usize = 0;
    var words = std.mem.tokenizeAny(u8, text, " \r\n\t");
    while (words.next()) |word| {
        const id = std.fmt.parseInt(usize, word, 10) catch return error.InvalidDraftVocabulary;
        if (id >= vocab) return error.InvalidDraftVocabulary;
        if (!seen[id]) count += 1;
        seen[id] = true;
    }
    if (count == 0) return error.InvalidDraftVocabulary;
    for (seen) |*present| {
        if (count % multiple == 0) break;
        if (!present.*) {
            present.* = true;
            count += 1;
        }
    }
    if (count % multiple != 0) return error.InvalidDraftVocabulary;
    const ids = try a.alloc(u32, count);
    var i: usize = 0;
    for (seen, 0..) |present, id| if (present) {
        ids[i] = @intCast(id);
        i += 1;
    };
    return ids;
}

pub fn install(w: *Store, text: []const u8, vocab: usize, multiple: usize) !void {
    const ids = try parse(mx.allocator, text, vocab, multiple);
    defer mx.allocator.free(ids);
    var s = mx.Scope{};
    defer s.deinit();
    const mapping = try s.data(ids.ptr, &.{@intCast(ids.len)}, mx.c.MLX_UINT32);
    inline for (.{ "weight", "scales", "biases" }) |suffix| {
        const value = try s.take(try w.field("lm_head", suffix), mapping, 0);
        try mx.eval(value);
        try w.put("draft_lm_head." ++ suffix, value);
    }
    try w.put("draft_ids", mapping);
}

test "draft vocabulary matches original sorting, deduplication and padding" {
    const a = std.testing.allocator;
    const ids = try parse(a, "9\n3 3 5", 10, 8);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 9 }, ids);
    for ([_][]const u8{ "", "10", "-1", "no" }) |invalid|
        try std.testing.expectError(error.InvalidDraftVocabulary, parse(a, invalid, 10, 8));
    try std.testing.expectError(error.InvalidDraftVocabulary, parse(a, "0", 3, 8));
    const nemotron = try parse(a, data.nemotron, 131072, 8);
    defer a.free(nemotron);
    try std.testing.expectEqual(@as(usize, 32768), nemotron.len);
    const flash = try parse(a, data.flash, 248320, 8);
    defer a.free(flash);
    try std.testing.expectEqual(@as(usize, 79592), flash.len);
}
