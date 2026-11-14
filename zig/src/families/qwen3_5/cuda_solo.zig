//! TF_QWEN35_CHUNK_SOLO: whole grid chunks a solo prompt pass takes, on lanes.solo_chunk's rule.
const std = @import("std");
const solo_chunk = @import("lanes").solo_chunk;
const Round = @import("cuda_round.zig");

pub const name = "TF_QWEN35_CHUNK_SOLO";
/// The grid chunk (a pass beside a decode), and the widest solo pass: the scratch rows and tap rings grow with it.
pub const limits: solo_chunk.Limits = .{ .busy = Round.grid_rows, .most = Round.max_rows, .grid = Round.grid_rows };

/// The prompt forward's rows at most, read once at load; a pipeline stage (`staged`) keeps the grid chunk.
pub fn rows(staged: bool) !usize {
    return stage(try solo_chunk.rows(name, limits), staged);
}

/// `raw` as the variable's text (null: unset), read as `rows` reads it.
pub fn rowsOf(raw: ?[]const u8, staged: bool) !usize {
    return stage(try solo_chunk.read(name, raw, limits), staged);
}

fn stage(n: usize, staged: bool) !usize {
    if (staged and n != limits.busy) return error.QwenSoloStage;
    return n;
}

/// Whether a stream besides `own` holds a cache with its prompt pass done: it decodes, so `own` takes the grid chunk.
pub fn othersDecode(self: anytype, own: anytype) bool {
    var it = self.caches.keyIterator();
    while (it.next()) |s| if (s.* != own and !self.fills.contains(s.*)) return true;
    return false;
}

test "TF_QWEN35_CHUNK_SOLO: unset and blank are the 4096-row grid, 8192 taken, every other text refused by name" {
    const t = std.testing;
    for ([_]?[]const u8{ null, "", " ", "\t \n", "4096" }) |v| try t.expectEqual(@as(usize, 4096), try rowsOf(v, false));
    try t.expectEqual(@as(usize, 8192), try rowsOf("8192", false));
    const bad = [_][]const u8{ "0", "off", "false", "no", "on", "yes", "2", " 0", "1", "2048", "4097", "6144", "12288", "16384", "-1", "8k", "0x2000", "8192x" };
    for (bad) |v| try t.expectError(error.BadChunkSolo, rowsOf(v, false));
    const why = "TF_QWEN35_CHUNK_SOLO: a solo prompt pass's rows, 4096 (off) or 8192";
    try t.expectEqualStrings(why, @import("cuda_native_explain.zig").explain(null, error.BadChunkSolo).?);
}

test "TF_QWEN35_CHUNK_SOLO: a pipeline stage refuses a solo value and keeps the grid unset" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 4096), try rowsOf(null, true));
    try t.expectEqual(@as(usize, 4096), try rowsOf("4096", true));
    try t.expectError(error.QwenSoloStage, rowsOf("8192", true));
    try t.expect(@import("cuda_native_explain.zig").explain(null, error.QwenSoloStage) != null);
}

test "TF_QWEN35_CHUNK_SOLO: decoding streams beside it count, filling ones and its own do not" {
    const t = std.testing;
    const S = struct { id: u8 };
    var a: S = .{ .id = 0 };
    var b: S = .{ .id = 1 };
    var x: struct { caches: std.AutoHashMapUnmanaged(*S, u8) = .empty, fills: std.AutoHashMapUnmanaged(*S, u8) = .empty } = .{};
    defer x.caches.deinit(t.allocator);
    defer x.fills.deinit(t.allocator);
    try x.caches.put(t.allocator, &a, 0);
    try x.fills.put(t.allocator, &a, 0);
    try t.expect(!othersDecode(&x, &a)); // alone
    try x.caches.put(t.allocator, &b, 0);
    try x.fills.put(t.allocator, &b, 0);
    try t.expect(!othersDecode(&x, &a)); // the other still fills its prompt
    _ = x.fills.remove(&b);
    try t.expect(othersDecode(&x, &a)); // the other decodes
    try t.expect(!othersDecode(&x, &b)); // its own lane never counts
}
