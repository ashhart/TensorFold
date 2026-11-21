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

/// Engine.scratch_rows: the context's, 32 to `solo`; unset that is main's @max(32, @min(4096, context)).
pub fn scratchRows(solo: usize, context: usize) usize {
    return @max(32, @min(solo, context));
}

/// The widest prompt pass: the grid chunk, or a solo pass's rows where the context admits more.
pub fn passRows(e: anytype) usize {
    return @max(Round.grid_rows, e.scratch_rows);
}

/// Rows a stream's tap ring holds (Engine.tapRows): the drafter's window behind the widest prompt pass.
pub fn tapRows(e: anytype) usize {
    return @import("cuda_cache.zig").tap_window + passRows(e);
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

test "TF_QWEN35_CHUNK_SOLO unset: scratch, pass and tap rows equal main's at every context, 8192 only when set" {
    const t = std.testing;
    const main_rows = 4096; // main: scratch @max(32, @min(4096, context)), tap ring 2048 + 4096
    const unset = try rowsOf(null, false);
    for ([_]usize{ 1, 32, 100, 4095, 4096, 4097, 8191, 8192, 8193, 65536, 262144 }) |context| {
        const e: struct { scratch_rows: usize } = .{ .scratch_rows = scratchRows(unset, context) };
        try t.expectEqual(@as(usize, @max(32, @min(main_rows, context))), e.scratch_rows);
        try t.expectEqual(@as(usize, main_rows), passRows(e));
        try t.expectEqual(@as(usize, 2048 + main_rows), tapRows(e));
    }
    const on = try rowsOf("8192", false);
    try t.expectEqual(@as(usize, 8192), scratchRows(on, 8192));
    try t.expectEqual(@as(usize, 6000), scratchRows(on, 6000)); // a context under the solo rows keeps its own
    try t.expectEqual(@as(usize, 4096), passRows(.{ .scratch_rows = scratchRows(on, 1000) }));
    try t.expectEqual(@as(usize, 2048 + 8192), tapRows(.{ .scratch_rows = scratchRows(on, 1 << 20) }));
}
