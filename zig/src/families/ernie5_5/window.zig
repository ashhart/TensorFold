//! ERNIE 5.5's sparse-attention key lists: each row's indexer picks, then its sliding window (vLLM's merge_swa_topk).
const std = @import("std");

/// The released config's sliding_window and index_topk.
pub const window = 128;
pub const topk = 2048;

/// rows rows, row r at position pos0 + r; k picks a row in, lists w wide out (at least k + window).
pub const Shape = struct { rows: usize, k: usize, w: usize, window: usize = window };

/// Sizes the kernel holds in i32, and lists wide enough for every pick and the window.
pub fn check(sh: Shape) error{WindowShape}!void {
    const most = std.math.maxInt(i32);
    if (sh.window == 0 or sh.k > most or sh.window > most or sh.w < sh.k + sh.window or sh.rows * sh.w > most)
        return error.WindowShape;
}

/// Row r: picks to picked[r] (in [0, k]) and the first negative, seq - min(seq, window) .. seq - 1, -1, its count.
pub fn merge(sh: Shape, picks: []const i32, picked: []const i32, pos0: i64, tok: []i32, cnt: []i32) void {
    for (0..sh.rows) |r| {
        const seq = pos0 + @as(i64, @intCast(r)) + 1;
        const held: usize = @intCast(std.math.clamp(picked[r], 0, @as(i32, @intCast(sh.k))));
        const n = for (picks[r * sh.k ..][0..held], 0..) |v, i| {
            if (v < 0) break i; // a negative pick inside the count ends the picks
        } else held;
        const sw: usize = @intCast(@min(seq, @as(i64, @intCast(sh.window))));
        const row = tok[r * sh.w ..][0..sh.w];
        @memcpy(row[0..n], picks[r * sh.k ..][0..n]);
        for (row[n..][0..sw], 0..) |*t, i| t.* = @intCast(seq - @as(i64, @intCast(sw)) + @as(i64, @intCast(i)));
        @memset(row[n + sw ..], -1);
        cnt[r] = @intCast(n + sw);
    }
}
