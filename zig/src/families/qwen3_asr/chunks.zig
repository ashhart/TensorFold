//! The tower's chunk and window plan: 100-frame chunks through the convolutions, then attention over whole windows.
const std = @import("std");
const Audio = @import("config.zig").Audio;

/// Mel frames of `samples` 16 kHz samples: the STFT's centered frames less the last.
pub fn frames(samples: usize) usize {
    return samples / @import("config.zig").hop;
}

fn floorHalf(x: i64) i64 {
    return @divFloor(x - 1, 2) + 1;
}

/// Rows the convolutions leave of `len` frames: three stride-2 steps of the remainder, 13 a whole chunk.
pub fn rows(len: usize) usize {
    const rem: i64 = @intCast(len % 100);
    const tail = floorHalf(floorHalf(floorHalf(rem)));
    return @as(usize, @intCast(tail)) + (len / 100) * 13;
}

/// A clip's plan: its chunks' frames and rows, and the attention windows over the rows.
pub const Plan = struct {
    frames: usize,
    chunks: usize,
    last_frames: usize, // the last chunk's frames (a full chunk when frames divide)
    rows: usize,
    chunk_rows: usize, // rows of the longest chunk: a row's position inside its chunk runs to this
    window_rows: usize, // rows an attention window spans

    pub fn init(a: Audio, n: usize) error{EmptyClip}!Plan {
        if (n == 0) return error.EmptyClip;
        const cf = a.chunkFrames();
        const chunks = (n + cf - 1) / cf;
        const last = if (n % cf == 0) cf else n % cf;
        const longest = if (chunks > 1) rows(cf) else rows(last);
        return .{ .frames = n, .chunks = chunks, .last_frames = last, .rows = rows(n), .chunk_rows = longest, .window_rows = longest * (a.window_infer / cf) };
    }

    /// Rows chunk `i` leaves.
    pub fn rowsOf(p: Plan, i: usize) usize {
        return rows(if (i + 1 == p.chunks) p.last_frames else 100);
    }

    pub fn windows(p: Plan) usize {
        return (p.rows + p.window_rows - 1) / p.window_rows;
    }

    /// Window `w`'s first row and its rows.
    pub fn window(p: Plan, w: usize) struct { start: usize, len: usize } {
        const start = w * p.window_rows;
        return .{ .start = start, .len = @min(p.window_rows, p.rows - start) };
    }
};

test "rows: the package's _get_feat_extract_output_lengths" {
    // values printed by qwen-asr 0.0.6's formula for these frame counts
    const cases = [_][2]usize{ .{ 1, 1 }, .{ 7, 1 }, .{ 8, 1 }, .{ 9, 2 }, .{ 99, 13 }, .{ 100, 13 }, .{ 101, 14 }, .{ 481, 63 }, .{ 585, 76 }, .{ 2940, 382 }, .{ 3000, 390 } };
    for (cases) |c| try std.testing.expectEqual(c[1], rows(c[0]));
}

test "plan: chunks, the short last chunk, windows of eight chunks" {
    const a: Audio = .{ .d = 896, .layers = 18, .heads = 14, .ffn = 3584, .mels = 128, .conv = 480, .out = 1024, .window = 50, .window_infer = 800, .positions = 1500, .conv_chunks = 500 };
    const p = try Plan.init(a, 2940);
    try std.testing.expectEqual(@as(usize, 30), p.chunks);
    try std.testing.expectEqual(@as(usize, 40), p.last_frames);
    try std.testing.expectEqual(@as(usize, 382), p.rows);
    try std.testing.expectEqual(@as(usize, 104), p.window_rows);
    try std.testing.expectEqual(@as(usize, 4), p.windows());
    try std.testing.expectEqual(@as(usize, 70), p.window(3).len);
    var sum: usize = 0;
    for (0..p.chunks) |i| sum += p.rowsOf(i);
    try std.testing.expectEqual(p.rows, sum);
    const short = try Plan.init(a, 60);
    try std.testing.expectEqual(@as(usize, 8), short.chunk_rows);
    try std.testing.expectEqual(@as(usize, 64), short.window_rows);
    try std.testing.expectEqual(@as(usize, 1), short.windows());
    try std.testing.expectError(error.EmptyClip, Plan.init(a, 0));
}
