//! A grammar bitmask on a rank's logits columns (disallowed tokens -inf before top-k), device and host forms.

const std = @import("std");

/// Row `row`'s columns [col0, col0 + 32) under `word` (bit b: token col0 + b allowed); columns >= width are skipped.
pub inline fn maskWord(row: [*]f32, word: u32, col0: u32, width: u32) void {
    if (word == 0xffff_ffff) return;
    var b: u32 = 0;
    while (b < 32) : (b += 1) {
        const col = col0 + b;
        if (col < width and (word >> @intCast(b)) & 1 == 0) row[col] = -std.math.inf(f32);
    }
}

/// Words of this rank's columns: [lo / 32, (lo + width + 31) / 32). `lo` must sit on a word (others are refused).
pub fn span(lo: u32, width: u32) struct { w0: u32, n: u32 } {
    return .{ .w0 = lo / 32, .n = (width + 31) / 32 };
}

/// Host form: logits rows `idx[r]` (stride `ld`) under bitmask rows `bits[r * words ..]`, columns from token `lo`.
pub fn maskRows(logits: []f32, ld: usize, idx: []const u32, bits: []const u32, words: u32, lo: u32, width: u32) void {
    const s = span(lo, width);
    for (idx, 0..) |row, r| {
        const base = logits[row * ld ..].ptr;
        const words_r = bits[r * words + s.w0 ..][0..s.n];
        for (words_r, 0..) |w, k| maskWord(base, w, @intCast(32 * k), width);
    }
}

test "maskRows equals the per-token rule on a rank's half" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const vocab: u32 = 1000;
    const words: u32 = (vocab + 31) / 32;
    const world: u32 = 2;
    const width: u32 = 512; // the halves of a 1024-column head (the last 24 columns past the vocabulary)
    const n = 5;
    const bits = try gpa.alloc(u32, n * words);
    defer gpa.free(bits);
    for (bits) |*b| b.* = switch (rnd.uintLessThan(u32, 3)) {
        0 => 0xffff_ffff,
        1 => 0,
        else => rnd.int(u32),
    };
    for (0..world) |rank| {
        const lo: u32 = @intCast(rank * width);
        const logits = try gpa.alloc(f32, 8 * width);
        defer gpa.free(logits);
        for (logits) |*x| x.* = rnd.float(f32);
        const want = try gpa.dupe(f32, logits);
        defer gpa.free(want);
        const idx = [_]u32{ 0, 2, 3, 6, 7 };
        for (idx, 0..) |row, r| for (0..width) |c| {
            const t = lo + c;
            const ok = t < words * 32 and (bits[r * words + t / 32] >> @intCast(t % 32)) & 1 == 1;
            if (!ok) want[row * width + c] = -std.math.inf(f32);
        };
        maskRows(logits, width, &idx, bits, words, lo, width);
        try std.testing.expectEqualSlices(f32, want, logits);
    }
}
