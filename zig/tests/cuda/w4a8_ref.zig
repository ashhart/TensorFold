//! Independent natural-order references for the finite W4A8 envelope, not the quantizer's warp reduction.
const std = @import("std");
const c = @import("core").contract.spec;
const rows = @import("core").contract.rows;
pub const Group = struct { scale: f32, sum: f64, magnitude: f64, codes: [64]i8 };
pub fn nearestEven(v: f32) i8 {
    const lo = @floor(v);
    const tail = v - lo;
    return @intFromFloat(if (tail > 0.5 or (tail == 0.5 and @mod(lo, 2) != 0)) lo + 1 else lo);
}
/// Ordinary inputs retain the reciprocal multiply. Tiny finite groups normalize before scaling to avoid infinity.
pub fn group(x: []const u16) !Group {
    std.debug.assert(x.len == 64);
    var top: f32 = 0;
    var sum: f64 = 0;
    var mag: f64 = 0;
    for (x) |h| {
        const v = c.fromBf16(h);
        if (!std.math.isFinite(v)) return error.ExceptionalQuantization;
        top = @max(top, @abs(v));
        sum += v;
        mag += @abs(v);
    }
    if (!std.math.isFinite(@as(f32, @floatCast(sum)))) return error.ExceptionalQuantization;
    var out: Group = .{ .scale = if (top > 0) top / 127 else 1, .sum = sum, .magnitude = mag, .codes = undefined };
    const inv: f32 = if (top > 0) 127 / top else 1;
    for (x, &out.codes) |h, *code| {
        const v = c.fromBf16(h);
        const scaled: f32 = if (std.math.isFinite(inv)) v * inv else (v / top) * 127;
        code.* = nearestEven(scaled);
    }
    return out;
}
/// Position in natural input order of one byte in a packed 32-input half-group.
pub fn inputAt(slot: usize) usize {
    const table = [32]u8{ 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 16, 17, 24, 25, 18, 19, 26, 27, 20, 21, 28, 29, 22, 23, 30, 31 };
    return table[slot];
}
/// The independent projection consumes natural-order signed codes and per-group FP32 scales and sums.
pub fn terms(xq: []const i8, xa: []const f32, xs: []const f32, codes: []const u8, scales: []const u16, biases: []const u16, out: []f64) void {
    for (out, 0..) |*term, g| {
        var dot: i32 = 0;
        for (0..64) |i| dot += @as(i32, xq[g * 64 + i]) * codes[g * 64 + i];
        term.* = @as(f64, c.fromBf16(scales[g])) * xa[g] * @as(f64, @floatFromInt(dot)) + @as(f64, c.fromBf16(biases[g])) * xs[g];
    }
}
/// Natural matrix codes packed directly into the GPU's layout, with zero words and metadata for padded columns.
pub fn pack(words: []u32, codes: []const u8, n: usize, k: usize) void {
    @memset(words, 0);
    const kg = k / 64;
    for (0..n) |row| for (0..k) |i| {
        const local = i % 64;
        const lane = (row % 8) * 4 + (local % 8) / 2;
        const at = (((row / 64 * kg + i / 64) * 8 + (row % 64) / 8) * 32 + lane) * 2 + local / 32;
        const nibble = (local % 32) / 8 + 4 * (local % 2);
        words[at] |= @as(u32, codes[row * k + i]) << @intCast(4 * nibble);
    };
}
pub fn sumTerms(acc: c.Accum, order: rows.Order, terms_: []const f64) f64 {
    return rows.sumIn(acc, order, terms_);
}
