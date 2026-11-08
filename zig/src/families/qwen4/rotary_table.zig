//! One stored FP32 rotary table, transformers' inverse frequencies from core's rope_freq, uploaded by either backend.
const std = @import("std");
const rope_freq = @import("core").rope_freq;
pub fn fill(theta: f64, rotary: usize, out: []f32) !void {
    if (!std.math.isFinite(theta) or theta <= 0 or rotary % 2 != 0 or out.len != 3 + rotary / 2) return error.Qwen4RotaryTable;
    out[0] = 0;
    out[1] = 1;
    out[2] = 1;
    rope_freq.inverseFrequencies(theta, rotary, out[3..]);
}
/// Returns the differing indices; when `loud`, prints old and new side by side with the largest differences.
fn receipt(name: []const u8, before: []const f32, after: []const f32, differ: []usize, loud: bool) []usize {
    if (loud) std.debug.print("\n{s}\n", .{name});
    if (loud) std.debug.print("    i  old word    new word    old             new             |old - new|  old - new\n", .{});
    var n: usize = 0;
    var abs: f64 = 0;
    var rel: f64 = 0;
    for (before, after, 0..) |o, w, i| {
        const a: u32 = @bitCast(o);
        const b: u32 = @bitCast(w);
        const d = @abs(@as(f64, o) - @as(f64, w));
        abs = @max(abs, d);
        rel = @max(rel, d / @abs(@as(f64, w)));
        const ulp: i64 = @as(i64, a) - @as(i64, b);
        if (ulp != 0) {
            differ[n] = i;
            n += 1;
        }
        const row = "  {d:>3}  0x{x:0>8}  0x{x:0>8}  {e:.8}  {e:.8}  {e:.3}    {d} ulp\n";
        if (loud) std.debug.print(row, .{ i, a, b, o, w, d, ulp });
    }
    const summary = "  {d} of {d} words differ; largest absolute difference {e:.3}, largest relative {e:.3}\n";
    if (loud) std.debug.print(summary, .{ n, before.len, abs, rel });
    return differ[0..n];
}
/// Names a failed comparison's first word: its position in the stored table (after three header words) and index.
fn mismatch(name: []const u8, i: usize, expected: []const u8, word: u32, got: u32) void {
    const line = "\n{s}: first mismatch at position {d}, index {d}: expected {s}0x{x:0>8}, got 0x{x:0>8}\n";
    std.debug.print(line, .{ name, 3 + i, i, expected, word, got });
}
/// For a failed word: names it, then prints the expected words (every `stride`-th of `want`) beside the table's.
fn wrongWord(name: []const u8, want: []const u32, stride: usize, got: []const f32, i: usize) void {
    mismatch(name, i, "", want[stride * i], @bitCast(got[i]));
    std.debug.print("    i  expected    got\n", .{});
    for (got, 0..) |g, k| {
        const w = want[stride * k];
        const b: u32 = @bitCast(g);
        std.debug.print("  {d:>3}  0x{x:0>8}  0x{x:0>8}{s}\n", .{ k, w, b, if (w == b) "" else "  <- differs" });
    }
}
/// For a failed move check: names the first word whose move disagrees with `moved`, then prints the receipt.
fn wrongMove(name: []const u8, before: []const f32, after: []const f32, moved: []const usize) void {
    for (before, after, 0..) |o, w, i| {
        const a: u32 = @bitCast(o);
        const b: u32 = @bitCast(w);
        const listed = std.mem.indexOfScalar(usize, moved, i) != null;
        if (listed == (a != b) and (if (a > b) a - b else b - a) <= 1) continue;
        mismatch(name, i, if (!listed) "unchanged " else if (a == b) "a 1-ulp move from " else "within 1 ulp of ", a, b);
        break;
    }
    var differ: [32]usize = undefined;
    _ = receipt(name, before, after, &differ, true);
}
test "Qwen4 rotary table: native devices consume transformers' FP32 words after the three header words" {
    // theta 1e4 over 64 (the native fixture: head 256, partial 0.25), each step correctly rounded (60-digit decimal)
    const expected = [_]u32{
        0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77,
        0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba15, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2,
        0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c, 0x3b8a2e77, 0x3b4f3e38, 0x3b1b690d, 0x3ae91528, 0x3aaec98e,
        0x3a83126f, 0x3a44948c, 0x3a136a16, 0x39dd1725, 0x39a5cb60, 0x3978a815, 0x393a7753, 0x390bd472,
    };
    var table: [35]f32 = undefined;
    try fill(10000, 64, &table);
    for (table[3..], expected, 0..) |value, bits, i| std.testing.expectEqual(bits, @as(u32, @bitCast(value))) catch |e| {
        wrongWord("Qwen4, theta 1e4 over 64 (native fixture)", &expected, 1, table[3..], i);
        return e;
    };
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 1 }, table[0..3]);
    var half: [19]f32 = undefined;
    try fill(10000, 32, &half);
    for (half[3..], 0..) |value, i| std.testing.expectEqual(expected[2 * i], @as(u32, @bitCast(value))) catch |e| {
        wrongWord("Qwen4, theta 1e4 over 32", &expected, 2, half[3..], i);
        return e;
    };
    var disabled: [3]f32 = undefined;
    try fill(10000, 0, &disabled);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 1 }, &disabled);
    try std.testing.expectError(error.Qwen4RotaryTable, fill(10000, 32, table[0..18]));
    try std.testing.expectError(error.Qwen4RotaryTable, fill(std.math.nan(f64), 32, &table));
}
test "Qwen4 rotary table: 10 of 32 words (theta 1e4 over 64) and 4 of 16 (over 32) move 1 ulp from the old ones" {
    const moved = [_][]const usize{ &.{ 5, 9, 10, 12, 13, 15, 20, 27, 28, 29 }, &.{ 5, 6, 10, 14 } };
    for ([_]usize{ 64, 32 }, moved) |rotary, want| {
        const n = rotary / 2;
        // the replaced expression: f32(theta^(-2i / rotary)), a negative f64 exponent and no reciprocal
        var before: [32]f32 = undefined;
        for (before[0..n], 0..) |*value, i| {
            const e = -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rotary));
            value.* = @floatCast(std.math.pow(f64, 10000, e));
        }
        var table: [35]f32 = undefined;
        try fill(10000, rotary, table[0 .. 3 + n]);
        const after = table[3 .. 3 + n];
        var differ: [32]usize = undefined;
        const name = if (rotary == 64) "Qwen4, theta 1e4 over 64 (native fixture)" else "Qwen4, theta 1e4 over 32";
        std.testing.expectEqualSlices(usize, want, receipt(name, before[0..n], after, &differ, false)) catch |e| {
            wrongMove(name, before[0..n], after, want);
            return e;
        };
        for (before[0..n], after) |o, w| {
            const a: u32 = @bitCast(o);
            const b: u32 = @bitCast(w);
            std.testing.expect((if (a > b) a - b else b - a) <= 1) catch |e| {
                wrongMove(name, before[0..n], after, want);
                return e;
            };
        }
    }
}
