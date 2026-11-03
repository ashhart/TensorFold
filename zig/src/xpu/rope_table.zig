//! The Qwen3.8-27B XPU runner's RoPE inverse frequencies (theta 1e7): core's rope_freq, transformers' words.
const std = @import("std");
const rope_freq = @import("core").rope_freq;

/// The target's attention rotates 64 of each 256-wide head's dims: 32 words, as attn_prep reads them.
pub fn target() [32]f32 {
    var out: [32]f32 = undefined;
    rope_freq.inverseFrequencies(1e7, 64, &out);
    return out;
}

/// The DFlash2 drafter rotates all 128 dims of its heads: 64 words, as dprep reads them.
pub fn drafter() [64]f32 {
    var out: [64]f32 = undefined;
    rope_freq.inverseFrequencies(1e7, 128, &out);
    return out;
}

/// Prints old and new side by side with the largest absolute and relative difference; returns the differing indices.
fn receipt(name: []const u8, old: []const f32, new: []const f32, differ: []usize) []usize {
    std.debug.print("\n{s}\n", .{name});
    std.debug.print("    i  old word    new word    old             new             |old - new|  old - new\n", .{});
    var n: usize = 0;
    var abs: f64 = 0;
    var rel: f64 = 0;
    for (old, new, 0..) |o, w, i| {
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
        std.debug.print(row, .{ i, a, b, o, w, d, ulp });
    }
    const summary = "  {d} of {d} words differ; largest absolute difference {e:.3}, largest relative {e:.3}\n";
    std.debug.print(summary, .{ n, old.len, abs, rel });
    return differ[0..n];
}

fn expectWords(want: []const u32, got: []const f32) !void {
    for (want, got) |w, g| try std.testing.expectEqual(w, @as(u32, @bitCast(g)));
}

fn expectUlps(old: []const f32, new: []const f32, most: u32) !void {
    for (old, new) |o, w| {
        const a: u32 = @bitCast(o);
        const b: u32 = @bitCast(w);
        try std.testing.expect((if (a > b) a - b else b - a) <= most);
    }
}

test "Qwen3.8-27B XPU RoPE: the target's 32 words are transformers', 15 of them 1 ulp from the loaders' old ones" {
    // the replaced expression (load.zig, both loaders): f32(1e7^(-i / 32)), a negative f64 exponent, no reciprocal
    var old: [32]f32 = undefined;
    for (&old, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    const new = target();
    // transformers 5.22.1, torch 2.15.0 CPU: the words core's rope_freq test pins for theta 1e7 over 64
    try expectWords(&.{
        0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
        0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
        0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
        0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
    }, &new);
    var differ: [32]usize = undefined;
    const at = receipt("Qwen3.8-27B target, theta 1e7 over 64 rotated dims (load.zig)", &old, &new, &differ);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 5, 7, 10, 13, 16, 18, 19, 20, 21, 22, 26, 28, 29 }, at);
    try expectUlps(&old, &new, 1);
}

test "Qwen3.8-27B XPU RoPE: the drafter's 64 words, 18 of them 1 ulp from its old ones" {
    // the replaced expression (drafter.zig): f32(1 / 1e7^(2i / 128)), power and reciprocal in f64, rounded once
    var old: [64]f32 = undefined;
    for (&old, 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, 1e7, @as(f64, @floatFromInt(2 * i)) / 128.0));
    const new = drafter();
    // each step correctly rounded (60-digit decimal); the even words are transformers' theta 1e7 over 64 above
    try expectWords(&.{
        0x3f800000, 0x3f470165, 0x3f1ab32b, 0x3ef0843c, 0x3ebaf81b, 0x3e9157e1, 0x3e61f835, 0x3e2fa92d,
        0x3e088d77, 0x3dd44d6c, 0x3da50956, 0x3d804b29, 0x3d47763f, 0x3d1b0e01, 0x3cf11177, 0x3cbb65e3,
        0x3c91ad39, 0x3c627ce5, 0x3c301052, 0x3c08dda5, 0x3bd4ca15, 0x3ba56a3f, 0x3b80967d, 0x3b47eb5e,
        0x3b1b690d, 0x3af19f02, 0x3abbd3ed, 0x3a9202c3, 0x3a6301e2, 0x3a3077b3, 0x3a092e02, 0x39d54706,
        0x39a5cb60, 0x3980e1fe, 0x394860c1, 0x391bc44e, 0x38f22ce2, 0x38bc4236, 0x3892587e, 0x3863872c,
        0x3830df52, 0x38097e8f, 0x37d5c441, 0x37a62cb9, 0x37812dab, 0x3748d669, 0x371c1fc4, 0x36f2bb16,
        0x36bcb0c1, 0x3692ae6d, 0x36640cc6, 0x3631472c, 0x3609cf4a, 0x35d641c6, 0x35a68e4c, 0x35817985,
        0x35494c57, 0x351c7b70, 0x34f3499d, 0x34bd1f8c, 0x3493048e, 0x346492ae, 0x3431af44, 0x340a2036,
    }, &new);
    for (0..32) |i| try std.testing.expectEqual(@as(u32, @bitCast(target()[i])), @as(u32, @bitCast(new[2 * i])));
    var differ: [64]usize = undefined;
    const at = receipt("Qwen3.8-27B DFlash2 drafter, theta 1e7 over 128 (drafter.zig)", &old, &new, &differ);
    const want = [_]usize{ 4, 6, 10, 14, 20, 25, 26, 32, 35, 36, 38, 39, 40, 42, 44, 52, 56, 58 };
    try std.testing.expectEqualSlices(usize, &want, at);
    try expectUlps(&old, &new, 1);
}
