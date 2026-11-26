//! Host-only Mini window, ring, tau and dispatch/ABI checks; no Metal import.
const std = @import("std");
const p = @import("attention_plan.zig");
const t = std.testing;

fn tau(pos: usize) f64 {
    return 1 + 0.1 * @log(@max(@as(f64, @floatFromInt(pos + 1)) / 8192, 1));
}

test "inkling2 metal ABI matches the shaders" {
    try t.expectEqual(@as(usize, 24), @sizeOf(p.Seg));
    try t.expectEqual(@as(usize, 36), @sizeOf(p.PrepArgs));
    try t.expectEqual(@as(usize, 24), @sizeOf(p.MergeArgs));
    try t.expectEqual(@as(usize, 28), @sizeOf(p.PromptArgs));
    try t.expectEqual(@as(u32, 24), (p.Rows{ .rows = 1, .last_pos = 0 }).heads);
    try t.expectEqual(@as(u32, 8), (p.Rows{ .rows = 1, .last_pos = 0 }).kv_heads);
    try t.expectEqual(@as(f32, 8192), (p.Prep{}).floor);
}

test "inkling2 metal Mini ring and window boundaries" {
    try t.expectEqual(@as(u32, 1024), p.ringFor(1024, 1));
    try t.expectEqual(@as(u32, 1088), p.ringFor(1024, 2));
    try t.expectEqual(@as(u32, 3072), p.Layer.sliding(1024, 2048).ring);
    const l = p.Layer.sliding(1024, 1);
    try t.expectEqual(@as(u32, 5), p.slotsFor(l, 1048576));
    try t.expectEqual(@as(?usize, 3), p.partialSlot(l, 3, 1023));
    try t.expectEqual(@as(?usize, null), p.partialSlot(l, 4, 1023));
    try t.expectEqual(@as(?usize, 4), p.partialSlot(l, 4, 1024));
    try t.expectEqual(@as(?usize, 3), p.partialSlot(l, 7, 2047));
    try t.expectEqual(@as(usize, 1023), @as(usize, 8191) % 1024);
    try t.expectEqual(@as(usize, 0), @as(usize, 8192) % 8192);
}

test "inkling2 metal tau starts strictly above its 8192 floor" {
    try t.expectEqual(@as(f64, 1), tau(0));
    try t.expectEqual(@as(f64, 1), tau(8191));
    try t.expect(tau(8192) > 1);
    try t.expectApproxEqAbs(1 + 0.1 * @log(@as(f64, 2)), tau(16383), 1e-15);
}

test "inkling2 metal decode and prompt dispatch geometry" {
    const sl = p.Layer.sliding(1024, 8);
    const sr = p.Rows{ .rows = 8, .last_pos = 4095, .streams = 2 };
    const sd = try p.attend(sr, sl, 5, 8 * 24 * 5);
    try t.expectEqual(p.Path.decode, sd.path);
    try t.expectEqual([3]u32{ 192, 5, 1 }, sd.grid);
    try t.expectEqual([3]u32{ 192, 1, 1 }, sd.merge_grid);
    try t.expectEqual(@as(u32, 32), sd.threads);
    const gl = p.Layer.global(1048576);
    const one = p.Rows{ .rows = 1, .last_pos = 1048575 };
    const gd = try p.attend(one, gl, 64, 24 * 64 * p.subs - 1);
    try t.expectEqual(p.Path.prompt, gd.path);
    try t.expectEqual([3]u32{ 1, 8, 64 }, gd.grid);
    try t.expectEqual(@as(u32, 256), gd.threads);
    const streamed = p.Rows{ .rows = 1, .last_pos = 1048575, .streams = 2 };
    const scalar = try p.attend(streamed, gl, 64, 0);
    try t.expectEqual(p.Path.decode, scalar.path);
    try t.expectEqual([3]u32{ 24, 64, 1 }, scalar.grid);
    try t.expectError(error.Inkling2Partials, p.attend(streamed, gl, 63, 0));
    const prompt = try p.attend(.{ .rows = 130, .last_pos = 20000 }, gl, 2, 130 * 24 * 2);
    try t.expectEqual(p.Path.prompt, prompt.path);
    try t.expectEqual([3]u32{ 4, 8, 2 }, prompt.grid); // ceil(130 * 3 / 128)
    const sl256 = p.Layer.sliding(1024, 256);
    const s1 = try p.attend(.{ .rows = 1, .last_pos = 1024 }, sl, 5, 0);
    try t.expectEqual(p.Path.decode, s1.path);
    const sp2 = try p.attend(.{ .rows = 2, .last_pos = 1024 }, sl, 5, 0);
    try t.expectEqual(p.Path.sliding_prompt, sp2.path);
    try t.expectEqual([3]u32{ 1, 8, 5 }, sp2.grid);
    try t.expectEqual(@as(u32, 0), sp2.c0);
    try t.expectEqual(@as(u32, 256), sp2.threads);
    const sp8 = try p.attend(.{ .rows = 8, .last_pos = 1024 }, sl, 5, 0);
    try t.expectEqual(p.Path.sliding_prompt, sp8.path);
    try t.expectEqual([3]u32{ 4, 8, 5 }, sp8.grid);
    try t.expectEqual(@as(u32, 0), sp8.c0);
    try t.expectEqual(@as(u32, 256), sp8.threads);
    const sp256 = try p.attend(.{ .rows = 256, .last_pos = 2255 }, sl256, 5, 0);
    try t.expectEqual(p.Path.sliding_prompt, sp256.path);
    try t.expectEqual([3]u32{ 128, 8, 6 }, sp256.grid);
    try t.expectEqual(@as(u32, 3), sp256.c0);
    const sp_short = try p.attend(.{ .rows = 64, .last_pos = 500 }, p.Layer.sliding(1024, 64), 5, 0);
    try t.expectEqual(p.Path.sliding_prompt, sp_short.path);
    try t.expectEqual([3]u32{ 32, 8, 2 }, sp_short.grid);
    try t.expectEqual(@as(u32, 0), sp_short.c0);
}
