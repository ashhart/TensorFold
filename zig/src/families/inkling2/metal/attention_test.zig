//! Host-only Mini window, ring, tau and dispatch/ABI checks; no Metal import.
const std = @import("std");
const p = @import("attention_plan.zig");
const t = std.testing;

fn tau(pos: usize) f64 {
    return 1 + 0.1 * @log(@max(@as(f64, @floatFromInt(pos + 1)) / 8192, 1));
}

test "inkling2 metal ABI matches the shaders" {
    try t.expectEqual(@as(usize, 24), @sizeOf(p.Seg));
    try t.expectEqual(@as(usize, 32), @sizeOf(p.BatchedSeg));
    try t.expectEqual(@as(usize, 0), @offsetOf(p.BatchedSeg, "k_cache"));
    try t.expectEqual(@as(usize, 8), @offsetOf(p.BatchedSeg, "v_cache"));
    try t.expectEqual(@as(usize, 16), @offsetOf(p.BatchedSeg, "row0"));
    try t.expectEqual(@as(usize, 20), @offsetOf(p.BatchedSeg, "n"));
    try t.expectEqual(@as(usize, 24), @offsetOf(p.BatchedSeg, "p0"));
    try t.expectEqual(@as(usize, 28), @offsetOf(p.BatchedSeg, "ring"));
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
    try t.expectEqual(p.Path.decode_gqa, sd.path);
    try t.expectEqual([3]u32{ 64, 5, 1 }, sd.grid);
    try t.expectEqual([3]u32{ 192, 1, 1 }, sd.merge_grid);
    try t.expectEqual(@as(u32, 32), sd.threads);
    const general = try p.attend(.{ .rows = 4, .heads = 16, .kv_heads = 8, .last_pos = 4095 }, sl, 5, 0);
    try t.expectEqual(p.Path.decode, general.path);
    try t.expectEqual([3]u32{ 64, 5, 1 }, general.grid);
    // Single-row sliding/short decode keeps 24 scalar TGs to avoid under-occupying wide GPUs.
    const lone_sl = try p.attend(.{ .rows = 1, .last_pos = 500 }, sl, 5, 0);
    try t.expectEqual(p.Path.decode, lone_sl.path);
    try t.expectEqual([3]u32{ 24, 5, 1 }, lone_sl.grid);
    const gl = p.Layer.global(1048576);
    const one = p.Rows{ .rows = 1, .last_pos = 1048575 };
    const gd = try p.attend(one, gl, 64, 24 * 64 * p.subs - 1);
    try t.expectEqual(p.Path.prompt, gd.path);
    try t.expectEqual([3]u32{ 1, 8, 64 }, gd.grid);
    try t.expectEqual(@as(u32, 256), gd.threads);
    const streamed = p.Rows{ .rows = 1, .last_pos = 1048575, .streams = 2 };
    const scalar = try p.attend(streamed, gl, 64, 0);
    try t.expectEqual(p.Path.decode_gqa, scalar.path);
    try t.expectEqual([3]u32{ 8, 64, 1 }, scalar.grid);
    try t.expectError(error.Inkling2Partials, p.attend(streamed, gl, 63, 0));
    const prompt = try p.attend(.{ .rows = 130, .last_pos = 20000 }, gl, 2, 130 * 24 * 2);
    try t.expectEqual(p.Path.prompt, prompt.path);
    try t.expectEqual([3]u32{ 4, 8, 2 }, prompt.grid); // ceil(130 * 3 / 128)
}

test "inkling2 metal GQA 3:1 guard, activeChunks, and batched segment planning" {
    try t.expect(p.useGqa3(.{ .rows = 1, .heads = 24, .kv_heads = 8, .last_pos = 0 }));
    try t.expect(!p.useGqa3(.{ .rows = 1, .heads = 16, .kv_heads = 8, .last_pos = 0 }));
    try t.expect(!p.useGqa3(.{ .rows = 1, .heads = 0, .kv_heads = 0, .last_pos = 0 }));

    const sl = p.Layer.sliding(1024, 1);
    try t.expectEqual(@as(u32, 1), p.activeChunks(sl, 0));
    try t.expectEqual(@as(u32, 1), p.activeChunks(sl, 255));
    try t.expectEqual(@as(u32, 2), p.activeChunks(sl, 256));
    try t.expectEqual(@as(u32, 4), p.activeChunks(sl, 1023));
    try t.expectEqual(@as(u32, 5), p.activeChunks(sl, 1024));
    try t.expectEqual(@as(u32, 5), p.activeChunks(sl, 99999));

    const gl = p.Layer.global(65536);
    try t.expectEqual(@as(u32, 1), p.activeChunks(gl, 16383));
    try t.expectEqual(@as(u32, 2), p.activeChunks(gl, 16384));

    const sl_segs = [_]p.BatchedSegSpec{
        .{ .rows = 1, .last_pos = 40, .layer = .{ .window = 1024, .ring = 1280, .chunk = 256, .tau = false }, .is_tree = false },
        .{ .rows = 2, .last_pos = 600, .layer = .{ .window = 1024, .ring = 1408, .chunk = 256, .tau = false }, .is_tree = false },
        .{ .rows = 1, .last_pos = 5000, .layer = .{ .window = 1024, .ring = 1536, .chunk = 256, .tau = false }, .is_tree = false },
    };
    const b = p.planBatched(&sl_segs, 24, 8, 64, 0).?;
    try t.expectEqual(@as(u32, 4), b.total_rows);
    try t.expectEqual(@as(u32, 1024), b.window);
    try t.expectEqual(@as(u32, 256), b.chunk);
    try t.expect(!b.tau);
    try t.expectEqual(@as(u32, 5), b.max_chunks);
    try t.expectEqual([3]u32{ 32, 5, 1 }, b.grid);
    try t.expectEqual([3]u32{ 96, 1, 1 }, b.merge_grid);
    try t.expectEqual(@as(u32, 32), b.threads);

    // Short-context batch uses tight max_chunks = 2 instead of 5 or 64.
    const short_b = p.planBatched(sl_segs[0..2], 24, 8, 64, 0).?;
    try t.expectEqual(@as(u32, 3), short_b.max_chunks);
    try t.expectEqual([3]u32{ 24, 3, 1 }, short_b.grid);

    // Reject single-segment, non-3:1 GQA, tree segment, mismatched layer, or stride overflow.
    try t.expect(p.planBatched(sl_segs[0..1], 24, 8, 64, 0) == null);
    try t.expect(p.planBatched(&sl_segs, 16, 8, 64, 0) == null);
    try t.expect(p.planBatched(&sl_segs, 24, 8, 4, 0) == null);
    var with_tree = sl_segs;
    with_tree[1].is_tree = true;
    try t.expect(p.planBatched(&with_tree, 24, 8, 64, 0) == null);
    var mismatched = sl_segs;
    mismatched[2].layer.window = 0;
    try t.expect(p.planBatched(&mismatched, 24, 8, 64, 0) == null);

    // Global segments fall back when slots == 0 (preserves prompt path), and batch when split decode is active.
    const gl_segs = [_]p.BatchedSegSpec{
        .{ .rows = 1, .last_pos = 500, .layer = p.Layer.global(4096), .is_tree = false },
        .{ .rows = 2, .last_pos = 17450, .layer = p.Layer.global(32768), .is_tree = false },
    };
    try t.expect(p.planBatched(&gl_segs, 24, 8, 64, 0) == null);
    const gl_b = p.planBatched(&gl_segs, 24, 8, 64, 64 * 24 * 64 * p.subs).?;
    try t.expectEqual(@as(u32, 3), gl_b.total_rows);
    try t.expectEqual(@as(u32, 2), gl_b.max_chunks);
    try t.expect(gl_b.tau);
    try t.expectEqual([3]u32{ 24, 2, 1 }, gl_b.grid);
}
