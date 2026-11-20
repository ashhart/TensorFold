//! Independent byte parity of pooled KDA against fast serial outputs and state.
const std = @import("std");
const pool_mod = @import("cpu_gguf_pool.zig");
const kda = @import("kda_fast.zig");
const Shape = kda.Shape;
const dim = kda.dim;
const taps = kda.taps;
const mixState = kda.mixState;
const mixStateOnPool = kda.mixStateOnPool;

test "KDA pool head-parallel mixStateOnPool is byte-equal to mixState across 2, 3, 4, and 8 threads" {
    const a = std.testing.allocator;
    const s = Shape{ .heads = 16, .lower_bound = -5.0, .stored_rate = true, .eps = 1e-6 };
    const W = s.width();
    const max_rows: usize = 4;

    const proj = try a.alloc(f32, max_rows * 3 * W);
    defer a.free(proj);
    const fgate = try a.alloc(f32, max_rows * W);
    defer a.free(fgate);
    const braw = try a.alloc(f32, max_rows * s.heads);
    defer a.free(braw);
    const g2 = try a.alloc(f32, max_rows * W);
    defer a.free(g2);
    const cq = try a.alloc(f32, W * taps);
    defer a.free(cq);
    const ck = try a.alloc(f32, W * taps);
    defer a.free(ck);
    const cv = try a.alloc(f32, W * taps);
    defer a.free(cv);
    var a_log: [16]f32 = undefined;
    const dt = try a.alloc(f32, W);
    defer a.free(dt);
    var on: [dim]f32 = undefined;

    for (proj, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 19;
    for (fgate, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 17;
    for (braw, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 13;
    for (g2, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 11;
    for (cq, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 21;
    for (ck, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 15)) - 7)) / 23;
    for (cv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 17;
    for (&a_log, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(i)) / 20;
    for (dt, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 10;
    for (&on, 0..) |*v, i| v.* = 0.75 + @as(f32, @floatFromInt(i % 9)) / 16;

    const init_S = try a.alloc(f32, s.heads * dim * dim);
    defer a.free(init_S);
    for (init_S, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 64;
    const init_hist = try a.alloc([3]f32, 3 * W);
    defer a.free(init_hist);
    for (init_hist, 0..) |*h, i| h.* = .{
        @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 15,
        @as(f32, @floatFromInt(@as(i32, @intCast((i + 3) % 13)) - 6)) / 15,
        @as(f32, @floatFromInt(@as(i32, @intCast((i + 7) % 13)) - 6)) / 15,
    };

    const want_y = try a.alloc(f32, max_rows * W);
    defer a.free(want_y);
    const want_S = try a.alloc(f32, init_S.len);
    defer a.free(want_S);
    const want_hist = try a.alloc([3]f32, init_hist.len);
    defer a.free(want_hist);

    const got_y = try a.alloc(f32, max_rows * W);
    defer a.free(got_y);
    const got_S = try a.alloc(f32, init_S.len);
    defer a.free(got_S);
    const got_hist = try a.alloc([3]f32, init_hist.len);
    defer a.free(got_hist);

    for ([_]u32{ 1, 4 }) |rows| {
        @memcpy(want_S, init_S);
        @memcpy(want_hist, init_hist);
        try mixState(want_y[0 .. rows * W], want_S, proj[0 .. rows * 3 * W], fgate[0 .. rows * W], braw[0 .. rows * s.heads], g2[0 .. rows * W], cq, ck, cv, &a_log, dt, &on, s, rows, want_hist);

        for ([_]usize{ 2, 3, 4, 8 }) |threads| {
            const pool = try pool_mod.Pool.create(a, std.testing.io, threads);
            defer pool.destroy();
            @memset(got_y[0 .. rows * W], std.math.nan(f32));
            @memcpy(got_S, init_S);
            @memcpy(got_hist, init_hist);
            try mixStateOnPool(pool, got_y[0 .. rows * W], got_S, proj[0 .. rows * 3 * W], fgate[0 .. rows * W], braw[0 .. rows * s.heads], g2[0 .. rows * W], cq, ck, cv, &a_log, dt, &on, s, rows, got_hist);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_y[0 .. rows * W]), std.mem.sliceAsBytes(got_y[0 .. rows * W]));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_S), std.mem.sliceAsBytes(got_S));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_hist), std.mem.sliceAsBytes(got_hist));
        }
    }
}
