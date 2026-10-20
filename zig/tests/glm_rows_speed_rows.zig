//! tf-glm-rows-speed's window rows (GLM_SPEED_ROWS): the 8-bit products at 1 to 16 rows, GPU time a launch.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const speed = @import("glm_rows_speed.zig");
const wide = tf.glm.wide;
const Pool = speed.Pool;
const Io = speed.Io;
const rf = speed.rf;

fn dense(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, pool: Pool, io: Io, m: u32) void {
    const q = pool.at(i);
    wide.dense(p, e, q, .{ .in = rf(io.x), .ldx = q.k, .out = rf(io.y), .ldo = q.n, .rows = m });
}

fn glu(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, pool: Pool, io: Io, m: u32) void {
    wide.glu(p, e, pool.at(i), pool.at(i).n / 2, rf(io.x), rf(io.y), m, 10.0);
}

pub fn multi(b: *const speed.Bench, p: *const wide.Pipes) !void {
    const shapes = [_]struct { name: []const u8, n: usize, k: usize, gated: bool }{
        .{ .name = "kda in", .n = speed.PS, .k = speed.D, .gated = false },
        .{ .name = "kda out", .n = speed.D, .k = speed.W, .gated = false },
        .{ .name = "mla out", .n = speed.D, .k = 16384, .gated = false },
        .{ .name = "shared gate|up", .n = 2 * speed.N, .k = speed.D, .gated = true },
        .{ .name = "shared down", .n = speed.D, .k = speed.N, .gated = false },
        .{ .name = "lm_head", .n = 154880, .k = speed.D, .gated = false },
    };
    std.debug.print("8-bit products at 1 to 16 rows (us a launch; x: over one row):\n", .{});
    for (shapes) |s| {
        const pool = try Pool.init(b.device, s.n, s.k, 8, 1, 1);
        defer pool.deinit();
        var one: f64 = 0;
        std.debug.print("  {s:<16}", .{s.name});
        for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |m| {
            const us = if (s.gated) try b.time(glu, .{ p, pool, b.io, m }) else try b.time(dense, .{ p, pool, b.io, m });
            if (m == 1) one = us;
            std.debug.print(" {d:8.1} ({d:4.2}x)", .{ us, us / one });
        }
        std.debug.print("  {d:6.1} GB/s at 1\n", .{@as(f64, @floatFromInt(pool.bytes)) / one / 1e3});
    }
}

