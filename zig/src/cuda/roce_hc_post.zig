//! TP=2's GLM hc_post epilogue inside the one-shot RoCE gather.
const std = @import("std");
const abi = @import("abi.zig");
const launch = @import("launch.zig");
const roce = @import("roce.zig");

/// Gathered fp32 rows and xout keep the separate gather and hc_post bytes.
pub fn gather(g: *roce.Gather, stream: abi.Stream, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize, x: abi.DevicePtr, xout: abi.DevicePtr, hc_post_values: abi.DevicePtr, comb: abi.DevicePtr, rows: usize, d: usize) !void {
    if (g.world != 2 or rows == 0 or d % 4 != 0 or !std.math.isPowerOfTwo(d) or bytes != rows * d * 4) return error.BadShape;
    if (send % 16 != 0 or recv % 16 != 0 or x % 16 != 0 or xout % 16 != 0) return error.Misaligned;
    const packs = bytes / 16;
    const need = @max(1, std.math.divCeil(usize, packs, 2 * roce.threads) catch unreachable);
    const grid: u32 = @intCast(@min(std.math.ceilPowerOfTwo(usize, need) catch roce.max_blocks, roce.max_blocks));
    const klass: i32 = @intCast(std.math.log2_int(u32, grid));
    const base = g.region_dev;
    var a: launch.Args = .{};
    a.add(send);
    a.add(recv);
    a.add(@as(i32, @intCast(packs)));
    a.add(@as(i32, g.rank));
    a.add(base + g.layout.recv);
    a.add(base + g.layout.flags);
    a.add(base + g.layout.send);
    a.add(base + g.layout.ctrl);
    a.add(@as(i64, @intCast(g.layout.slot_bytes)));
    a.add(g.counters.ptr);
    a.add(klass);
    a.add(roce.spin_limit);
    a.add(@as(i32, @intCast(g.plan.stripes)));
    a.add(@as(i32, g.world));
    a.add(x);
    a.add(xout);
    a.add(hc_post_values);
    a.add(comb);
    a.add(@as(i64, @intCast(d)));
    a.add(@as(i32, @intCast(std.math.log2_int(usize, d))));
    try launch.launch(g.hc_post, .{ .grid = .{ .x = grid }, .block = .{ .x = roce.threads } }, .{ .handle = stream, .d = g.d }, &a);
    g.gathers += 1;
}
