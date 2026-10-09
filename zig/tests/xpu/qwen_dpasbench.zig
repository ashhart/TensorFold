//! DPAS ceiling: matrix-engine rate (dpas_loop) and a 32 x 64 micro-tile (dpas_tile). usage: xpu-qwen_dpasbench-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

const spv = @import("xpu").kernels.qwen_dpasbench;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try rt.open();
    defer r.deinit();
    var m = try r.moduleWith(spv, "-cl-intel-256-GRF-per-thread");
    var loop = try m.kernel("dpas_loop", .{ 64, 1, 1 });
    var tile = try m.kernel("dpas_tile", .{ 64, 1, 1 });
    const gpa = std.heap.page_allocator;
    const nA: usize = 64 * 4 * 16 * 8; // shorts
    const host = try gpa.alloc(u16, 64 * 4 * 128 * 2);
    for (host, 0..) |*v, i| v.* = 0x3c00 + @as(u16, @truncate(i & 0x3f));
    const dA = try r.alloc(nA * 2 + 4096);
    const dB = try r.alloc(host.len * 2);
    try r.upload(dA, std.mem.sliceAsBytes(host[0..nA]));
    try r.upload(dB, std.mem.sliceAsBytes(host));
    const out = try r.alloc(1 << 24);
    try r.sync();
    const iters: u32 = 4096;
    for ([_]u32{ 64, 256, 1024, 4096 }) |wgs| {
        for ([_]u8{ 0, 1 }) |which| {
            var best: u64 = std.math.maxInt(u64);
            for (0..4) |_| {
                const t0 = nowNs();
                if (which == 0) {
                    try loop.setBuffer(0, out);
                    try loop.setU32(1, iters);
                    try loop.launch(.{ wgs, 1, 1 });
                } else {
                    try tile.setBuffer(0, dA);
                    try tile.setBuffer(1, dB);
                    try tile.setBuffer(2, out);
                    try tile.setU32(3, iters / 4);
                    try tile.launch(.{ wgs, 1, 1 });
                }
                try r.sync();
                best = @min(best, nowNs() - t0);
            }
            // 4 sub-groups a workgroup; 4096 flop a DPAS; loop: 8 DPAS an iteration, tile: 16 DPAS an iteration
            const per_it: f64 = if (which == 0) 8 else 16;
            const n_it: f64 = if (which == 0) @floatFromInt(iters) else @floatFromInt(iters / 4);
            const flop = @as(f64, @floatFromInt(wgs)) * 4 * n_it * per_it * 4096;
            std.debug.print("{s}: {d:>5} workgroups: {d:>8.2} ms = {d:>6.1} TFLOP/s\n", .{ if (which == 0) "dpas_loop" else "dpas_tile", wgs, @as(f64, @floatFromInt(best)) / 1e6, flop / @as(f64, @floatFromInt(best)) / 1e3 });
        }
    }
}
