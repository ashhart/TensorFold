//! Launch overhead microbenchmark: N tiny kernels behind one sync. usage: xpu-qwen_launch-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

const spv = @import("xpu").kernels.qwen_basic;

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
    var m = try r.module(spv);
    var k = try m.kernel("add_bf16", .{ 64, 1, 1 });
    var x = try r.alloc(5120 * 2);
    var d = try r.alloc(5120 * 2);
    const n: u32 = 5120;
    for ([_]u32{ 100, 1000, 4000 }) |count| {
        inline for (.{ true, false }) |set_args| {
            var best_host: u64 = std.math.maxInt(u64);
            var best_tot: u64 = std.math.maxInt(u64);
            for (0..5) |_| {
                const t0 = nowNs();
                if (!set_args) {
                    try k.setBuffer(0, x);
                    try k.setBuffer(1, d);
                    try k.setU32(2, n);
                }
                for (0..count) |_| {
                    if (set_args) {
                        try k.setBuffer(0, x);
                        try k.setBuffer(1, d);
                        try k.setU32(2, n);
                    }
                    try k.launch(.{ n / 64, 1, 1 });
                }
                const t1 = nowNs();
                try r.sync();
                const t2 = nowNs();
                best_host = @min(best_host, t1 - t0);
                best_tot = @min(best_tot, t2 - t0);
            }
            std.debug.print("{d:>5} launches, args {s}: host submit {d:.2} us/launch, with sync {d:.2} us/launch\n", .{ count, if (set_args) "set each launch" else "set once      ", @as(f64, @floatFromInt(best_host)) / 1e3 / @as(f64, @floatFromInt(count)), @as(f64, @floatFromInt(best_tot)) / 1e3 / @as(f64, @floatFromInt(count)) });
        }
    }
    x.free();
    d.free();
}
