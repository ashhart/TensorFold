//! LFM3 prefill and decode timing on the device: a prompt of N random ids in windows of W rows, then 64 greedy tokens.

const std = @import("std");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const stop = @import("xpu").stop;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("skipped: usage tf-xpu-lfm3_bench-test MODEL_DIR [PROMPT_TOKENS] [WINDOW_ROWS]\n", .{});
        return;
    }
    const n: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 1024;
    const w: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 64;
    var r = try rt.open();
    defer r.deinit();
    var c = try lfm3.config.Config.read(gpa, init.io, args[1]);
    defer c.deinit();
    var l = try @import("xpu").loader.Loader.init(gpa, &r, args[1]);
    var m = try lfm3.model.Model.load(gpa, &r, &l, &c, @intCast(n + 128), w);
    var prng = std.Random.DefaultPrng.init(5);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*v| v.* = prng.random().uintLessThan(u32, 120000);
    try m.reset();
    var t0 = nowNs();
    try m.prefill(ids, null);
    var tok = try m.argmax();
    const pre = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    t0 = nowNs();
    for (0..64) |_| {
        try m.forward(&.{tok}, true, null);
        tok = try m.argmax();
    }
    const dec = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    std.debug.print("prompt {d} tokens in windows of {d}: prefill {d:.2} s ({d:.1} tok/s); decode after it {d:.1} tok/s; device {d:.0} MiB\n", .{ n, w, pre, @as(f64, @floatFromInt(n)) / pre, 64.0 / dec, @as(f64, @floatFromInt(m.device_bytes)) / (1 << 20) });
}
