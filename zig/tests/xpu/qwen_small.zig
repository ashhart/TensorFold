//! The 4-sub-group fp16 matvec (qwen_small.cl) vs mv_f16 on 48 x 5120: bf16 agreement and time per launch.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

const spv_new = @import("xpu").kernels.qwen_small;
const spv_old = @import("xpu").kernels.qwen_basic;
const alloc = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn toBf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    var mn = try r.module(spv_new);
    defer mn.deinit();
    var mo = try r.module(spv_old);
    defer mo.deinit();
    var rng = std.Random.DefaultPrng.init(5);
    const rnd = rng.random();
    const rows: u32 = 48;
    const in: u32 = 5120;
    const w = try alloc.alloc(f16, rows * in);
    const x = try alloc.alloc(u16, in);
    for (w) |*v| v.* = @floatCast(rnd.floatNorm(f32) * 0.02);
    for (x) |*v| v.* = toBf(rnd.floatNorm(f32));
    const bw = try r.alloc(w.len * 2);
    const bx = try r.alloc(x.len * 2);
    const yo = try r.alloc(rows * 2);
    const yn = try r.alloc(rows * 2);
    try r.upload(bw, std.mem.sliceAsBytes(w));
    try r.upload(bx, std.mem.sliceAsBytes(x));
    try r.sync();
    var ko = try mo.kernel("mv_f16", .{ 64, 1, 1 });
    var kn = try mn.kernel("mv16x", .{ 64, 1, 1 });
    inline for (.{ &ko, &kn }, .{ yo, yn }) |k, y| {
        try k.setBuffer(0, bw);
        try k.setBuffer(1, bx);
        try k.setBuffer(2, y);
        try k.setU32(3, in);
        try k.setU32(4, 0);
        try k.setU32(5, rows);
    }
    try ko.launch(.{ (rows + 3) / 4, 1, 1 });
    try kn.launch(.{ rows, 1, 1 });
    var ho: [48]u16 = undefined;
    var hn: [48]u16 = undefined;
    try r.download(std.mem.sliceAsBytes(&ho), yo);
    try r.download(std.mem.sliceAsBytes(&hn), yn);
    try r.sync();
    var differ: usize = 0;
    var worst: f64 = 0;
    var max_y: f64 = 0;
    for (0..rows) |row| {
        var acc: f64 = 0;
        for (0..in) |i| acc += @as(f64, bf(x[i])) * @as(f64, @floatCast(w[row * in + i]));
        max_y = @max(max_y, @abs(acc));
        differ += @intFromBool(ho[row] != hn[row]);
        worst = @max(worst, @abs(@as(f64, bf(hn[row])) - acc));
    }
    var best_o: u64 = std.math.maxInt(u64);
    var best_n: u64 = std.math.maxInt(u64);
    for (0..7) |_| {
        var t0 = nowNs();
        for (0..200) |_| try ko.launch(.{ (rows + 3) / 4, 1, 1 });
        try r.sync();
        best_o = @min(best_o, (nowNs() - t0) / 200);
        t0 = nowNs();
        for (0..200) |_| try kn.launch(.{ rows, 1, 1 });
        try r.sync();
        best_n = @min(best_n, (nowNs() - t0) / 200);
    }
    std.debug.print("mv_f16 48 x 5120: old {d:.1} us, new {d:.1} us a launch; {d} of 48 bf16 differ, new error vs fp64 {e:.2} (max |y| {e:.2})\n", .{ @as(f64, @floatFromInt(best_o)) / 1000, @as(f64, @floatFromInt(best_n)) / 1000, differ, worst, max_y });
    if (worst > 0.02 * max_y) return error.TooInaccurate;
}
