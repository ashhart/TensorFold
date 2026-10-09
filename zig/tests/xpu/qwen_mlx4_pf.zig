//! MLX 4-bit prefill GEMM: accuracy vs float64, chunk invariance, layouts agree. usage: xpu-qwen_mlx4_pf-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const pf = @import("qwen_xpu").mlx4_pf;
const mxb = @import("qwen_xpu").mlx4b;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn bfBits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

const Shape = struct { name: []const u8, rows: u32, in: u32 };

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var p = try pf.Pf.init(&r);
    const shapes = [_]Shape{
        .{ .name = "gate/up 17408x5120", .rows = 17408, .in = 5120 },
        .{ .name = "down 5120x17408", .rows = 5120, .in = 17408 },
        .{ .name = "qkv 10240x5120", .rows = 10240, .in = 5120 },
        .{ .name = "q 12288x5120", .rows = 12288, .in = 5120 },
        .{ .name = "z 6144x5120", .rows = 6144, .in = 5120 },
        .{ .name = "out 5120x6144", .rows = 5120, .in = 6144 },
        .{ .name = "k/v 1024x5120", .rows = 1024, .in = 5120 },
    };
    const Rmax: u32 = 2048;
    for (shapes) |sh| {
        const n: usize = @as(usize, sh.rows) * sh.in;
        const words = try gpa.alloc(u32, n / 8);
        const sc = try gpa.alloc(u16, n / 64);
        const bi = try gpa.alloc(u16, n / 64);
        for (words) |*w| w.* = rnd.int(u32);
        for (sc) |*s| s.* = bfBits(0.01 + rnd.float(f32) * 0.02);
        for (bi) |*b| b.* = bfBits(-0.15 - rnd.float(f32) * 0.05);
        const wb = try gpa.alloc(u8, n / 2);
        const sb = try gpa.alloc(u8, n / 32);
        const bb = try gpa.alloc(u8, n / 32);
        mxb.repack(sh.in, sh.rows, std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(sc), std.mem.sliceAsBytes(bi), wb, sb, bb);
        const xh = try gpa.alloc(u16, @as(usize, Rmax) * sh.in);
        for (xh) |*x| x.* = bfBits(rnd.floatNorm(f32));
        const dw = try r.alloc(n / 2);
        const ds = try r.alloc(n / 32);
        const db = try r.alloc(n / 32);
        const dwb = try r.alloc(n / 2);
        const dsb = try r.alloc(n / 32);
        const dbb = try r.alloc(n / 32);
        const dx = try r.alloc(xh.len * 2);
        try r.upload(dw, std.mem.sliceAsBytes(words));
        try r.upload(ds, std.mem.sliceAsBytes(sc));
        try r.upload(db, std.mem.sliceAsBytes(bi));
        try r.upload(dwb, wb);
        try r.upload(dsb, sb);
        try r.upload(dbb, bb);
        try r.upload(dx, std.mem.sliceAsBytes(xh));
        const y = try r.alloc(@as(usize, Rmax) * sh.rows * 2);
        const y2 = try r.alloc(@as(usize, Rmax) * sh.rows * 2);
        try r.sync();
        // accuracy on 40 rows x 24 outputs, R = 40
        const R: u32 = 40;
        try p.run(false, dw, ds, db, dx, R, y, sh.in, 0, sh.rows);
        try r.sync();
        const yh = try gpa.alloc(u16, @as(usize, R) * sh.rows);
        try r.download(std.mem.sliceAsBytes(yh), y);
        try r.sync();
        var maxe: f64 = 0;
        var maxy: f64 = 0;
        const groups = sh.in / 64;
        for (0..R) |row| for (0..24) |c| {
            const col = (c * 7919 + 13) % sh.rows;
            var acc: f64 = 0;
            for (0..sh.in) |k| {
                const wd = words[col * (sh.in / 8) + k / 8];
                const q: f64 = @floatFromInt((wd >> @intCast(4 * (k % 8))) & 15);
                const w = q * bf(sc[col * groups + k / 64]) + bf(bi[col * groups + k / 64]);
                acc += w * bf(xh[row * sh.in + k]);
            }
            maxe = @max(maxe, @abs(bf(yh[row * sh.rows + col]) - acc));
            maxy = @max(maxy, @abs(acc));
        };
        // chunk invariance: rows [0,13) then [13,40) == all 40; row layout == block layout
        try p.run(false, dw, ds, db, dx, 13, y2, sh.in, 0, sh.rows);
        const dx2: rt.Buffer = .{ .rt = dx.rt, .ptr = @ptrFromInt(@intFromPtr(dx.ptr.?) + 13 * sh.in * 2), .len = dx.len - 13 * sh.in * 2 };
        const y2b: rt.Buffer = .{ .rt = y2.rt, .ptr = @ptrFromInt(@intFromPtr(y2.ptr.?) + 13 * sh.rows * 2), .len = y2.len - 13 * sh.rows * 2 };
        try p.run(false, dw, ds, db, dx2, 27, y2b, sh.in, 0, sh.rows);
        try r.sync();
        const y2h = try gpa.alloc(u16, @as(usize, R) * sh.rows);
        try r.download(std.mem.sliceAsBytes(y2h), y2);
        try r.sync();
        const chunk_same = std.mem.eql(u16, yh, y2h);
        try p.run(true, dwb, dsb, dbb, dx, R, y2, sh.in, 0, sh.rows);
        try r.sync();
        try r.download(std.mem.sliceAsBytes(y2h), y2);
        try r.sync();
        const blk_same = std.mem.eql(u16, yh, y2h);
        std.debug.print("{s}: max abs error vs fp64 {e:.2} (max |y| {d:.2}); chunked == one pass: {s}; block layout == row layout: {s}\n", .{ sh.name, maxe, maxy, if (chunk_same) "yes" else "NO", if (blk_same) "yes" else "NO" });
        // speed
        for ([_]u32{ 128, 512, 2048 }) |RR| {
            var best: u64 = std.math.maxInt(u64);
            for (0..4) |_| {
                const t0 = nowNs();
                for (0..4) |_| try p.run(false, dw, ds, db, dx, RR, y, sh.in, 0, sh.rows);
                try r.sync();
                best = @min(best, nowNs() - t0);
            }
            const us = @as(f64, @floatFromInt(best)) / 1e3 / 4.0;
            const fl = 2.0 * @as(f64, @floatFromInt(RR)) * @as(f64, @floatFromInt(sh.rows)) * @as(f64, @floatFromInt(sh.in));
            std.debug.print("    R={d:>4}: {d:>8.1} us, {d:>5.1} TFLOP/s (decode + GEMM)\n", .{ RR, us, fl / us / 1e6 });
        }
        for ([_]rt.Buffer{ dw, ds, db, dwb, dsb, dbb, dx, y, y2 }) |bufx| {
            var bb2 = bufx;
            bb2.free();
        }
        gpa.free(words);
        gpa.free(sc);
        gpa.free(bi);
        gpa.free(wb);
        gpa.free(sb);
        gpa.free(bb);
        gpa.free(xh);
        gpa.free(yh);
        gpa.free(y2h);
    }
}
