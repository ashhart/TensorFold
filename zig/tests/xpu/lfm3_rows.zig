//! LFM3's window matvecs (pdense, bf16 and fp32 out) against the one-row kernels, bit for bit, on random data.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;
const kernels = @import("xpu").kernels;

const ROWS: u32 = 1024;

fn randWords(r: *o.rt.Runtime, n: usize, rnd: std.Random) !o.Buf {
    const h = try o.gpa.alloc(u32, n);
    for (h) |*v| v.* = rnd.int(u32);
    const b = try r.alloc(n * 4);
    try r.upload(b, std.mem.sliceAsBytes(h));
    return b;
}

fn fetch32(r: *o.rt.Runtime, b: o.Buf, n: usize) ![]u32 {
    const h = try o.gpa.alloc(u32, n);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(h), b);
    try r.sync();
    return h;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var am = try r.module(kernels.attn);
    var pm = try r.module(if (kernels.dpas) kernels.nem_prow else kernels.nem_prow_gen12);
    const tile_outs: u32 = if (kernels.dpas) 8 else 4;
    var k_bf = try am.kernel("qmv4_bf", .{ 64, 1, 1 });
    var k_f32 = try am.kernel("qmv4_f32", .{ 64, 1, 1 });
    var k_pd = try pm.kernel("pdense", .{ 16 * (16 / tile_outs), 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_rows" };
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    std.debug.print("window matvecs (pdense) against the one-row kernels, {d} output rows, bit for bit\n", .{ROWS});
    for ([_]u32{ 2048, 7680 }) |in_dim| {
        const sh = try o.gpa.alloc(u16, ROWS * in_dim / 64);
        const bh = try o.gpa.alloc(u16, sh.len);
        o.randBf(sh, rnd, 0.02);
        o.randBf(bh, rnd, 0.1);
        const w = try randWords(&r, ROWS * in_dim / 8, rnd);
        const s = try o.upload(&r, sh);
        const b = try o.upload(&r, bh);
        const xh = try o.gpa.alloc(u16, 16 * in_dim);
        o.randBf(xh, rnd, 1.0);
        const x = try o.upload(&r, xh);
        const y1 = try r.alloc(16 * ROWS * 4);
        const yn = try r.alloc(16 * ROWS * 4);
        for ([_]u32{ 1, 2, 3, 4, 8, 13, 16 }) |n| {
            var buf: [96]u8 = undefined;
            for (0..n) |z| try o.run(&k_bf, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, y1, in_dim, @as(u32, @intCast(z)) * in_dim, @as(u32, @intCast(z)) * ROWS, ROWS });
            try o.run(&k_pd, .{ ROWS / 16, (n + 7) / 8, 1 }, .{ w, s, b, x, yn, in_dim, ROWS, n, ROWS, @as(u32, 0), @as(u32, 0) });
            t.same(try std.fmt.bufPrint(&buf, "pdense bf16 in {d} n {d:>2} vs qmv4_bf", .{ in_dim, n }), std.mem.eql(u16, try o.fetch(&r, y1, n * ROWS), try o.fetch(&r, yn, n * ROWS)));
            for (0..n) |z| try o.run(&k_f32, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, y1, in_dim, @as(u32, @intCast(z)) * in_dim, @as(u32, @intCast(z)) * ROWS, ROWS });
            try o.run(&k_pd, .{ ROWS / 16, (n + 7) / 8, 1 }, .{ w, s, b, x, yn, in_dim, ROWS, n, ROWS, @as(u32, 0), @as(u32, 1) });
            t.same(try std.fmt.bufPrint(&buf, "pdense fp32 in {d} n {d:>2} vs qmv4_f32", .{ in_dim, n }), std.mem.eql(u32, try fetch32(&r, y1, n * ROWS), try fetch32(&r, yn, n * ROWS)));
        }
    }
    try t.finish();
}
