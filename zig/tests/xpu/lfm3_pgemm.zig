//! pgemm, the prompt windows' GEMM: MLX 4-bit in groups of 64 and 32 against FP64; each row alone = in a window.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;
const kernels = @import("xpu").kernels;

/// W[out][i] = q * scale + bias of an MLX 4-bit table in groups of `group`, FP64.
fn dequant(w: []const u32, s: []const f64, b: []const f64, outs: usize, in_dim: usize, group: usize) ![]f64 {
    const d = try o.gpa.alloc(f64, outs * in_dim);
    for (0..outs) |r| for (0..in_dim) |i| {
        const q: f64 = @floatFromInt((w[r * (in_dim / 8) + i / 8] >> @intCast(4 * (i % 8))) & 15);
        const g = r * (in_dim / group) + i / group;
        d[r * in_dim + i] = q * s[g] + b[g];
    };
    return d;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var vm = try r.module(kernels.lfm3_mv);
    var k = try vm.kernel("pgemm", .{ 16, 16, 1 });
    var t: o.Tally = .{ .name = "lfm3_pgemm" };
    var prng = std.Random.DefaultPrng.init(64);
    const rnd = prng.random();
    std.debug.print("pgemm: 4-bit tables in groups of 64 and 32, against FP64; rows alone and in a window\n", .{});
    var buf: [96]u8 = undefined;
    const outs: u32 = 1000; // not a multiple of the 64-wide tile
    const ld: u32 = outs + 24;
    const off: u32 = 16;
    for ([_]u32{ 2048, 7680 }) |in_dim| for ([_]u32{ 64, 32 }) |group| {
        const wh = try o.gpa.alloc(u32, outs * in_dim / 8);
        for (wh) |*v| v.* = rnd.int(u32);
        const w = try r.alloc(wh.len * 4);
        try r.upload(w, std.mem.sliceAsBytes(wh));
        const sh = try o.gpa.alloc(u16, outs * in_dim / group);
        const bh = try o.gpa.alloc(u16, sh.len);
        o.randBf(sh, rnd, 0.02);
        o.randBf(bh, rnd, 0.1);
        const s = try o.upload(&r, sh);
        const b = try o.upload(&r, bh);
        const wd = try dequant(wh, try o.widen(o.gpa, sh), try o.widen(o.gpa, bh), outs, in_dim, group);
        const rows: u32 = 130;
        const xh = try o.gpa.alloc(u16, rows * in_dim);
        o.randBf(xh, rnd, 1.0);
        const x = try o.upload(&r, xh);
        const xf = try o.widen(o.gpa, xh);
        const want = try o.gpa.alloc(f64, rows * outs);
        for (0..rows) |i| for (0..outs) |j| {
            var acc: f64 = 0;
            for (0..in_dim) |c| acc += xf[i * in_dim + c] * wd[j * in_dim + c];
            want[i * outs + j] = acc;
        };
        const y = try r.alloc(rows * ld * 2);
        const one = try r.alloc(rows * outs * 2);
        for (0..rows) |i| try o.run(&k, .{ (outs + 63) / 64, 1, 1 }, .{ w, s, b, sub(x, i * in_dim * 2), sub(one, i * outs * 2), in_dim, outs, @as(u32, 1), outs, @as(u32, 0), group });
        const alone = try o.fetch(&r, one, rows * outs);
        t.check(try std.fmt.bufPrint(&buf, "groups of {d}, in {d}, rows one at a time", .{ group, in_dim }), o.relErr(alone, want));
        for ([_]u32{ 5, 64, 130 }) |m| {
            try o.run(&k, .{ (outs + 63) / 64, (m + 63) / 64, 1 }, .{ w, s, b, x, y, in_dim, outs, m, ld, off, group });
            const got = try o.fetch(&r, y, m * ld);
            var eq = true;
            for (0..m) |i| eq = eq and std.mem.eql(u16, got[i * ld + off ..][0..outs], alone[i * outs ..][0..outs]);
            t.same(try std.fmt.bufPrint(&buf, "groups of {d}, in {d}, a window of {d:>3} rows (stride, offset) = alone", .{ group, in_dim, m }), eq);
        }
        // planted control: the table read in the other group size must not match
        try o.run(&k, .{ (outs + 63) / 64, 1, 1 }, .{ w, s, b, x, one, in_dim, outs, @as(u32, 1), outs, @as(u32, 0), 96 - group });
        t.control(try std.fmt.bufPrint(&buf, "groups of {d} read as {d}, in {d}", .{ group, 96 - group, in_dim }), o.relErr(try o.fetch(&r, one, outs), want[0..outs]));
    };
    try t.finish();
}

fn sub(b: o.Buf, off: usize) o.Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}
