//! Tuned MLX 4-bit decode matvec vs the earlier qmv4_bf and an fp64 host reference on Qwen3.8-27B shapes.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

const spv_new = @import("xpu").kernels.qwen_mlx4;
const spv_old = @import("xpu").kernels.qwen_basic;
const alloc = std.heap.page_allocator;

const Shape = struct { name: []const u8, in: u32, rows: u32 };
const shapes = [_]Shape{
    .{ .name = "gate/up", .in = 5120, .rows = 17408 },
    .{ .name = "down", .in = 17408, .rows = 5120 },
    .{ .name = "in_proj_qkv", .in = 5120, .rows = 10240 },
    .{ .name = "in_proj_z", .in = 5120, .rows = 6144 },
    .{ .name = "out_proj/o_proj", .in = 6144, .rows = 5120 },
    .{ .name = "q_proj", .in = 5120, .rows = 12288 },
    .{ .name = "k/v", .in = 5120, .rows = 1024 },
    .{ .name = "lm_head", .in = 5120, .rows = 248320 },
};

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

fn upload(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(@max(bytes.len, 16));
    try r.upload(b, bytes);
    try r.sync();
    return b;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    var mn = try r.module(spv_new);
    defer mn.deinit();
    var mo = try r.module(spv_old);
    defer mo.deinit();
    var rng = std.Random.DefaultPrng.init(42);
    const rnd = rng.random();
    var worst: f64 = 0;
    for (shapes) |sh| {
        const words = sh.in / 8;
        const groups = sh.in / 64;
        const nw = @as(usize, sh.rows) * words;
        const ns = @as(usize, sh.rows) * groups;
        const w = try alloc.alloc(u32, nw);
        defer alloc.free(w);
        rnd.bytes(std.mem.sliceAsBytes(w));
        const sc = try alloc.alloc(u16, ns);
        defer alloc.free(sc);
        const bi = try alloc.alloc(u16, ns);
        defer alloc.free(bi);
        for (sc, bi) |*s, *b| {
            s.* = toBf((rnd.float(f32) - 0.5) * 0.02);
            b.* = toBf((rnd.float(f32) - 0.5) * 0.2);
        }
        const x = try alloc.alloc(u16, sh.in);
        defer alloc.free(x);
        for (x) |*v| v.* = toBf(rnd.floatNorm(f32));
        var bw = try upload(&r, std.mem.sliceAsBytes(w));
        defer bw.free();
        var bs = try upload(&r, std.mem.sliceAsBytes(sc));
        defer bs.free();
        var bb = try upload(&r, std.mem.sliceAsBytes(bi));
        defer bb.free();
        var bx = try upload(&r, std.mem.sliceAsBytes(x));
        defer bx.free();
        var y_old = try r.alloc(@as(usize, sh.rows) * 2);
        defer y_old.free();
        var y_new = try r.alloc(@as(usize, sh.rows) * 2);
        defer y_new.free();
        const R: u32 = 1;
        var kn = try mn.kernel("qmv4x", .{ 16, 1, 1 });
        defer kn.deinit();
        var ko = try mo.kernel("qmv4_bf", .{ 64, 1, 1 });
        defer ko.deinit();
        try ko.setBuffer(0, bw);
        try ko.setBuffer(1, bs);
        try ko.setBuffer(2, bb);
        try ko.setBuffer(3, bx);
        try ko.setBuffer(4, y_old);
        try ko.setU32(5, sh.in);
        try ko.setU32(6, 0);
        try ko.setU32(7, 0);
        try ko.setU32(8, sh.rows);
        try kn.setBuffer(0, bw);
        try kn.setBuffer(1, bs);
        try kn.setBuffer(2, bb);
        try kn.setBuffer(3, bx);
        try kn.setBuffer(4, y_new);
        try kn.setU32(5, sh.in);
        try kn.setU32(6, 0);
        try kn.setU32(7, 0);
        try kn.setU32(8, sh.rows);
        try kn.setU32(9, 0);
        try ko.launch(.{ (sh.rows + 3) / 4, 1, 1 });
        try kn.launch(.{ (sh.rows + R - 1) / R, 1, 1 });
        const ho = try alloc.alloc(u16, sh.rows);
        defer alloc.free(ho);
        const hn = try alloc.alloc(u16, sh.rows);
        defer alloc.free(hn);
        try r.download(std.mem.sliceAsBytes(ho), y_old);
        try r.download(std.mem.sliceAsBytes(hn), y_new);
        try r.sync();
        // fp64 reference on every 97th row; relative to the row's magnitude scale (max |y| of the checked rows)
        var max_ref: f64 = 0;
        var e_old: f64 = 0;
        var e_new: f64 = 0;
        var differ: usize = 0;
        for (ho, hn) |a, b| differ += @intFromBool(a != b);
        var row: usize = 0;
        while (row < sh.rows) : (row += 97) {
            var acc: f64 = 0;
            for (0..sh.in) |i| {
                const q: f64 = @floatFromInt((w[row * words + i / 8] >> @intCast(4 * (i % 8))) & 15);
                const g = i / 64;
                acc += @as(f64, bf(x[i])) * (q * @as(f64, bf(sc[row * groups + g])) + @as(f64, bf(bi[row * groups + g])));
            }
            max_ref = @max(max_ref, @abs(acc));
            e_old = @max(e_old, @abs(@as(f64, bf(ho[row])) - acc));
            e_new = @max(e_new, @abs(@as(f64, bf(hn[row])) - acc));
        }
        worst = @max(worst, e_new / max_ref);
        // timing: weights are one copy, larger than L2 for all shapes but k/v
        const bytes: f64 = @floatFromInt(nw * 4 + ns * 4);
        const iters = 40;
        var best_o: u64 = std.math.maxInt(u64);
        var best_n: u64 = std.math.maxInt(u64);
        // small shapes cycle over copies of the weights so the 18 MB L2 does not serve them
        const ncopy: usize = @as(usize, @intFromFloat(200e6 / bytes)) + 1;
        const cw = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(cw);
        const cs = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(cs);
        const cb = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(cb);
        cw[0] = bw;
        cs[0] = bs;
        cb[0] = bb;
        for (1..ncopy) |i| {
            cw[i] = try upload(&r, std.mem.sliceAsBytes(w));
            cs[i] = try upload(&r, std.mem.sliceAsBytes(sc));
            cb[i] = try upload(&r, std.mem.sliceAsBytes(bi));
        }
        for (0..7) |_| {
            var t0 = nowNs();
            for (0..iters) |i| {
                try ko.setBuffer(0, cw[i % ncopy]);
                try ko.setBuffer(1, cs[i % ncopy]);
                try ko.setBuffer(2, cb[i % ncopy]);
                try ko.launch(.{ (sh.rows + 3) / 4, 1, 1 });
            }
            try r.sync();
            best_o = @min(best_o, (nowNs() - t0) / iters);
            t0 = nowNs();
            for (0..iters) |i| {
                try kn.setBuffer(0, cw[i % ncopy]);
                try kn.setBuffer(1, cs[i % ncopy]);
                try kn.setBuffer(2, cb[i % ncopy]);
                try kn.launch(.{ (sh.rows + R - 1) / R, 1, 1 });
            }
            try r.sync();
            best_n = @min(best_n, (nowNs() - t0) / iters);
        }
        for (1..ncopy) |i| {
            var a = cw[i];
            a.free();
            a = cs[i];
            a.free();
            a = cb[i];
            a.free();
        }
        std.debug.print("{s:16} {d:>6} x {d:<6} R={d}: old {d:7.1} us {d:6.1} GB/s | new {d:7.1} us {d:6.1} GB/s ({d:.1} MB) | {d} of {d} bf16 differ, err vs fp64 old {e:.2} new {e:.2} (of max |y| {e:.2})\n", .{ sh.name, sh.in, sh.rows, R, @as(f64, @floatFromInt(best_o)) / 1000, bytes / @as(f64, @floatFromInt(best_o)), @as(f64, @floatFromInt(best_n)) / 1000, bytes / @as(f64, @floatFromInt(best_n)), bytes / 1e6, differ, sh.rows, e_old, e_new, max_ref });
    }
    std.debug.print("worst new error / max|y|: {e}\n", .{worst});
    if (worst > 5e-3) return error.TooInaccurate;
}
