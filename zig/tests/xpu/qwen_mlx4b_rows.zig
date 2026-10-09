//! Systolic MLX 4-bit matvec (qmv4d) for m=1..16 rows: bit-identical row invariance and accuracy vs fp64.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const mx = @import("qwen_xpu").mlx4b;

const spv = @import("xpu").kernels.qwen_mlx4;
const alloc = std.heap.page_allocator;

const Shape = struct { name: []const u8, in: u32, rows: u32 };
const shapes = [_]Shape{
    .{ .name = "gate/up", .in = 5120, .rows = 17408 },
    .{ .name = "down", .in = 17408, .rows = 5120 },
    .{ .name = "in_proj_qkv", .in = 5120, .rows = 10240 },
    .{ .name = "in_proj_z", .in = 5120, .rows = 6144 },
    .{ .name = "out_proj", .in = 6144, .rows = 5120 },
    .{ .name = "q_proj", .in = 5120, .rows = 12288 },
    .{ .name = "k/v", .in = 5120, .rows = 1024 },
    .{ .name = "lm_head", .in = 5120, .rows = 248320 },
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn toBf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn at(b: rt.Buffer, off: usize) rt.Buffer {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
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
    var m = try r.module(spv);
    defer m.deinit();
    var mr = try mx.Block.init(&r, 17408, 17408);
    defer mr.deinit();
    var k1 = try m.kernel("qmv4x", .{ 16, 1, 1 });
    defer k1.deinit();
    var rng = std.Random.DefaultPrng.init(11);
    const rnd = rng.random();
    var bad: usize = 0;
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
        const x = try alloc.alloc(u16, 16 * sh.in);
        defer alloc.free(x);
        for (x) |*v| v.* = toBf(rnd.floatNorm(f32));
        const xrev = try alloc.alloc(u16, 16 * sh.in);
        defer alloc.free(xrev);
        for (0..16) |i| @memcpy(xrev[i * sh.in ..][0..sh.in], x[(15 - i) * sh.in ..][0..sh.in]);
        const wb = try alloc.alloc(u32, nw);
        defer alloc.free(wb);
        const sb = try alloc.alloc(u16, ns);
        defer alloc.free(sb);
        const bb2 = try alloc.alloc(u16, ns);
        defer alloc.free(bb2);
        mx.repack(sh.in, sh.rows, std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(sc), std.mem.sliceAsBytes(bi), std.mem.sliceAsBytes(wb), std.mem.sliceAsBytes(sb), std.mem.sliceAsBytes(bb2));
        var bw = try upload(&r, std.mem.sliceAsBytes(w));
        defer bw.free();
        var bs = try upload(&r, std.mem.sliceAsBytes(sc));
        defer bs.free();
        var bbi = try upload(&r, std.mem.sliceAsBytes(bi));
        defer bbi.free();
        var dw = try upload(&r, std.mem.sliceAsBytes(wb));
        defer dw.free();
        var ds = try upload(&r, std.mem.sliceAsBytes(sb));
        defer ds.free();
        var db = try upload(&r, std.mem.sliceAsBytes(bb2));
        defer db.free();
        var bx = try upload(&r, std.mem.sliceAsBytes(x));
        defer bx.free();
        var bxr = try upload(&r, std.mem.sliceAsBytes(xrev));
        defer bxr.free();
        var by = try r.alloc(@as(usize, 16) * sh.rows * 2);
        defer by.free();
        var bq = try r.alloc(@as(usize, sh.rows) * 2);
        defer bq.free();
        const bytes: f64 = @floatFromInt(nw * 4 + ns * 4);
        // earlier single-row result of row 0 and its time
        try k1.setBuffer(0, bw);
        try k1.setBuffer(1, bs);
        try k1.setBuffer(2, bbi);
        try k1.setBuffer(3, bx);
        try k1.setBuffer(4, bq);
        try k1.setU32(5, sh.in);
        try k1.setU32(6, 0);
        try k1.setU32(7, 0);
        try k1.setU32(8, sh.rows);
        try k1.setU32(9, 0);
        try k1.launch(.{ sh.rows, 1, 1 });
        const proto = try alloc.alloc(u16, sh.rows);
        defer alloc.free(proto);
        try r.download(std.mem.sliceAsBytes(proto), bq);
        try r.sync();
        // each row alone through m = 1: the reference of the invariance test
        const ref = try alloc.alloc(u16, @as(usize, 16) * sh.rows);
        defer alloc.free(ref);
        for (0..16) |i| try mr.matvec(dw, ds, db, at(bx, i * sh.in * 2), at(by, i * sh.rows * 2), sh.in, sh.rows, 1, 0, false);
        try r.download(std.mem.sliceAsBytes(ref), by);
        try r.sync();
        const got = try alloc.alloc(u16, @as(usize, 16) * sh.rows);
        defer alloc.free(got);
        for ([_]u32{ 1, 2, 3, 4, 5, 7, 8, 9, 12, 16 }) |mm| {
            for ([_]bool{ false, true }) |rev| {
                try mr.matvec(dw, ds, db, if (rev) bxr else bx, by, sh.in, sh.rows, mm, 0, false);
                try r.download(std.mem.sliceAsBytes(got), by);
                try r.sync();
                for (0..mm) |i| {
                    const src = if (rev) 15 - i else i;
                    const ne = !std.mem.eql(u16, got[i * sh.rows ..][0..sh.rows], ref[src * sh.rows ..][0..sh.rows]);
                    if (ne) std.debug.print("  DIFF {s} m={d} rev={} row {d}\n", .{ sh.name, mm, rev, i });
                    bad += @intFromBool(ne);
                }
            }
        }
        // accuracy (every 97th output of row 0)
        var max_ref: f64 = 0;
        var e_d: f64 = 0;
        var e_p: f64 = 0;
        var row: usize = 0;
        while (row < sh.rows) : (row += 97) {
            var acc: f64 = 0;
            for (0..sh.in) |i| {
                const q: f64 = @floatFromInt((w[row * words + i / 8] >> @intCast(4 * (i % 8))) & 15);
                const g = i / 64;
                acc += @as(f64, bf(x[i])) * (q * @as(f64, bf(sc[row * groups + g])) + @as(f64, bf(bi[row * groups + g])));
            }
            max_ref = @max(max_ref, @abs(acc));
            e_d = @max(e_d, @abs(@as(f64, bf(ref[row])) - acc));
            e_p = @max(e_p, @abs(@as(f64, bf(proto[row])) - acc));
        }
        var differ: usize = 0;
        for (ref[0..sh.rows], proto) |a, b| differ += @intFromBool(a != b);
        // times
        const ncopy: usize = @as(usize, @intFromFloat(160e6 / bytes)) + 1;
        const cw = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(cw);
        const cs = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(cs);
        const cb = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(cb);
        cw[0] = dw;
        cs[0] = ds;
        cb[0] = db;
        for (1..ncopy) |i| {
            cw[i] = try upload(&r, std.mem.sliceAsBytes(wb));
            cs[i] = try upload(&r, std.mem.sliceAsBytes(sb));
            cb[i] = try upload(&r, std.mem.sliceAsBytes(bb2));
        }
        const pw = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(pw);
        const ps = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(ps);
        const pb = try alloc.alloc(rt.Buffer, ncopy);
        defer alloc.free(pb);
        pw[0] = bw;
        ps[0] = bs;
        pb[0] = bbi;
        for (1..ncopy) |i| {
            pw[i] = try upload(&r, std.mem.sliceAsBytes(w));
            ps[i] = try upload(&r, std.mem.sliceAsBytes(sc));
            pb[i] = try upload(&r, std.mem.sliceAsBytes(bi));
        }
        var t_proto: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t0 = nowNs();
            for (0..30) |i| {
                try k1.setBuffer(0, pw[i % ncopy]);
                try k1.setBuffer(1, ps[i % ncopy]);
                try k1.setBuffer(2, pb[i % ncopy]);
                try k1.launch(.{ sh.rows, 1, 1 });
            }
            try r.sync();
            t_proto = @min(t_proto, (nowNs() - t0) / 30);
        }
        std.debug.print("{s} {d} x {d} ({d:.1} MB): qmv4x R=1 {d:.1} us ({d:.0} GB/s); systolic err vs fp64 {e:.2} (qmv4x {e:.2}, max |y| {e:.2}), {d} of {d} bf16 differ from qmv4x\n", .{ sh.name, sh.in, sh.rows, bytes / 1e6, @as(f64, @floatFromInt(t_proto)) / 1000, bytes / @as(f64, @floatFromInt(t_proto)), e_d, e_p, max_ref, differ, sh.rows });
        for ([_]u32{ 1, 2, 4, 8, 16 }) |mm| {
            var best: u64 = std.math.maxInt(u64);
            for (0..5) |_| {
                const t0 = nowNs();
                for (0..30) |i| try mr.matvec(cw[i % ncopy], cs[i % ncopy], cb[i % ncopy], bx, by, sh.in, sh.rows, mm, 0, false);
                try r.sync();
                best = @min(best, (nowNs() - t0) / 30);
            }
            const us = @as(f64, @floatFromInt(best)) / 1000;
            std.debug.print("  m={d:<2}: {d:7.1} us = {d:.2}x of qmv4x R=1, {d:.0} GB/s of weight bytes\n", .{ mm, us, us / (@as(f64, @floatFromInt(t_proto)) / 1000), bytes / (us * 1000) });
        }
        for (1..ncopy) |i| {
            var a = pw[i];
            a.free();
            a = ps[i];
            a.free();
            a = pb[i];
            a.free();
            a = cw[i];
            a.free();
            a = cs[i];
            a.free();
            a = cb[i];
            a.free();
        }
    }
    std.debug.print("windows that differ from the single-row result: {d}\n", .{bad});
    if (bad != 0) return error.NotRowInvariant;
}
