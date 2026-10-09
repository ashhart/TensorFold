//! Row-invariant multi-row MLX 4-bit matvec (qmv4r) must be bit-identical to single-row qmv4x on every window row.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

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
const Variant = struct { r: u32, w: u32 };
const variants = [_]Variant{  .{ .r = 1, .w = 1 }, .{ .r = 1, .w = 2 }, .{ .r = 1, .w = 4 }, .{ .r = 2, .w = 1 }, .{ .r = 2, .w = 2 }, .{ .r = 2, .w = 4 }, .{ .r = 4, .w = 1 }, .{ .r = 4, .w = 2 }, .{ .r = 4, .w = 4 }, .{ .r = 8, .w = 1 }, .{ .r = 8, .w = 2 }, .{ .r = 16, .w = 1 } };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn toBf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// K splits of the systolic kernel: about 1400 sub-groups, each a multiple of 4 groups.
fn splitsFor(in: u32, rows: u32) u32 {
    const groups = in / 64;
    var best: u32 = 1;
    var score: f64 = 1e9;
    var s: u32 = 1;
    while (s <= 16) : (s += 1) {
        if (groups % s != 0 or (groups / s) % 4 != 0) continue;
        const sc = @abs(@log(@as(f64, @floatFromInt(rows / 16 * s)) / 1400.0));
        if (sc < score) {
            score = sc;
            best = s;
        }
    }
    return best;
}

fn at(b: rt.Buffer, off: usize) rt.Buffer {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn upload(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(@max(bytes.len, 16));
    try r.upload(b, bytes);
    try r.sync();
    return b;
}

fn launchRows(k: *rt.Kernel, v: Variant, bw: rt.Buffer, bs: rt.Buffer, bb: rt.Buffer, bx: rt.Buffer, by: rt.Buffer, in: u32, rows: u32, m: u32) !void {
    try k.setBuffer(0, bw);
    try k.setBuffer(1, bs);
    try k.setBuffer(2, bb);
    try k.setBuffer(3, bx);
    try k.setBuffer(4, by);
    try k.setU32(5, in);
    try k.setU32(6, 0);
    try k.setU32(7, rows);
    try k.setU32(8, rows);
    try k.setU32(9, m);
    try k.setU32(10, 0);
    try k.launch(.{ (rows + v.w - 1) / v.w, 1, 1 });
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    var m = try r.module(spv);
    defer m.deinit();
    var rng = std.Random.DefaultPrng.init(11);
    const rnd = rng.random();
    var bad: usize = 0;
    var k1 = try m.kernel("qmv4x", .{ 16, 1, 1 });
    defer k1.deinit();
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
        var bw = try upload(&r, std.mem.sliceAsBytes(w));
        defer bw.free();
        var bs = try upload(&r, std.mem.sliceAsBytes(sc));
        defer bs.free();
        var bb = try upload(&r, std.mem.sliceAsBytes(bi));
        defer bb.free();
        var bx = try upload(&r, std.mem.sliceAsBytes(x));
        defer bx.free();
        var bxr = try upload(&r, std.mem.sliceAsBytes(xrev));
        defer bxr.free();
        var by = try r.alloc(@as(usize, 16) * sh.rows * 2);
        defer by.free();
        // the single-row reference of each of the 16 rows
        const ref = try alloc.alloc(u16, @as(usize, 16) * sh.rows);
        defer alloc.free(ref);
        for (0..16) |i| {
            try k1.setBuffer(0, bw);
            try k1.setBuffer(1, bs);
            try k1.setBuffer(2, bb);
            try k1.setBuffer(3, bx);
            try k1.setBuffer(4, by);
            try k1.setU32(5, sh.in);
            try k1.setU32(6, @intCast(i * sh.in));
            try k1.setU32(7, @intCast(i * sh.rows));
            try k1.setU32(8, sh.rows);
            try k1.setU32(9, 0);
            try k1.launch(.{ sh.rows, 1, 1 });
        }
        try r.download(std.mem.sliceAsBytes(ref), by);
        try r.sync();
        const got = try alloc.alloc(u16, @as(usize, 16) * sh.rows);
        defer alloc.free(got);
        const ncopy: usize = @as(usize, @intFromFloat(160e6 / @as(f64, @floatFromInt(nw * 4 + ns * 4)))) + 1;
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
        const bytes: f64 = @floatFromInt(nw * 4 + ns * 4);
        std.debug.print("{s} {d} x {d} ({d:.1} MB): single-row qmv4x time first\n", .{ sh.name, sh.in, sh.rows, bytes / 1e6 });
        var t_single: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t0 = nowNs();
            for (0..30) |i| {
                try k1.setBuffer(0, cw[i % ncopy]);
                try k1.setBuffer(1, cs[i % ncopy]);
                try k1.setBuffer(2, cb[i % ncopy]);
                try k1.setBuffer(3, bx);
                try k1.setBuffer(4, by);
                try k1.setU32(6, 0);
                try k1.setU32(7, 0);
                try k1.launch(.{ sh.rows, 1, 1 });
            }
            try r.sync();
            t_single = @min(t_single, (nowNs() - t0) / 30);
        }
        std.debug.print("  R=1 qmv4x: {d:.1} us, {d:.0} GB/s\n", .{ @as(f64, @floatFromInt(t_single)) / 1000, bytes / @as(f64, @floatFromInt(t_single)) });
        var best: [5]struct { us: f64, w: u32 } = @splat(.{ .us = 1e9, .w = 0 });
        for (variants) |v| {
            var nm: [32]u8 = undefined;
            var kv = try m.kernel(try std.fmt.bufPrintSentinel(&nm, "qmv4r_{d}_{d}", .{ v.r, v.w }, 0), .{ 16, 1, 1 });
            defer kv.deinit();
            // windows of m = 1..R rows and the reversed rows must match the single-row result bit for bit
            for ([_]u32{ 1, @max(1, v.r / 2), v.r - @min(v.r - 1, 1), v.r }) |mm| {
                for ([_]bool{ false, true }) |rev| {
                    try launchRows(&kv, v, bw, bs, bb, if (rev) bxr else bx, by, sh.in, sh.rows, mm);
                    try r.download(std.mem.sliceAsBytes(got), by);
                    try r.sync();
                    for (0..mm) |i| {
                        const src = if (rev) 15 - i else i;
                        bad += @intFromBool(!std.mem.eql(u16, got[i * sh.rows ..][0..sh.rows], ref[src * sh.rows ..][0..sh.rows]));
                    }
                }
            }
            var best_t: u64 = std.math.maxInt(u64);
            for (0..5) |_| {
                const t0 = nowNs();
                for (0..30) |i| {
                    try kv.setBuffer(0, cw[i % ncopy]);
                    try kv.setBuffer(1, cs[i % ncopy]);
                    try kv.setBuffer(2, cb[i % ncopy]);
                    try kv.setBuffer(3, bx);
                    try kv.setBuffer(4, by);
                    try kv.setU32(5, sh.in);
                    try kv.setU32(6, 0);
                    try kv.setU32(7, sh.rows);
                    try kv.setU32(8, sh.rows);
                    try kv.setU32(9, v.r);
                    try kv.setU32(10, 0);
                    try kv.launch(.{ (sh.rows + v.w - 1) / v.w, 1, 1 });
                }
                try r.sync();
                best_t = @min(best_t, (nowNs() - t0) / 30);
            }
            const us = @as(f64, @floatFromInt(best_t)) / 1000;
            const ri: usize = switch (v.r) {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                else => 4,
            };
            if (us < best[ri].us) best[ri] = .{ .us = us, .w = v.w };
        }
        for (best, [_]u32{ 1, 2, 4, 8, 16 }) |b, rr| std.debug.print("  R={d:<2} best W={d}: {d:7.1} us ({d:.2}x of R=1), {d:.0} GB/s of weight bytes\n", .{ rr, b.w, b.us, b.us / (@as(f64, @floatFromInt(t_single)) / 1000), bytes / (b.us * 1000) });

        // systolic variant: row invariance (windows, reversed rows, each row alone), error against fp64, time
        {
            var kx = try m.kernel("qmv4_xprep", .{ 16, 1, 1 });
            defer kx.deinit();
            var bxt = try r.alloc(@as(usize, 2) * sh.in * 8 * 2);
            defer bxt.free();
            var bz = try r.alloc(@as(usize, 16) * 16 * sh.rows * 4);
            defer bz.free();
            var kfin = try m.kernel("qmv4_finish", .{ 16, 1, 1 });
            defer kfin.deinit();
            const refd = try alloc.alloc(u16, @as(usize, 16) * sh.rows);
            defer alloc.free(refd);
            for ([_]u32{ 1, 2 }) |G| {
                var nm: [32]u8 = undefined;
                var kd = try m.kernel(try std.fmt.bufPrintSentinel(&nm, "qmv4d_{d}", .{G}, 0), .{ 16, 1, 1 });
                defer kd.deinit();
                const run = struct {
                    fn go(kxx: *rt.Kernel, kdd: *rt.Kernel, kf: *rt.Kernel, xs: rt.Buffer, xt: rt.Buffer, zb: rt.Buffer, w_: rt.Buffer, s_: rt.Buffer, b_: rt.Buffer, y_: rt.Buffer, in: u32, rows: u32, mm: u32, g: u32) !void {
                        const S = splitsFor(in, rows);
                        try kxx.setBuffer(0, xs);
                        try kxx.setBuffer(1, xt);
                        try kxx.setU32(2, in);
                        try kxx.setU32(3, mm);
                        try kxx.launch(.{ in / 16, g * 8, 1 });
                        try kdd.setBuffer(0, w_);
                        try kdd.setBuffer(1, s_);
                        try kdd.setBuffer(2, b_);
                        try kdd.setBuffer(3, xt);
                        try kdd.setBuffer(4, y_);
                        try kdd.setBuffer(5, zb);
                        try kdd.setU32(6, in);
                        try kdd.setU32(7, 0);
                        try kdd.setU32(8, rows);
                        try kdd.setU32(9, rows);
                        try kdd.setU32(10, mm);
                        try kdd.setU32(11, 0);
                        try kdd.setU32(12, S);
                        try kdd.launch(.{ rows / 16, S, 1 });
                        if (S > 1) {
                            try kf.setBuffer(0, zb);
                            try kf.setBuffer(1, y_);
                            try kf.setU32(2, rows);
                            try kf.setU32(3, S);
                            try kf.setU32(4, 0);
                            try kf.setU32(5, rows);
                            try kf.setU32(6, 0);
                            try kf.launch(.{ (rows + 15) / 16, mm, 1 });
                        }
                    }
                }.go;
                if (G == 1) {
                    // each row alone (m = 1, x at its own offset): the reference of the invariance test
                    for (0..16) |i| {
                        try run(&kx, &kd, &kfin, at(bx, i * sh.in * 2), bxt, bz, bw, bs, bb, at(by, i * sh.rows * 2), sh.in, sh.rows, 1, 1);
                    }
                    try r.download(std.mem.sliceAsBytes(refd), by);
                    try r.sync();
                }
                for ([_]u32{ 1, G * 4, G * 8 - 1, G * 8 }) |mm| {
                    for ([_]bool{ false, true }) |rev| {
                        try run(&kx, &kd, &kfin, if (rev) bxr else bx, bxt, bz, bw, bs, bb, by, sh.in, sh.rows, mm, G);
                        try r.download(std.mem.sliceAsBytes(got), by);
                        try r.sync();
                        for (0..mm) |i| {
                            const src = if (rev) 15 - i else i;
                            bad += @intFromBool(!std.mem.eql(u16, got[i * sh.rows ..][0..sh.rows], refd[src * sh.rows ..][0..sh.rows]));
                        }
                    }
                }
                // error against fp64 (every 97th output of row 0) and bits against qmv4x
                try run(&kx, &kd, &kfin, bx, bxt, bz, bw, bs, bb, by, sh.in, sh.rows, G * 8, G);
                try r.download(std.mem.sliceAsBytes(got), by);
                try r.sync();
                var max_ref: f64 = 0;
                var e_d: f64 = 0;
                var e_f: f64 = 0;
                var row: usize = 0;
                while (row < sh.rows) : (row += 97) {
                    var accd: f64 = 0;
                    for (0..sh.in) |ii| {
                        const q: f64 = @floatFromInt((w[row * words + ii / 8] >> @intCast(4 * (ii % 8))) & 15);
                        const gg = ii / 64;
                        accd += @as(f64, bf(x[ii])) * (q * @as(f64, bf(sc[row * groups + gg])) + @as(f64, bf(bi[row * groups + gg])));
                    }
                    max_ref = @max(max_ref, @abs(accd));
                    e_d = @max(e_d, @abs(@as(f64, bf(got[row])) - accd));
                    e_f = @max(e_f, @abs(@as(f64, bf(ref[row])) - accd));
                }
                var differ: usize = 0;
                for (got[0..sh.rows], ref[0..sh.rows]) |a, b| differ += @intFromBool(a != b);
                var best_t: u64 = std.math.maxInt(u64);
                for (0..5) |_| {
                    const t0 = nowNs();
                    for (0..30) |i| try run(&kx, &kd, &kfin, bx, bxt, bz, cw[i % ncopy], cs[i % ncopy], cb[i % ncopy], by, sh.in, sh.rows, G * 8, G);
                    try r.sync();
                    best_t = @min(best_t, (nowNs() - t0) / 30);
                }
                const us = @as(f64, @floatFromInt(best_t)) / 1000;
                std.debug.print("  DPAS m={d:<2}: {d:7.1} us ({d:.2}x of R=1 qmv4x, includes the x prepass), {d:.0} GB/s; err vs fp64 {e:.2} (qmv4x {e:.2}, max |y| {e:.2}); {d} of {d} bf16 differ from qmv4x\n", .{ G * 8, us, us / (@as(f64, @floatFromInt(t_single)) / 1000), bytes / (us * 1000), e_d, e_f, max_ref, differ, sh.rows });
            }
        }
        for (1..ncopy) |i| {
            var a = cw[i];
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
