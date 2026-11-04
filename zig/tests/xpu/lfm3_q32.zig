//! LFM3-VL's matvecs: MLX 4-bit in groups of 32 (the _g32 images) and the 8-bit embedding and head, against FP64.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;
const kernels = @import("xpu").kernels;

const ROWS: u32 = 1024;
const G: u32 = 32;

fn randWords(r: *o.rt.Runtime, h: []u32, rnd: std.Random) !o.Buf {
    for (h) |*v| v.* = rnd.int(u32);
    const b = try r.alloc(h.len * 4);
    try r.upload(b, std.mem.sliceAsBytes(h));
    return b;
}

fn fetch32(r: *o.rt.Runtime, b: o.Buf, n: usize) ![]f32 {
    const h = try o.gpa.alloc(f32, n);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(h), b);
    try r.sync();
    return h;
}

/// max|got - want| / max|want| over fp32 outputs.
fn relErr32(got: []const f32, want: []const f64) f64 {
    var e: f64 = 0;
    var m: f64 = 0;
    for (got, want) |g, w| {
        e = @max(e, @abs(@as(f64, g) - w));
        m = @max(m, @abs(w));
    }
    return if (m == 0) e else e / m;
}

/// Weight (row, i) of an MLX table of `bits` in groups of `group`; `swap` reads a word's values in reverse (control).
fn weight(w: []const u32, s: []const f64, b: []const f64, row: usize, i: usize, in_dim: usize, bits: u5, group: usize, swap: bool) f64 {
    const per: usize = 32 / @as(usize, bits);
    const word = w[row * (in_dim / per) + i / per];
    const j: usize = if (swap) per - 1 - i % per else i % per;
    const mask: u32 = (@as(u32, 1) << bits) - 1;
    const q: f64 = @floatFromInt((word >> @intCast(j * bits)) & mask);
    const g = row * (in_dim / group) + i / group;
    return q * s[g] + b[g];
}

/// y[row] = x . W[row] in FP64 for every row.
fn matvec(w: []const u32, s: []const f64, b: []const f64, x: []const f64, in_dim: usize, bits: u5, group: usize, swap: bool, out: []f64) void {
    for (out, 0..) |*y, row| {
        var acc: f64 = 0;
        for (0..in_dim) |i| acc += x[i] * weight(w, s, b, row, i, in_dim, bits, group, swap);
        y.* = acc;
    }
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var am = try r.module(kernels.attn_g32);
    var am64 = try r.module(kernels.attn);
    var vm = try r.module(kernels.lfm3_mv_g32);
    var pm = try r.module(if (kernels.dpas) kernels.nem_prow_g32 else kernels.nem_prow_gen12_g32);
    var qm = try r.module(kernels.lfm3_q8);
    const tile_outs: u32 = if (kernels.dpas) 8 else 4;
    var k_bf = try am.kernel("qmv4_bf", .{ 64, 1, 1 });
    var k_bf64 = try am64.kernel("qmv4_bf", .{ 64, 1, 1 });
    var k_x4 = try vm.kernel("qmv4_bf_x4", .{ 64, 1, 1 });
    var k_swi = try vm.kernel("qmv4_swiglu", .{ 64, 1, 1 });
    var k_pd = try pm.kernel("pdense", .{ 16 * (16 / tile_outs), 1, 1 });
    var k_e8 = try qm.kernel("embed8", .{ 64, 1, 1 });
    var k_e8t = try qm.kernel("embed8_tok", .{ 64, 1, 1 });
    var k_h8 = try qm.kernel("qmv8_f32", .{ 64, 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_q32" };
    var prng = std.Random.DefaultPrng.init(32);
    const rnd = prng.random();
    std.debug.print("MLX group 32: 4-bit projections and the 8-bit embedding, {d} output rows, against FP64\n", .{ROWS});
    var buf: [96]u8 = undefined;
    for ([_]u32{ 2048, 7680 }) |in_dim| {
        const wh = try o.gpa.alloc(u32, ROWS * in_dim / 8);
        const w = try randWords(&r, wh, rnd);
        const sh = try o.gpa.alloc(u16, ROWS * in_dim / G);
        const bh = try o.gpa.alloc(u16, sh.len);
        o.randBf(sh, rnd, 0.02);
        o.randBf(bh, rnd, 0.1);
        const s = try o.upload(&r, sh);
        const b = try o.upload(&r, bh);
        const sf = try o.widen(o.gpa, sh);
        const bfv = try o.widen(o.gpa, bh);
        const xh = try o.gpa.alloc(u16, 16 * in_dim);
        o.randBf(xh, rnd, 1.0);
        const x = try o.upload(&r, xh);
        const xf = try o.widen(o.gpa, xh);
        const want = try o.gpa.alloc(f64, ROWS);
        matvec(wh, sf, bfv, xf[0..in_dim], in_dim, 4, G, false, want);
        const y1 = try r.alloc(16 * 4 * ROWS * 4);
        try o.run(&k_bf, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, y1, in_dim, @as(u32, 0), @as(u32, 0), ROWS });
        const got = try o.fetch(&r, y1, ROWS);
        t.check(try std.fmt.bufPrint(&buf, "qmv4_bf g32, in {d}", .{in_dim}), o.relErr(got, want));
        // planted control: the same table read in groups of 64 (attn.spv) must not match
        try o.run(&k_bf64, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, y1, in_dim, @as(u32, 0), @as(u32, 0), ROWS });
        t.control(try std.fmt.bufPrint(&buf, "groups of 64, in {d}", .{in_dim}), o.relErr(try o.fetch(&r, y1, ROWS), want));
        // window rows: pdense sums each as qmv4_bf does, bit for bit
        const yn = try r.alloc(16 * ROWS * 4);
        for ([_]u32{ 1, 3, 8, 16 }) |n| {
            for (0..n) |z| try o.run(&k_bf, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, y1, in_dim, @as(u32, @intCast(z)) * in_dim, @as(u32, @intCast(z)) * ROWS, ROWS });
            try o.run(&k_pd, .{ ROWS / 16, (n + 7) / 8, 1 }, .{ w, s, b, x, yn, in_dim, ROWS, n, ROWS, @as(u32, 0), @as(u32, 0) });
            t.same(try std.fmt.bufPrint(&buf, "pdense g32 in {d} n {d:>2} vs qmv4_bf", .{ in_dim, n }), std.mem.eql(u16, try o.fetch(&r, y1, n * ROWS), try o.fetch(&r, yn, n * ROWS)));
        }
        if (in_dim != 2048) continue;
        // q, k, v and gate in one launch: four quarters of the table, each row as qmv4_bf sums it
        const q4 = ROWS / 4;
        const wq = [4]o.Buf{ w, from(w, q4 * in_dim / 2), from(w, 2 * q4 * in_dim / 2), from(w, 3 * q4 * in_dim / 2) };
        const sq = [4]o.Buf{ s, from(s, q4 * in_dim / G * 2), from(s, 2 * q4 * in_dim / G * 2), from(s, 3 * q4 * in_dim / G * 2) };
        const bq = [4]o.Buf{ b, from(b, q4 * in_dim / G * 2), from(b, 2 * q4 * in_dim / G * 2), from(b, 3 * q4 * in_dim / G * 2) };
        try o.run(&k_x4, .{ ROWS / 4, 1, 1 }, .{ wq[0], sq[0], bq[0], yn, q4, @as(u32, 0), wq[1], sq[1], bq[1], yn, q4, q4, wq[2], sq[2], bq[2], yn, q4, 2 * q4, wq[3], sq[3], bq[3], yn, q4, 3 * q4, x, in_dim });
        try o.run(&k_bf, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, y1, in_dim, @as(u32, 0), @as(u32, 0), ROWS });
        t.same("qmv4_bf_x4 g32 vs qmv4_bf", std.mem.eql(u16, try o.fetch(&r, y1, ROWS), try o.fetch(&r, yn, ROWS)));
        // w1, w3 and SwiGLU in one launch: silu(x . w1) * (x . w3), halves of the table as w1 and w3
        const half = ROWS / 2;
        try o.run(&k_swi, .{ half / 4, 1, 1 }, .{ w, s, b, from(w, half * in_dim / 2), from(s, half * in_dim / G * 2), from(b, half * in_dim / G * 2), x, yn, in_dim, half });
        const sw = try o.gpa.alloc(f64, half);
        for (sw, 0..) |*v, i| v.* = want[i] / (1.0 + @exp(-want[i])) * want[half + i];
        t.check("qmv4_swiglu g32", o.relErr(try o.fetch(&r, yn, half), sw));
    }
    // the 8-bit embedding: rows as bf16, and the tied head in fp32
    {
        const D: u32 = 2048;
        const wh = try o.gpa.alloc(u32, ROWS * D / 4);
        const w = try randWords(&r, wh, rnd);
        const sh = try o.gpa.alloc(u16, ROWS * D / G);
        const bh = try o.gpa.alloc(u16, sh.len);
        o.randBf(sh, rnd, 0.002);
        o.randBf(bh, rnd, 0.1);
        const s = try o.upload(&r, sh);
        const b = try o.upload(&r, bh);
        const sf = try o.widen(o.gpa, sh);
        const bfv = try o.widen(o.gpa, bh);
        const ids = [_]u32{ 3, 0, ROWS - 1, 517 };
        const idb = try r.alloc(ids.len * 4);
        try r.upload(idb, std.mem.sliceAsBytes(&ids));
        const y = try r.alloc(ids.len * D * 2);
        try o.run(&k_e8, .{ ids.len, 1, 1 }, .{ w, s, b, idb, y, D });
        const rows_want = try o.gpa.alloc(f64, ids.len * D);
        const rows_swap = try o.gpa.alloc(f64, ids.len * D);
        for (ids, 0..) |id, z| for (0..D) |i| {
            rows_want[z * D + i] = weight(wh, sf, bfv, id, i, D, 8, G, false);
            rows_swap[z * D + i] = weight(wh, sf, bfv, id, i, D, 8, G, true);
        };
        const got = try o.fetch(&r, y, ids.len * D);
        t.check("embed8 rows", o.relErr(got, rows_want));
        t.control("embed8 bytes of a word reversed", o.relErr(got, rows_swap));
        try o.run(&k_e8t, .{ 1, 1, 1 }, .{ w, s, b, ids[3], y, D });
        t.check("embed8_tok row", o.relErr(try o.fetch(&r, y, D), rows_want[3 * D ..][0..D]));
        const xh = try o.gpa.alloc(u16, D);
        o.randBf(xh, rnd, 1.0);
        const x = try o.upload(&r, xh);
        const xf = try o.widen(o.gpa, xh);
        const want = try o.gpa.alloc(f64, ROWS);
        const ctl = try o.gpa.alloc(f64, ROWS);
        matvec(wh, sf, bfv, xf, D, 8, G, false, want);
        matvec(wh, sf, bfv, xf, D, 8, 64, false, ctl);
        const yl = try r.alloc(ROWS * 4);
        try o.run(&k_h8, .{ ROWS / 4, 1, 1 }, .{ w, s, b, x, yl, D, @as(u32, 0), @as(u32, 0), ROWS });
        const lg = try fetch32(&r, yl, ROWS);
        t.check("qmv8_f32 head", relErr32(lg, want));
        t.control("qmv8_f32 against groups of 64", relErr32(lg, ctl));
    }
    try t.finish();
}

fn from(b: o.Buf, off: usize) o.Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}
