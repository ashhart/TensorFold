//! ggml block-quant kernels on real GGUF slices: bit-exact dequant, matvec vs float64. usage: xpu-ggml-test [--bench]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const gg = @import("xpu").ggml;
const bench_mod = @import("ggml_bench.zig");

pub const Fix = struct { key: []const u8, t: gg.Type, bytes: []const u8 };
fn fixes() ![]const Fix {
    const S = struct {
        var v: [14]Fix = undefined;
        var done = false;
    };
    if (!S.done) {
        S.v = [_]Fix{
    .{ .key = "q2_k", .t = .q2_k, .bytes = (try tfix.load("ggml_q2_k")) },
    .{ .key = "q4_k", .t = .q4_k, .bytes = (try tfix.load("ggml_q4_k")) },
    .{ .key = "iq4_xs", .t = .iq4_xs, .bytes = (try tfix.load("ggml_iq4_xs")) },
    .{ .key = "iq2_xxs", .t = .iq2_xxs, .bytes = (try tfix.load("ggml_iq2_xxs")) },
    .{ .key = "iq2_xs", .t = .iq2_xs, .bytes = (try tfix.load("ggml_iq2_xs")) },
    .{ .key = "iq2_s", .t = .iq2_s, .bytes = (try tfix.load("ggml_iq2_s")) },
    .{ .key = "iq2_s_embd", .t = .iq2_s, .bytes = (try tfix.load("ggml_iq2_s_embd")) },
    .{ .key = "iq3_xxs", .t = .iq3_xxs, .bytes = (try tfix.load("ggml_iq3_xxs")) },
    .{ .key = "iq3_s", .t = .iq3_s, .bytes = (try tfix.load("ggml_iq3_s")) },
    .{ .key = "iq1_m", .t = .iq1_m, .bytes = (try tfix.load("ggml_iq1_m")) },
    .{ .key = "q6_k", .t = .q6_k, .bytes = (try tfix.load("ggml_q6_k")) },
    .{ .key = "q5_k", .t = .q5_k, .bytes = (try tfix.load("ggml_q5_k")) },
    .{ .key = "q8_0", .t = .q8_0, .bytes = (try tfix.load("ggml_q8_0")) },
    .{ .key = "iq4_nl", .t = .iq4_nl, .bytes = (try tfix.load("ggml_iq4_nl")) },
};
        S.done = true;
    }
    return &S.v;
}

pub const gpa = std.heap.page_allocator;

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn up(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(bytes.len);
    try r.upload(b, bytes);
    try r.sync();
    return b;
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn toBf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Copies a fixture section into aligned storage of T.
pub fn section(comptime T: type, bytes: []const u8, off: usize, n: usize) ![]T {
    const out = try gpa.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[off..][0 .. n * @sizeOf(T)]);
    return out;
}

const Result = struct { differ: usize, quant_differ: usize, exact_rel: f64, mv_rel: f64, noise_rel: f64, embed_differ: usize };

/// Round to nearest even, as convert_int_rte.
fn rne(v: f32) f32 {
    const r = @floor(v + 0.5);
    return if (r - v == 0.5 and @mod(r, 2.0) != 0) r - 1 else r;
}

/// The fixture's file rows in the kernels' interleaved layout.
fn packed_(f: Fix) ![]u8 {
    const hdr = try section(u32, f.bytes, 0, 4);
    const rows: usize = hdr[0];
    const nb: usize = hdr[2];
    const out = try gpa.alloc(u8, rows * nb * gg.packedBytes(f.t));
    gg.packRows(f.t, f.bytes[16..][0 .. rows * nb * hdr[3]], out, rows, nb);
    return out;
}

fn checkOne(r: *rt.Runtime, set: *gg.Set, f: Fix) !Result {
    const hdr = try section(u32, f.bytes, 0, 4);
    const rows: usize = hdr[0];
    const k: usize = hdr[1];
    const nb: usize = hdr[2];
    if (hdr[3] != gg.blockBytes(f.t) or nb * 256 != k) return error.BadFixture;
    const raw_len = rows * nb * hdr[3];
    const deq = try section(u32, f.bytes, 16 + raw_len, rows * k);
    const xs = try section(u16, f.bytes, 16 + raw_len + rows * k * 4, k);
    const yref = try section(f64, f.bytes, 16 + raw_len + rows * k * 4 + k * 2, rows);
    const w = try up(r, try packed_(f));
    const x = try up(r, std.mem.sliceAsBytes(xs));

    // dequantization, bit for bit
    const dqb = try r.alloc(rows * k * 4);
    try set.dequant(f.t, w, dqb, @intCast(rows), @intCast(nb));
    const got = try gpa.alloc(u32, rows * k);
    try r.download(std.mem.sliceAsBytes(got), dqb);
    try r.sync();
    var differ: usize = 0;
    for (got, deq, 0..) |g, d, i| {
        if (g != d) {
            if (differ < 3) std.debug.print("  {s}: element {d} got {e} want {e}\n", .{ f.key, i, @as(f32, @bitCast(g)), @as(f32, @bitCast(d)) });
            differ += 1;
        }
    }

    // activation quantization (the quant32 the matvec runs) against the same arithmetic on the host
    const xq = try r.alloc(k + k / 8);
    try set.quantize(x, xq, @intCast(k));
    const hq = try gpa.alloc(i8, k);
    const hd = try gpa.alloc(f32, k / 32);
    try r.download(std.mem.sliceAsBytes(hq), xq);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(hd), gg.sub(xq, k));
    try r.sync();
    var quant_differ: usize = 0;
    var quant_ties: usize = 0;
    const xdq = try gpa.alloc(f64, k); // dequantized activation the kernels see
    for (0..k / 32) |blk| {
        var amax: f32 = 0;
        for (0..32) |i| amax = @max(amax, @abs(bf(xs[blk * 32 + i])));
        const d = amax * @as(f32, 0.007874015718698502);
        const id: f32 = if (amax > 0) 127.0 / amax else 0;
        if (hd[blk] != d) quant_differ += 1;
        for (0..32) |i| {
            const want: i8 = @intFromFloat(rne(bf(xs[blk * 32 + i]) * id));
            // 127 / amax is not an IEEE-exact divide on the device: a quotient one ulp off can move a tie by one level
            const diff = @abs(@as(i32, hq[blk * 32 + i]) - @as(i32, want));
            if (diff == 1) quant_ties += 1;
            if (diff > 1) quant_differ += 1;
            xdq[blk * 32 + i] = @as(f64, hd[blk]) * @as(f64, @floatFromInt(hq[blk * 32 + i]));
        }
    }
    if (quant_ties * 500 > k) quant_differ += 1;

    // matvec: fp32 output vs the float64 dot with the quantized activation, bf16 output, quantization noise vs x
    const yb = try r.alloc(rows * 2);
    const yf = try r.alloc(rows * 4);
    try set.matvec(f.t, w, x, yb, @intCast(k), 0, @intCast(rows), false);
    try set.matvec(f.t, w, x, yf, @intCast(k), 0, @intCast(rows), true);
    const hb = try gpa.alloc(u16, rows);
    const hf = try gpa.alloc(f32, rows);
    try r.download(std.mem.sliceAsBytes(hb), yb);
    try r.download(std.mem.sliceAsBytes(hf), yf);
    try r.sync();
    var mx: f64 = 0;
    for (yref) |v| mx = @max(mx, @abs(v));
    var rel_exact: f64 = 0;
    var rel_b: f64 = 0;
    var rel_noise: f64 = 0;
    for (0..rows) |row| {
        var acc: f64 = 0;
        for (0..k) |i| acc += @as(f64, @as(f32, @bitCast(deq[row * k + i]))) * xdq[i];
        rel_exact = @max(rel_exact, @abs(@as(f64, hf[row]) - acc) / mx);
        rel_b = @max(rel_b, @abs(bf(hb[row]) - acc) / mx);
        rel_noise = @max(rel_noise, @abs(@as(f64, hf[row]) - yref[row]) / mx);
    }

    // embedding row 5 of the slice
    var embed_differ: usize = 0;
    {
        const ids = try up(r, std.mem.sliceAsBytes(&[_]u32{5}));
        const eb = try r.alloc(k * 2);
        try set.embedRow(f.t, w, ids, eb, @intCast(k));
        const he = try gpa.alloc(u16, k);
        try r.download(std.mem.sliceAsBytes(he), eb);
        try r.sync();
        for (he, 0..) |v, i| {
            if (v != toBf(@bitCast(deq[5 * k + i]))) embed_differ += 1;
        }
    }
    return .{ .differ = differ, .quant_differ = quant_differ, .exact_rel = rel_exact, .mv_rel = rel_b, .noise_rel = rel_noise, .embed_differ = embed_differ };
}

/// A weight buffer of rows x in in kernel layout filled by cycling the fixture's real 16-row group blocks.
pub fn tableBuf(f: Fix, rows: u32, in: u32) ![]u8 {
    const pk = try packed_(f);
    const pb = gg.packedBytes(f.t);
    const chunk = 16 * pb;
    const n_chunks_src = pk.len / chunk;
    const n: usize = @as(usize, rows / 16) * (in / 256);
    const host = try gpa.alloc(u8, n * chunk);
    for (0..n) |i| @memcpy(host[i * chunk ..][0..chunk], pk[(i % n_chunks_src) * chunk ..][0..chunk]);
    return host;
}

pub fn fixFor(t: gg.Type) !Fix {
    for (try fixes()) |f| {
        const h = try section(u32, f.bytes, 0, 4);
        if (f.t == t and h[1] == 5120) return f;
    }
    return error.NoFixture;
}

pub fn randomX(r: *rt.Runtime, n: usize) !rt.Buffer {
    const xh = try gpa.alloc(u16, n);
    var prng = std.Random.DefaultPrng.init(5);
    for (xh) |*v| v.* = toBf(prng.random().floatNorm(f32));
    return up(r, std.mem.sliceAsBytes(xh));
}

/// matvecMulti vs the single-type kernels on every ordered pair of fixture types (K = 5120, no IQ1_M): same results.
fn checkMulti(r: *rt.Runtime, set: *gg.Set) !bool {
    var bad = false;
    var worst: f64 = 0;
    for (try fixes()) |fa| {
        for (try fixes()) |fb| {
            const ha = try section(u32, fa.bytes, 0, 4);
            const hb = try section(u32, fb.bytes, 0, 4);
            if (ha[1] != 5120 or hb[1] != 5120 or fa.t == .iq1_m or fb.t == .iq1_m or fa.t == .q6_k or fb.t == .q6_k or fa.t == .q5_k or fb.t == .q5_k or fa.t == .q8_0 or fb.t == .q8_0 or fa.t == .iq4_nl or fb.t == .iq4_nl) continue;
            const wa = try up(r, try packed_(fa));
            const wb = try up(r, try packed_(fb));
            const x = try randomX(r, 5120);
            const ra: u32 = ha[0];
            const rb: u32 = hb[0];
            const ya = try r.alloc(ra * 2);
            const yb = try r.alloc(rb * 2);
            const fa_y = try r.alloc(ra * 2);
            const fb_y = try r.alloc(rb * 2);
            try set.matvec(fa.t, wa, x, ya, 5120, 0, ra, false);
            try set.matvec(fb.t, wb, x, yb, 5120, 0, rb, false);
            try set.matvecMulti(&.{ .{ .t = fa.t, .w = wa, .y = fa_y, .rows = ra, .y_off = 0 }, .{ .t = fb.t, .w = wb, .y = fb_y, .rows = rb, .y_off = 0 } }, x, 5120);
            const ha_y = try gpa.alloc(u16, ra);
            const hb_y = try gpa.alloc(u16, rb);
            const ga = try gpa.alloc(u16, ra);
            const gb = try gpa.alloc(u16, rb);
            try r.download(std.mem.sliceAsBytes(ha_y), ya);
            try r.download(std.mem.sliceAsBytes(hb_y), yb);
            try r.download(std.mem.sliceAsBytes(ga), fa_y);
            try r.download(std.mem.sliceAsBytes(gb), fb_y);
            try r.sync();
            var mx: f32 = 0;
            for (ha_y) |v| mx = @max(mx, @abs(bf(v)));
            for (hb_y) |v| mx = @max(mx, @abs(bf(v)));
            for (ha_y, ga) |a, b| worst = @max(worst, @abs(bf(a) - bf(b)) / mx);
            for (hb_y, gb) |a, b| worst = @max(worst, @abs(bf(a) - bf(b)) / mx);
        }
    }
    std.debug.print("matvecMulti vs single kernels over all type pairs: worst difference {e:.2} of max|y| (bf16 output; 1 ulp = 4e-3)\n", .{worst});
    if (worst > 8e-3) bad = true;
    return bad;
}

/// The bf16 gate projections (48 x 5120): correctness against float64 and launch time, single and as a pair.
fn checkGates(r: *rt.Runtime, set: *gg.Set) !bool {
    const rows: usize = 48;
    const k: usize = 5120;
    var prng = std.Random.DefaultPrng.init(11);
    const wh = try gpa.alloc(u16, 2 * rows * k);
    for (wh) |*v| v.* = toBf(prng.random().floatNorm(f32) * 0.05);
    const xh = try gpa.alloc(u16, k);
    for (xh) |*v| v.* = toBf(prng.random().floatNorm(f32));
    const w0 = try up(r, std.mem.sliceAsBytes(wh[0 .. rows * k]));
    const w1 = try up(r, std.mem.sliceAsBytes(wh[rows * k ..]));
    const x = try up(r, std.mem.sliceAsBytes(xh));
    const y0 = try r.alloc(rows * 2);
    const y1 = try r.alloc(rows * 2);
    const z0 = try r.alloc(rows * 2);
    const z1 = try r.alloc(rows * 2);
    try set.matvecBf16(w0, x, y0, k, 0, rows);
    try set.matvecBf16(w1, x, y1, k, 0, rows);
    try set.matvecBf16Pair(w0, w1, x, z0, z1, k, rows);
    const g = try gpa.alloc(u16, 4 * rows);
    try r.download(std.mem.sliceAsBytes(g[0..rows]), y0);
    try r.download(std.mem.sliceAsBytes(g[rows .. 2 * rows]), y1);
    try r.download(std.mem.sliceAsBytes(g[2 * rows .. 3 * rows]), z0);
    try r.download(std.mem.sliceAsBytes(g[3 * rows ..]), z1);
    try r.sync();
    var worst: f64 = 0;
    var mismatch: usize = 0;
    for (0..2 * rows) |row| {
        var acc: f64 = 0;
        for (0..k) |i| acc += @as(f64, bf(wh[row * k + i])) * @as(f64, bf(xh[i]));
        worst = @max(worst, @abs(bf(g[row]) - acc) / @max(@abs(acc), 0.5));
        if (g[row] != g[2 * rows + row]) mismatch += 1;
    }
    var best_s: u64 = std.math.maxInt(u64);
    var best_p: u64 = std.math.maxInt(u64);
    for (0..3) |_| {
        var t0 = nowNs();
        for (0..200) |_| {
            try set.matvecBf16(w0, x, y0, k, 0, rows);
            try set.matvecBf16(w1, x, y1, k, 0, rows);
        }
        try r.sync();
        best_s = @min(best_s, (nowNs() - t0) / 200);
        t0 = nowNs();
        for (0..200) |_| try set.matvecBf16Pair(w0, w1, x, z0, z1, k, rows);
        try r.sync();
        best_p = @min(best_p, (nowNs() - t0) / 200);
    }
    std.debug.print("gates bf16 48 x 5120: worst relative error {e:.2}, pair vs singles differing {d}; two launches {d:.1} us, one pair launch {d:.1} us\n", .{ worst, mismatch, @as(f64, @floatFromInt(best_s)) / 1e3, @as(f64, @floatFromInt(best_p)) / 1e3 });
    return worst > 8e-3 or mismatch != 0;
}

/// gateUp against the two matvecs + the host swiglu formula, and its time against the two-launch alternative.
fn checkGateUp(r: *rt.Runtime, set: *gg.Set) !bool {
    const pairs = [_][2]gg.Type{ .{ .iq3_xxs, .iq3_xxs }, .{ .iq4_xs, .iq3_s }, .{ .q4_k, .iq2_xs }, .{ .iq2_s, .iq3_xxs } };
    const rows: u32 = 17408;
    const x = try randomX(r, 5120);
    var bad = false;
    for (pairs) |pr| {
        const hg = try tableBuf(try fixFor(pr[0]), rows, 5120);
        defer gpa.free(hg);
        const hu = try tableBuf(try fixFor(pr[1]), rows, 5120);
        defer gpa.free(hu);
        var wg: [4]rt.Buffer = undefined;
        var wu: [4]rt.Buffer = undefined;
        for (&wg) |*b| b.* = try up(r, hg);
        for (&wu) |*b| b.* = try up(r, hu);
        const g = try r.alloc(rows * 2);
        const u = try r.alloc(rows * 2);
        const act = try r.alloc(rows * 2);
        try set.matvec(pr[0], wg[0], x, g, 5120, 0, rows, false);
        try set.matvec(pr[1], wu[0], x, u, 5120, 0, rows, false);
        try set.gateUp(pr[0], wg[0], pr[1], wu[0], x, act, 5120, rows);
        const hg_y = try gpa.alloc(u16, rows);
        const hu_y = try gpa.alloc(u16, rows);
        const ha = try gpa.alloc(u16, rows);
        try r.download(std.mem.sliceAsBytes(hg_y), g);
        try r.download(std.mem.sliceAsBytes(hu_y), u);
        try r.download(std.mem.sliceAsBytes(ha), act);
        try r.sync();
        var diff: usize = 0;
        for (0..rows) |i| {
            const a = bf(hg_y[i]);
            const want = toBf(a / (1.0 + @exp(-a)) * bf(hu_y[i]));
            if (@abs(@as(i32, want) - @as(i32, ha[i])) > 1) diff += 1;
        }
        var sep: u64 = std.math.maxInt(u64);
        var fus: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            var t0 = nowNs();
            for (0..8) |it| {
                try set.matvec(pr[0], wg[it % 4], x, g, 5120, 0, rows, false);
                try set.matvec(pr[1], wu[it % 4], x, u, 5120, 0, rows, false);
            }
            try r.sync();
            sep = @min(sep, (nowNs() - t0) / 8);
            t0 = nowNs();
            for (0..8) |it| try set.gateUp(pr[0], wg[it % 4], pr[1], wu[it % 4], x, act, 5120, rows);
            try r.sync();
            fus = @min(fus, (nowNs() - t0) / 8);
        }
        std.debug.print("gateUp {s}+{s}: {d} of {d} rows off by more than 1 bf16 ulp; two matvecs {d:.1} us (+ swiglu launch), fused {d:.1} us\n", .{ @tagName(pr[0]), @tagName(pr[1]), diff, rows, @as(f64, @floatFromInt(sep)) / 1e3, @as(f64, @floatFromInt(fus)) / 1e3 });
        if (diff != 0) bad = true;
        for (&wg) |*b| b.free();
        for (&wu) |*b| b.free();
    }
    return bad;
}

/// Row invariance: every window (m = 2..8) over 8 activation rows must reproduce the single-row kernel bit for bit.
fn checkRowsInvariance(r: *rt.Runtime, set: *gg.Set) !bool {
    var bad = false;
    for (try fixes()) |f| {
        const hdr = try section(u32, f.bytes, 0, 4);
        const rows: u32 = hdr[0];
        const k: u32 = hdr[1];
        const w = try up(r, try packed_(f));
        const xh = try gpa.alloc(u16, 8 * k);
        var prng = std.Random.DefaultPrng.init(21);
        for (xh, 0..) |*v, i| v.* = toBf(prng.random().floatNorm(f32) * (if (i % 101 == 0) @as(f32, 8.0) else 1.0));
        const x = try up(r, std.mem.sliceAsBytes(xh));
        const single = try gpa.alloc(u16, 8 * rows);
        const ys = try r.alloc(rows * 2);
        for (0..8) |ri| {
            try set.matvec(f.t, w, gg.sub(x, ri * k * 2), ys, k, 0, rows, false);
            try r.download(std.mem.sliceAsBytes(single[ri * rows ..][0..rows]), ys);
            try r.sync();
        }
        const yb = try r.alloc(8 * rows * 2);
        const got = try gpa.alloc(u16, 8 * rows);
        var differ: usize = 0;
        var windows: usize = 0;
        for (2..9) |m| {
            for (0..9 - m) |r0| {
                try set.matvecRows(f.t, w, gg.sub(x, r0 * k * 2), @intCast(m), yb, k, 0, rows);
                try r.download(std.mem.sliceAsBytes(got[0 .. m * rows]), yb);
                try r.sync();
                windows += 1;
                for (0..m) |ri| {
                    for (0..rows) |c| {
                        if (got[ri * rows + c] != single[(r0 + ri) * rows + c]) differ += 1;
                    }
                }
            }
        }
        std.debug.print("rows {s:<11}: {d} windows (m = 2..8, every start), {d} differing values vs the single-row kernel\n", .{ f.key, windows, differ });
        if (differ != 0) bad = true;
    }
    return bad;
}

/// Fused multi-row kernels vs the single-row fused kernels, bit for bit, every row and window (m = 2..4).
fn checkFusedRows(r: *rt.Runtime, set: *gg.Set) !bool {
    var bad = false;
    const x = try randomX(r, 8 * 5120);
    const pairs = [_][2]gg.Type{ .{ .iq3_xxs, .iq3_xxs }, .{ .iq4_xs, .iq3_s }, .{ .q4_k, .iq2_xs }, .{ .iq2_s, .iq3_xxs }, .{ .q2_k, .iq2_xxs } };
    for (pairs) |pr| {
        const rows: u32 = 5120;
        const wa = try up(r, try tableBuf(try fixFor(pr[0]), rows, 5120));
        const wb = try up(r, try tableBuf(try fixFor(pr[1]), rows, 5120));
        const act1 = try r.alloc(rows * 2);
        const g1 = try gpa.alloc(u16, 8 * rows);
        var ya = try r.alloc(rows * 2);
        var yb = try r.alloc(rows * 2);
        const sa = try gpa.alloc(u16, 8 * rows);
        const sb = try gpa.alloc(u16, 8 * rows);
        for (0..8) |ri| {
            const xr = gg.sub(x, ri * 5120 * 2);
            try set.gateUp(pr[0], wa, pr[1], wb, xr, act1, 5120, rows);
            try r.download(std.mem.sliceAsBytes(g1[ri * rows ..][0..rows]), act1);
            try set.matvecMulti(&.{ .{ .t = pr[0], .w = wa, .y = ya, .rows = rows, .y_off = 0 }, .{ .t = pr[1], .w = wb, .y = yb, .rows = rows, .y_off = 0 } }, xr, 5120);
            try r.download(std.mem.sliceAsBytes(sa[ri * rows ..][0..rows]), ya);
            try r.download(std.mem.sliceAsBytes(sb[ri * rows ..][0..rows]), yb);
            try r.sync();
        }
        const mact = try r.alloc(8 * rows * 2);
        const ma = try r.alloc(8 * rows * 2);
        const mb = try r.alloc(8 * rows * 2);
        const hg = try gpa.alloc(u16, 8 * rows);
        const ha = try gpa.alloc(u16, 8 * rows);
        const hb = try gpa.alloc(u16, 8 * rows);
        var differ: usize = 0;
        for (2..9) |m| {
            try set.gateUpRows(pr[0], wa, pr[1], wb, x, mact, 5120, rows, @intCast(m));
            try set.matvecMultiRows(&.{ .{ .t = pr[0], .w = wa, .y = ma, .rows = rows, .y_off = 0 }, .{ .t = pr[1], .w = wb, .y = mb, .rows = rows, .y_off = 0 } }, x, 5120, @intCast(m));
            try r.download(std.mem.sliceAsBytes(hg[0 .. m * rows]), mact);
            try r.download(std.mem.sliceAsBytes(ha[0 .. m * rows]), ma);
            try r.download(std.mem.sliceAsBytes(hb[0 .. m * rows]), mb);
            try r.sync();
            for (0..m * rows) |i| {
                if (hg[i] != g1[i]) differ += 1;
                if (ha[i] != sa[i]) differ += 1;
                if (hb[i] != sb[i]) differ += 1;
            }
        }
        ya.free();
        yb.free();
        std.debug.print("fused rows {s}+{s}: gateUpRows and matvecMultiRows (m = 2..8) differing values vs the single-row fused kernels: {d}\n", .{ @tagName(pr[0]), @tagName(pr[1]), differ });
        if (differ != 0) bad = true;
    }
    return bad;
}

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(p: *const anyopaque, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;

/// Prefill GEMM: accuracy vs float64 on dequantized fixture weights, diff to decode-path rows, chunk invariance.
fn checkPrefill(r: *rt.Runtime, set: *gg.Set) !bool {
    var bad = false;
    const R: u32 = 40;
    for (try fixes()) |f| {
        if (f.t == .iq1_m) continue; // no kernel (the GEMM takes any multiple of 16 rows)
        const hdr = try section(u32, f.bytes, 0, 4);
        const rows: u32 = hdr[0];
        const k: u32 = hdr[1];
        const raw_len = @as(usize, rows) * hdr[2] * hdr[3];
        const deq = try section(u32, f.bytes, 16 + raw_len, @as(usize, rows) * k);
        const w = try up(r, try packed_(f));
        const xh = try gpa.alloc(u16, R * k);
        var prng = std.Random.DefaultPrng.init(33);
        for (xh, 0..) |*v, i| v.* = toBf(prng.random().floatNorm(f32) * (if (i % 97 == 0) @as(f32, 6.0) else 1.0));
        const x = try up(r, std.mem.sliceAsBytes(xh));
        const yp = try r.alloc(R * rows * 2);
        const ym = try r.alloc(R * rows * 2);
        try set.prefillRows(f.t, w, x, R, yp, k, 0, rows);
        var done: u32 = 0;
        while (done < R) {
            const n = @min(8, R - done);
            try set.matvecRows(f.t, w, gg.sub(x, @as(usize, done) * k * 2), n, gg.sub(ym, @as(usize, done) * rows * 2), k, 0, rows);
            done += n;
        }
        // chunked: rows [0, 13) then [13, 40)
        const yc = try r.alloc(R * rows * 2);
        try set.prefillRows(f.t, w, x, 13, yc, k, 0, rows);
        try set.prefillRows(f.t, w, gg.sub(x, 13 * k * 2), R - 13, gg.sub(yc, 13 * rows * 2), k, 0, rows);
        const gp = try gpa.alloc(u16, R * rows);
        const gm = try gpa.alloc(u16, R * rows);
        const gc = try gpa.alloc(u16, R * rows);
        try r.download(std.mem.sliceAsBytes(gp), yp);
        try r.download(std.mem.sliceAsBytes(gm), ym);
        try r.download(std.mem.sliceAsBytes(gc), yc);
        try r.sync();
        if (std.c.getenv("GG_PF_DUMP")) |dir| { // bits of the prefill output, to compare two builds
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}.bin", .{ std.mem.span(dir), f.key }, 0);
            const fp = fopen(path.ptr, "wb") orelse return error.DumpFailed;
            _ = fwrite(gp.ptr, 2, gp.len, fp);
            _ = fclose(fp);
        }
        var mx: f64 = 0;
        var e_ref: f64 = 0;
        var e_dec: f64 = 0;
        var chunk_diff: usize = 0;
        const refs = try gpa.alloc(f64, R * rows);
        for (0..R) |ri| {
            for (0..rows) |c| {
                var acc: f64 = 0;
                for (0..k) |i| acc += @as(f64, @as(f32, @bitCast(deq[c * k + i]))) * @as(f64, bf(xh[ri * k + i]));
                refs[ri * rows + c] = acc;
                mx = @max(mx, @abs(acc));
            }
        }
        for (0..R * rows) |i| {
            e_ref = @max(e_ref, @abs(bf(gp[i]) - refs[i]) / mx);
            e_dec = @max(e_dec, @abs(bf(gp[i]) - bf(gm[i])) / mx);
            if (gp[i] != gc[i]) chunk_diff += 1;
        }
        std.debug.print("prefill {s:<11}: vs float64 {e:.2}, vs decode-path rows {e:.2} of max|y|; chunked [0,13)+[13,40) vs one pass: {d} differing values\n", .{ f.key, e_ref, e_dec, chunk_diff });
        if (e_ref > 8e-3 or chunk_diff != 0) bad = true;
    }
    return bad;
}

/// The lm_head rows kernel (Q4_K, fp32 logits) vs the single-row fp32 kernel, bit for bit, m = 2..4 at every start.
fn checkHeadRowsOf(r: *rt.Runtime, set: *gg.Set, ty: gg.Type) !bool {
    const f = try fixFor(ty);
    const hdr = try section(u32, f.bytes, 0, 4);
    const rows: u32 = hdr[0];
    const k: u32 = hdr[1];
    const w = try up(r, try packed_(f));
    const x = try randomX(r, 8 * k);
    const single = try gpa.alloc(f32, 8 * rows);
    const ys = try r.alloc(rows * 4);
    for (0..8) |ri| {
        try set.matvec(ty, w, gg.sub(x, ri * k * 2), ys, k, 0, rows, true);
        try r.download(std.mem.sliceAsBytes(single[ri * rows ..][0..rows]), ys);
        try r.sync();
    }
    const yb = try r.alloc(8 * rows * 4);
    const got = try gpa.alloc(f32, 8 * rows);
    var differ: usize = 0;
    for (2..9) |m| {
        for (0..9 - m) |r0| {
            try set.headRows(ty, w, gg.sub(x, r0 * k * 2), @intCast(m), yb, k, rows);
            try r.download(std.mem.sliceAsBytes(got[0 .. m * rows]), yb);
            try r.sync();
            for (0..m * rows) |i| if (@as(u32, @bitCast(got[i])) != @as(u32, @bitCast(single[r0 * rows + i]))) {
                differ += 1;
            };
        }
    }
    std.debug.print("headRows {s} fp32 (m = 2..8, every start): {d} differing values vs the single-row kernel\n", .{ @tagName(ty), differ });
    return differ != 0;
}

fn checkHeadRows(r: *rt.Runtime, set: *gg.Set) !bool {
    const a = try checkHeadRowsOf(r, set, .q4_k);
    const b = try checkHeadRowsOf(r, set, .iq4_xs);
    return a or b;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const do_bench = args.len > 1 and std.mem.eql(u8, args[1], "--bench");
    var r = try rt.open();
    defer r.deinit();
    var set = try gg.Set.init(&r);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--replay")) return bench_mod.replay(&r, &set);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--fuse")) return bench_mod.benchFuse(&r, &set);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--info")) return bench_mod.deviceInfo(&r);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--pfbench")) return bench_mod.benchPrefill(&r, &set);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--rowscost")) return bench_mod.benchRows(&r, &set);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--occ")) return bench_mod.occProbe(&r);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--probe2")) return bench_mod.probe2(&r);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--probe")) return bench_mod.probeBandwidth(&r);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--shapes")) return bench_mod.benchShapes(&r, &set);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--prefill")) { // only the prefill GEMM checks
        if (try checkPrefill(&r, &set)) return error.TooInaccurate;
        std.debug.print("prefill ok\n", .{});
        return;
    }
    var bad = false;
    for (try fixes()) |f| {
        const res = try checkOne(&r, &set, f);
        std.debug.print("{s:<11} dequant differing {d}, quant_x differing {d}, matvec vs quantized-x float64: fp32 {e:.2} bf16 {e:.2}, vs original x (quantization noise) {e:.2} of max|y|, embed row differing {d}\n", .{ f.key, res.differ, res.quant_differ, res.exact_rel, res.mv_rel, res.noise_rel, res.embed_differ });
        if (res.differ != 0 or res.quant_differ != 0 or res.embed_differ != 0 or res.exact_rel > 1e-5 or res.mv_rel > 5e-3 or res.noise_rel > 2e-2) bad = true;
        if (do_bench) try bench_mod.bench(&r, &set, f);
    }
    if (try checkMulti(&r, &set)) bad = true;
    if (try checkGates(&r, &set)) bad = true;
    if (try checkGateUp(&r, &set)) bad = true;
    if (try checkRowsInvariance(&r, &set)) bad = true;
    if (try checkFusedRows(&r, &set)) bad = true;
    if (try checkHeadRows(&r, &set)) bad = true;
    if (try checkPrefill(&r, &set)) bad = true;
    if (bad) return error.TooInaccurate;
    std.debug.print("ok\n", .{});
}
