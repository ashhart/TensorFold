//! LFM3 GQA attention (partial + merge with gate) over a KV ring, full and sliding, against FP64 on random data.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;

const HD: usize = 128;
const NQ: u32 = 16;
const NKV: u32 = 4;
const CHUNK: u32 = 64;

/// FP64 attention of row z (position pos0 + z) over keys by absolute position, window 0 = all, times sigmoid(gate).
fn refRows(q: []const f64, k: []const f64, v: []const f64, g: []const f64, n: usize, pos0: usize, window: usize, out: []f64) !void {
    const scale = 1.0 / @sqrt(@as(f64, HD));
    const s = try o.gpa.alloc(f64, pos0 + n);
    defer o.gpa.free(s);
    for (0..n) |z| {
        const p = pos0 + z;
        const lo = if (window == 0 or p + 1 <= window) 0 else p + 1 - window;
        for (0..NQ) |h| {
            const kvh = h / (NQ / NKV);
            const qh = q[(z * NQ + h) * HD ..][0..HD];
            var mx: f64 = -std.math.inf(f64);
            for (lo..p + 1) |key| {
                var a: f64 = 0;
                for (0..HD) |d| a += qh[d] * k[(key * NKV + kvh) * HD + d];
                s[key] = a * scale;
                mx = @max(mx, s[key]);
            }
            var den: f64 = 0;
            const oh = out[(z * NQ + h) * HD ..][0..HD];
            @memset(oh, 0);
            for (lo..p + 1) |key| {
                const e = @exp(s[key] - mx);
                den += e;
                for (0..HD) |d| oh[d] += e * v[(key * NKV + kvh) * HD + d];
            }
            const gate = 1.0 / (1.0 + @exp(-g[z * NQ + h]));
            for (oh) |*x| x.* = x.* / den * gate;
        }
    }
}

const Case = struct { what: []const u8, pos0: u32, n: u32, window: u32, ring: u32 };

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var m = try r.module(@import("xpu").kernels.lfm3);
    var kp = try m.kernel("lfm3_attn_partial", .{ 64, 1, 1 });
    var km = try m.kernel("lfm3_attn_merge_gate", .{ 16, 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_attn" };
    var prng = std.Random.DefaultPrng.init(13);
    const rnd = prng.random();
    const maxc: u32 = 64;
    const max_rows: u32 = 8;
    const po = try r.alloc(@as(usize, max_rows) * NQ * maxc * HD * 4);
    const pm = try r.alloc(@as(usize, max_rows) * NQ * maxc * 4);
    const pl = try r.alloc(@as(usize, max_rows) * NQ * maxc * 4);
    std.debug.print("GQA attention, 16 q / 4 kv heads, chunks of {d} keys, sigmoid gate per head, against FP64\n", .{CHUNK});

    const cases = [_]Case{
        .{ .what = "full layer, 6 rows at 700 (12 chunks)", .pos0 = 700, .n = 6, .window = 0, .ring = 4096 },
        .{ .what = "full layer, decode row at 0 (one key)", .pos0 = 0, .n = 1, .window = 0, .ring = 4096 },
        .{ .what = "sliding 256, 8 rows at 900 in a 512-slot ring (wraps)", .pos0 = 900, .n = 8, .window = 256, .ring = 512 },
        .{ .what = "sliding 256, 3 rows at 100 (shorter than the window)", .pos0 = 100, .n = 3, .window = 256, .ring = 512 },
    };
    for (cases, 0..) |c, ci| {
        const npos: usize = c.pos0 + c.n;
        const kh = try o.gpa.alloc(u16, npos * NKV * HD);
        const vh = try o.gpa.alloc(u16, kh.len);
        o.randBf(kh, rnd, 1.0);
        o.randBf(vh, rnd, 1.0);
        const qh = try o.gpa.alloc(u16, c.n * NQ * HD);
        o.randBf(qh, rnd, 1.0);
        const gh = try o.gpa.alloc(u16, c.n * NQ);
        o.randBf(gh, rnd, 2.0);
        const planted = c.window != 0 and c.pos0 >= c.window;
        if (planted) { // a key just outside each row's window, aligned with its query: any row that reads it is dominated by it
            for (0..c.n) |z| {
                const key = c.pos0 + z - c.window;
                for (0..NKV) |kvh| for (0..HD) |d| {
                    kh[(key * NKV + kvh) * HD + d] = o.toBf(3.0 * o.bf(qh[(z * NQ + kvh * (NQ / NKV)) * HD + d]));
                    vh[(key * NKV + kvh) * HD + d] = o.toBf(8.0);
                };
            }
        }
        // the device ring holds position p at slot p % ring; only the positions a row may read must survive
        const ring_k = try o.gpa.alloc(u16, @as(usize, c.ring) * NKV * HD);
        const ring_v = try o.gpa.alloc(u16, ring_k.len);
        @memset(ring_k, 0);
        @memset(ring_v, 0);
        const first = if (npos > c.ring) npos - c.ring else 0;
        for (first..npos) |p| {
            const slot = p % c.ring;
            @memcpy(ring_k[slot * NKV * HD ..][0 .. NKV * HD], kh[p * NKV * HD ..][0 .. NKV * HD]);
            @memcpy(ring_v[slot * NKV * HD ..][0 .. NKV * HD], vh[p * NKV * HD ..][0 .. NKV * HD]);
        }
        const kc = try o.upload(&r, ring_k);
        const vc = try o.upload(&r, ring_v);
        const q = try o.upload(&r, qh);
        const g = try o.upload(&r, gh);
        const out = try r.alloc(c.n * NQ * HD * 2);
        const span = if (c.window == 0) c.pos0 + c.n else @min(c.pos0 + c.n, c.window);
        const nch = (span + CHUNK - 1) / CHUNK;
        try o.run(&kp, .{ NKV, nch, c.n }, .{ q, kc, vc, po, pm, pl, c.pos0, CHUNK, c.ring, c.window, NKV, maxc, @as(f32, 1.0 / @sqrt(128.0)) });
        try o.run(&km, .{ NQ, c.n, 1 }, .{ po, pm, pl, g, out, c.pos0, CHUNK, c.window, NQ, maxc });
        const oh = try o.fetch(&r, out, c.n * NQ * HD);
        const qf = try o.widen(o.gpa, qh);
        const kf = try o.widen(o.gpa, kh);
        const vf = try o.widen(o.gpa, vh);
        const gf = try o.widen(o.gpa, gh);
        const want = try o.gpa.alloc(f64, oh.len);
        try refRows(qf, kf, vf, gf, c.n, c.pos0, c.window, want);
        t.check(c.what, o.relErr(oh, want));
        if (ci == 2) { // planted control: a window of 257 reads the planted key and must not match
            try refRows(qf, kf, vf, gf, c.n, c.pos0, c.window + 1, want);
            t.control("sliding layer replayed with a window of 257", o.relErr(oh, want));
        }
    }
    try t.finish();
}
