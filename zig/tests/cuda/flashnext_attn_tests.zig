//! Flash Next's own attention glue and its indexer's pooling and scores against float64 references.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Rig = rig.Rig;
const bf = rig.bf;
const rbf = rig.rbf;
const ulps = rig.ulps;
const sumTol = rig.sumTol;

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const index_heads = 4;
const index_dim = 128;
const half = 32;
const proj_n = 13952;
const eps: f32 = 1e-6;
const cap = 3072;
const idw = 2052;

/// RoPE's inverse frequencies as the pass uploads them (base 10,000,000 over 32 pairs).
fn inverse(r: *Rig) ![]f32 {
    const inv = try r.a.alloc(f32, half);
    for (inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
    return inv;
}

/// The kernels' rotate-half RoPE of normed values xn, xpn (feature d of 2 * half rotated) at `pos`.
fn rope(xn: f64, xpn: f64, d: usize, pos: usize, inv: []const f32) f64 {
    if (d >= 2 * half) return xn;
    const i = if (d < half) d else d - half;
    const ang: f64 = @as(f32, @floatFromInt(pos)) * inv[i];
    return if (d < half) xn * @cos(ang) - xpn * @sin(ang) else xpn * @sin(ang) + xn * @cos(ang);
}

fn partner(d: usize) usize {
    return if (d < half) d + half else if (d < 2 * half) d - half else d;
}

/// A head of `width` features at src normed by w, rounded, roped at pos and rounded: the float64 chain.
fn prepped(p: []const u16, src: usize, width: usize, w: []const f32, pos: usize, inv: []const f32, d: usize) f64 {
    var ss: f64 = 0;
    for (0..width) |i| ss += bf(p[src + i]) * bf(p[src + i]);
    const rinv = 1 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
    const xn = rbf(bf(p[src + d]) * rinv * w[d]);
    const xpn = rbf(bf(p[src + partner(d)]) * rinv * w[partner(d)]);
    return rbf(rope(xn, xpn, d, pos, inv));
}

fn prep(r: *Rig, g: *const flash.Glue) !void {
    const counts = [_]usize{ 1, 3, 17 };
    const most = counts[counts.len - 1];
    const pos0 = 700;
    const p = r.bfs(most * proj_n, 2);
    const qw = r.f32s(head_dim, 0.5, 1.5);
    const kw = r.f32s(head_dim, 0.5, 1.5);
    const iw = r.f32s(index_dim, 0.5, 1.5);
    const inv = try inverse(r);
    const dp = try r.dev(std.mem.sliceAsBytes(p));
    const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
    const dqw = try r.dev(std.mem.sliceAsBytes(qw));
    const dkw = try r.dev(std.mem.sliceAsBytes(kw));
    const diw = try r.dev(std.mem.sliceAsBytes(iw));
    const dinv = try r.dev(std.mem.sliceAsBytes(inv));
    var qs: [counts.len]u64 = undefined;
    var iqs: [counts.len]u64 = undefined;
    var kcs: [counts.len]u64 = undefined;
    var vcs: [counts.len]u64 = undefined;
    var ikcs: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        qs[c] = try r.out(m * q_heads * head_dim * 2);
        iqs[c] = try r.out(m * index_heads * index_dim * 2);
        kcs[c] = try r.out(cap * kv_heads * head_dim * 2);
        vcs[c] = try r.out(cap * kv_heads * head_dim * 2);
        ikcs[c] = try r.out(cap * index_dim * 2);
        try g.attnPrep(r.s, dp, dpos, dqw, dkw, diw, dinv, qs[c], kcs[c], vcs[c], iqs[c], ikcs[c], eps, m, .{});
    }
    const last = counts.len - 1;
    const q = try r.back(u16, qs[last], most * q_heads * head_dim);
    const iq = try r.back(u16, iqs[last], most * index_heads * index_dim);
    const kc = try r.back(u16, kcs[last], cap * kv_heads * head_dim);
    const vc = try r.back(u16, vcs[last], cap * kv_heads * head_dim);
    const ikc = try r.back(u16, ikcs[last], cap * index_dim);
    for (0..most) |row| {
        const pos = pos0 + row;
        const base = row * proj_n;
        for (0..q_heads) |h| for (0..head_dim) |d| {
            const want = prepped(p, base + h * 2 * head_dim, head_dim, qw, pos, inv, d);
            try r.within("prep queries", bf(q[(row * q_heads + h) * head_dim + d]), want, ulps(want, 1) + 1e-3);
        };
        for (0..kv_heads) |h| for (0..head_dim) |d| {
            const want = prepped(p, base + q_heads * 2 * head_dim + h * head_dim, head_dim, kw, pos, inv, d);
            const at = (pos * kv_heads + h) * head_dim + d;
            try r.within("prep keys", bf(kc[at]), want, ulps(want, 1) + 1e-3);
            try check.expect(vc[at] == p[base + q_heads * 2 * head_dim + kv_heads * head_dim + h * head_dim + d], "prep values: row {d} head {d} copied", .{ row, h });
        };
        for (0..index_heads) |h| for (0..index_dim) |d| {
            const want = prepped(p, base + q_heads * 2 * head_dim + 2 * kv_heads * head_dim + h * index_dim, index_dim, iw, pos, inv, d);
            try r.within("prep indexer queries", bf(iq[(row * index_heads + h) * index_dim + d]), want, ulps(want, 1) + 1e-3);
        };
        for (0..index_dim) |d| try check.expect(ikc[pos * index_dim + d] == p[base + q_heads * 2 * head_dim + 2 * kv_heads * head_dim + index_heads * index_dim + d], "prep indexer keys: row {d} copied", .{row});
    }
    for (counts[0..last], 0..) |m, c| {
        try r.sameRows("prep queries", u16, qs[c], q[0 .. m * q_heads * head_dim]);
        try r.sameRows("prep keys", u16, kcs[c] + pos0 * kv_heads * head_dim * 2, kc[pos0 * kv_heads * head_dim ..][0 .. m * kv_heads * head_dim]);
    }
    // the output gate: bf16(o / (1 + e^-gate)) and its group sums
    const o = r.bfs(most * q_heads * head_dim, 2);
    const out = try r.out(most * q_heads * head_dim * 2);
    const xs = try r.out(most * q_heads * head_dim / 32 * 4);
    try g.attnGate(r.s, try r.dev(std.mem.sliceAsBytes(o)), dp, out, xs, most, proj_n, q_heads, head_dim);
    const go = try r.back(u16, out, most * q_heads * head_dim);
    const gx = try r.back(f32, xs, most * q_heads * head_dim / 32);
    for (0..most) |row| for (0..q_heads) |h| {
        for (0..head_dim / 32) |gi| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..32) |k| {
                const d = gi * 32 + k;
                const at = (row * q_heads + h) * head_dim + d;
                const want = bf(o[at]) * rig.sigmoid(bf(p[row * proj_n + h * 2 * head_dim + head_dim + d]));
                try r.within("output gate", bf(go[at]), want, ulps(want, 0.5) + @abs(want) * 0x1p-18);
                sum += bf(go[at]);
                abs += @abs(bf(go[at]));
            }
            try r.within("output gate sums", gx[(row * q_heads + h) * (head_dim / 32) + gi], sum, sumTol(abs, 32));
        }
    };
}

/// The indexer's pooled blocks and each row's block scores.
fn indexer(r: *Rig, g: *const flash.Glue) !void {
    const pos0 = 2196;
    const rows = 5;
    const ikc = r.bfs(cap * index_dim, 2);
    const w = r.f32s(index_dim, 0.5, 1.5);
    const inv = try inverse(r);
    const dikc = try r.dev(std.mem.sliceAsBytes(ikc));
    const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
    const pooled = try r.out((cap / 4) * index_dim * 2);
    try g.qsaPool(r.s, dikc, pooled, dpos, try r.dev(std.mem.sliceAsBytes(w)), try r.dev(std.mem.sliceAsBytes(inv)), eps, rows, .{});
    const got = try r.back(u16, pooled, (cap / 4) * index_dim);
    var b: usize = pos0 / 4;
    while (4 * b + 4 <= pos0 + rows) : (b += 1) {
        var x: [index_dim]f64 = undefined;
        var ss: f64 = 0;
        for (&x, 0..) |*v, d| {
            var acc: f64 = 0;
            for (0..4) |k| acc += bf(ikc[(4 * b + k) * index_dim + d]);
            v.* = rbf(acc / 4);
            ss += v.* * v.*;
        }
        const rinv = 1 / @sqrt(ss / index_dim + eps);
        for (0..index_dim) |d| {
            const xn = rbf(x[d] * rinv * w[d]);
            const xpn = rbf(x[partner(d)] * rinv * w[partner(d)]);
            const want = rbf(rope(xn, xpn, d, 4 * b, inv));
            try r.within("indexer pool", bf(got[b * index_dim + d]), want, ulps(want, 1) + 1e-3);
        }
    }
    // scores of every complete block for rows past 512 of them
    const keys = r.bfs((cap / 4) * index_dim, 1);
    const iq = r.bfs(rows * index_heads * index_dim, 1);
    const nb = 784;
    const dkeys = try r.dev(std.mem.sliceAsBytes(keys));
    const diq = try r.dev(std.mem.sliceAsBytes(iq));
    var scs: [2]u64 = undefined;
    for ([_]usize{ 1, rows }, 0..) |m, c| {
        scs[c] = try r.out(m * nb * 4);
        try g.qsaScores(r.s, diq, dkeys, dpos, scs[c], nb, m, (pos0 + rows) / 4, .{});
    }
    const sc = try r.back(f32, scs[1], rows * nb);
    for (0..rows) |row| {
        const complete = (pos0 + row + 1) / 4;
        for (0..complete) |blk| {
            var total: f64 = 0;
            var abs: f64 = 0;
            for (0..index_heads) |h| {
                var dot: f64 = 0;
                for (0..index_dim) |d| {
                    const t = bf(iq[(row * index_heads + h) * index_dim + d]) * bf(keys[blk * index_dim + d]);
                    dot += t;
                    abs += @abs(t);
                }
                total += @max(dot, 0);
            }
            const want = total / @sqrt(@as(f64, index_dim));
            try r.within("indexer scores", sc[row * nb + blk], want, sumTol(abs, 2 * index_dim) / @sqrt(@as(f64, index_dim)));
        }
    }
    try r.sameRows("indexer scores", f32, scs[0], sc[0 .. (pos0 + 1) / 4]);
}

/// Each row's attention over its keys (dense from position 0, or a sparse row's list) in 512-key chunks, merged.
fn attention(r: *Rig, g: *const flash.Glue, sparse: bool) !void {
    const counts = [_]usize{ 1, 3, 20 }; // up to 16 rows read the next tile ahead; 20 take one tile at a time
    const rows = counts[counts.len - 1];
    const pos0: usize = if (sparse) 2600 else 1100;
    const q = r.bfs(rows * q_heads * head_dim, 1);
    const kc = r.bfs(cap * kv_heads * head_dim, 1);
    const vc = r.bfs(cap * kv_heads * head_dim, 1);
    const ids = try r.a.alloc(i32, rows * idw);
    const nk = try r.a.alloc(i32, rows);
    const spr = try r.a.alloc(i32, rows);
    for (0..rows) |row| {
        nk[row] = @intCast(if (sparse) idw - 1 - row % 3 else pos0 + row + 1);
        spr[row] = @intFromBool(sparse);
        // a sparse row's list: distinct keys before its position, in a scrambled order
        for (0..idw) |i| ids[row * idw + i] = @intCast((i * 1237 + row * 31) % (pos0 + row + 1));
    }
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const dkc = try r.dev(std.mem.sliceAsBytes(kc));
    const dvc = try r.dev(std.mem.sliceAsBytes(vc));
    const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ @intCast(pos0), 0, 0, 0 }));
    const dids = try r.dev(std.mem.sliceAsBytes(ids));
    const dnk = try r.dev(std.mem.sliceAsBytes(nk));
    const dspr = try r.dev(std.mem.sliceAsBytes(spr));
    const nch = 5;
    const po = try r.out(rows * nch * q_heads * head_dim * 4);
    const pm = try r.out(rows * nch * q_heads * 4);
    const pl = try r.out(rows * nch * q_heads * 4);
    var outs: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        outs[c] = try r.out(m * q_heads * head_dim * 2);
        const chunks = if (sparse) (idw + 511) / 512 else (pos0 + m + 511) / 512;
        try g.attnChunks(r.s, dq, dkc, dvc, dpos, po, pm, pl, dids, dnk, dspr, m, kv_heads, chunks, true, .{});
        try g.attnMerge(r.s, po, pm, pl, dpos, outs[c], dnk, dspr, m, kv_heads, true);
    }
    const got = try r.back(u16, outs[counts.len - 1], rows * q_heads * head_dim);
    const name = if (sparse) "attention sparse" else "attention dense";
    const s = try r.a.alloc(f64, idw + pos0 + rows);
    for (0..rows) |row| for (0..q_heads) |h| {
        const kvh = h / (q_heads / kv_heads);
        const n: usize = @intCast(nk[row]);
        var top = -std.math.inf(f64);
        for (0..n) |j| {
            const key: usize = if (sparse) @intCast(ids[row * idw + j]) else j;
            var dot: f64 = 0;
            for (0..head_dim) |d| dot += bf(q[(row * q_heads + h) * head_dim + d]) * bf(kc[(key * kv_heads + kvh) * head_dim + d]);
            s[j] = dot * 0.0625;
            top = @max(top, s[j]);
        }
        var l: f64 = 0;
        for (s[0..n]) |*v| {
            v.* = @exp(v.* - top);
            l += v.*;
        }
        for (0..head_dim) |d| {
            var o: f64 = 0;
            var abs: f64 = 0;
            for (0..n) |j| {
                const key: usize = if (sparse) @intCast(ids[row * idw + j]) else j;
                const v = bf(vc[(key * kv_heads + kvh) * head_dim + d]);
                o += s[j] * v;
                abs += s[j] * @abs(v);
            }
            const want = o / l;
            // p goes into p . v as bf16 (2^-9 of each term), its sum l stays fp32
            try r.within(name, bf(got[(row * q_heads + h) * head_dim + d]), want, abs / l * 0x1p-8 + ulps(want, 0.5));
        }
    };
    for (counts[0 .. counts.len - 1], 0..) |m, c| try r.sameRows(name, u16, outs[c], got[0 .. m * q_heads * head_dim]);
    // prompt rows' fused launch (every chunk merged in the program): the chunks then merge's bytes
    const fused = try r.out(rows * q_heads * head_dim * 2);
    try g.attnFused(r.s, dq, dkc, dvc, dpos, fused, dids, dnk, dspr, rows, kv_heads, .{});
    try r.sameRows(if (sparse) "attention sparse fused" else "attention dense fused", u16, fused, got);
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(59);
    var r = try Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    try prep(&r, &g);
    try indexer(&r, &g);
    try attention(&r, &g, false);
    try attention(&r, &g, true);
    r.report("flashnext-attn: q, k and indexer heads, the output gate, pooled blocks and scores, dense and sparse attention");
}
