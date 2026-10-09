//! Host references for the glue kernels (nemotron_*.cu): every output's bits as the kernel computes them.

const std = @import("std");
const m = @import("glue_math.zig");

const fma = m.fma;
const bf = m.bf16ToF32;
const tobf = m.f32ToBf16;

const nt = 256; // the norm kernels' threads a row

/// An xor butterfly over n lanes (n / 2 down to 1), each lane adding its partner to its own value.
fn butterfly(comptime T: type, comptime n: usize, v: [n]T) T {
    var a = v;
    var off: usize = n / 2;
    while (off > 0) : (off >>= 1) {
        var b: [n]T = undefined;
        for (0..n) |l| b[l] = a[l] + a[l ^ off];
        a = b;
    }
    return a[0];
}

/// A row's sum of squares as `threads` threads take it: element i on thread i % threads, warps summed in order.
fn sumSquares(x: []const f32, comptime threads: usize) f32 {
    var lane: [threads]f32 = @splat(0);
    for (x, 0..) |v, i| lane[i % threads] = fma(v, v, lane[i % threads]);
    var ss: f32 = undefined;
    for (0..threads / 32) |w| {
        const s = butterfly(f32, 32, lane[w * 32 ..][0..32].*);
        ss = if (w == 0) s else ss + s;
    }
    return ss;
}

fn invRms(ss: f32, n: usize, eps: f32) f32 {
    return 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
}

/// y = bf16((x * inv) * w), and each 64-group's sum of the stored values in order.
fn scaleStore(x: []const f32, inv: f32, w: []const u16, out: []u16, xs: []f32) void {
    for (x, w, out) |v, wv, *o| o.* = tobf((v * inv) * bf(wv));
    for (xs, 0..) |*s, g| {
        var acc: f32 = 0;
        for (out[g * 64 ..][0..64]) |y| acc = acc + bf(y);
        s.* = acc;
    }
}

/// tf_nemo_embed: MLX 4-bit rows, fma(q, scale, bias) rounded to bf16.
pub fn embed(ids: []const i32, w: []const u32, s: []const u16, b: []const u16, out: []u16, d: usize) void {
    for (ids, 0..) |id, r| {
        const tok: usize = @intCast(id);
        for (0..d) |i| {
            const q: f32 = @floatFromInt((w[tok * (d / 8) + i / 8] >> @intCast(4 * (i % 8))) & 0xF);
            const g = tok * (d / 64) + i / 64;
            out[r * d + i] = tobf(fma(q, bf(s[g]), bf(b[g])));
        }
    }
}

/// tf_nemo_add_rmsnorm: h = bf16(x + r) unless r is null, then y and xs.
pub fn addRmsnorm(gpa: std.mem.Allocator, x: []const u16, r: ?[]const u16, w: []const u16, h: []u16, y: []u16, xs: []f32, rows: usize, d: usize, eps: f32) !void {
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..rows) |row| {
        for (0..d) |i| {
            v[i] = bf(x[row * d + i]);
            if (r) |rr| {
                h[row * d + i] = tobf(v[i] + bf(rr[row * d + i]));
                v[i] = bf(h[row * d + i]);
            }
        }
        scaleStore(v, invRms(sumSquares(v, nt), d, eps), w, y[row * d ..][0..d], xs[row * (d / 64) ..][0..(d / 64)]);
    }
}

/// tf_nemo_add_moe_norm: delta from the slots in order (routed by fma, shared added), h, then y and xs.
pub fn addMoeNorm(gpa: std.mem.Allocator, x: []const u16, y32: ?[]const f32, y16: ?[]const u16, wt: []const f32, w: []const u16, hn: []u16, out: []u16, xs: []f32, rows: usize, d: usize, eps: f32, nr: usize, ns: usize) !void {
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..rows) |row| {
        for (0..d) |i| {
            var routed: f32 = 0;
            var shared: f32 = 0;
            for (0..ns) |s| {
                const at = (row * ns + s) * d + i;
                const yv = if (y32) |a| a[at] else bf(y16.?[at]);
                if (s < nr) routed = fma(wt[row * ns + s], yv, routed) else shared = shared + yv;
            }
            hn[row * d + i] = tobf(bf(x[row * d + i]) + m.rbf(routed + shared));
            v[i] = bf(hn[row * d + i]);
        }
        scaleStore(v, invRms(sumSquares(v, nt), d, eps), w, out[row * d ..][0..d], xs[row * (d / 64) ..][0..(d / 64)]);
    }
}

/// tf_nemo_concat_norms: [rmsnorm(e) * we | rmsnorm(h) * wh] and its group sums.
pub fn concatNorms(gpa: std.mem.Allocator, e: []const u16, h: []const u16, we: []const u16, wh: []const u16, out: []u16, xs: []f32, rows: usize, d: usize, eps: f32) !void {
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..rows) |row| for (0..2) |part| {
        const src = if (part == 0) e else h;
        for (0..d) |i| v[i] = bf(src[row * d + i]);
        const g = d / 64;
        scaleStore(v, invRms(sumSquares(v, nt), d, eps), if (part == 0) we else wh, out[row * 2 * d + part * d ..][0..d], xs[row * 2 * g + part * g ..][0..g]);
    };
}

/// tf_nemo_group_rmsnorm: per group of gs, n = bf16(v * inv), out = bf16(w * n), and the 64-group sums.
pub fn groupRmsnorm(gpa: std.mem.Allocator, x: []const u16, w: []const u16, out: []u16, xs: []f32, rows: usize, xd: usize, groups: usize, eps: f32) !void {
    const gs = xd / groups;
    const v = try gpa.alloc(f32, gs);
    defer gpa.free(v);
    for (0..rows) |row| for (0..groups) |grp| {
        const at = row * xd + grp * gs;
        for (0..gs) |i| v[i] = bf(x[at + i]);
        const inv = invRms(sumSquares(v, 128), gs, eps);
        for (0..gs) |i| out[at + i] = tobf(bf(w[grp * gs + i]) * m.rbf(v[i] * inv));
        for (0..gs / 64) |g| {
            var acc: f32 = 0;
            for (out[at + g * 64 ..][0..64]) |o| acc = acc + bf(o);
            xs[at / 64 + g] = acc;
        }
    };
}

/// tf_nemo_router: PART[s][row][e], one fma chain over the slice's inputs in order.
pub fn router(x: []const u16, w: []const u16, part: []f32, rows: usize, d: usize, e: usize, sk: usize) void {
    const per = d / sk;
    for (0..sk) |s| for (0..rows) |r| for (0..e) |ex| {
        var acc: f32 = 0;
        for (s * per..(s + 1) * per) |k| acc = fma(bf(x[r * d + k]), bf(w[ex * d + k]), acc);
        part[(s * rows + r) * e + ex] = acc;
    };
}

/// Sum slices in order, rank sigmoid scores plus bias (lowest id on ties), then add shared slots.
pub fn topk(part: []const f32, bias: []const f32, idx: []i32, wt: []f32, rows: usize, scaling: f32, e: usize, sk: usize, top_k: usize, ns: usize, norm: bool) void {
    var score: [256]f32 = undefined;
    var sel: [256]f32 = undefined;
    for (0..rows) |r| {
        for (0..e) |ex| {
            var logit: f32 = 0;
            for (0..sk) |s| logit = logit + part[(s * rows + r) * e + ex];
            score[ex] = m.sigmoid(logit);
            const v = score[ex] + bias[ex];
            sel[ex] = if (std.math.isNan(v)) -std.math.inf(f32) else v;
        }
        var total: f32 = 0;
        var probs: [8]f32 = undefined;
        var ids: [8]i32 = undefined;
        for (0..top_k) |k| {
            var bi: usize = 0;
            for (1..e) |ex| if (sel[ex] > sel[bi]) {
                bi = ex;
            };
            probs[k] = score[bi];
            ids[k] = @intCast(bi);
            total = total + score[bi];
            sel[bi] = -std.math.inf(f32);
        }
        for (0..ns) |k| {
            idx[r * ns + k] = if (k < top_k) ids[k] else @intCast(e + k - top_k);
            wt[r * ns + k] = if (k >= top_k) 1.0 else if (norm) (probs[k] / (total + 1e-20)) * scaling else probs[k] * scaling;
        }
    }
}

/// The conv taps in fma order, rounded to bf16, through SiLU, rounded again.
fn conv4(bias: f32, w: [4]f32, t0: f32, t1: f32, t2: f32, cur: f32) u16 {
    var acc = bias;
    acc = fma(w[0], t0, acc);
    acc = fma(w[1], t1, acc);
    acc = fma(w[2], t2, acc);
    acc = fma(w[3], cur, acc);
    return tobf(m.silu(m.rbf(acc)));
}

/// The Mamba layer's shapes as the kernels take them (Tri.Shape's fields).
pub const Shape = struct { proj: usize, xd: usize, cd: usize, heads: usize, dh: usize, groups: usize, rmax: usize };

/// tf_nemo_conv: kept rows of RAW[1 - parity] replay, then the window's rows; BASE commits after the last kept row.
pub fn conv(p: []const u16, base: []u16, raw: []u16, xc: []u16, cw: []const f32, cb: []const f32, parity: usize, pk: usize, rows: usize, s: Shape) void {
    const cd = s.cd;
    for (0..cd) |ch| {
        var t0 = bf(base[ch]);
        var t1 = bf(base[cd + ch]);
        var t2 = bf(base[2 * cd + ch]);
        const w = [4]f32{ cw[ch], cw[cd + ch], cw[2 * cd + ch], cw[3 * cd + ch] };
        for (0..pk + rows) |i| {
            const prev = i < pk;
            const cur = if (prev) bf(raw[((1 - parity) * s.rmax + i) * cd + ch]) else bf(p[(i - pk) * s.proj + s.xd + ch]);
            if (!prev) {
                const at = (parity * s.rmax + (i - pk)) * cd + ch;
                xc[at] = conv4(cb[ch], w, t0, t1, t2, cur);
                raw[at] = tobf(cur);
            }
            t0 = t1;
            t1 = t2;
            t2 = cur;
            if (i + 1 == pk) {
                base[ch] = tobf(t0);
                base[cd + ch] = tobf(t1);
                base[2 * cd + ch] = tobf(t2);
            }
        }
    }
}

/// tf_nemo_conv_rows then tf_nemo_conv_commit: a chunk's rows from its own inputs or BASE, then BASE's new last three.
pub fn convRows(p: []const u16, base: []u16, xc: []u16, cw: []const f32, cb: []const f32, rows: usize, s: Shape) void {
    const cd = s.cd;
    for (0..rows) |r| for (0..cd) |ch| {
        var t: [4]f32 = undefined;
        for (0..4) |j| {
            const src = @as(isize, @intCast(r)) - 3 + @as(isize, @intCast(j));
            t[j] = if (src >= 0) bf(p[@as(usize, @intCast(src)) * s.proj + s.xd + ch]) else bf(base[@as(usize, @intCast(src + 3)) * cd + ch]);
        }
        xc[r * cd + ch] = conv4(cb[ch], .{ cw[ch], cw[cd + ch], cw[2 * cd + ch], cw[3 * cd + ch] }, t[0], t[1], t[2], t[3]);
    };
    for (0..cd) |ch| {
        var v: [3]u16 = undefined;
        for (0..3) |j| {
            const src = @as(isize, @intCast(rows)) - 3 + @as(isize, @intCast(j));
            v[j] = if (src >= 0) p[@as(usize, @intCast(src)) * s.proj + s.xd + ch] else base[(rows + j) * cd + ch];
        }
        for (0..3) |j| base[j * cd + ch] = v[j];
    }
}

/// tf_nemo_scan: per head and value row, the state replays the kept rows, then the window's; y as the kernel sums it.
pub fn scan(p: []const u16, xc: []const u16, dt: []f32, state: []f32, a: []const f32, dsk: []const f32, dtb: []const f32, y: []u16, parity: usize, pk: usize, rows: usize, lo: f32, hi: f32, s: Shape) void {
    const ds = 128;
    for (0..s.heads) |h| for (0..s.dh) |d| {
        const g = h / (s.heads / s.groups);
        var st: [ds]f32 = state[(h * s.dh + d) * ds ..][0..ds].*;
        var dt_row: f32 = 0;
        for (0..pk + rows) |i| {
            const prev = i < pk;
            const buf = if (prev) 1 - parity else parity;
            const row = if (prev) i else i - pk;
            const base = (buf * s.rmax + row) * s.cd;
            const dt_at = (buf * s.rmax + row) * s.heads + h;
            if (prev) dt_row = dt[dt_at] else {
                const v = bf(p[row * s.proj + s.xd + s.cd + h]) + dtb[h];
                dt_row = @min(@max(m.softplus(v), lo), hi);
                if (d == 0) dt[dt_at] = dt_row;
            }
            const x = bf(xc[base + h * s.dh + d]);
            const da = m.exp(a[h] * dt_row);
            const xdt = x * dt_row;
            var outs: [4]f32 = undefined;
            for (0..4) |q| {
                var acc = [4]f32{ 0, 0, 0, 0 };
                for (0..8) |j| for (0..4) |c| {
                    const n = 4 * (4 * j + q) + c;
                    st[n] = fma(xdt, bf(xc[base + s.xd + g * ds + n]), st[n] * da);
                    acc[c] = fma(st[n], bf(xc[base + s.xd + s.groups * ds + g * ds + n]), acc[c]);
                };
                outs[q] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
            if (i + 1 == pk) state[(h * s.dh + d) * ds ..][0..ds].* = st;
            if (prev) continue;
            const out = (outs[0] + outs[1]) + (outs[2] + outs[3]);
            const yv = m.rbf(fma(x, dsk[h], out));
            const gz = m.rbf(m.silu(bf(p[row * s.proj + h * s.dh + d])));
            y[row * s.xd + h * s.dh + d] = tobf(gz * yv);
        }
    };
}

/// The attention layer's shapes as the kernels take them (Tri.Attn's fields).
pub const Attn = struct { nqkv: usize, heads: usize, kv_heads: usize, dim: usize, nch: usize };

const tile = 64;
const chunk = 512;

/// tf_nemo_attn_chunk then tf_nemo_attn_merge for rows at positions pos.., caches already written through them.
pub fn attention(qkv: []const u16, kc: []const u16, vc: []const u16, pos: usize, rows: usize, at: Attn, po: []f32, pm: []f32, pl: []f32, out: []u16, xs: []f32) void {
    const hd = 128;
    const grp = at.heads / at.kv_heads;
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(at.dim), -0.5));
    const inf = std.math.inf(f32);
    for (0..rows) |r| for (0..at.kv_heads) |hk| for (0..grp) |g| {
        const head = hk * grp + g;
        const limit = pos + r + 1;
        const nch = (limit + chunk - 1) / chunk;
        for (0..nch) |c| {
            var mx = -inf;
            var den: f32 = 0;
            var o: [hd]f32 = @splat(0);
            var key0 = c * chunk;
            while (key0 < (c + 1) * chunk and key0 < limit) : (key0 += tile) {
                var sc: [tile]f32 = undefined;
                var tile_m = -inf;
                for (0..tile) |j| {
                    const key = key0 + j;
                    if (key >= limit) {
                        sc[j] = -inf;
                        continue;
                    }
                    var acc: f32 = 0;
                    for (0..hd) |d| acc = fma(bf(qkv[r * at.nqkv + head * hd + d]), bf(kc[(key * at.kv_heads + hk) * hd + d]), acc);
                    sc[j] = acc * scale;
                    tile_m = @max(tile_m, sc[j]);
                }
                const active = tile_m != -inf;
                const next = if (active) @max(mx, tile_m) else mx;
                const alpha: f32 = if (!active) 1 else if (mx == -inf) 0 else m.exp(mx - next);
                var p: [tile]f32 = undefined;
                for (0..tile) |j| p[j] = if (active and key0 + j < limit) m.exp(sc[j] - next) else 0;
                var local: [16]f32 = undefined;
                for (0..16) |jj| {
                    var acc: f32 = 0;
                    for (0..4) |u| acc = acc + p[jj + 16 * u];
                    local[jj] = acc;
                }
                den = fma(den, alpha, butterfly(f32, 16, local));
                mx = next;
                for (0..hd) |d| {
                    var pv: f32 = 0;
                    for (0..tile) |j| {
                        const key = key0 + j;
                        const v: f32 = if (key < limit) bf(vc[(key * at.kv_heads + hk) * hd + d]) else 0;
                        pv = fma(p[j], v, pv);
                    }
                    o[d] = fma(o[d], alpha, pv);
                }
            }
            const base = (r * at.nch + c) * at.heads + head;
            @memcpy(po[base * hd ..][0..hd], &o);
            pm[base] = mx;
            pl[base] = den;
        }
        var mx = -inf;
        var den: f32 = 0;
        var o: [hd]f32 = @splat(0);
        for (0..nch) |c| {
            const base = (r * at.nch + c) * at.heads + head;
            const active = pl[base] > 0;
            const next = if (active) @max(mx, pm[base]) else mx;
            const a: f32 = if (!active) 1 else if (mx == -inf) 0 else m.exp(mx - next);
            const b: f32 = if (active) m.exp(pm[base] - next) else 0;
            for (0..hd) |d| o[d] = fma(o[d], a, po[base * hd + d] * b);
            den = fma(den, a, pl[base] * b);
            mx = next;
        }
        for (0..hd) |d| out[(r * at.heads + head) * hd + d] = tobf(o[d] / den);
        for (0..hd / 64) |gi| {
            var acc: f32 = 0;
            for (0..64) |j| acc = acc + bf(out[(r * at.heads + head) * hd + gi * 64 + j]);
            xs[r * (at.heads * hd / 64) + head * (hd / 64) + gi] = acc;
        }
    };
}

/// tf_nemo_keyed_greedy: rank 0 by (value desc, id asc) and, given `prob`, 1 / sum over the top k of e^(v - v0) in f64.
pub fn keyedGreedy(vals: []const f32, ids: []const i64, out: []i32, prob: ?[]f32, rows: usize, count: usize, k: usize) void {
    for (0..rows) |r| {
        var v: [32]f64 = @splat(-std.math.inf(f64));
        var id: [32]i64 = @splat(1 << 40);
        for (0..count) |i| {
            v[i] = vals[r * count + i];
            id[i] = ids[r * count + i];
        }
        var e: [32]f64 = @splat(0);
        var top = -std.math.inf(f64);
        var rank: [32]usize = undefined;
        for (0..32) |i| {
            rank[i] = 0;
            for (0..count) |j| {
                if (v[j] > v[i] or (v[j] == v[i] and id[j] < id[i])) rank[i] += 1;
            }
            if (i < count and rank[i] == 0) out[r] = @intCast(id[i]);
            if (i < count and rank[i] < k) top = @max(top, v[i]);
        }
        const p = prob orelse continue;
        for (0..count) |i| e[i] = if (rank[i] < k) @exp(v[i] - top) else 0;
        p[r] = @floatCast(1.0 / butterfly(f64, 32, e));
    }
}

test "one key: every head's output is that key's value, and the partials say one chunk of weight 1" {
    const gpa = std.testing.allocator;
    const a: Attn = .{ .nqkv = 32 * 128 + 2 * 2 * 128, .heads = 32, .kv_heads = 2, .dim = 128, .nch = 1 };
    const qkv = try gpa.alloc(u16, a.nqkv);
    defer gpa.free(qkv);
    for (qkv, 0..) |*v, i| v.* = tobf(@as(f32, @floatFromInt(i % 7)) - 3);
    const kv = 2 * 128;
    var kc: [kv]u16 = undefined;
    var vc: [kv]u16 = undefined;
    for (&kc, &vc, 0..) |*k, *v, i| {
        k.* = tobf(@as(f32, @floatFromInt(i % 5)) * 0.25);
        v.* = tobf(@as(f32, @floatFromInt(i % 11)) - 5);
    }
    var po: [32 * 128]f32 = undefined;
    var pm: [32]f32 = undefined;
    var pl: [32]f32 = undefined;
    var out: [32 * 128]u16 = undefined;
    var xs: [64]f32 = undefined;
    attention(qkv, &kc, &vc, 0, 1, a, &po, &pm, &pl, &out, &xs);
    for (0..32) |h| {
        try std.testing.expectEqual(@as(f32, 1), pl[h]);
        try std.testing.expectEqualSlices(u16, vc[(h / 16) * 128 ..][0..128], out[h * 128 ..][0..128]);
    }
}

test "top-k picks by score plus bias, the lowest id on a tie, then the shared slots" {
    var part: [8]f32 = .{ 0, 2, -1, 2, 0.5, -3, 1, 2 };
    const bias: [8]f32 = .{ 0, 0, 0, 0, 0, 0, 3, 0 };
    var idx: [4]i32 = undefined;
    var wt: [4]f32 = undefined;
    topk(&part, &bias, &idx, &wt, 1, 2.0, 8, 1, 2, 4, false);
    try std.testing.expectEqualSlices(i32, &.{ 6, 1, 8, 9 }, &idx);
    try std.testing.expectEqual(m.sigmoid(1) * 2.0, wt[0]);
    try std.testing.expectEqual(@as(f32, 1), wt[3]);
}
