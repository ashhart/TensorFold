//! Float64 formulas of Inkling 2's norms, attention residual, router, softcap, residual joins, from modeling_inkling2.
const std = @import("std");
const contract = @import("contract");
const g = contract.gates;
const c = contract.spec;

const unit: f64 = 0x1p-24;

/// Longest fp32 chain of a norm row sum: ceil(n/256) squares a thread, then five butterfly steps and seven adds.
pub fn rowSumDepth(n: usize) usize {
    return (n + 255) / 256 + 12;
}

/// RMS norm: weight * x * rsqrt(mean(x^2) + eps); `mag` = half |out| (a sum-of-squares error moves it by half).
pub fn rmsRow(x: []const f64, w: []const f64, eps: f64, out: []f64, mag: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, w, out, mag) |v, wi, *o, *m| {
        o.* = wi * (v * inv);
        m.* = @abs(o.*) / 2;
    }
}

/// The same row with the sum of squares replayed in `acc` (gate 4); `terms` is scratch of x.len.
pub fn rmsRowIn(acc: c.Accum, order: g.Order, x: []const f64, w: []const f64, eps: f64, terms: []f64, out: []f64) void {
    for (x, terms[0..x.len]) |v, *t| t.* = v * v;
    const ss = contract.rows.sumIn(acc, order, terms[0..x.len]);
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, w, out) |v, wi, *o| o.* = wi * (v * inv);
}

/// Normed embedding of a token: the RMS norm of its table row; an id past the vocabulary yields a NaN row.
pub fn embedRow(table: []const f64, id: u32, vocab: usize, hidden: usize, w: []const f64, eps: f64, out: []f64, mag: []f64) void {
    if (id >= vocab) {
        @memset(out[0..hidden], std.math.nan(f64));
        @memset(mag[0..hidden], 0);
        return;
    }
    rmsRow(table[@as(usize, id) * hidden ..][0..hidden], w, eps, out, mag);
}

pub fn embedRowIn(acc: c.Accum, order: g.Order, table: []const f64, id: u32, vocab: usize, hidden: usize, w: []const f64, eps: f64, terms: []f64, out: []f64) void {
    if (id >= vocab) return @memset(out[0..hidden], std.math.nan(f64));
    rmsRowIn(acc, order, table[@as(usize, id) * hidden ..][0..hidden], w, eps, terms, out);
}

pub const AttnRes = struct { hidden: usize, sources: usize, queries: usize, eps_u: f64, eps_n: f64 };

/// Block attention residual: mix = sum_s softmax_s(hidden^-0.5 q.u_s) u_s, u_s = rms(x_s, eps); out = norm(mix).
pub fn attnResRow(a: std.mem.Allocator, p: AttnRes, src: []const f64, q: []const f64, norms: []const f64, out: []f64, err: ?[]f64) !void {
    const H = p.hidden;
    const S = p.sources;
    const Hf: f64 = @floatFromInt(H);
    const buf = try a.alloc(f64, S * H + 4 * S + 2 * H);
    defer a.free(buf);
    const un = buf[0 .. S * H];
    const inv = buf[S * H ..][0..S];
    const score = buf[S * H + S ..][0..S];
    const prob = buf[S * H + 2 * S ..][0..S];
    const ds = buf[S * H + 3 * S ..][0..S];
    const mix = buf[S * H + 4 * S ..][0..H];
    const del = buf[S * H + 4 * S + H ..][0..H];
    for (0..S) |s| {
        var ss: f64 = 0;
        for (0..H) |i| ss += src[s * H + i] * src[s * H + i];
        inv[s] = 1 / @sqrt(ss / Hf + p.eps_u);
        for (0..H) |i| un[s * H + i] = src[s * H + i] * inv[s];
    }
    const scale = 1 / @sqrt(Hf);
    // fp32 chains of a dot or a sum of squares over the hidden axis: 256 threads, then 5 + 7 adds.
    const gd = c.gamma(@as(f64, @floatFromInt((H + 255) / 256 + 13)), unit);
    // 1 / rms of a source: its sum of squares is all-positive (rel. gd), then / n, + eps, rsqrt, (<= 5 roundings).
    const e_inv = gd / 2 + 5 * unit;
    for (0..p.queries) |qi| {
        const qv = q[qi * H ..][0..H];
        var dmax: f64 = 0;
        for (0..S) |s| {
            var t: f64 = 0;
            var ab: f64 = 0;
            for (0..H) |i| {
                t += qv[i] * un[s * H + i];
                ab += @abs(qv[i] * src[s * H + i]);
            }
            score[s] = scale * t;
            // The kernel dots the raw source (error gd * sum |q x|), then scales by inv_s and hidden^-0.5.
            ds[s] = scale * inv[s] * gd * ab + @abs(score[s]) * (e_inv + 3 * unit);
            dmax = @max(dmax, ds[s]);
        }
        var m: f64 = -std.math.inf(f64);
        for (score) |v| m = if (std.math.isNan(v) or std.math.isNan(m)) std.math.nan(f64) else @max(m, v);
        var l: f64 = 0;
        for (score, prob) |v, *e| {
            e.* = @exp(v - m);
            l += e.*;
        }
        for (prob) |*e| e.* /= l;
        // A score moved by dmax moves p_s by exp(+-2 dmax) plus exp/sum roundings; p_s * inv_s adds inverse's error.
        const rp = @exp(2 * dmax) - 1 + @as(f64, @floatFromInt(S + 6)) * unit;
        const rc = rp + e_inv + rp * e_inv + 3 * unit;
        const rmix = rc + c.gamma(@as(f64, @floatFromInt(S + 1)), unit);
        var n2: f64 = 0;
        for (0..H) |i| {
            var sum: f64 = 0;
            var mass: f64 = 0;
            for (0..S) |s| {
                sum += prob[s] * un[s * H + i];
                mass += prob[s] * @abs(un[s * H + i]);
            }
            mix[i] = sum;
            del[i] = rmix * mass;
            n2 += sum * sum;
        }
        const inv2 = 1 / @sqrt(n2 / Hf + p.eps_n);
        var cross: f64 = 0;
        var square: f64 = 0;
        for (0..H) |i| {
            cross += @abs(mix[i]) * del[i];
            square += del[i] * del[i];
        }
        // The second norm's sum moves by 2 sum |mix| del + sum del^2; (n2 + H eps)^-1/2 then moves by half as much.
        const ds2 = (2 * cross + square) * (1 + gd) + gd * n2;
        const x = ds2 / (n2 + Hf * p.eps_n);
        const ri = if (x < 0.5) 0.5 * x / (1 - x) else 1;
        for (0..H) |i| {
            const w = norms[qi * H + i];
            out[qi * H + i] = w * (mix[i] * inv2);
            if (err) |e| e[qi * H + i] = @abs(w) * inv2 * (@abs(mix[i]) * (ri + 6 * unit) + del[i] * (1 + ri));
        }
    }
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

pub fn dot(x: []const f64, w: []const f64) f64 {
    var s: f64 = 0;
    for (x, w) |a, b| s += a * b;
    return s;
}

/// Expert a ranks before b: a NaN above every number, a larger value above a smaller, equal values by the lower index.
pub fn before(biased: []const f64, a: u32, b: u32) bool {
    const an = std.math.isNan(biased[a]);
    const bn = std.math.isNan(biased[b]);
    if (an != bn) return an;
    if (!an and biased[a] != biased[b]) return biased[a] > biased[b];
    return a < b;
}

/// TopKRouter's picks: the first `ids.len` of the full ranking by score + bias, in rank order (`order` is scratch).
pub fn rank(biased: []const f64, order: []u32, ids: []u32) void {
    for (order[0..biased.len], 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order[0..biased.len], biased, before);
    @memcpy(ids, order[0..ids.len]);
}

/// weights = scores[picks] / (sum + 1e-20) * route_scale.
pub fn routeWeights(scores: []const f64, ids: []const u32, scale: f64, out: []f64) void {
    var sum: f64 = 0;
    for (ids) |id| sum += scores[id];
    for (ids, out[0..ids.len]) |id, *o| o.* = scores[id] / (sum + 1e-20) * scale;
}

/// Inkling2Softcap: tanh(y / cap) * cap.
pub fn softcap(y: f64, cap: f64) f64 {
    return std.math.tanh(y / cap) * cap;
}

test "rms norm of 3, 4 and the replayed sum" {
    const x = [_]f64{ 3, 4 };
    const w = [_]f64{ 2, 1 };
    var out: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    rmsRow(&x, &w, 0, &out, &mag);
    try std.testing.expectApproxEqAbs(2 * 3 / @sqrt(12.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(4 / @sqrt(12.5), out[1], 1e-12);
    var terms: [2]f64 = undefined;
    var again: [2]f64 = undefined;
    rmsRowIn(.f32, .sequential, &x, &w, 0, &terms, &again);
    try std.testing.expectApproxEqAbs(out[0], again[0], 1e-12);
}

test "an id outside the vocabulary is a NaN row" {
    const table = [_]f64{ 1, 2, 3, 4 };
    const w = [_]f64{ 1, 1 };
    var out: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    embedRow(&table, 2, 2, 2, &w, 0, &out, &mag);
    try std.testing.expect(std.math.isNan(out[0]) and std.math.isNan(out[1]));
    embedRow(&table, 1, 2, 2, &w, 0, &out, &mag);
    try std.testing.expectApproxEqAbs(3 / @sqrt(12.5), out[0], 1e-12);
}

test "attention residual over one source is that source normed twice" {
    const a = std.testing.allocator;
    const x = [_]f64{ 3, 4 };
    const q = [_]f64{ 1, -1 };
    const w = [_]f64{ 1, 1 };
    var out: [2]f64 = undefined;
    var err: [2]f64 = undefined;
    try attnResRow(a, .{ .hidden = 2, .sources = 1, .queries = 1, .eps_u = 0, .eps_n = 0 }, &x, &q, &w, &out, &err);
    try std.testing.expectApproxEqAbs(3 / @sqrt(12.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(4 / @sqrt(12.5), out[1], 1e-12);
    try std.testing.expect(err[0] > 0 and err[0] < 1e-5);
}

test "attention residual weighs two sources by their scaled dot products" {
    const a = std.testing.allocator;
    // Both sources already have rms 1 (eps 0) and the query (1, 0) scores each at 1 / sqrt 2: equal weights.
    const x = [_]f64{ 1, 1, 1, -1 };
    const q = [_]f64{ 1, 0 };
    const w = [_]f64{ 1, 1 };
    var out: [2]f64 = undefined;
    try attnResRow(a, .{ .hidden = 2, .sources = 2, .queries = 1, .eps_u = 0, .eps_n = 0 }, &x, &q, &w, &out, null);
    // scores are 1/sqrt 2 each: equal weights, mix = (1, 0), normed: (sqrt 2, 0)
    try std.testing.expectApproxEqAbs(@sqrt(2.0), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.0, out[1], 1e-12);
}

test "router ranks NaN first and ties to the lower index, weights over the picks" {
    const biased = [_]f64{ 0.5, 0.7, 0.7, std.math.nan(f64), 0.1 };
    var order: [5]u32 = undefined;
    var ids: [3]u32 = undefined;
    rank(&biased, &order, &ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 2 }, &ids);
    const scores = [_]f64{ 0.5, 0.25, 0.25, 0.5, 0.1 };
    var wt: [3]f64 = undefined;
    routeWeights(&scores, &ids, 2, &wt);
    try std.testing.expectApproxEqAbs(@as(f64, 1), wt[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), wt[1], 1e-12);
    try std.testing.expect(sigmoid(0) == 0.5);
}

test "softcap saturates at the cap" {
    try std.testing.expectEqual(@as(f64, 30), softcap(std.math.inf(f64), 30));
    try std.testing.expectEqual(@as(f64, -30), softcap(-std.math.inf(f64), 30));
    try std.testing.expect(std.math.isNan(softcap(std.math.nan(f64), 30)));
    try std.testing.expectEqual(@as(f64, 0), softcap(0, 30));
}
