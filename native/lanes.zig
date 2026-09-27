//! Host-side ports of kernels/qwen/dense/v1. Metal source is shared verbatim.
const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;
const td = mx.td;
const tb = mx.tb;
pub const Act = struct { x: A, sums: ?A = null };
pub const Linear = struct {
    weight: A,
    sb: A,
    n: i32,
    k: i32,
    tiled: bool,
    scales: A = mx.empty,
    biases: A = mx.empty,
    pub fn init(s: *mx.Scope, weight: A, scales: A, biases: A) !Linear {
        const n = mx.dim(weight, 0);
        const k = mx.dim(weight, 1) * 8;
        const sb = try s.cast(try s.stack(&.{ try s.transpose(scales, &.{ 1, 0 }), try s.transpose(biases, &.{ 1, 0 }) }, -1), mx.bf16);
        const tiled = mx.tensor_units and @mod(n, 32) == 0;
        const w = if (tiled) try s.contiguous(try s.reshape(try s.transpose(try s.reshape(weight, &.{ @divExact(n, 32), 32, @divExact(k, 64), 8 }), &.{ 0, 2, 1, 3 }), &.{ n, @divExact(k, 8) })) else weight;
        try mx.evalMany(&.{ w, sb }, false);
        const own_w = try mx.retain(w);
        errdefer mx.free(own_w);
        const own_sb = try mx.retain(sb);
        errdefer mx.free(own_sb);
        const own_sc = if (!mx.tensor_units) try mx.retain(scales) else mx.empty;
        errdefer mx.free(own_sc);
        return .{ .weight = own_w, .sb = own_sb, .n = n, .k = k, .tiled = tiled, .scales = own_sc, .biases = if (!mx.tensor_units) try mx.retain(biases) else mx.empty };
    }
    pub fn deinit(l: *Linear) void {
        mx.free(l.weight);
        mx.free(l.sb);
        mx.free(l.scales);
        mx.free(l.biases);
    }
    pub fn apply(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, x: Act) !A {
        const m: i32 = @intCast(mx.c.mlx_array_size(x.x) / @as(usize, @intCast(l.k)));
        if (m < 1 or m > 128) return error.InvalidLaneWidth;
        if (!mx.tensor_units) {
            const split: i32 = if (l.n <= 64) 32 else if (l.n <= 2048) 16 else 8;
            const rt = @min(2, @divTrunc(m + 7, 8));
            var nt: i32 = if (@mod(l.n, 32) == 0) 4 else if (@mod(l.n, 16) == 0) 2 else 1;
            while (nt > 1 and split * rt * nt * 64 * 4 > 16384) nt = @divExact(nt, 2);
            const out = (try kernels.run(s, src.simd_qmm_mma, &.{ try s.reshape(x.x, &.{ m, l.k }), l.weight, l.scales, l.biases, try s.scalar(1) }, &.{ ti("K", l.k), ti("N", l.n), ti("S", split), ti("NT", nt), ti("RT", rt) }, .{ @divTrunc(l.n + 8 * nt - 1, 8 * nt) * split * 32, @divTrunc(m + 8 * rt - 1, 8 * rt), 1 }, .{ split * 32, 1, 1 }, &.{.{ .shape = &.{ m, l.n } }}))[0];
            return s.reshape(out, &.{ 1, m, l.n });
        }
        const mp = @divTrunc(m + 15, 16) * 16;
        const dims = try s.ints(&.{ m, mp });
        const x2 = try s.reshape(x.x, &.{ m, l.k });
        const sums = x.sums orelse (try kernels.run(s, src.lane_qmm_xsum, &.{ x2, dims }, &.{ti("K", l.k)}, .{ @divExact(l.k, 64), mp, 1 }, .{ @min(@divExact(l.k, 64), 256), 1, 1 }, &.{.{ .shape = &.{ @divExact(l.k, 64), mp }, .dtype = mx.f32t }}))[0];
        const tiles = @divTrunc(l.n + 31, 32);
        var sk: i32 = 1;
        while (sk < 8 and tiles * sk < 1024 and @divTrunc(@divExact(l.k, 64), sk * 2) >= 8) sk *= 2;
        const block = @min(mp, 32);
        const out = (try kernels.run(s, if (l.tiled) src.lane_qmm_main_tiled else src.lane_qmm_main, &.{ x2, sums, l.weight, l.sb, dims }, &.{ ti("TMR", @divExact(block, 16)), ti("N", l.n), ti("K", l.k), ti("NT", 32), ti("SK", sk) }, .{ tiles * 32 * sk, @divTrunc(mp + block - 1, block), 1 }, .{ 32 * sk, 1, 1 }, &.{.{ .shape = &.{ m, l.n } }}))[0];
        return s.reshape(out, &.{ 1, m, l.n });
    }
};

pub fn norm(k: *mx.Kernels, s: *mx.Scope, h: A, r: ?A, w: A) !struct { h: A, x: Act } {
    const width = mx.dim(h, -1);
    const m: i32 = @intCast(mx.c.mlx_array_size(h) / @as(usize, @intCast(width)));
    const mp = @divTrunc(m + 15, 16) * 16;
    const eps = try s.scalar(1e-6);
    const dims = try s.ints(&.{ m, mp });
    const hh = try s.reshape(h, &.{ m, width });
    const out = if (r) |res| try k.run(s, src.lane_glue_norm, &.{ hh, try s.reshape(res, &.{ m, width }), w, eps, dims }, &.{ti("K", width)}, .{ @divExact(width, 16), mp, 1 }, .{ @divExact(width, 16), 1, 1 }, &.{ .{ .shape = &.{ m, width } }, .{ .shape = &.{ m, width } }, .{ .shape = &.{ @divExact(width, 64), mp }, .dtype = mx.f32t } }) else try k.run(s, src.lane_glue_norm_nores, &.{ hh, w, eps, dims }, &.{ti("K", width)}, .{ @divExact(width, 16), mp, 1 }, .{ @divExact(width, 16), 1, 1 }, &.{ .{ .shape = &.{ m, width } }, .{ .shape = &.{ @divExact(width, 64), mp }, .dtype = mx.f32t } });
    return .{ .h = if (r != null) try s.reshape(out[0], &.{ 1, m, width }) else h, .x = .{ .x = try s.reshape(out[@intFromBool(r != null)], &.{ 1, m, width }), .sums = out[1 + @as(usize, @intFromBool(r != null))] } };
}
pub fn mlp(k: *mx.Kernels, s: *mx.Scope, gate: A, up: A) !Act {
    const m = mx.dim(gate, 1);
    const width = mx.dim(gate, 2);
    const mp = @divTrunc(m + 15, 16) * 16;
    const out = try k.run(s, src.lane_glue_mlp_act, &.{ gate, up, try s.ints(&.{ m, mp }) }, &.{ti("N", width)}, .{ width, mp, 1 }, .{ 64, 1, 1 }, &.{ .{ .shape = &.{ 1, m, width } }, .{ .shape = &.{ @divExact(width, 64), mp }, .dtype = mx.f32t } });
    return .{ .x = out[0], .sums = out[1] };
}

pub const Tree = struct {
    parents: []const i32,
    depths: [128]i32 = @splat(0),
    paths: [128 * 128]i32 = @splat(0),
    windows: [128 * 4]i32 = undefined,
    chain: bool = true,
    max_depth: i32 = 0,
    pub fn init(parents: []const i32) !Tree {
        if (parents.len == 0 or parents.len > 128 or parents[0] != -1) return error.InvalidTree;
        var t = Tree{ .parents = parents };
        for (parents, 0..) |p, i| {
            if (i > 0 and (p < 0 or p >= i)) return error.InvalidTree;
            if (p != @as(i32, @intCast(i)) - 1) t.chain = false;
            if (p >= 0) {
                const pp: usize = @intCast(p);
                t.depths[i] = t.depths[pp] + 1;
                @memcpy(t.paths[i * 128 ..][0..@intCast(t.depths[i])], t.paths[pp * 128 ..][0..@intCast(t.depths[i])]);
            }
            t.paths[i * 128 + @as(usize, @intCast(t.depths[i]))] = @intCast(i);
            t.max_depth = @max(t.max_depth, t.depths[i]);
            for (0..4) |j| {
                const d = t.depths[i] - 3 + @as(i32, @intCast(j));
                t.windows[i * 4 + j] = if (d < 0) 3 + d else 3 + t.paths[i * 128 + @as(usize, @intCast(d))];
            }
        }
        if (!t.chain and parents.len > 32) return error.InvalidTree;
        return t;
    }
};

pub fn attention(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, t: *const Tree) !A {
    return attentionCapacity(k, s, q, keys, values, t, mx.dim(keys, 2));
}
pub fn attentionCapacity(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, t: *const Tree, used: i32) !A {
    if (used > mx.dim(keys, 2) or used < t.parents.len) return error.InvalidAttentionShape;
    if (!mx.tensor_units) return rowAttention(k, s, q, keys, values, t, used);
    const w: i32 = @intCast(t.parents.len);
    const h = mx.dim(q, 1);
    const d = mx.dim(q, 3);
    const hkv = mx.dim(keys, 1);
    const g = @divExact(h, hkv);
    const len = used;
    const p = len - w;
    const pt = @divTrunc(p, 64) * 64;
    const ca = @divTrunc(pt + 511, 512);
    const ncb = @divTrunc(p + t.max_depth, 512) - @divTrunc(pt, 512) + 1;
    const r = g * w;
    const rp = @divTrunc(r + 15, 16) * 16;
    const sga = @divExact(rp, 16);
    const sg = @min(sga, 16);
    const scale = try s.scalar(0.0625);
    const qb0 = try s.transpose(try s.reshape(q, &.{ hkv, g, w, d }), &.{ 0, 2, 1, 3 });
    var qa = try s.reshape(qb0, &.{ hkv, r, d });
    if (rp != r) qa = try s.cat(&.{ qa, try s.zeros(&.{ hkv, rp - r, d }, mx.bf16) }, 1);
    qa = try s.contiguous(qa);
    const zero = try s.zeros(&.{1}, mx.f32t);
    var a = [_]A{ zero, zero, zero, mx.empty, mx.empty };
    if (ca > 0) a = try k.run(s, src.lane_attention_partial_direct, &.{ qa, keys, values, scale, try s.ints(&.{ pt, ca, w, 0, sga }) }, &.{ ti("G", g), ti("D", d), ti("SG", sg), ti("CK", 512), ti("TK", 64) }, .{ hkv * 32 * sg, ca, @divTrunc(sga + sg - 1, sg) }, .{ 32 * sg, 1, 1 }, &.{ .{ .shape = &.{hkv * ca * rp * d}, .dtype = mx.f32t }, .{ .shape = &.{hkv * ca * rp}, .dtype = mx.f32t }, .{ .shape = &.{hkv * ca * rp}, .dtype = mx.f32t } });
    const qb = try s.contiguous(try s.cat(&.{ qb0, try s.zeros(&.{ hkv, w, 16 - g, d }, mx.bf16) }, 2));
    const dims = try s.ints(&.{ len, p, pt, ncb, w, rp, ca });
    const b = try k.run(s, src.lane_attention_tail, &.{ qb, keys, values, scale, dims, try s.ints(t.paths[0 .. t.parents.len * 128]), try s.ints(t.depths[0..t.parents.len]), a[0], a[1], a[2] }, &.{ ti("G", g), ti("D", d), ti("CK", 512), ti("TK", 64), ti("MAXD", 128) }, .{ hkv * 32, ncb, w }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{hkv * ncb * w * 16 * d}, .dtype = mx.f32t }, .{ .shape = &.{hkv * ncb * w * 16}, .dtype = mx.f32t }, .{ .shape = &.{hkv * ncb * w * 16}, .dtype = mx.f32t } });
    return (try k.run(s, src.lane_attention_tree_merge, &.{ a[0], a[1], a[2], b[0], b[1], b[2], dims }, &.{ ti("G", g), ti("D", d), ti("CK", 512) }, .{ hkv * 32, r, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, h, w, d } }}))[0];
}

/// Causal attention for the last query rows, including Nemotron's 128-wide heads.
pub fn sdpa(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, scale: f32) !A {
    const h = mx.dim(q, 1);
    const w = mx.dim(q, 2);
    const d = mx.dim(q, 3);
    const hkv = mx.dim(keys, 1);
    const len = mx.dim(keys, 2);
    if ((d != 128 and d != 256) or w < 1 or w > 128 or w > len or @mod(h, hkv) != 0) return error.InvalidAttentionShape;
    const g = @divExact(h, hkv);
    const r = g * w;
    const rp = @divTrunc(r + 15, 16) * 16;
    const sga = @divExact(rp, 16);
    const sg = @min(sga, 16);
    var qp = try s.reshape(try s.transpose(try s.reshape(q, &.{ hkv, g, w, d }), &.{ 0, 2, 1, 3 }), &.{ hkv, r, d });
    if (rp != r) qp = try s.cat(&.{ qp, try s.zeros(&.{ hkv, rp - r, d }, mx.bf16) }, 1);
    qp = try s.contiguous(qp);
    const nch = @divTrunc(len + 511, 512);
    const dims = try s.ints(&.{ len, nch, w, 1, sga });
    const part = try k.run(s, if (d == 128) src.lane_attention_partial_direct_128 else src.lane_attention_partial_direct, &.{ qp, keys, values, try s.scalar(scale), dims }, &.{ ti("G", g), ti("D", d), ti("SG", sg), ti("CK", 512), ti("TK", 64) }, .{ hkv * 32 * sg, nch, @divTrunc(sga + sg - 1, sg) }, .{ 32 * sg, 1, 1 }, &.{ .{ .shape = &.{hkv * nch * rp * d}, .dtype = mx.f32t }, .{ .shape = &.{hkv * nch * rp}, .dtype = mx.f32t }, .{ .shape = &.{hkv * nch * rp}, .dtype = mx.f32t } });
    return (try k.run(s, src.lane_attention_merge, &.{ part[0], part[1], part[2], dims }, &.{ ti("G", g), ti("D", d) }, .{ hkv * 32, r, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, h, w, d } }}))[0];
}

fn rowAttention(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, t: *const Tree, used: i32) !A {
    const w: i32 = @intCast(t.parents.len);
    const h = mx.dim(q, 1);
    const d = mx.dim(q, 3);
    const hkv = mx.dim(keys, 1);
    const g = @divExact(h, hkv);
    const cap = mx.dim(keys, 2);
    const start = used - w;
    const maxd = t.max_depth + 1;
    const nch = @divTrunc(start + maxd + 127, 128);
    var paths: [128 * 128]i32 = undefined;
    for (0..t.parents.len) |i| @memcpy(paths[i * @as(usize, @intCast(maxd)) ..][0..@intCast(maxd)], t.paths[i * 128 ..][0..@intCast(maxd)]);
    const dims = try s.ints(&.{ start, w, cap, nch, maxd });
    const out = try k.run(s, src.row_attention_partial, &.{ try s.contiguous(q), try s.contiguous(keys), try s.contiguous(values), try s.ints(t.depths[0..t.parents.len]), try s.ints(paths[0 .. t.parents.len * @as(usize, @intCast(maxd))]), try s.scalar(0.0625), dims }, &.{ ti("D", d), ti("G", g), ti("CK", 128), ti("SPLIT", 4), ti("BLK", 4) }, .{ 32 * g * 4, nch, hkv }, .{ 32 * g * 4, 1, 1 }, &.{ .{ .shape = &.{h * w * nch}, .dtype = mx.f32t }, .{ .shape = &.{h * w * nch}, .dtype = mx.f32t }, .{ .shape = &.{ h * w * nch, d }, .dtype = mx.f32t } });
    return (try k.run(s, src.row_attention_merge, &.{ out[0], out[1], out[2], dims }, &.{ti("D", d)}, .{ 32, h, w }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, h, w, d } }}))[0];
}

test "tree ancestry, convolution windows, and invalid parents" {
    const t = try Tree.init(&.{ -1, 0, 0, 1, 3 });
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1, 2, 3 }, t.depths[0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 3, 4 }, t.paths[4 * 128 ..][0..4]);
    try std.testing.expectEqualSlices(i32, &.{ 3, 4, 6, 7 }, t.windows[16..20]);
    try std.testing.expectError(error.InvalidTree, Tree.init(&.{ -1, 2 }));
}
