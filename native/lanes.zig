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
    format: ?@import("quantization.zig").Spec = .{},
    generic: bool = false,
    signs: A = mx.empty,
    prism_dense: bool = false,
    pub fn initFormat(s: *mx.Scope, weight: A, scales: A, biases: A, format: ?@import("quantization.zig").Spec) !Linear {
        const f = format orelse {
            if (mx.shape(weight).len != 2 or mx.dim(weight, 0) < 1 or mx.dim(weight, 1) < 1) return error.InvalidTensorShape;
            if (mx.dtype(weight) != mx.bf16 and mx.dtype(weight) != mx.f32t and mx.dtype(weight) != mx.c.MLX_FLOAT16) return error.InvalidTensorDType;
            return .{ .weight = try mx.retain(weight), .sb = mx.empty, .n = mx.dim(weight, 0), .k = mx.dim(weight, 1), .tiled = false, .format = null, .generic = true };
        };
        const shape = try f.shape(mx.shape(weight), mx.shape(scales), mx.shape(biases));
        if (mx.dtype(weight) != mx.c.MLX_UINT32 or mx.dtype(scales) != mx.dtype(biases)) return error.InvalidTensorDType;
        if (mx.dtype(scales) != mx.bf16 and mx.dtype(scales) != mx.f32t and mx.dtype(scales) != mx.c.MLX_FLOAT16) return error.InvalidTensorDType;
        if (f.bits == 4 and f.group_size == 64 and mx.dtype(scales) == mx.bf16 and @mod(shape.n, if (mx.tensor_units) @as(i32, 4) else 8) == 0 and @mod(shape.k, 64) == 0) return init(s, weight, scales, biases);
        const w = try mx.retain(weight);
        errdefer mx.free(w);
        const sc = try mx.retain(scales);
        errdefer mx.free(sc);
        const tensor = mx.tensor_units and mx.dtype(scales) == mx.bf16 and @mod(shape.n, 4) == 0 and @mod(shape.k, 64) == 0 and (f.group_size == 64 or (f.bits == 4 and f.group_size == 32));
        const pairs = if (tensor) try mx.retain(try s.cast(try s.stack(&.{ try s.transpose(scales, &.{ 1, 0 }), try s.transpose(biases, &.{ 1, 0 }) }, -1), mx.bf16)) else mx.empty;
        errdefer mx.free(pairs);
        return .{ .weight = w, .scales = sc, .biases = try mx.retain(biases), .sb = pairs, .n = shape.n, .k = shape.k, .tiled = false, .format = f, .generic = true };
    }
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
        mx.free(l.signs);
    }
    pub fn apply(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, x: Act) !A {
        return l.applyWithReduction(kernels, s, x, null);
    }
    /// Match a stacked projection's SIMD reduction without duplicating its weights.
    pub fn applyWithReduction(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, input: Act, reduction: ?i32) !A {
        const x = if (l.signs.ctx != null) Act{ .x = try l.rotate(kernels, s, input.x) } else input;
        const m: i32 = @intCast(mx.c.mlx_array_size(x.x) / @as(usize, @intCast(l.k)));
        if (m < 1 or m > 128) return error.InvalidLaneWidth;
        if (l.prism_dense) return (try kernels.run(s, src.prism_dense, &.{ try s.reshape(x.x, &.{ m, l.k }), l.weight }, &.{ ti("K", l.k), ti("N", l.n) }, .{ 256 * @divTrunc(l.n + 7, 8), m, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ 1, m, l.n } }}))[0];
        if (l.generic) {
            if (l.sb.ctx != null) return l.tensorRows(kernels, s, x);
            if (!mx.tensor_units and l.format != null and l.format.?.bits == 4 and (l.format.?.group_size == 32 or l.format.?.group_size == 64) and mx.dtype(l.scales) == mx.bf16 and @mod(l.n, 8) == 0 and @mod(l.k, 64) == 0) return l.simdRows(kernels, s, x.x, reduction, l.format.?.group_size);
            return l.rows(kernels, s, x.x);
        }
        if (!mx.tensor_units) return l.simdRows(kernels, s, x.x, reduction, 64);
        const mp = @divTrunc(m + 15, 16) * 16;
        const dims = try s.ints(&.{ m, mp });
        const x2 = try s.reshape(x.x, &.{ m, l.k });
        const sums = x.sums orelse (try kernels.run(s, src.lane_qmm_xsum, &.{ x2, dims }, &.{ ti("K", l.k), ti("GS", 64) }, .{ @divExact(l.k, 64), mp, 1 }, .{ @min(@divExact(l.k, 64), 256), 1, 1 }, &.{.{ .shape = &.{ @divExact(l.k, 64), mp }, .dtype = mx.f32t }}))[0];
        const tiles = @divTrunc(l.n + 31, 32);
        var sk: i32 = 1;
        while (sk < 8 and tiles * sk < 1024 and @divTrunc(@divExact(l.k, 64), sk * 2) >= 8) sk *= 2;
        const block = @min(mp, 32);
        const out = (try kernels.run(s, if (l.tiled) src.lane_qmm_main_tiled else src.lane_qmm_main, &.{ x2, sums, l.weight, l.sb, dims }, &.{ ti("TMR", @divExact(block, 16)), ti("N", l.n), ti("K", l.k), ti("NT", 32), ti("SK", sk), ti("GS", 64), ti("EDGE", @intFromBool(@mod(mp, block) != 0)) }, .{ tiles * 32 * sk, @divTrunc(mp + block - 1, block), 1 }, .{ 32 * sk, 1, 1 }, &.{.{ .shape = &.{ m, l.n } }}))[0];
        return s.reshape(out, &.{ 1, m, l.n });
    }
    fn simdRows(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, x: A, reduction: ?i32, group: i32) !A {
        const m: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(l.k)));
        const split: i32 = reduction orelse (if (l.n <= 64) @as(i32, 32) else if (l.n <= 6144) 16 else 8);
        const rt = @min(2, @divTrunc(m + 7, 8));
        var nt: i32 = if (@mod(l.n, 32) == 0) 4 else if (@mod(l.n, 16) == 0) 2 else 1;
        while (nt > 1 and split * rt * nt * 64 * 4 > 16384) nt = @divExact(nt, 2);
        const out = (try kernels.run(s, src.simd_qmm_mma, &.{ try s.reshape(x, &.{ m, l.k }), l.weight, l.scales, l.biases, try s.scalar(1) }, &.{ ti("K", l.k), ti("N", l.n), ti("S", split), ti("GS", group), ti("SGS", split), ti("NT", nt), ti("RT", rt) }, .{ @divTrunc(l.n + 8 * nt - 1, 8 * nt) * split * 32, @divTrunc(m + 8 * rt - 1, 8 * rt), 1 }, .{ split * 32, 1, 1 }, &.{.{ .shape = &.{ m, l.n } }}))[0];
        return s.reshape(out, &.{ 1, m, l.n });
    }
    /// Python's installed lane linear falls back to MLX above 128 prompt rows.
    pub fn prefill(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, input: A) !A {
        if (l.prism_dense and mx.dim(input, 1) <= 128) return l.apply(kernels, s, .{ .x = input });
        const x = if (l.signs.ctx != null) try l.rotate(kernels, s, input) else input;
        if (l.generic) {
            if (l.sb.ctx != null and mx.dim(x, 1) <= 128) return l.tensorRows(kernels, s, .{ .x = x });
            return s.cast(try l.stock(s, x), mx.bf16);
        }
        if (mx.tensor_units and mx.dim(x, 1) <= 128) return l.apply(kernels, s, .{ .x = x });
        const w = if (l.tiled) try s.contiguous(try s.reshape(try s.transpose(try s.reshape(l.weight, &.{ @divExact(l.n, 32), @divExact(l.k, 64), 32, 8 }), &.{ 0, 2, 1, 3 }), &.{ l.n, @divExact(l.k, 8) })) else l.weight;
        const sb = try s.transpose(l.sb, &.{ 1, 0, 2 });
        const sc = try s.contiguous(try s.reshape(try s.slice(sb, 2, 0, 1), &.{ l.n, @divExact(l.k, 64) }));
        const bs = try s.contiguous(try s.reshape(try s.slice(sb, 2, 1, 2), &.{ l.n, @divExact(l.k, 64) }));
        var result = mx.c.mlx_array_new();
        const rc = mx.c.mlx_quantized_matmul(&result, x, w, sc, bs, true, mx.opt(64), mx.opt(4), "affine", mx.stream);
        return s.result(rc, result);
    }

    fn stock(l: Linear, s: *mx.Scope, x: A) !A {
        var result = mx.c.mlx_array_new();
        const rc = if (l.format) |f|
            mx.c.mlx_quantized_matmul(&result, x, l.weight, l.scales, l.biases, true, mx.opt(f.group_size), mx.opt(f.bits), "affine", mx.stream)
        else
            mx.c.mlx_matmul(&result, x, try s.transpose(l.weight, &.{ 1, 0 }), mx.stream);
        return s.result(rc, result);
    }

    fn rotate(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, x: A) !A {
        if (@mod(l.k, 1024) != 0 or mx.c.mlx_array_size(l.signs) != l.k) return error.InvalidHadamardSigns;
        const count: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(l.k)));
        return (try kernels.run(s, src.prism_rotate, &.{ try s.reshape(x, &.{ count, l.k }), l.signs }, &.{ti("K", l.k)}, .{ 512 * @divExact(l.k, 1024), count, 1 }, .{ 512, 1, 1 }, &.{.{ .shape = mx.shape(x) }}))[0];
    }

    fn tensorRows(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, input: Act) !A {
        const x = input.x;
        const f = l.format.?;
        const m: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(l.k)));
        const mp = @divTrunc(m + 15, 16) * 16;
        const dims = try s.ints(&.{ m, mp });
        const x2 = try s.reshape(x, &.{ m, l.k });
        const kg = @divExact(l.k, f.group_size);
        const sums = (if (f.group_size == 64) input.sums else null) orelse (try kernels.run(s, src.lane_qmm_xsum, &.{ x2, dims }, &.{ ti("K", l.k), ti("GS", f.group_size) }, .{ kg, mp, 1 }, .{ @min(kg, 256), 1, 1 }, &.{.{ .shape = &.{ kg, mp }, .dtype = mx.f32t }}))[0];
        const tiles = @divTrunc(l.n + 31, 32);
        var sk: i32 = 1;
        while (sk < 8 and tiles * sk < 1024 and @divTrunc(@divExact(l.k, 64), sk * 2) >= 8) sk *= 2;
        const block = @min(mp, 32);
        const args = [_]mx.Template{ ti("TMR", @divExact(block, 16)), ti("N", l.n), ti("K", l.k), ti("NT", 32), ti("SK", sk), ti(if (f.bits == 4) "GS" else "BITS", if (f.bits == 4) f.group_size else f.bits), ti(if (f.bits == 4) "EDGE" else "TILED", if (f.bits == 4) @intFromBool(@mod(mp, block) != 0) else 0) };
        const spec = if (f.bits == 4) src.lane_qmm_main else if (f.bits < 4) src.lane_qmm_lowbit else src.lane_qmm_bytes;
        return s.reshape((try kernels.run(s, spec, &.{ x2, sums, l.weight, l.sb, dims }, &args, .{ tiles * 32 * sk, @divTrunc(mp + block - 1, block), 1 }, .{ 32 * sk, 1, 1 }, &.{.{ .shape = &.{ m, l.n } }}))[0], &.{ 1, m, l.n });
    }

    pub fn rows(l: Linear, kernels: *mx.Kernels, s: *mx.Scope, x: A) !A {
        if (mx.dim(x, -1) != l.k or mx.dtype(x) != mx.bf16) return error.InvalidProjectionInput;
        const rows_: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(l.k)));
        if (rows_ < 1 or rows_ > 65536) return error.InvalidLaneWidth;
        const f = l.format orelse {
            const inputs = try s.reshape(x, &.{ rows_, l.k });
            const outputs = try mx.allocator.alloc(A, @intCast(rows_));
            defer mx.allocator.free(outputs);
            for (outputs, 0..) |*out, i| out.* = try l.stock(s, try s.slice(inputs, 0, @intCast(i), @intCast(i + 1)));
            return s.reshape(try s.cat(outputs, 0), &.{ 1, rows_, l.n });
        };
        const rt: i32 = if (rows_ == 1) 1 else if (rows_ == 2) 2 else if (rows_ <= 4) 4 else 8;
        const out = (try kernels.run(s, src.affine_rows, &.{ try s.contiguous(try s.reshape(x, &.{ rows_, l.k })), try padded(s, l.weight), try padded(s, l.scales), try padded(s, l.biases) }, &.{ ti("K", l.k), ti("N", l.n), ti("BITS", f.bits), ti("GS", f.group_size), ti("OPS", 2), ti("RT", rt), ti("SG", 8) }, .{ @divTrunc(l.n + 15, 16) * 256, @divTrunc(rows_ + rt - 1, rt), 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ 1, rows_, l.n } }}))[0];
        return out;
    }

    pub fn selectRanges(l: Linear, s: *mx.Scope, ranges: []const [2]i32) !Linear {
        const weight = if (l.tiled) try s.contiguous(try s.reshape(try s.transpose(try s.reshape(l.weight, &.{ @divExact(l.n, 32), @divExact(l.k, 64), 32, 8 }), &.{ 0, 2, 1, 3 }), &.{ l.n, @divExact(l.k, 8) })) else l.weight;
        var scales = l.scales;
        var biases = l.biases;
        if (!l.generic) {
            const sb = try s.transpose(l.sb, &.{ 1, 0, 2 });
            scales = try s.reshape(try s.slice(sb, 2, 0, 1), &.{ l.n, @divExact(l.k, 64) });
            biases = try s.reshape(try s.slice(sb, 2, 1, 2), &.{ l.n, @divExact(l.k, 64) });
        }
        if (ranges.len == 0 or ranges.len > 16) return error.InvalidProjectionRange;
        var ws: [16]A = undefined;
        var ss: [16]A = undefined;
        var bs: [16]A = undefined;
        for (ranges, 0..) |range, i| {
            if (range[0] < 0 or range[1] > l.n or range[0] >= range[1]) return error.InvalidProjectionRange;
            ws[i] = try s.slice(weight, 0, range[0], range[1]);
            if (l.format != null) {
                ss[i] = try s.slice(scales, 0, range[0], range[1]);
                bs[i] = try s.slice(biases, 0, range[0], range[1]);
            }
        }
        var selected = try initFormat(s, try s.cat(ws[0..ranges.len], 0), if (l.format != null) try s.cat(ss[0..ranges.len], 0) else mx.empty, if (l.format != null) try s.cat(bs[0..ranges.len], 0) else mx.empty, l.format);
        errdefer selected.deinit();
        if (l.signs.ctx != null) selected.signs = try mx.retain(l.signs);
        selected.prism_dense = l.prism_dense;
        return selected;
    }
};

fn padded(s: *mx.Scope, a: A) !A {
    const count: i32 = @intCast(mx.c.mlx_array_size(a));
    if (count >= 8) return a;
    return s.cat(&.{ try s.reshape(a, &.{count}), try s.zeros(&.{8 - count}, mx.dtype(a)) }, 0);
}

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
    if (!mx.tensor_units) return serialAttention(s, q, keys, values, t, used);
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

// Production SIMD uses MLX's single-query attention arithmetic. A tree node
// must see its ancestors in serial order, without unused buffer capacity.
fn serialAttention(s: *mx.Scope, q: A, keys: A, values: A, t: *const Tree, used: i32) !A {
    const start = used - @as(i32, @intCast(t.parents.len));
    var outputs: [128]A = undefined;
    for (0..t.parents.len) |row| {
        const r: i32 = @intCast(row);
        var k: A = undefined;
        var v: A = undefined;
        if (t.chain) {
            k = try s.slice(keys, 2, 0, start + r + 1);
            v = try s.slice(values, 2, 0, start + r + 1);
        } else {
            const ids = try s.ints(t.paths[row * 128 ..][0..@intCast(t.depths[row] + 1)]);
            k = try s.cat(&.{ try s.slice(keys, 2, 0, start), try s.take(try s.slice(keys, 2, start, used), ids, 2) }, 2);
            v = try s.cat(&.{ try s.slice(values, 2, 0, start), try s.take(try s.slice(values, 2, start, used), ids, 2) }, 2);
        }
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, try s.slice(q, 2, r, r + 1), k, v, 0.0625, "", mx.empty, mx.empty, false, mx.stream);
        outputs[row] = try s.result(rc, out);
    }
    return if (t.parents.len == 1) outputs[0] else s.cat(outputs[0..t.parents.len], 2);
}

test "tree ancestry, convolution windows, and invalid parents" {
    const t = try Tree.init(&.{ -1, 0, 0, 1, 3 });
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1, 2, 3 }, t.depths[0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 3, 4 }, t.paths[4 * 128 ..][0..4]);
    try std.testing.expectEqualSlices(i32, &.{ 3, 4, 6, 7 }, t.windows[16..20]);
    try std.testing.expectError(error.InvalidTree, Tree.init(&.{ -1, 2 }));
}
