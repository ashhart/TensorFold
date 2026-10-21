//! qk_rows.cu's attention inputs for every CUDA family: QK norm, RoPE (rotate-half, partial) and the cache write.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// A packed row's heads: h query, hk key and hk value heads of `dim` (a multiple of 32, at most 512).
pub const Heads = struct { h: usize, hk: usize, dim: usize };

/// Where a row's heads come from: q, k and v (0: none) at row stride *_ld and head stride *_hs (elements).
pub const Src = extern struct { q: u64, q_ld: i64, q_hs: i64, k: u64, k_ld: i64, k_hs: i64, v: u64 = 0, v_ld: i64 = 0, v_hs: i64 = 0 };

/// q/k norms [2, dim] (or q's with norm_k), fp32 or bf16; RoPE on `rot` dims: cos/sin tables or inv [rot / 2].
pub const Shape = struct {
    norms: ?u64 = null,
    norm_k: ?u64 = null,
    norms_bf16: bool = false,
    cos: ?u64 = null,
    sin: ?u64 = null,
    inv: ?u64 = null,
    rot: usize = 0,
    eps: f32 = 1e-6,
    /// MLX's rows: q and k = bf16(w * bf16(x * inv)) before RoPE.
    round: bool = false,
    /// v by a unit RMS norm, bf16(x * inv); else v as is.
    v_norm: bool = false,
};

/// qk_rows.cu's QkW.
const W = extern struct { q: u64, k: u64, bf16: i32 };

/// Where the rows go: q [n, h, dim], k and v [n, hk, dim]; k and v also into kc/vc at (slot * len + pos % len).
pub const Out = struct {
    q: u64,
    k: u64,
    v: u64,
    kc: ?u64 = null,
    vc: ?u64 = null,
    len: usize = 1,
    /// Row and head strides in elements (0: packed [n, heads, dim]); head-major [heads, n, dim]: ld dim, hs n dim.
    q_ld: usize = 0,
    q_hs: usize = 0,
    kv_ld: usize = 0,
    kv_hs: usize = 0,
};

/// qk_rows.cu's QkDst.
const Dst = extern struct { q: u64, q_ld: i64, q_hs: i64, k: u64, k_ld: i64, k_hs: i64, v: u64, v_ld: i64, v_hs: i64 };

pub const QkRows = struct {
    mod: Module,
    fn_: Function,
    /// tf_qk_rows_w: dim 128 with no or full rotation, a warp a head.
    warp_fn: Function,
    /// tf_qk_rows_once and tf_qk_rows_round: dims past 256, a unit v norm, MLX's rounding points.
    once_fn: Function,
    round_fn: Function,

    pub fn load(d: *const Driver) !QkRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qk_rows);
        errdefer mod.unload();
        return .{ .mod = mod, .fn_ = try mod.function("tf_qk_rows"), .warp_fn = try mod.function("tf_qk_rows_w"), .once_fn = try mod.function("tf_qk_rows_once"), .round_fn = try mod.function("tf_qk_rows_round") };
    }

    pub fn unload(g: *QkRows) void {
        g.mod.unload();
    }

    /// n packed rows qkv [n, (h + 2 hk) dim] bf16 at positions pos (int32 [n]) in cache slots slot (int32 [n] or none).
    pub fn rows(g: *const QkRows, s: Stream, qkv: u64, hs: Heads, shape: Shape, pos: u64, slot: ?u64, out: Out, n: usize) !void {
        const ld: i64 = @intCast((hs.h + 2 * hs.hk) * hs.dim);
        const dim: i64 = @intCast(hs.dim);
        const src: Src = .{ .q = qkv, .q_ld = ld, .q_hs = dim, .k = qkv + hs.h * hs.dim * 2, .k_ld = ld, .k_hs = dim, .v = qkv + (hs.h + hs.hk) * hs.dim * 2, .v_ld = ld, .v_hs = dim };
        return g.rowsFrom(s, src, hs, shape, pos, slot, out, n);
    }

    /// As `rows`, the heads read from `src` (e.g. [q | gate] rows: q_hs = 2 dim); no v source: no v out or vc.
    pub fn rowsFrom(g: *const QkRows, s: Stream, src: Src, hs: Heads, shape: Shape, pos: u64, slot: ?u64, out: Out, n: usize) !void {
        return g.rowsVia(null, 0, s, src, hs, shape, pos, slot, out, n);
    }

    /// rowsFrom through `f` (null: the kernel the shape selects) on tf_qk_rows_round's arguments, `last` its final int.
    pub fn rowsVia(g: *const QkRows, f: ?Function, last: i32, s: Stream, src: Src, hs: Heads, shape: Shape, pos: u64, slot: ?u64, out: Out, n: usize) !void {
        if (hs.dim % 32 != 0 or hs.dim > 512 or shape.rot % 2 != 0 or shape.rot > hs.dim) return error.HeadShape;
        if ((shape.cos == null) != (shape.sin == null) or (shape.cos != null and shape.inv != null)) return error.HeadShape;
        const roped = shape.cos != null or shape.inv != null;
        if (roped and shape.rot == 0) return error.HeadShape;
        if (src.v == 0 and out.vc != null) return error.HeadShape;
        if (n == 0) return;
        var a: launch_.Args = .{};
        a.add(src);
        const wsize: usize = if (shape.norms_bf16) 2 else 4;
        const nq = shape.norms orelse 0;
        a.add(W{ .q = nq, .k = shape.norm_k orelse (if (nq != 0) nq + hs.dim * wsize else 0), .bf16 = @intFromBool(shape.norms_bf16) });
        a.add(shape.cos orelse 0);
        a.add(shape.sin orelse 0);
        a.add(shape.inv orelse 0);
        a.add(@as(i32, @intCast(if (roped) shape.rot else 0)));
        a.add(pos);
        a.add(slot orelse 0);
        const dim: i64 = @intCast(hs.dim);
        const qld: i64 = if (out.q_ld != 0) @intCast(out.q_ld) else @as(i64, @intCast(hs.h)) * dim;
        const qhs: i64 = if (out.q_hs != 0) @intCast(out.q_hs) else dim;
        const kld: i64 = if (out.kv_ld != 0) @intCast(out.kv_ld) else @as(i64, @intCast(hs.hk)) * dim;
        const khs: i64 = if (out.kv_hs != 0) @intCast(out.kv_hs) else dim;
        a.add(Dst{ .q = out.q, .q_ld = qld, .q_hs = qhs, .k = out.k, .k_ld = kld, .k_hs = khs, .v = out.v, .v_ld = kld, .v_hs = khs });
        a.add(out.kc orelse 0);
        a.add(out.vc orelse 0);
        a.add(@as(i64, @intCast(out.len)));
        a.add(@as(i32, @intCast(hs.h)));
        a.add(@as(i32, @intCast(hs.hk)));
        a.add(shape.eps);
        const heads = hs.h + (if (src.v != 0) 2 * hs.hk else hs.hk);
        if (f != null or hs.dim > 256 or shape.round or shape.v_norm) {
            a.add(if (f != null) last else @as(i32, @intFromBool(shape.v_norm)));
            const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast(heads), .z = 1 };
            return launch_.launch(f orelse if (shape.round) g.round_fn else g.once_fn, .{ .grid = grid, .block = .{ .x = @intCast(hs.dim) } }, s, &a);
        }
        if (hs.dim == 128 and (!roped or shape.rot == 128)) {
            const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast((heads + 3) / 4), .z = 1 };
            return launch_.launch(g.warp_fn, .{ .grid = grid, .block = .{ .x = 128 } }, s, &a);
        }
        const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast(heads), .z = 1 };
        try launch_.launch(g.fn_, .{ .grid = grid, .block = .{ .x = @intCast(hs.dim) } }, s, &a);
    }
};
