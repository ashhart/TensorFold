//! MOSS-Image's Metal pipelines: core_qwen_vision's bf16 and fp32-sum projections, then its own source.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
pub const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

/// Each pipeline by its function: the tower's four bf16 projections, the two fp32-sum ones, then this family's.
pub const names = [_][]const u8{
    "qwv_gemm_bf16_nt_00",
    "qwv_gemm_bf16_nt_01",
    "qwv_gemm_bf16_nt_10",
    "qwv_gemm_bf16_nt_11",
    "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_n_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float",
    "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_t_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float",
    "mi_norm",
    "mi_qk",
    "mi_situ",
    "mi_gate_add",
    "mi_bias",
    "mi_attention_00",
    "mi_attention_01",
    "mi_attention_11",
    "me_norm",
    "me_head",
    "me_attention",
    "me_gate",
    "me_silu_mul",
    "me_conv",
    "me_gdn_prep",
    "me_gdn",
    "me_gated_norm",
};
const tower_count = 6;
const sums = 4;
const attention_first = 11;

pub const Kernels = struct {
    tower: mtl.Library,
    own: mtl.Library,
    p: [names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        const src = try frags.source(device, gpa, sources.core_qwen_vision);
        defer gpa.free(src);
        var k: Kernels = undefined;
        k.tower = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer k.tower.deinit();
        const own = try frags.source(device, gpa, sources.moss_image);
        defer gpa.free(own);
        k.own = try mtl.Library.fromSource(device, own, mtl.CompileOptions.mlx());
        errdefer k.own.deinit();
        var made: usize = 0;
        errdefer for (k.p[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            k.p[i] = try mtl.Pipeline.init(device, if (i < tower_count) k.tower else k.own, name, false);
            made += 1;
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (k.p) |p| p.deinit();
        k.own.deinit();
        k.tower.deinit();
    }

    pub fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }

    /// y [M, N] = x [M, K] w^T in bf16 with fp32 sums (tfp::gemm_nax), w as stored [N, K].
    pub fn linear(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: usize, N: usize, K: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(k.p[@as(usize, @intFromBool(M % 64 == 0)) * 2 + @intFromBool(N % 128 == 0)]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, N, tn, tm, 2, K / 256 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(y.buf, y.off, 3);
        e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }

    /// y [M, N] = bf16(x w^T + b): one split-K part's fp32 sums into `sum`, the bias, one rounding (N % 64 = 0).
    pub fn linearBias(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, sum: Ref, y: Ref, M: usize, N: usize, K: usize) void {
        std.debug.assert(N % 64 == 0);
        const tn = N / 64;
        const tm = (M + 63) / 64;
        const swz: u6 = if (tm <= 3) 0 else 1;
        e.setPipeline(k.p[sums + @as(usize, @intFromBool(M % 64 == 0))]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, N, tn, tm, 1, M * N, K, swz, 0, 0 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(sum.buf, sum.off, 3);
        const tiles = (tn << swz) * ((tm + (@as(usize, 1) << swz) - 1) >> swz);
        e.dispatchThreads(mtl.Size.of(tiles * 32, 2, 2), mtl.Size.of(32, 2, 2));
        bias(e, k.get("mi_bias"), sum, b, y, M, N);
    }

    /// o = softmax(q k^T / sqrt(128)) v, every row to every row: q, k, v [count, L, 128], o rows [L, count * 128].
    pub fn attention(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, o: Ref, L: usize, count: usize) void {
        attend(e, k.p[attention_first + attentionForm(L)], q, kb, vb, o, L, count);
    }

    /// One kernel's buffers in order, then its parameters at the next index.
    pub fn bind(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref, params: anytype) void {
        e.setPipeline(p);
        for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(params, refs.len);
    }
};

/// The attention form for L rows: 0 for L % 32 > 0, 1 for 32-aligned, 2 for 64-aligned (its query blocks whole too).
pub fn attentionForm(L: usize) usize {
    return if (L % 64 == 0) 2 else @intFromBool(L % 32 == 0);
}

/// mi_attention_* `p` (the form for L) over `count` heads of L rows.
pub fn attend(e: mtl.ComputeEncoder, p: mtl.Pipeline, q: Ref, kb: Ref, vb: Ref, o: Ref, L: usize, count: usize) void {
    const nq = (L + 63) / 64;
    const nk = (L + 31) / 32;
    e.setPipeline(p);
    for ([_]Ref{ q, kb, vb }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    const d = 128;
    const vals = [_]usize{ 1, count, L, L, 1, nq, nk, L / 64, L / 32, L % 64, L % 32, 0, 0, L * d, d, 0, L * d, d, 0, L * d, d, 0, d, count * d };
    var params: [32]i32 = @splat(0);
    for (vals, 0..) |v, i| params[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&params), 3);
    var f: [16]f32 = @splat(0);
    f[0] = 1.0 / @sqrt(@as(f32, d));
    e.setBytes(std.mem.asBytes(&f), 4);
    e.setBuffer(o.buf, o.off, 5);
    e.dispatchThreads(mtl.Size.of(nq * 32, count * 4, 1), mtl.Size.of(32, 4, 1));
}

/// y = mi_norm of x's `rows` rows of `width` with weight w, then by s and b when `modulated`; a threadgroup a row.
pub fn norm(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, s: Ref, b: Ref, y: Ref, rows: usize, width: usize, eps: f32, modulated: bool) void {
    const params = extern struct { width: u32, eps: f32, modulated: u32 }{ .width = @intCast(width), .eps = eps, .modulated = @intFromBool(modulated) };
    Kernels.bind(e, p, &.{ x, w, s, b, y }, params);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// q, k, v rows `stride` apart to normed, rotated heads [count, rows, 128] in q, k and v (copied).
pub fn heads(e: mtl.ComputeEncoder, p: mtl.Pipeline, qr: Ref, kr: Ref, vr: Ref, nq: Ref, nk: Ref, cos: Ref, sin: Ref, q: Ref, k: Ref, v: Ref, rows: usize, count: usize, stride: usize, eps: f32) void {
    const params = extern struct { rows: u32, heads: u32, stride: u32, eps: f32 }{ .rows = @intCast(rows), .heads = @intCast(count), .stride = @intCast(stride), .eps = eps };
    Kernels.bind(e, p, &.{ qr, kr, vr, nq, nk, cos, sin, q, k, v }, params);
    e.dispatchGroups(mtl.Size.of(rows, (count + 7) / 8, 1), mtl.Size.of(256, 1, 1));
}

/// o [rows, width] = situ(h, g) of rows `stride` apart.
pub fn situ(e: mtl.ComputeEncoder, p: mtl.Pipeline, h: Ref, g: Ref, o: Ref, rows: usize, width: usize, stride: usize) void {
    Kernels.bind(e, p, &.{ h, g, o }, [2]u32{ @intCast(width), @intCast(stride) });
    e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
}

/// x [rows, width] += y gated by the vector g, in place.
pub fn gateAdd(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, y: Ref, g: Ref, rows: usize, width: usize) void {
    Kernels.bind(e, p, &.{ x, y, g }, @as(u32, @intCast(width)));
    e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
}

/// y [rows, width] = bf16(c + b): fp32 sums and a bias vector, rounded once.
pub fn bias(e: mtl.ComputeEncoder, p: mtl.Pipeline, c: Ref, b: Ref, y: Ref, rows: usize, width: usize) void {
    Kernels.bind(e, p, &.{ c, b, y }, @as(u32, @intCast(width)));
    e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
}

/// y = Qwen3.5's zero-centered RMSNorm of `rows` rows of `width` (rows `in_stride` and `out_stride` apart).
pub fn encoderNorm(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, y: Ref, rows: usize, width: usize, in_stride: usize, out_stride: usize, eps: f32) void {
    const params = extern struct { width: u32, in_stride: u32, out_stride: u32, eps: f32 }{ .width = @intCast(width), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride), .eps = eps };
    Kernels.bind(e, p, &.{ x, w, y }, params);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// 256-wide heads to [count, rows, 256]: normed and turned by cs, sn [rows, 32] when `normed`, else copied.
pub fn encoderHeads(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, cs: Ref, sn: Ref, y: Ref, rows: usize, count: usize, in_stride: usize, head_stride: usize, eps: f32, normed: bool) void {
    const params = extern struct { rows: u32, heads: u32, in_stride: u32, head_stride: u32, eps: f32, normed: u32 }{ .rows = @intCast(rows), .heads = @intCast(count), .in_stride = @intCast(in_stride), .head_stride = @intCast(head_stride), .eps = eps, .normed = @intFromBool(normed) };
    Kernels.bind(e, p, &.{ x, w, cs, sn, y }, params);
    e.dispatchGroups(mtl.Size.of(rows, (count + 7) / 8, 1), mtl.Size.of(256, 1, 1));
}

/// o [rows, count * 256] = causal attention of q [count, rows, 256] over k, v [kv_count, rows, 256].
pub fn encoderAttention(e: mtl.ComputeEncoder, p: mtl.Pipeline, q: Ref, k: Ref, v: Ref, o: Ref, rows: usize, count: usize, kv_count: usize, scale: f32) void {
    const params = extern struct { rows: u32, heads: u32, kv_heads: u32, scale: f32 }{ .rows = @intCast(rows), .heads = @intCast(count), .kv_heads = @intCast(kv_count), .scale = scale };
    Kernels.bind(e, p, &.{ q, k, v, o }, params);
    e.dispatchGroups(mtl.Size.of(rows, count, 1), mtl.Size.of(256, 1, 1));
}

/// One of the elementwise steps (me_gate, me_silu_mul, me_conv) over `rows` rows of `width`.
pub fn encoderRows(e: mtl.ComputeEncoder, p: mtl.Pipeline, a: Ref, b: Ref, y: Ref, rows: usize, width: usize) void {
    Kernels.bind(e, p, &.{ a, b, y }, @as(u32, @intCast(width)));
    e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
}

/// The DeltaNet's geometry for me_gdn_prep and me_gdn.
pub const Gdn = extern struct { rows: u32, kheads: u32, vheads: u32, mix_stride: u32, scale: f32 };

/// qn, kn [rows, kheads, 128] and g, beta [rows, vheads] in fp32 from the conv's rows and the a, b projections.
pub fn encoderGdnPrep(e: mtl.ComputeEncoder, p: mtl.Pipeline, mix: Ref, a: Ref, b: Ref, a_log: Ref, dt_bias: Ref, qn: Ref, kn: Ref, g: Ref, beta: Ref, params: Gdn) void {
    Kernels.bind(e, p, &.{ mix, a, b, a_log, dt_bias, qn, kn, g, beta }, params);
    e.dispatchGroups(mtl.Size.of(params.rows, (params.kheads + 7) / 8, 1), mtl.Size.of(256, 1, 1));
}

/// The gated delta rule's out [rows, vheads, 128] bf16: a threadgroup a value head.
pub fn encoderGdn(e: mtl.ComputeEncoder, p: mtl.Pipeline, qn: Ref, kn: Ref, mix: Ref, g: Ref, beta: Ref, out: Ref, params: Gdn) void {
    Kernels.bind(e, p, &.{ qn, kn, mix, g, beta, out }, params);
    e.dispatchGroups(mtl.Size.of(params.vheads, 1, 1), mtl.Size.of(128, 1, 1));
}

/// y [rows, width] = the gated norm of x by its 128-wide heads with gate z and weight w.
pub fn encoderGatedNorm(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, z: Ref, w: Ref, y: Ref, rows: usize, width: usize, eps: f32) void {
    const params = extern struct { width: u32, in_stride: u32, out_stride: u32, eps: f32 }{ .width = @intCast(width), .in_stride = @intCast(width), .out_stride = @intCast(width), .eps = eps };
    Kernels.bind(e, p, &.{ x, z, w, y }, params);
    e.dispatchGroups(mtl.Size.of(rows, (width / 128 + 7) / 8, 1), mtl.Size.of(256, 1, 1));
}

test "every MOSS-Image pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var k = try Kernels.load(std.testing.allocator, device);
    k.deinit();
}
