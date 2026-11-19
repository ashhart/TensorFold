//! Idol's Metal pipelines: its fp16 projections and steps, fp32 projections and head-64 attention.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const cfg = @import("config.zig");
pub const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

/// Each pipeline by its function.
pub const names = [_][]const u8{
    "id_linear_00",
    "id_linear_01",
    "id_linear_10",
    "id_linear_11",
    "id_linear_gelu_00",
    "id_linear_gelu_01",
    "id_linear_gelu_10",
    "id_linear_gelu_11",
    "id_norm",
    "id_norm32",
    "id_repo",
    "id_qk",
    "id_gate_add",
    "id_embed",
    "id_cond",
    "id_bias32",
    "id_gemm32",
    "id_split_0",
    "id_split_1",
    "id_norm_split",
    "id_attention_00",
    "id_attention_01",
    "id_attention_10",
    "id_attention_11",
};
const linear_first = 0;
const attention_first = blk: {
    for (names, 0..) |n, i| if (std.mem.eql(u8, n, "id_attention_00")) break :blk i;
    unreachable;
};

pub const Norm = enum(u32) { plain = 0, affine = 1, modulated = 2 };

pub const Kernels = struct {
    own: mtl.Library,
    p: [names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        var k: Kernels = undefined;
        const own = try frags.source(device, gpa, sources.idol);
        defer gpa.free(own);
        k.own = try mtl.Library.fromSource(device, own, mtl.CompileOptions.mlx());
        errdefer k.own.deinit();
        var made: usize = 0;
        errdefer for (k.p[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            k.p[i] = try mtl.Pipeline.init(device, k.own, name, false);
            made += 1;
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (k.p) |p| p.deinit();
        k.own.deinit();
    }

    pub fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }

    /// y [M, N] (rows `ldd` apart) = x [M, K] w^T in fp32 FMA, w as stored [N, K]; K % 32, N % 64 (or N < 64, % 8) = 0.
    pub fn linear32(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: usize, N: usize, K: usize, ldd: usize) void {
        std.debug.assert(K % 32 == 0 and (N % 64 == 0 or (N < 64 and N % 8 == 0)));
        const params = [4]u32{ @intCast(M), @intCast(N), @intCast(K), @intCast(ldd) };
        bind(e, k.get("id_gemm32"), &.{ x, w, y }, params);
        e.dispatchGroups(mtl.Size.of((N + 63) / 64, (M + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }

    /// c [M, 256] fp32 = x3 [M, 3072] w3^T over fp16: a split LayerNorm's rows and a RePo layer's split maps.
    pub fn split(k: *const Kernels, e: mtl.ComputeEncoder, x3: Ref, w3: Ref, c: Ref, M: usize) void {
        const N = 2 * cfg.repo_hidden;
        const K = 3 * cfg.hidden;
        const tn: usize = N / 64;
        const tm = (M + 63) / 64;
        const swz: u6 = if (tm <= 3) 0 else 1;
        e.setPipeline(if (M % 64 == 0) k.get("id_split_1") else k.get("id_split_0"));
        e.setBuffer(x3.buf, x3.off, 0);
        e.setBuffer(w3.buf, w3.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, N, tn, tm, 1, M * N, K, swz, 0, 0 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(c.buf, c.off, 3);
        const tiles = (tn << swz) * ((tm + (@as(usize, 1) << swz) - 1) >> swz);
        e.dispatchThreads(mtl.Size.of(tiles * 32, 2, 2), mtl.Size.of(32, 2, 2));
    }

    /// x3 rows [hi | hi | lo] of the affine LayerNorm32 of x's rows, for `split`.
    pub fn normSplit(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, x3: Ref, rows: usize, eps: f32) void {
        const params = extern struct { width: u32, eps: f32, mode: u32 }{ .width = cfg.hidden, .eps = eps, .mode = 1 };
        bind(e, k.get("id_norm_split"), &.{ x, w, b, x3 }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [M, N] = fp16(x w^T + b) in fp16 with fp32 sums, then GELU (rounded again) when `gelu`.
    pub fn linear(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, M: usize, N: usize, K: usize, gelu: bool) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        const form = @as(usize, @intFromBool(M % 64 == 0)) * 2 + @intFromBool(N % 128 == 0);
        e.setPipeline(k.p[linear_first + @as(usize, @intFromBool(gelu)) * 4 + form]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, N, tn, tm, 0, K / 256 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(y.buf, y.off, 3);
        e.setBuffer(b.buf, b.off, 4);
        e.dispatchGroups(mtl.Size.of(tn, tm, 1), mtl.Size.of(256, 1, 1));
    }

    /// o rows [Lq, heads * 64] = softmax(q k^T / 8) v for q [heads, Lq, 64] and k, v [heads, Lk, 64].
    pub fn attention(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, o: Ref, lq: usize, lk: usize) void {
        const form = @as(usize, @intFromBool(lq % 64 == 0)) * 2 + @intFromBool(lk % 32 == 0);
        const nq = (lq + 63) / 64;
        const nk = (lk + 31) / 32;
        e.setPipeline(k.p[attention_first + form]);
        for ([_]Ref{ q, kb, vb }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // heads, q rows, k rows, k blocks, last q block, last k block, rows in the last q block and in the last k block
        var params: [8]i32 = .{ cfg.heads, @intCast(lq), @intCast(lk), @intCast(nk), @intCast(nq - 1), @intCast(nk - 1), @intCast(lq - (nq - 1) * 64), @intCast(lk - (nk - 1) * 32) };
        e.setBytes(std.mem.asBytes(&params), 3);
        var f: [4]f32 = .{ 1.0 / @sqrt(@as(f32, cfg.head_dim)), 0, 0, 0 };
        e.setBytes(std.mem.asBytes(&f), 4);
        e.setBuffer(o.buf, o.off, 5);
        e.dispatchGroups(mtl.Size.of(nq, cfg.heads, 1), mtl.Size.of(32, 4, 1));
    }

    /// y = LayerNorm32 of x's rows in fp32 rounded to fp16, then affine (w, b fp32) or modulated (s1, sh fp16).
    pub fn norm(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, s1: Ref, sh: Ref, y: Ref, rows: usize, eps: f32, mode: Norm) void {
        const params = extern struct { width: u32, eps: f32, mode: u32 }{ .width = cfg.hidden, .eps = eps, .mode = @backingInt(mode) };
        bind(e, k.get("id_norm"), &.{ x, w, b, s1, sh, y }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// y fp32 = LayerNorm of x's rows times w plus b (affine), or times (1 + w) plus b (modulated).
    pub fn norm32(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: usize, eps: f32, mode: Norm) void {
        const params = extern struct { width: u32, eps: f32, mode: u32 }{ .width = cfg.hidden, .eps = eps, .mode = @backingInt(mode) };
        bind(e, k.get("id_norm32"), &.{ x, w, b, y }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// cos, sin [rows, 16, 32] of RePo from its gate and content sums (fp32 rows of 256) and final map.
    pub fn repo(k: *const Kernels, e: mtl.ComputeEncoder, gc: Ref, inv: f32, final: Ref, freqs: Ref, cos: Ref, sin: Ref, rows: usize) void {
        const params = extern struct { rows: u32, stride: u32, inv: f32 }{ .rows = @intCast(rows), .stride = 2 * cfg.repo_hidden, .inv = inv };
        bind(e, k.get("id_repo"), &.{ gc, final, freqs, cos, sin }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(64, 1, 1));
    }

    /// q, k, v heads [16, rows, 64] from packed qkv rows: rotated by cos, sin, then normed by gq, gk.
    pub fn heads(k: *const Kernels, e: mtl.ComputeEncoder, qkv: Ref, gq: Ref, gk: Ref, cos: Ref, sin: Ref, q: Ref, kk: Ref, v: Ref, rows: usize) void {
        const params = extern struct { rows: u32, scale: f32 }{ .rows = @intCast(rows), .scale = @sqrt(@as(f32, cfg.head_dim)) };
        bind(e, k.get("id_qk"), &.{ qkv, gq, gk, cos, sin, q, kk, v }, params);
        e.dispatchGroups(mtl.Size.of(rows, 2, 1), mtl.Size.of(256, 1, 1));
    }

    /// x [rows, 1024] += y gated by the fp16 vector g, in place.
    pub fn gateAdd(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, y: Ref, g: Ref, rows: usize) void {
        bind(e, k.get("id_gate_add"), &.{ x, y, g }, @as(u32, cfg.hidden));
        e.dispatchThreads(mtl.Size.of(cfg.hidden, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// x [8192, 1024] = fp16(fp16(z w^T + b) + pe).
    pub fn embed(k: *const Kernels, e: mtl.ComputeEncoder, z: Ref, w: Ref, b: Ref, pe: Ref, x: Ref) void {
        e.setPipeline(k.get("id_embed"));
        for ([_]Ref{ z, w, b, pe, x }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchThreads(mtl.Size.of(cfg.hidden, cfg.latent_rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [rows, 1024] = fp16((a + ba) + (c + bc)).
    pub fn cond(k: *const Kernels, e: mtl.ComputeEncoder, a: Ref, ba: Ref, c: Ref, bc: Ref, y: Ref, rows: usize) void {
        bind(e, k.get("id_cond"), &.{ a, ba, c, bc, y }, @as(u32, cfg.hidden));
        e.dispatchThreads(mtl.Size.of(cfg.hidden, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [rows, width] += b by column, fp32.
    pub fn bias32(k: *const Kernels, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: usize, width: usize) void {
        bind(e, k.get("id_bias32"), &.{ y, b }, @as(u32, @intCast(width)));
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(@min(width, 256), 1, 1));
    }
};

/// One kernel's buffers in order, then its parameters at the next index.
pub fn bind(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref, params: anytype) void {
    e.setPipeline(p);
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(params, refs.len);
}

test "every Idol pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var k = try Kernels.load(std.testing.allocator, device);
    k.deinit();
}
