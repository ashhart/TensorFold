//! Unlimited-OCR-2's Metal pipelines: core_qwen_vision's bf16 and fp32-sum projections, then its own steps.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
pub const Ref = @import("weights.zig").Ref;

/// Each pipeline by its function: the four bf16 projections, the two fp32-sum ones, then this family's.
pub const names = [_][]const u8{
    "qwv_gemm_bf16_nt_00",
    "qwv_gemm_bf16_nt_01",
    "qwv_gemm_bf16_nt_10",
    "qwv_gemm_bf16_nt_11",
    "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_n_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float",
    "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_t_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float",
    "uv_im2col",
    "uv_add",
    "uv_bias",
    "uv_layer_norm",
    "uv_layer_norm2d",
    "uv_gelu",
    "uv_quick_gelu",
    "uv_window",
    "uv_unwindow_add",
    "uv_rel_pos",
    "uv_attention",
    "uv_concat",
    "uv_clip_embed",
};
const tower_count = 6;
const sums = 4;

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
        k.own = try mtl.Library.fromSource(device, sources.uocr, mtl.CompileOptions.mlx());
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

    /// y [M, N] = bf16(x [M, K] w^T) with fp32 sums (tfp::gemm_nax), w as stored [N, K]; K a multiple of 256.
    pub fn linear(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: usize, N: usize, K: usize) void {
        std.debug.assert(K % 256 == 0);
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

    /// y [M, N] = bf16(x w^T + b): one split-K part's fp32 sums into `sum`, then the bias, one rounding (N % 64 = 0).
    pub fn linearBias(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, sum: Ref, y: Ref, M: usize, N: usize, K: usize) void {
        std.debug.assert(N % 64 == 0 and K % 256 == 0);
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
        bind(e, k.get("uv_bias"), &.{ sum, b, y }, @as(u32, @intCast(N)));
        e.dispatchThreads(mtl.Size.of(N, M, 1), mtl.Size.of(256, 1, 1));
    }

    /// One kernel's buffers in order, then its parameters at the next index.
    pub fn bind(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref, params: anytype) void {
        e.setPipeline(p);
        for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(params, refs.len);
    }
};

test "every Unlimited-OCR-2 pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var k = try Kernels.load(std.testing.allocator, device);
    k.deinit();
}
