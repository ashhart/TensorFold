//! The core Metal kernels Qwen3-ASR shares: shared_attention (tower and LM) and the Qwen-VL tower's GEMM and GELU.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags");
const attn = @import("shared_attention");
const cfg = @import("../config.zig");

pub const Attention = attn.Attention;

/// The rows one shared_attention encode takes.
pub const encode_rows = 16;

pub const Shared = struct {
    lib: mtl.Library,
    gemm: [2][2]mtl.Pipeline, // qwv_gemm_bf16_nt_{M % 64 == 0}{N % 128 == 0}: bf16 x w^T, fp32 sums
    gelu: mtl.Pipeline, // qwv_bias_gelu: y = gelu(y + b) in fp32, one rounding
    tower: Attention, // bidirectional over a window, all keys unmasked
    lm: Attention, // causal GQA over the caches, by mask; decode by `ordinary`

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, a: cfg.Audio, t: cfg.Text, span: usize, window: usize) !Shared {
        const src = try frags.source(device, gpa, sources.core_qwen_vision);
        defer gpa.free(src);
        var s: Shared = undefined;
        s.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer s.lib.deinit();
        var made: usize = 0;
        errdefer for (0..made) |i| s.gemm[i / 2][i % 2].deinit();
        for (0..2) |m| for (0..2) |n| {
            var name: [32]u8 = undefined;
            s.gemm[m][n] = try mtl.Pipeline.init(device, s.lib, try std.fmt.bufPrint(&name, "qwv_gemm_bf16_nt_{d}{d}", .{ m, n }), false);
            made += 1;
        };
        s.gelu = try mtl.Pipeline.init(device, s.lib, "qwv_bias_gelu", false);
        errdefer s.gelu.deinit();
        s.tower = try Attention.init(gpa, device, .{ .query_heads = a.heads, .kv_heads = a.heads, .head_dim = a.headDim(), .rotary_dim = 0, .capacity = window, .output = .bf16 });
        errdefer s.tower.deinit();
        s.lm = try Attention.init(gpa, device, .{ .query_heads = t.heads, .kv_heads = t.kv_heads, .head_dim = t.head_dim, .rotary_dim = 0, .capacity = span, .output = .bf16 });
        return s;
    }

    pub fn deinit(s: *Shared) void {
        s.lm.deinit();
        s.tower.deinit();
        s.gelu.deinit();
        for (&s.gemm) |*r| for (r) |p| p.deinit();
        s.lib.deinit();
    }

    /// y [M, N] = bf16(x [M, K] w^T) with fp32 sums, w as stored [N, K] (tfp::gemm_nax, as the Qwen-VL tower runs it).
    pub fn linear(s: *const Shared, e: mtl.ComputeEncoder, x: attn.Ref, w: attn.Ref, y: attn.Ref, m: usize, n: usize, k: usize) void {
        const tn = (n + 127) / 128;
        const tm = (m + 63) / 64;
        e.setPipeline(s.gemm[@intFromBool(m % 64 == 0)][@intFromBool(n % 128 == 0)]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ m, n, k, k, k, n, tn, tm, 2, k / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(y.buf, y.off, 3);
        e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }

    /// y = bf16(gelu(y + b)), the exact (erf) GELU, in place over `rows` x `width`.
    pub fn biasGelu(s: *const Shared, e: mtl.ComputeEncoder, y: attn.Ref, b: attn.Ref, rows: usize, width: usize) void {
        e.setPipeline(s.gelu);
        e.setBuffer(y.buf, y.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        e.setBytes(std.mem.asBytes(&[2]u32{ @intCast(width), 1 }), 2);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }
};
