//! Prompt matmuls with int8 activations on MLX affine 4-bit tiles (w4a8_prefill.cu): the quantizer, then the int8 mma.
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch_ = @import("launch.zig");
const Affine4 = @import("qlinear.zig").Affine4;

/// tf_w4a8_prefill's four stages: 128 rows of 64 inputs, their scales and sums, 128 columns' words, scales, biases.
pub const smem = 4 * (128 * 64 + 128 * 8 + 128 * 32 + 128 * 4);

pub const Kernels = struct {
    quantize: Function,
    prefill: Function,

    pub fn resolve(m: Module) !Kernels {
        const k: Kernels = .{ .quantize = try m.function("tf_w4a8_quantize"), .prefill = try m.function("tf_w4a8_prefill") };
        try k.prefill.allowDynamicShared(smem);
        return k;
    }

    /// Scratch bytes for `rows` rows of up to `k` inputs: their int8 codes, then their group scales and sums.
    pub fn scratch(rows: usize, k: usize) usize {
        return rows * k + 2 * rows * (k / 64) * 4;
    }

    /// Quantize and read its verdict before any consumer, with the same seven-argument CUDA ABI for every caller.
    pub fn quantizeRows(kk: Kernels, s: Stream, x: u64, codes: u64, scales: u64, sums: u64, rows: usize, k: usize, ldx: usize, flag: u64, exceptional: *u32) !void {
        const width: c_int = @intCast(k);
        const stride: c_int = @intCast(ldx);
        // the mark starts clear, and the quantizer raises it for an exceptional group
        try s.d.check(s.d.api.cuMemsetD8Async(flag, 0, 4, s.handle), "cuMemsetD8Async");
        var a: launch_.Args = .{};
        inline for (.{ x, codes, scales, sums, width, stride, flag }) |v| a.add(v);
        try launch_.launch(kk.quantize, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((k / 64 + 3) / 4) }, .block = .{ .x = 128 } }, s, &a);
        // the quantizer's verdict decides the chunk's precision
        try s.d.check(s.d.api.cuMemcpyDtoHAsync_v2(exceptional, flag, 4, s.handle), "cuMemcpyDtoHAsync");
        try s.synchronize();
    }

    /// out (rows, q.n) bf16 = x (rows, q.k) bf16 through q: x quantized a 64-input group into `ws`, then the int8 mma.
    pub fn prompt(kk: Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize, ws: u64, flag: u64, exceptional: *u32) !void {
        const groups = q.k / 64;
        const xa = ws + rows * q.k;
        const xs = xa + rows * groups * 4;
        try kk.quantizeRows(s, x, ws, xa, xs, rows, q.k, q.k, flag, exceptional);
        if (exceptional.* != 0) return;
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2); // the bf16 prompt GEMM's row bands
        const group = @max(1, @min(rows_t, band));
        var b: launch_.Args = .{};
        inline for (.{ ws, xa, xs, q.w, q.s, q.b, out }) |v| b.add(v);
        inline for (.{ rows, q.n, q.k, q.npad, group }) |v| b.add(@as(c_int, @intCast(v)));
        try launch_.launch(kk.prefill, .{ .grid = .{ .x = @intCast(rows_t * ((q.n + 127) / 128)) }, .block = .{ .x = 256 }, .shared = smem }, s, &b);
    }
};
