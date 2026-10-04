//! The prefill kernels' launches as tools/zig/prefill_launch.py and prefill_glue.py build them: variant, params, grid.
const std = @import("std");
const mtl = @import("metal");
const pk = @import("prefill_kernels.zig");
const Enc = @import("encoder.zig").Enc;

const Buffer = mtl.Buffer;

/// A buffer and a byte offset into it.
pub const At = struct {
    b: Buffer,
    off: usize = 0,

    pub fn of(b: Buffer) At {
        return .{ .b = b };
    }

    pub fn plus(a: At, bytes: usize) At {
        return .{ .b = a.b, .off = a.off + bytes };
    }
};

fn tag(flag: bool) []const u8 {
    return if (flag) "t" else "n";
}

pub const Launch = struct {
    k: *const pk.Kernels,
    e: *Enc,

    /// Bind `inputs`, the int32 P array (`n` entries, zero-padded), `extra` bytes, then `outputs`, and dispatch.
    pub fn go(l: Launch, name: []const u8, inputs: []const At, p: []const usize, n: usize, extra: ?[]const u8, outputs: []const At, grid: [3]usize, group: [3]usize) void {
        var params: [32]i32 = @splat(0);
        for (p, 0..) |v, i| params[i] = @intCast(v);
        l.e.pipe(l.k.get(name));
        var slot: usize = 0;
        for (inputs) |a| {
            l.e.buf(a.b, a.off, slot);
            slot += 1;
        }
        l.e.e.setBytes(std.mem.sliceAsBytes(params[0..n]), slot);
        slot += 1;
        if (extra) |bytes| {
            l.e.e.setBytes(bytes, slot);
            slot += 1;
        }
        for (outputs) |a| {
            l.e.buf(a.b, a.off, slot);
            slot += 1;
        }
        l.e.run(grid, group);
    }

    /// A glue or sort kernel (prefill/glue.metal, sort.metal) by its entry point after `custom_kernel_tf_`.
    pub fn glue(l: Launch, comptime name: []const u8, inputs: []const At, p: []const usize, outputs: []const At, grid: [3]usize, group: [3]usize) void {
        l.go("custom_kernel_tf_" ++ name, inputs, p, 16, null, outputs, grid, group);
    }

    /// steel_gemm_fused_nax (bm64 bn128 bk256 wm2 wn4, swizzle 2): d [batch, M, N] = a x b, operands at element offsets.
    pub fn gemm(l: Launch, fp32: bool, a: At, b: At, d: At, M: usize, N: usize, K: usize, lda: usize, ldb: usize, ta: bool, tb: bool, batch: usize, strides: [2]usize, offsets: [2]usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        const t = if (fp32) "f32" else "bf16";
        const T = if (fp32) "float" else "bfloat16_t";
        var name: [160]u8 = undefined;
        const full = std.fmt.bufPrint(&name, "custom_kernel_tf_gemm_nax_{s}_{s}_{s}_{s}_{s}_{s}_64_128_256_2_4_{s}_{s}_int32_t_{s}", .{ t, tag(ta), tag(tb), tag(M % 64 == 0), tag(N % 128 == 0), tag(K % 256 == 0), T, T, T }) catch unreachable;
        const p = [_]usize{ M, N, K, lda, ldb, N, tn, tm, 2, K / 256, strides[0], strides[1], M * N, offsets[0], offsets[1] };
        l.go(full, &.{ a, b }, &p, 16, null, &.{d}, .{ (tn << 2) * 32, (tm + 3) / 4 * 4, batch * 2 }, .{ 32, 4, 2 });
    }

    /// x [M, K] @ w [N, K]^T in bf16 as MLX routes it on an M5: NAX split-K and its fp32 sum while K is long.
    pub fn matmulNt(l: Launch, x: At, w: At, parts_buf: At, d: At, M: usize, N: usize, K: usize) void {
        const big = @max(M, N);
        if (!(K >= 3 * big or (big <= 1024 and K > 2 * big))) return l.gemm(false, x, w, d, M, N, K, K, K, false, true, 1, .{ 0, 0 }, .{ 0, 0 });
        const size: usize = if (K <= 1024) K / 2 else if (K <= 2048) 1024 else if (K <= 4096) 2048 else 4096;
        const parts = (K + size - 1) / size;
        const tn = (N + 63) / 64;
        const tm = (M + 63) / 64;
        const swz: u6 = if (tm <= 3) 0 else 1;
        var name: [160]u8 = undefined;
        const full = std.fmt.bufPrint(&name, "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_{s}_{s}_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float", .{ tag(M % 64 == 0), tag(N % 64 == 0) }) catch unreachable;
        const p = [_]usize{ M, N, K, K, K, N, tn, tm, parts, M * N, size, swz, 0, 0 };
        const tiles = (tn << swz) * ((tm + (@as(usize, 1) << swz) - 1) >> swz);
        l.go(full, &.{ x, w }, &p, 16, null, &.{parts_buf}, .{ tiles * parts * 32, 2, 2 }, .{ 32, 2, 2 });
        l.go("custom_kernel_tf_gemm_splitk_sum_bf16_float_int32_t_bfloat16_t", &.{parts_buf}, &.{ parts, M * N, N }, 16, null, &.{d}, .{ N, M, 1 }, .{ 32, 8, 1 });
    }

    /// Causal NAX attention (bq64 bk32 d128) of L queries on kL keys; strides: (batch, head, row) of Q, K, V, O.
    pub fn attention(l: Launch, q: At, k: At, v: At, o: At, L: usize, kL: usize, heads: usize, kv_heads: usize, strides: [12]usize, scale: f32) void {
        const nq = (L + 63) / 64;
        const nk = (kL + 31) / 32;
        var name: [160]u8 = undefined;
        const full = std.fmt.bufPrint(&name, "custom_kernel_tf_attention_nax_bf16_{s}_{s}_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t", .{ tag(L % 64 == 0), tag(kL % 32 == 0) }) catch unreachable;
        const head = [_]usize{ 1, heads, L, kL, heads / kv_heads, nq, nk, L / 64, kL / 32, L % 64, kL % 32, kL - L };
        const p = head ++ strides;
        var f: [16]f32 = @splat(0);
        f[0] = scale;
        l.go(full, &.{ q, k, v }, &p, 32, std.mem.sliceAsBytes(&f), &.{o}, .{ nq * 32, heads * 4, 1 }, .{ 32, 4, 1 });
    }

    /// Inclusive fp32 cumsum along the middle axis of a contiguous [outer, axis, stride] array from `offset`.
    pub fn scan(l: Launch, x: At, y: At, outer: usize, axis: usize, stride: usize, offset: usize) void {
        const blocks = (stride + 31) / 32;
        l.go("custom_kernel_tf_scan_sum_strided_f32_float_int32_t_float", &.{x}, &.{ axis, stride, blocks, offset }, 16, null, &.{y}, .{ 256, outer * blocks, 1 }, .{ 256, 1, 1 });
    }

    /// Depthwise conv over time of [L + taps - 1, C] padded rows with weights [C, taps, 1].
    pub fn conv(l: Launch, x: At, w: At, y: At, L: usize, C: usize, taps: usize) void {
        l.go("custom_kernel_tf_conv1d_depthwise_bf16_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ x, w }, &.{ (L + taps - 1) * C, C, 1, taps }, 16, null, &.{y}, .{ C, L, 1 }, .{ 32, 32, 1 });
    }

    /// MLX's 4-bit x [M, K] @ w^T past lane_qmm's rows: affine_qmm_t_nax, or split-K and its bf16 sum.
    pub fn qmm(l: Launch, x: At, w: [3]At, y: At, parts_buf: At, M: usize, N: usize, K: usize) void {
        const parts = splitkParts(M, N, K);
        if (parts == 1) {
            return l.go("custom_kernel_tf_qmm_t_nax_bf16_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ w[0], w[1], w[2], x }, &.{ K, N, M }, 16, null, &.{y}, .{ N / 64 * 32, (M + 63) / 64 * 2, 2 }, .{ 32, 2, 2 });
        }
        l.go("custom_kernel_tf_qmm_splitk_part_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ w[0], w[1], w[2], x }, &.{ K, N, M, K / parts, M * N }, 16, null, &.{parts_buf}, .{ N / 32 * 32, (M + 31) / 32 * 4, parts }, .{ 32, 4, 1 });
        l.go("custom_kernel_tf_qmm_splitk_sum_bfloat16_t_int32_t_bfloat16_t", &.{parts_buf}, &.{ parts, M * N, M * N }, 16, null, &.{y}, .{ M * N, 1, 1 }, .{ 256, 1, 1 });
    }

    /// Rows sorted by expert times their expert's 4-bit W^T (affine_gather_qmm_rhs_nax) from the experts' first rows.
    pub fn gatherQmm(l: Launch, xs: At, w: [3]At, offsets: At, y: At, n: usize, N: usize, K: usize, E: usize) void {
        const bm: usize = if (n / E < 64) 32 else 64;
        var name: [160]u8 = undefined;
        const full = std.fmt.bufPrint(&name, "custom_kernel_tf_gather_qmm_rhs_nax_bf16_{d}_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_t_int32_t_bfloat16_t", .{bm}) catch unreachable;
        l.go(full, &.{ xs, w[0], w[1], w[2], offsets }, &.{ n, N, K, E }, 16, null, &.{y}, .{ (N + 63) / 64 * 32, @min(n, (n + bm - 1) / bm + E - 1) * 2, 2 }, .{ 32, 2, 2 });
    }

    /// y_prev = state @ C per (row, group, head of the group): fp32 [heads, head, dstate] states, C [s, groups, dstate].
    pub fn gemv(l: Launch, state: At, c: At, y: At, s: usize, groups: usize, heads: usize, head: usize, dstate: usize) void {
        const rep = heads / groups;
        const p = [_]usize{ dstate, head, dstate, 3, s, groups, rep, 1, groups * dstate, dstate, 0, 0, 0, rep * head * dstate, head * dstate, 0 };
        l.go("custom_kernel_tf_gemv_rows_f32_4_4_4_float_float_int32_t_float", &.{ state, c }, &p, 16, null, &.{y}, .{ head / 16 * 32, 1, s * groups * rep * 4 }, .{ 32, 1, 4 });
    }
};

/// MLX's qmm_splitk partition count: ~512 threadgroups, whole groups of 64, dividing K (1: not split).
pub fn splitkParts(M: usize, N: usize, K: usize) usize {
    const tiles = ((N + 31) / 32) * ((M + 31) / 32);
    var split = @min(@max(1, 512 / tiles), K / 64);
    while (split > 1 and K % (split * 64) != 0) split -= 1;
    return split;
}

test "split-K partitions follow MLX's rule" {
    try std.testing.expectEqual(@as(usize, 7), splitkParts(129, 256, 2688));
    try std.testing.expectEqual(@as(usize, 2), splitkParts(1024, 256, 2688));
    try std.testing.expectEqual(@as(usize, 1), splitkParts(2048, 256, 2688));
    try std.testing.expectEqual(@as(usize, 1), splitkParts(129, 10304, 2688));
}
