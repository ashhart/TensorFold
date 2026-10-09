//! The training kernels as calls: products with 4-bit weights the other way round, norms, the low-rank change, Adam.
const std = @import("std");
const mtl = @import("metal");
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const wts = @import("weights.zig");
const pl = @import("prefill_launch.zig");
const pk = @import("prefill_kernels.zig");
const layers = @import("layers.zig");

const Buffer = mtl.Buffer;
const At = pl.At;
const Enc = fwd.Enc;

/// A Mamba layer's shape as the mixer kernels read it.
pub const Shape = extern struct { rows: u32, heads: u32, dh: u32, groups: u32, n: u32, inner: u32, conv: u32, proj: u32 };

/// Attention's shape as its kernels read it.
pub const Heads = extern struct { rows: u32, heads: u32, kv_heads: u32, dim: u32, scale: f32 };

/// Scratch the products borrow: x in bf16, a weight dequantized transposed, an f32 sum, the experts' partial sums.
pub const Tmp = struct { xb: Buffer, wt: Buffer, acc: Buffer, part: Buffer };

/// Slices an expert's rows are streamed in, each its own threadgroup.
pub const slices = 8;

/// d [m, n] (row stride ldd) = a [m, k] times b [n, k]^T, bf16 on the tensor units (MLX's NAX GEMM, any k).
pub fn mm(l: pl.Launch, a: At, b: At, d: At, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldd: usize) void {
    const whole = m % 64 == 0 and n % 128 == 0;
    const name = if (whole) "custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t" else "custom_kernel_tf_gemm_nax_bf16_n_t_n_n_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
    const tn = (n + 127) / 128;
    const tm = (m + 63) / 64;
    const p = [_]usize{ m, n, k, lda, ldb, ldd, tn, tm, 2, k / 256, 0, 0, m * n, 0, 0 };
    l.go(name, &.{ a, b }, &p, 16, null, &.{d}, .{ (tn << 2) * 32, (tm + 3) / 4 * 4, 2 }, .{ 32, 4, 2 });
}

pub const Ops = struct {
    k: *const kern.Kernels,
    pk: *const pk.Kernels,
    e: *Enc,
    tmp: *const Tmp,

    fn go(o: Ops, comptime name: []const u8) void {
        o.e.pipe(o.k.get(name));
    }

    /// y [rows, k] (+)= x [rows, n] times the 4-bit [n, k] matrix w: w dequantized transposed, then the NAX product.
    pub fn product(o: Ops, x: Buffer, w: [3]At, y: Buffer, rows: usize, n: usize, k: usize, add: bool) void {
        o.go("tf_train_narrow");
        o.e.buf(x, 0, 0);
        o.e.buf(o.tmp.xb, 0, 1);
        o.e.run(.{ rows * n, 1, 1 }, .{ 256, 1, 1 });
        o.go("tf_train_dequant_t");
        for (w, 0..) |t, i| o.e.buf(t.b, t.off, i);
        o.e.buf(o.tmp.wt, 0, 3);
        o.e.run(.{ n, k, 1 }, .{ 256, 1, 1 });
        const out = if (add) o.tmp.acc else y;
        const tn = (k + 63) / 64;
        const tm = (rows + 63) / 64;
        const swz: u6 = if (tm <= 3) 0 else 1;
        const name = if (rows % 64 == 0) "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_t_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float" else "custom_kernel_tf_gemm_splitk_nax_bf16_n_t_n_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float";
        const p = [_]usize{ rows, k, n, n, n, k, tn, tm, 1, rows * k, n, swz, 0, 0 };
        const tiles = (tn << swz) * ((tm + (@as(usize, 1) << swz) - 1) >> swz);
        const l: pl.Launch = .{ .k = o.pk, .e = o.e };
        l.go(name, &.{ At.of(o.tmp.xb), At.of(o.tmp.wt) }, &p, 16, null, &.{At.of(out)}, .{ tiles * 32, 2, 2 }, .{ 32, 2, 2 });
        if (!add) return;
        o.go("tf_train_add");
        o.e.buf(o.tmp.acc, 0, 0);
        o.e.buf(y, 0, 1);
        o.e.run(.{ rows * k, 1, 1 }, .{ 256, 1, 1 });
    }

    /// y [pairs, k] = x [pairs, n] times each pair's expert of the 4-bit w, an expert's rows streamed once, by slice.
    pub fn experts(o: Ops, x: Buffer, w: [3]At, offsets: Buffer, y: Buffer, pairs: usize, n: usize, k: usize, count: usize) void {
        std.debug.assert(n % (slices * 4) == 0 and k % 64 == 0);
        const dims = [4]u32{ @intCast(pairs), @intCast(n), @intCast(k), @intCast(count) };
        const width = (k / 8 + 31) / 32 * 32;
        o.go("tf_train_experts_part");
        o.e.buf(x, 0, 0);
        for (w, 1..) |t, i| o.e.buf(t.b, t.off, i);
        o.e.buf(offsets, 0, 4);
        o.e.buf(o.tmp.part, 0, 5);
        o.e.bytes(dims, 6);
        o.e.bytes(@as(u32, slices), 7);
        o.e.run(.{ width, slices * count, 1 }, .{ width, 1, 1 });
        o.go("tf_train_experts_sum");
        o.e.buf(o.tmp.part, 0, 0);
        o.e.buf(y, 0, 1);
        o.e.bytes(dims, 2);
        o.e.bytes(@as(u32, slices), 3);
        o.e.run(.{ k, pairs, 1 }, .{ 256, 1, 1 });
    }

    /// g [rows, dim] += the residual's gradient through an input RMS norm of h (bf16) with weight w, given dx.
    pub fn rmsBack(o: Ops, h: At, w: At, dx: Buffer, g: Buffer, rows: usize, dim: usize, eps: f32) void {
        o.go("tf_train_rms_back");
        o.e.buf(h.b, h.off, 0);
        o.e.buf(w.b, w.off, 1);
        o.e.buf(dx, 0, 2);
        o.e.buf(g, 0, 3);
        o.e.bytes(@as(u32, @intCast(dim)), 4);
        o.e.bytes(eps, 5);
        o.e.run(.{ 256 * rows, 1, 1 }, .{ 256, 1, 1 });
    }

    /// The open block's gradient (ranks from first) from its site's bf16 input x and gradient gy; dx gains every block.
    pub fn adapterBack(o: Ops, ad: wts.Adapter, gb: Buffer, first: usize, dxa: Buffer, x: Buffer, gy: Buffer, dx: Buffer, rows: usize) void {
        const r: u32 = @intCast(ad.rank);
        const out_dims = [4]u32{ @intCast(rows), @intCast(ad.out), r, @intCast(first) };
        layers.project(o.k, o.e, ad, x, rows);
        o.go("tf_train_lora_db");
        o.e.buf(ad.xa, 0, 0);
        o.e.buf(gy, 0, 1);
        o.e.buf(gb, 0, 2);
        o.e.bytes(out_dims, 3);
        o.e.bytes(ad.scale, 4);
        o.e.run(.{ ad.out, ad.rank - first, 1 }, .{ 256, 1, 1 });
        o.go("tf_train_lora_dxa");
        o.e.buf(gy, 0, 0);
        o.e.buf(ad.b, 0, 1);
        o.e.buf(dxa, 0, 2);
        o.e.bytes(out_dims, 3);
        o.e.bytes(ad.scale, 4);
        o.e.buf(ad.gates, 0, 5);
        o.e.run(.{ 32 * ad.rank, rows, 1 }, .{ 256, 1, 1 });
        o.go("tf_train_lora_dx");
        o.e.buf(dxa, 0, 0);
        o.e.buf(ad.a, 0, 1);
        o.e.buf(dx, 0, 2);
        o.e.bytes([4]u32{ @intCast(rows), @intCast(ad.in), r, 0 }, 3);
        o.e.run(.{ ad.in, rows, 1 }, .{ 256, 1, 1 });
    }

    /// Each row of a site's bf16 input x, made unit, along the k candidate directions f [k, in], into p [rows, k].
    pub fn project(o: Ops, x: Buffer, f: Buffer, p: At, rows: usize, in: usize, k: usize) void {
        o.go("tf_train_project");
        o.e.buf(x, 0, 0);
        o.e.buf(f, 0, 1);
        o.e.buf(p.b, p.off, 2);
        o.e.bytes([4]u32{ @intCast(rows), @intCast(in), @intCast(k), 0 }, 3);
        o.e.run(.{ 32 * k, rows, 1 }, .{ 256, 1, 1 });
    }

    /// rows rows of a site's bf16 input x added to a random sketch y [k, in] (first: the rows' place in the sketch).
    pub fn sketch(o: Ops, x: Buffer, y: Buffer, rows: usize, in: usize, k: usize, first: usize, seed: u32) void {
        o.go("tf_train_sketch");
        o.e.buf(x, 0, 0);
        o.e.buf(y, 0, 1);
        o.e.bytes([4]u32{ @intCast(rows), @intCast(in), @intCast(k), @intCast(first) }, 2);
        o.e.bytes(seed, 3);
        o.e.run(.{ in, k, 1 }, .{ 256, 1, 1 });
    }

    /// du = da 2 relu(u) over n values (u bf16).
    pub fn relu2Back(o: Ops, u: At, da: Buffer, du: Buffer, n: usize) void {
        o.go("tf_train_relu2_back");
        o.e.buf(u.b, u.off, 0);
        o.e.buf(da, 0, 1);
        o.e.buf(du, 0, 2);
        o.e.run(.{ n, 1, 1 }, .{ 256, 1, 1 });
    }

    /// The routed experts' output gradients in sorted pair order: wt[p] g[p / top_k] at each slot of order.
    pub fn pairsIn(o: Ops, order: Buffer, wt: Buffer, g: Buffer, dy: Buffer, pairs: usize, dim: usize, top_k: usize) void {
        o.go("tf_train_pairs_in");
        o.e.buf(order, 0, 0);
        o.e.buf(wt, 0, 1);
        o.e.buf(g, 0, 2);
        o.e.buf(dy, 0, 3);
        o.e.bytes([2]u32{ @intCast(dim), @intCast(top_k) }, 4);
        o.e.run(.{ dim, pairs, 1 }, .{ 256, 1, 1 });
    }

    /// dx [rows, dim] += each row's top_k pairs' input gradients (sorted slots through inv), in pair order.
    pub fn pairsOut(o: Ops, inv: Buffer, dxp: Buffer, dx: Buffer, rows: usize, dim: usize, top_k: usize) void {
        o.go("tf_train_pairs_out");
        o.e.buf(inv, 0, 0);
        o.e.buf(dxp, 0, 1);
        o.e.buf(dx, 0, 2);
        o.e.bytes([2]u32{ @intCast(dim), @intCast(top_k) }, 3);
        o.e.run(.{ dim, rows, 1 }, .{ 256, 1, 1 });
    }

    /// One Adam step on n parameters; their gradient is cleared after.
    pub fn adam(o: Ops, p: At, g: Buffer, m: Buffer, v: Buffer, n: usize, hp: [4]f32, corr: [2]f32) void {
        o.go("tf_train_adam");
        o.e.buf(p.b, p.off, 0);
        o.e.buf(g, 0, 1);
        o.e.buf(m, 0, 2);
        o.e.buf(v, 0, 3);
        o.e.bytes(hp, 4);
        o.e.bytes(corr, 5);
        o.e.run(.{ n, 1, 1 }, .{ 256, 1, 1 });
    }

    pub fn zero(o: Ops, x: Buffer, n: usize) void {
        o.go("tf_train_zero");
        o.e.buf(x, 0, 0);
        o.e.run(.{ n, 1, 1 }, .{ 256, 1, 1 });
    }

    pub fn widen(o: Ops, x: At, out: Buffer, n: usize) void {
        o.go("tf_train_widen");
        o.e.buf(x.b, x.off, 0);
        o.e.buf(out, 0, 1);
        o.e.run(.{ n, 1, 1 }, .{ 256, 1, 1 });
    }

    /// A mixer kernel with a second constant after the first (an epsilon or a tap count).
    pub fn mixer2(o: Ops, comptime name: []const u8, bufs: []const At, constant: anytype, extra: anytype, grid: [3]usize, group: [3]usize) void {
        o.go(name);
        for (bufs, 0..) |t, i| o.e.buf(t.b, t.off, i);
        o.e.bytes(constant, bufs.len);
        o.e.bytes(extra, bufs.len + 1);
        o.e.run(grid, group);
    }

    /// A mixer kernel over a grid, its buffers bound in order and its constant after them.
    pub fn mixer(o: Ops, comptime name: []const u8, bufs: []const At, constant: anytype, grid: [3]usize, group: [3]usize) void {
        o.go(name);
        for (bufs, 0..) |t, i| o.e.buf(t.b, t.off, i);
        o.e.bytes(constant, bufs.len);
        o.e.run(grid, group);
    }
};
