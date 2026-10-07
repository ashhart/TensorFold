//! Flash Next's Triton launches from the captured cubins: each grid and constexpr matches the Python wrapper.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

pub const embed = @import("cuda_embed.zig");
pub const hc = @import("cuda_hc.zig");
pub const qmm = @import("cuda_qmm.zig");

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

/// One program per token row and per group of 32 hidden features.
pub fn embedGrid(rows: usize, dims: usize) [3]u32 {
    return .{ u(rows), u(dims / 32), 1 };
}

/// RMSNorm's BLOCK is the next power of two at or above the group width.
pub fn rmsBlock(group: usize) usize {
    return std.math.ceilPowerOfTwo(usize, group) catch unreachable;
}

pub const Tri = struct {
    set: *const aot.Set,
    s: cuda.Stream,

    fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try t.set.run(t.s, name, grid, args, consts);
    }

    /// glue.embed: ids (R,) int32 into (R, copies * dims) bf16, MLX 4-bit group 32.
    pub fn embed(t: Tri, ids: u64, weight: u64, scales: u64, biases: u64, out: u64, dims: usize, copies: usize, rows: usize) !void {
        try t.run("_embed", embedGrid(rows, dims), &.{
            p("IDS", "*i32", ids),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("OUT", "*bf16", out),
        }, &.{ ci("D", dims), ci("S_COPIES", copies) });
    }

    /// glue.hc_reduce_act: split-K sum fused with hc_act. M folds to a constexpr when it is 1.
    pub fn hcReduceAct(t: Tri, part: u64, act: u64, xs: u64, inj: u64, sk: usize, rows: usize, streams: usize, low: usize, ndn: usize, has_inj: usize) !void {
        try t.run("_hc_reduce_act", .{ u(rows), 1, 1 }, &.{
            p("PART", "*fp32", part),
            p("ACT", "*bf16", act),
            p("XS", "*fp32", xs),
            p("INJ", "*bf16", inj),
            int("M", rows),
        }, &.{ ci("SK", sk), ci("S", streams), ci("LOW", low), ci("LOWP", rmsBlock(low)), ci("HAS_INJ", has_inj), ci("NDN", ndn) });
    }

    /// qmm.hc_upmix: the up projection and the stream mix. N is the wide row, K is the low rank. M folds when it is 1.
    pub fn hcUpmix(t: Tri, act: u64, xs: u64, weight: u64, scales: u64, biases: u64, normed: u64, mixed: u64, xs_mixed: u64, rows: usize, n: usize, k: usize, dims: usize, streams: usize) !void {
        try t.run("_qmm_upmix", .{ cdiv(rows, 16), u(dims / 32), 1 }, &.{
            p("X", "*bf16", act),
            p("XS", "*fp32", xs),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("NORMED", "*bf16", normed),
            p("MIXED", "*bf16", mixed),
            p("XSM", "*fp32", xs_mixed),
            int("M", rows),
        }, &.{ ci("N", n), ci("K", k), ci("D", dims), ci("SS", streams), ci("BM", 16), ci("DB", 32), ci("GPI", 2), ci("SBN", 64) });
    }

    /// qmm.hc_down for one decode row. N=324 is the layer down stacked with inject. M folds when it is 1.
    pub fn hcDown(t: Tri, h: u64, pss: u64, scale: u64, normed: u64, weight: u64, scales: u64, biases: u64, out: u64, part: u64, eps: f32, rows: usize, n: usize, k: usize, dims: usize, nc: usize, streams: usize, sk: usize) !void {
        try t.run("_qmm_hcdown", .{ cdiv(rows, 16), cdiv(n, 64), u(sk) }, &.{
            p("H", "*bf16", h),
            p("PSS", "*fp32", pss),
            p("SCALE", "*fp32", scale),
            p("NORMED", "*bf16", normed),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("OUT", "*bf16", out),
            p("PART", "*fp32", part),
            int("M", rows),
            aot.float("eps", eps),
        }, &.{ ci("N", n), ci("K", k), ci("D", dims), ci("NC", nc), ci("SS", streams), ci("SK", sk), ci("BM", 16), ci("BLOCK_N", 64), ci("GPI", 2), ci("SBN", 64) });
    }

    /// glue.hc_writeback. Pointer types follow the tensor dtypes Triton specialized (mode 0..4).
    pub fn hcWriteback(t: Tri, h: u64, hout: u64, pss: u64, branch: u64, branch_ty: []const u8, inj: u64, y: u64, y_ty: []const u8, wts: u64, wts_ty: []const u8, rs: usize, rows: usize, dims: usize, streams: usize, mode: usize, topk: usize, slots: usize, world: usize) !void {
        try t.run("_hc_writeback", .{ u(rows), u(dims / 256), 1 }, &.{
            p("H", "*bf16", h),
            p("HOUT", "*bf16", hout),
            p("PSS", "*fp32", pss),
            p("BR", branch_ty, branch),
            p("INJ", "*bf16", inj),
            p("Y", y_ty, y),
            p("WTS", wts_ty, wts),
            int("RS", rs),
        }, &.{ ci("D", dims), ci("S", streams), ci("MODE", mode), ci("TOPK", topk), ci("SLOTS", slots), ci("BLOCK", 256), ci("WORLD", world) });
    }
};

test {
    _ = embed;
    _ = hc;
    _ = qmm;
}

test "embed grid is one program per row and per group of 32" {
    try std.testing.expectEqual([3]u32{ 4, 80, 1 }, embedGrid(4, 2560));
}

test "rmsnorm block is the next power of two at or above the group" {
    try std.testing.expectEqual(@as(usize, 4096), rmsBlock(2560));
    try std.testing.expectEqual(@as(usize, 16384), rmsBlock(10240));
}
