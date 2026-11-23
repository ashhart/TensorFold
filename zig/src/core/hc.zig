//! Hyper-connection block boundaries (hc_boundary.metal): expand with partial sums then split; 2 launches, 3 in place.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// The boundary's shape: width (4 streams, 1024s), Sinkhorn iterations, split eps in 1e-9, the MoE's picks.
pub const Shape = struct { width: u32, sinkhorn: u32, eps_e9: u32, topk: u32 = 8 };

/// The partial sums a row: 24 mixes and the squares, for each 1024-column slice of the 4 streams.
pub fn partBytes(s: Shape) usize {
    return @as(usize, 4 * s.width / 1024) * 25 * 4;
}

pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    if (s.width % 1024 != 0) return error.UnsupportedHcShape;
    return std.fmt.allocPrint(a, "#define TF_D {d}\n#define TF_ITERS {d}\n#define TF_HC_EPS_INT {d}\n#define TF_HC_TOPK {d}\n{s}", .{ s.width, s.sinkhorn, s.eps_e9, s.topk, ks.core_hc_boundary });
}

pub const names = [8][:0]const u8{ "tf_hc_expand_mix", "tf_hc_first_mix", "tf_hc_split", "tf_hc_expand_inplace", "tf_hc_expand_mix_parts", "tf_hc_first_mix_parts", "tf_hc_expand_mix_moe", "tf_hc_expand_mix_parts_moe" };
pub const Pipes = [names.len]mtl.Pipeline;

/// Windows of up to 4 rows: the expand's parts on six groups a slice (same sums; M5 Ultra 1.1-1.35 us a boundary less).
pub const parts_rows = 4;

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// Streams written over themselves: one buffer for both (prompt chunks), the same bits as two.
pub fn inPlace(x_old: anytype, x_new: anytype) bool {
    return x_old.buf.id == x_new.buf.id and x_old.off == x_new.off;
}

/// `x_old` expanded to `x_new`, mix sums to `part`, split; with `b.wts`, the expand combines a MoE's rows `branch`.
pub fn boundary(e: mtl.ComputeEncoder, pipes: Pipes, s: Shape, expand: bool, rows: u32, eps: f32, b: anytype) void {
    boundaryWith(e, pipes, s, expand, rows, eps, b, rows <= @as(u32, if (moeWts(b) != null) moe_parts_rows else parts_rows));
}

/// Windows of up to 2 rows take the combining expand on six groups a slice (its reads a group grow with the combine).
pub const moe_parts_rows = 2;

fn moeWts(b: anytype) ?@TypeOf(b.x_old) {
    return if (comptime @hasField(@TypeOf(b), "wts")) b.wts else null;
}

/// `boundary` with the expand's parts on six threadgroups a slice or on one (the same bits either way).
pub fn boundaryWith(e: mtl.ComputeEncoder, pipes: Pipes, s: Shape, expand: bool, rows: u32, eps: f32, b: anytype, parts: bool) void {
    const one = expand and inPlace(b.x_old, b.x_new);
    const wts = moeWts(b);
    std.debug.assert(wts == null or (expand and !one));
    if (one) expandInPlace(e, pipes, s, rows, b.x_old, b.branch, b.post, b.comb);
    e.setPipeline(pipes[if (wts != null) @as(usize, if (parts) 7 else 6) else @as(usize, if (expand and !one) 0 else 1) + @as(usize, if (parts) 4 else 0)]);
    inline for (.{ b.x_old, b.branch, b.post, b.comb, b.fn_packed, b.x_new, b.part }, 0..) |r, i| bind(e, i, r);
    if (wts) |w| bind(e, 7, w);
    e.dispatchGroups(mtl.Size.of(4 * s.width / 1024, rows, if (parts) 6 else 1), mtl.Size.of(256, 1, 1));
    e.setPipeline(pipes[2]);
    inline for (.{ if (expand) b.x_new else b.x_old, b.part }, 0..) |r, i| bind(e, i, r);
    bind(e, 2, b.scale);
    bind(e, 3, b.base);
    bind(e, 4, b.norm);
    e.setValue(eps, 5);
    inline for (.{ b.normed, b.post, b.comb }, 6..) |r, i| bind(e, i, r);
    e.dispatchGroups(mtl.Size.of(rows, split_groups, 1), mtl.Size.of(1024, 1, 1));
}

/// The split's threadgroups a row: the collapse and norm, and post with the Sinkhorn comb beside it (the same bits).
pub const split_groups = 2;

/// A pending `branch` into the streams `x` in place (width / 4 threads a row, so width <= 4096).
pub fn expandInPlace(e: mtl.ComputeEncoder, pipes: Pipes, s: Shape, rows: u32, x: anytype, branch: anytype, post: anytype, comb: anytype) void {
    std.debug.assert(s.width <= 4096);
    e.setPipeline(pipes[3]);
    inline for (.{ x, branch, post, comb }, 0..) |r, i| bind(e, i, r);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(s.width / 4, 1, 1));
}

test "the boundary's shape is checked" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const src = try source(arena.allocator(), .{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 });
    try std.testing.expect(std.mem.startsWith(u8, src, "#define TF_D 4096\n#define TF_ITERS 20\n#define TF_HC_EPS_INT 1000\n#define TF_HC_TOPK 8\n"));
    try std.testing.expectError(error.UnsupportedHcShape, source(arena.allocator(), .{ .width = 4000, .sinkhorn = 20, .eps_e9 = 1000 }));
    try std.testing.expectEqual(@as(usize, 1600), partBytes(.{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 }));
}
