//! Hyper-connection block boundaries in two launches (kernels/metal/core/hc_boundary.metal): the expand with partial sums, then the split (in place: three).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// The boundary's shape: stream width (4 streams, a multiple of 1024), Sinkhorn iterations, the split's eps in 1e-9.
pub const Shape = struct { width: u32, sinkhorn: u32, eps_e9: u32 };

/// The partial sums a row: 24 mixes and the squares, for each 1024-column slice of the 4 streams.
pub fn partBytes(s: Shape) usize {
    return @as(usize, 4 * s.width / 1024) * 25 * 4;
}

pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    if (s.width % 1024 != 0) return error.UnsupportedHcShape;
    return std.fmt.allocPrint(a, "#define TF_D {d}\n#define TF_ITERS {d}\n#define TF_HC_EPS_INT {d}\n{s}", .{ s.width, s.sinkhorn, s.eps_e9, ks.core_hc_boundary });
}

pub const names = [4][:0]const u8{ "tf_hc_expand_mix", "tf_hc_first_mix", "tf_hc_split", "tf_hc_expand_inplace" };

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// Streams written over themselves: one buffer for both (prompt chunks), the same bits as two.
pub fn inPlace(x_old: anytype, x_new: anytype) bool {
    return x_old.buf.id == x_new.buf.id and x_old.off == x_new.off;
}

/// `x_old` expanded into `x_new` (one buffer: in place, then read as a first boundary, the same sums), its mix sums into `part`, the split.
pub fn boundary(e: mtl.ComputeEncoder, pipes: [4]mtl.Pipeline, s: Shape, expand: bool, rows: u32, eps: f32, b: anytype) void {
    const one = expand and inPlace(b.x_old, b.x_new);
    if (one) expandInPlace(e, pipes, s, rows, b.x_old, b.branch, b.post, b.comb);
    e.setPipeline(pipes[if (expand and !one) 0 else 1]);
    inline for (.{ b.x_old, b.branch, b.post, b.comb, b.fn_packed, b.x_new, b.part }, 0..) |r, i| bind(e, i, r);
    e.dispatchGroups(mtl.Size.of(4 * s.width / 1024, rows, 1), mtl.Size.of(256, 1, 1));
    e.setPipeline(pipes[2]);
    inline for (.{ if (expand) b.x_new else b.x_old, b.part }, 0..) |r, i| bind(e, i, r);
    bind(e, 2, b.scale);
    bind(e, 3, b.base);
    bind(e, 4, b.norm);
    e.setValue(eps, 5);
    inline for (.{ b.normed, b.post, b.comb }, 6..) |r, i| bind(e, i, r);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// A pending `branch` into the streams `x` in place (width / 4 threads a row, so width <= 4096).
pub fn expandInPlace(e: mtl.ComputeEncoder, pipes: [4]mtl.Pipeline, s: Shape, rows: u32, x: anytype, branch: anytype, post: anytype, comb: anytype) void {
    std.debug.assert(s.width <= 4096);
    e.setPipeline(pipes[3]);
    inline for (.{ x, branch, post, comb }, 0..) |r, i| bind(e, i, r);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(s.width / 4, 1, 1));
}

test "the boundary's shape is checked" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const src = try source(arena.allocator(), .{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 });
    try std.testing.expect(std.mem.startsWith(u8, src, "#define TF_D 4096\n#define TF_ITERS 20\n#define TF_HC_EPS_INT 1000\n"));
    try std.testing.expectError(error.UnsupportedHcShape, source(arena.allocator(), .{ .width = 4000, .sinkhorn = 20, .eps_e9 = 1000 }));
    try std.testing.expectEqual(@as(usize, 1600), partBytes(.{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 }));
}
