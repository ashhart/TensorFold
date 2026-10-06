//! Hyper-connection block boundaries in one launch (kernels/metal/core/hc_boundary.metal): expand, mix, split and norm.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// The boundary's shape: stream width (4 streams), Sinkhorn iterations, the split's eps in 1e-9, the mix unroll.
pub const Shape = struct { width: u32, sinkhorn: u32, eps_e9: u32, unroll: u32 = 8, sq_fma: bool = false };

pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    if (s.width % 1024 != 0 or (4 * s.width / 1024) % s.unroll != 0) return error.UnsupportedHcShape;
    return std.fmt.allocPrint(a, "#define TF_D {d}\n#define TF_ITERS {d}\n#define TF_HC_EPS_INT {d}\n#define TF_U {d}\n#define TF_SQ_FMA {d}\n{s}", .{ s.width, s.sinkhorn, s.eps_e9, s.unroll, @intFromBool(s.sq_fma), ks.core_hc_boundary });
}

pub const names = [2][:0]const u8{ "tf_hc_boundary_expand", "tf_hc_boundary_first" };

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// The streams `x_old` (with the pending `branch` written in by `post` and `comb` when `expand`) into `x_new`, mixed by
/// the packed `fn`, split by `scale` and `base`, collapsed and normed by `norm` into `normed`; the new post and comb
/// replace the old ones; `mixes` and `count` (zeroed, a word a row) are scratch.
pub fn boundary(e: mtl.ComputeEncoder, pipes: [2]mtl.Pipeline, expand: bool, rows: u32, eps: f32, b: anytype) void {
    e.setPipeline(pipes[if (expand) 0 else 1]);
    inline for (.{ b.x_old, b.branch, b.post, b.comb, b.fn_packed, b.scale, b.base, b.norm }, 0..) |r, i| bind(e, i, r);
    e.setValue(eps, 8);
    inline for (.{ b.x_new, b.mixes, b.normed, b.post, b.comb, b.count }, 9..) |r, i| bind(e, i, r);
    e.dispatchGroups(mtl.Size.of(6, rows, 1), mtl.Size.of(256, 1, 1));
}

test "the boundary's shape is checked" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const src = try source(arena.allocator(), .{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 });
    try std.testing.expect(std.mem.startsWith(u8, src, "#define TF_D 4096\n#define TF_ITERS 20\n#define TF_HC_EPS_INT 1000\n#define TF_U 8\n#define TF_SQ_FMA 0\n"));
    try std.testing.expectError(error.UnsupportedHcShape, source(arena.allocator(), .{ .width = 4000, .sinkhorn = 20, .eps_e9 = 1000 }));
}
