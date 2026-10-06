//! A MoE layer's route on Metal in two launches for 1-16 rows (kernels/metal/core/moe_route.metal): sigmoid + bias routers.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// The router's shape: hidden width, experts (a multiple of 4, at most 512), picks a row, rows a launch at most.
pub const Shape = struct { hidden: u32, experts: u32, topk: u32, max_rows: u32 = 16, out_bf16: bool = false };

/// The kernel source with `shape`'s constants.
pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    if (s.experts % 4 != 0 or s.experts > 512 or s.hidden % 64 != 0 or s.max_rows > 16 or s.topk * s.max_rows > 512) return error.UnsupportedRouterShape;
    return std.fmt.allocPrint(a, "#define TF_K {d}\n#define TF_E {d}\n#define TF_TOPK {d}\n#define TF_MAXR {d}\n{s}{s}", .{ s.hidden, s.experts, s.topk, s.max_rows, if (s.out_bf16) "#define TF_OUT_T bfloat\n" else "", ks.core_moe_route });
}

pub const names = [2][:0]const u8{ "tf_route_logits", "tf_route_select" };

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// Logits [rows, experts] fp32 from bf16 rows `x` and the router packed by `pack` (a 4-expert block a threadgroup).
pub fn logits(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, s: Shape, x: anytype, packed_router: anytype, out: anytype, rows: u32) void {
    e.setPipeline(pipe);
    bind(e, 0, x);
    bind(e, 1, packed_router);
    e.setValue(@as(i32, @intCast(rows)), 2);
    bind(e, 3, out);
    e.dispatchGroups(mtl.Size.of(s.experts / 4, (rows + 3) / 4, 1), mtl.Size.of(256, 1, 1));
}

pub const Args = extern struct { rows: i32, lo: i32, hi: i32, scale: f32 };

/// Where the selection writes: picks and weights, this Mac's picked experts with their members, both Macs' pick lists, counts, word.
pub fn Outputs(comptime B: type) type {
    return struct { pick: B, wts: B, ids: B, members: B, count: B, mine: B, theirs: B, counts: B, word: B };
}

/// The top-k by sigmoid + bias (ties to the lower id), weights normalized and scaled, and the groups for experts `own`.
pub fn select(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, logits_in: anytype, bias: anytype, scale: f32, rows: u32, own: [2]u32, out: anytype) void {
    e.setPipeline(pipe);
    bind(e, 0, logits_in);
    bind(e, 1, bias);
    e.setValue(Args{ .rows = @intCast(rows), .lo = @intCast(own[0]), .hi = @intCast(own[1]), .scale = scale }, 2);
    inline for (.{ out.pick, out.wts, out.ids, out.members, out.count, out.mine, out.theirs, out.counts, out.word }, 3..) |b, i| bind(e, i, b);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
}

/// A [experts, hidden] bf16 router repacked for `logits`: packed[q][m][i][tm][tn] = router[4q + tn][32i + 4m + tm].
pub fn pack(in: []const u16, out: []u16, experts: usize, hidden: usize) void {
    var o: usize = 0;
    for (0..experts / 4) |q| for (0..8) |m| for (0..hidden / 32) |i| for (0..4) |tm| for (0..4) |tn| {
        out[o] = in[(4 * q + tn) * hidden + 32 * i + 4 * m + tm];
        o += 1;
    };
}

test "the router's shape is checked and the pack is a permutation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const src = try source(arena.allocator(), .{ .hidden = 4096, .experts = 288, .topk = 8 });
    try std.testing.expect(std.mem.startsWith(u8, src, "#define TF_K 4096\n#define TF_E 288\n#define TF_TOPK 8\n#define TF_MAXR 16\n"));
    try std.testing.expectError(error.UnsupportedRouterShape, source(arena.allocator(), .{ .hidden = 4096, .experts = 290, .topk = 8 }));
    const E = 8;
    const K = 64;
    var in: [E * K]u16 = undefined;
    for (&in, 0..) |*v, i| v.* = @intCast(i);
    var out: [E * K]u16 = undefined;
    pack(&in, &out, E, K);
    var seen: [E * K]bool = @splat(false);
    for (out) |v| seen[v] = true;
    for (seen) |b| try std.testing.expect(b);
    try std.testing.expectEqual(@as(u16, 1 * K + 0), out[1]); // q 0, m 0, i 0, tm 0, tn 1: expert 1, k 0
    try std.testing.expectEqual(@as(u16, 0 * K + 1), out[4]); // tm 1, tn 0: expert 0, k 1
}
