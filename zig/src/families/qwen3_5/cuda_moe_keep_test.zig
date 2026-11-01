//! Packed-row routing keeps use byte strides, independent of a forward's chunk width.
const std = @import("std");
const cuda = @import("cuda");
const hooks = @import("cuda_learn_hooks.zig");
const shape = @import("cuda_learn_shape.zig");
const R = struct {
    var calls: usize = 0;
    var dst: [2]u64 = undefined;
    var bytes: [2]usize = undefined;
    fn copy(d: u64, _: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        dst[calls] = d;
        bytes[calls] = n;
        calls += 1;
        return 0;
    }
};
test "MoE learning: routing keeps copy chunk rows at the packed row offset and skip absent buffers" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemcpyDtoDAsync_v2 = R.copy;
    var g = shape.the_27b;
    g.experts = 256;
    g.top_k = 8;
    const ks = [_]hooks.LayerKeep{ .{ .ids = 1000, .wts = 2000 }, .{} };
    const k: hooks.Keep = .{ .d = &d, .g = g, .layers = &ks, .at = 3, .t = .{ .k = undefined, .s = .{ .d = &d, .handle = null } } };
    R.calls = 0;
    try k.moe(0, 10, 20, 2);
    try std.testing.expectEqualSlices(u64, &.{ 1108, 2108 }, &R.dst);
    try std.testing.expectEqualSlices(usize, &.{ 72, 72 }, &R.bytes);
    try k.moe(1, 10, 20, 2);
    try std.testing.expectEqual(@as(usize, 2), R.calls);
}
test "MoE learning: memory replaces dense MLP copies with bounded pair scratch" {
    var g = shape.the_27b;
    g.hidden = 2048;
    g.intermediate = 512;
    g.layers = 40;
    const dense = @import("cuda_train.zig").memory(g);
    g.experts = 256;
    g.top_k = 8;
    const moe = @import("cuda_train.zig").memory(g);
    try std.testing.expect(moe > dense);
    try std.testing.expect(moe - dense < 1 << 30);
    // Routed weights remain borrowed: growing the count adds logits and metadata, never another expert table.
    g.experts = 512;
    const larger = @import("cuda_train.zig").memory(g);
    try std.testing.expect(larger - moe < 2 << 20);
}

test {
    _ = @import("cuda_learn_routes.zig");
    _ = @import("cuda_moe_adjoint_test.zig");
}
