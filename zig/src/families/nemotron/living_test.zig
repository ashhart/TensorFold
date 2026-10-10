//! The living-weights file: blocks and routers read back as written, a layer with nothing acting left out, another
//! version refused.
const std = @import("std");
const living = @import("living.zig");
const cfg = @import("config.zig");
const shape = @import("learn_shape.zig");
const core = struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
};
const Living = living.Io(core);

test "saved blocks and routers read back as written; a layer with nothing acting is left out; another version is refused" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    var c: cfg.Config = .{ .hidden = 8, .vocab = 16, .layers = 2, .mamba_heads = 2, .mamba_head_dim = 4, .groups = 1, .state = 4, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 4, .experts = 4, .top_k = 2, .expert_width = 4, .shared_width = 8, .routed_scaling = 1, .eps = 1e-5 };
    c.kinds[0] = .mamba;
    c.kinds[1] = .attention;
    const ranks = shape.block;
    var a0: [ranks * 8]f32 = undefined;
    var b0: [ranks * 8]f32 = undefined;
    for (&a0, &b0, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i);
        y.* = -@as(f32, @floatFromInt(i));
    }
    const zeros: [ranks * 8]f32 = @splat(0);
    _ = try Living.write(gpa, dir, &.{
        .{ .layer = 0, .in = 8, .out = 8, .a = &a0, .b = &b0, .tau = &.{0.3} },
        .{ .layer = 1, .in = 8, .out = 8, .a = &zeros, .b = &zeros, .tau = &.{shape.shut} },
    }, &.{7}, ranks);
    var saved = (try Living.open(gpa, io, dir)).?;
    const got = (try saved.layer(c, 0)).?;
    try std.testing.expectEqual(@as(usize, ranks), saved.ranks);
    try std.testing.expectEqualSlices(f32, &a0, got.a);
    try std.testing.expectEqualSlices(f32, &b0, got.b);
    try std.testing.expectEqualSlices(f32, &.{0.3}, got.tau);
    try std.testing.expectEqualSlices(f32, &.{7}, try saved.need());
    try std.testing.expect((try saved.layer(c, 1)) == null);
    saved.close(io);
    const path = try std.fs.path.join(gpa, &.{ dir, living.file_name });
    defer gpa.free(path);
    try core.shard_edit.create(gpa, path, &.{ .{ "format", "tensorfold living weights" }, .{ "version", "0" } }, &.{.{ .name = "layers.0.a", .dtype = .f32, .shape = &.{ ranks, 8 }, .bytes = std.mem.sliceAsBytes(&a0) }});
    try std.testing.expectError(error.LivingWeightsFromAnotherVersion, Living.open(gpa, io, dir));
}
