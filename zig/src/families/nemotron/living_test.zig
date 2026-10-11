//! Lesson files: read back as written, idle layers left out, a taken-back lesson removed, other versions refused.
const std = @import("std");
const living = @import("living.zig");
const cfg = @import("config.zig");
const shape = @import("learn_shape.zig");
const core = struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
};
const Living = living.Io(core);

test "lessons read back as written, a file each; a layer a lesson leaves be is left out; a lesson taken back goes; another version is refused" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    var c: cfg.Config = .{ .hidden = 8, .vocab = 16, .layers = 2, .mamba_heads = 2, .mamba_head_dim = 4, .groups = 1, .state = 4, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 4, .experts = 4, .top_k = 2, .expert_width = 4, .shared_width = 8, .routed_scaling = 1, .eps = 1e-5 };
    c.kinds[0] = .mamba;
    c.kinds[1] = .attention;
    const rows = 2 * shape.block; // two lessons
    var a0: [rows * 8]f32 = undefined;
    var b0: [rows * 8]f32 = undefined;
    for (&a0, &b0, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i + 1);
        y.* = -@as(f32, @floatFromInt(i + 1));
    }
    // layer 1: the first lesson leaves it be, the second reads and changes it
    var a1: [rows * 8]f32 = @splat(0);
    var b1: [rows * 8]f32 = @splat(0);
    for (a1[shape.block * 8 ..], b1[shape.block * 8 ..]) |*x, *y| {
        x.* = 0.5;
        y.* = 0.25;
    }
    const layers = [_]living.Layer{
        .{ .layer = 0, .in = 8, .out = 8, .a = &a0, .b = &b0, .tau = &.{ 0.3, 0.4 } },
        .{ .layer = 1, .in = 8, .out = 8, .a = &a1, .b = &b1, .tau = &.{ 0, 0.6 } },
    };
    _ = try Living.write(gpa, dir, &layers, &.{ 7, 9 }, 0, 1);
    _ = try Living.write(gpa, dir, &layers, &.{ 7, 9 }, 1, 2); // the second lesson alone: the first file stays
    {
        var saved = (try Living.open(gpa, io, dir)).?;
        defer saved.close(io);
        try std.testing.expectEqual(@as(usize, 2), saved.lessons.len);
        const first = (try saved.lessons[0].layer(c, 0)).?;
        try std.testing.expectEqualSlices(f32, a0[0 .. shape.block * 8], first.a);
        try std.testing.expectEqualSlices(f32, b0[0 .. shape.block * 8], first.b);
        try std.testing.expectEqual(@as(f32, 0.3), first.tau);
        try std.testing.expectEqual(@as(f32, 7), try saved.lessons[0].need());
        try std.testing.expect((try saved.lessons[0].layer(c, 1)) == null);
        const second = (try saved.lessons[1].layer(c, 1)).?;
        try std.testing.expectEqualSlices(f32, a1[shape.block * 8 ..], second.a);
        try std.testing.expectEqual(@as(f32, 0.6), second.tau);
        try std.testing.expectEqual(@as(f32, 9), try saved.lessons[1].need());
    }
    // the second lesson taken back: its file goes
    _ = try Living.write(gpa, dir, &layers, &.{ 7, 9 }, 1, 1);
    {
        var saved = (try Living.open(gpa, io, dir)).?;
        defer saved.close(io);
        try std.testing.expectEqual(@as(usize, 1), saved.lessons.len);
    }
    var buf: [64]u8 = undefined;
    const path = try std.fs.path.join(gpa, &.{ dir, try living.fileName(&buf, 0) });
    defer gpa.free(path);
    try core.shard_edit.create(gpa, path, &.{ .{ "format", "tensorfold living weights" }, .{ "version", "0" } }, &.{.{ .name = "layers.0.a", .dtype = .f32, .shape = &.{ shape.block, 8 }, .bytes = std.mem.sliceAsBytes(a0[0 .. shape.block * 8]) }});
    try std.testing.expectError(error.LivingWeightsFromAnotherVersion, Living.open(gpa, io, dir));
}
