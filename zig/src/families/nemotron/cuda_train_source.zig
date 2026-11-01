//! Raw backward weights use the same language-model scope as the serving loader.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;

pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: Config) !core.Checkpoint {
    var ck = try core.Checkpoint.openModel(a, io, dir);
    ck.scope = c.prefix();
    return ck;
}

test "Omni learner: raw weights select the wrapped language model, leaving other tensors alone" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    defer a.free(path);
    const lm = [_]u16{ 0x3f80, 0x4000 };
    const other = [_]u16{ 0x4040, 0x4080 };
    const name = "backbone.layers.0.mixer.shared_experts.up_proj.weight";
    try core.shard_edit.create(a, path, &.{.{ "format", "mlx" }}, &.{
        .{ .name = "language_model." ++ name, .dtype = .bf16, .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&lm) },
        .{ .name = name, .dtype = .bf16, .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&other) },
        .{ .name = "vision_model.weight", .dtype = .bf16, .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&other) },
    });
    var c: Config = undefined;
    for ([_]bool{ false, true }) |omni| {
        c.omni = omni;
        var ck = try open(a, io, dir, c);
        defer ck.close();
        const got = try ck.expect(name, .bf16, &.{ 1, 2 });
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(if (omni) &lm else &other), got.bytes);
        try std.testing.expectEqual(@as(usize, 1), ck.used.count());
    }
}
