//! Which CUDA family opens an Inkling 2 Mini checkpoint: the two-rank engine or main's one-GPU engine.
const std = @import("std");
const sys = @import("sys");
const core = @import("core");
const inkling2 = @import("inkling2");
const registry = @import("cuda_families.zig").registry;
const pick = @import("cuda_alias.zig").pick;

fn index(comptime F: type) usize {
    inline for (registry, 0..) |G, i| if (G == F) return i;
    unreachable;
}

fn place(world: ?[:0]const u8) void {
    _ = if (world) |w| sys.env.setenv("TF_INKLING2_WORLD", w, 1) else sys.env.unsetenv("TF_INKLING2_WORLD");
    _ = if (world != null) sys.env.setenv("TF_INKLING2_RANK", "0", 1) else sys.env.unsetenv("TF_INKLING2_RANK");
}

test "inkling2_loop routing: one rank or another pack layout opens on main's engine, two ranks of ours on the two-rank engine" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const config = "{\"model_type\":\"inkling2_loop\",\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"1.6.6\",\"codebook\":\"mul1\"}}";
    const ours = inkling2.native_exl3.mini_marker;
    const theirs = "0.self_attn.q_proj.linear.trellis";
    const main_mini = comptime index(inkling2.native);
    const two_rank = comptime index(inkling2.native_exl3);
    const cases = [_]struct { sub: []const u8, tensor: []const u8, world: ?[:0]const u8, want: usize }{
        .{ .sub = "ours2", .tensor = ours, .world = "2", .want = two_rank },
        .{ .sub = "ours1", .tensor = ours, .world = null, .want = main_mini },
        .{ .sub = "theirs2", .tensor = theirs, .world = "2", .want = main_mini },
        .{ .sub = "theirs1", .tensor = theirs, .world = null, .want = main_mini },
    };
    defer place(null);
    for (cases) |c| {
        try tmp.dir.createDirPath(io, c.sub);
        var d = try tmp.dir.openDir(io, c.sub, .{});
        defer d.close(io);
        try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
        try core.checkpoint_fixture.write(gpa, io, d, "model.safetensors", &.{.{ .name = c.tensor, .shape = &.{4}, .bytes = "abcd" }});
        const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, c.sub });
        defer gpa.free(path);
        place(c.world);
        try std.testing.expectEqual(@as(?usize, c.want), pick(registry, gpa, io, "inkling2_loop", path));
    }
    try std.testing.expectEqualStrings("inkling2_loop", registry[main_mini].model_type);
    try std.testing.expectEqual(@as(?usize, index(inkling2.native_exl3)), pick(registry, gpa, io, "inkling2", "unused"));
}
