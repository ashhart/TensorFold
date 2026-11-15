//! The release format's rank plans: the real release pack when TF_DSV41_RELEASE_PACK names it (headers only).
const std = @import("std");
const testing = std.testing;
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");

test "release plan: the real release pack plans at worlds 1, 2 and 4 with no tensor left over (TF_DSV41_RELEASE_PACK)" {
    const env = std.c.getenv("TF_DSV41_RELEASE_PACK") orelse return error.SkipZigTest;
    const dir = std.mem.span(env);
    const a = testing.allocator;
    const io = testing.io;
    const cfg = try Config.read(a, io, dir);
    try testing.expect(cfg.format == .release_fp8);
    var pack = try Pack.open(a, io, dir);
    defer pack.deinit();
    for ([_]u32{ 1, 2, 4 }) |world| for (0..world) |rank| {
        var p = try plan.build(a, &cfg, &pack, .{ .rank = @intCast(rank), .world = world });
        defer p.deinit();
        for (p.leftovers) |n| std.debug.print("world {d} rank {d} leftover {s}\n", .{ world, rank, n });
        try testing.expectEqual(@as(usize, 0), p.leftovers.len);
        var experts: u64 = 0;
        for (p.layers) |*l| experts += l.expertBytes();
        std.debug.print("world {d} rank {d}: {d:.2} GiB planned, {d:.2} GiB experts\n", .{ world, rank, @as(f64, @floatFromInt(p.bytes())) / (1 << 30), @as(f64, @floatFromInt(experts)) / (1 << 30) });
    };
}
