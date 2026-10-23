const std = @import("std");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");

fn defaultPack(a: std.mem.Allocator) !Pack {
    const cfg: Config = .{};
    const header = try @import("host_test.zig").tinyPack(a, &cfg, "", "");
    defer a.free(header);
    var p = Pack.init(a);
    errdefer p.deinit();
    try p.addHeader("model.safetensors", header, std.math.maxInt(u64));
    return p;
}

fn fingerprint(p: *const plan.Plan, hash: *std.crypto.hash.sha2.Sha256) !void {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try std.json.Stringify.value(.{ .rank = p.rank, .world = p.world, .layers = p.layers, .top = p.top, .head = p.head, .dspark = p.dspark, .leftovers = p.leftovers }, .{}, &out.writer);
    hash.update(out.written());
}

test "world four: default configuration plans every rank without payload reads" {
    const cfg: Config = .{};
    var pack = try defaultPack(std.testing.allocator);
    defer pack.deinit();
    for (0..4) |rank| {
        var p = try plan.build(std.testing.allocator, &cfg, &pack, .{ .rank = @intCast(rank), .world = 4 });
        defer p.deinit();
        try std.testing.expectEqual(@as(usize, cfg.allLayers()), p.layers.len);
        try std.testing.expectEqual(@as(usize, 0), p.leftovers.len);
    }
}

test "world four: world one and two plans retain their main fingerprint" {
    const cfg: Config = .{};
    var pack = try defaultPack(std.testing.allocator);
    defer pack.deinit();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_]u32{ 1, 2 }) |world| for (0..world) |rank| {
        var p = try plan.build(std.testing.allocator, &cfg, &pack, .{ .rank = @intCast(rank), .world = world });
        defer p.deinit();
        try fingerprint(&p, &hash);
    };
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    std.debug.print("world one/two plan fingerprint: {s}\n", .{hex});
    try std.testing.expectEqualStrings("0b2e082e4d244eec2d631a0da6f4495a5ccf73259f9384faea2167507169b5f5", &hex);
}
