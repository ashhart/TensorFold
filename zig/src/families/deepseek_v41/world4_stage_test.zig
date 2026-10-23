const std = @import("std");
const pk = @import("pack.zig");
const partition = @import("partition.zig");
const named = @import("named.zig");

test "world four: captured programs retain their rank and world" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "meta.json", .data = "{\"env\":{\"TF_TP_WORLD\":\"4\",\"TF_TP_RANK\":\"3\"}}" });
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const o = try @import("calls_check.zig").optionsOf(arena.allocator(), std.testing.io, dir);
    try std.testing.expectEqual(@as(u32, 4), o.world);
    try std.testing.expectEqual(@as(u32, 3), o.rank);
}

test "world four: embedding shard reads use each rank's global offset" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const width = 768;
    const hidden = 2;
    const file = try pk.image(a, &.{.{ .name = "embed.weight", .dtype = "BF16", .shape = &.{ width, hidden } }});
    defer a.free(file);
    const start = 8 + std.mem.readInt(u64, file[0..8], .little);
    for (0..width * hidden) |i| std.mem.writeInt(u16, file[start + i * 2 ..][0..2], @intCast(i), .little);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var pack = try pk.Pack.open(a, io, dir);
    defer pack.deinit();
    for (0..4) |rank| {
        const part = try partition.share(width, rank, 4);
        var b: named.Builder = .{ .gpa = a, .io = io, .pack = &pack };
        defer b.deinit();
        var top = [_]@import("plan.zig").NativePlan{.{ .name = "embed.weight", .info = try pack.need("embed.weight"), .slice = .vocab_rows, .bytes = part.len * hidden * 2 }};
        const plan: @import("plan.zig").Plan = .{ .arena = undefined, .rank = @intCast(rank), .world = 4, .layers = &.{}, .top = &top, .head = null, .dspark = null, .leftovers = &.{} };
        try b.top(&plan);
        const embed = b.find("embed").?;
        try std.testing.expectEqual(@as(usize, part.len), embed.shape[0]);
        try std.testing.expectEqual(@as(usize, part.len * hidden * 2), embed.bytes.len);
        for (0..part.len * hidden) |i| try std.testing.expectEqual(@as(u16, @intCast(part.lo * hidden + i)), std.mem.readInt(u16, embed.bytes[i * 2 ..][0..2], .little));
    }
}
