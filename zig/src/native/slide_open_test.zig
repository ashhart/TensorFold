//! Unsupported learning is refused before an absent checkpoint or device can be opened.
const std = @import("std");
const api = @import("engine_api");

pub fn check(comptime open: anytype, backend: []const u8, models: []const []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (models) |model| {
        var why: []const u8 = "";
        const o: api.Open = .{ .dir = "missing-slide-checkpoint", .model_type = model, .slide = true };
        try std.testing.expect((try open(a, std.testing.allocator, std.testing.io, o, &why)) == null);
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on {s} yet", .{ model, backend }), why);
    }
}
