//! XPU startup refusal checks for --vision and unsupported --tp values before device access.
const std = @import("std");
const native = @import("xpu.zig");

test "draft sources: XPU refuses --vision and unsupported --tp before device access" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (native.families) |f| {
        var problem: []const u8 = "";
        try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = f.model_type, .vision = .{} }, &problem) == null);
        try std.testing.expect(std.mem.indexOf(u8, problem, "--vision") != null);
    }
    if (@import("xpu_engine_options").qwen27) {
        for ([_][]const u8{"qwen3_5"}) |mt| {
            var problem: []const u8 = "";
            try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = mt, .tp = 3 }, &problem) == null);
            try std.testing.expect(std.mem.indexOf(u8, problem, "--tp 1, 2 or 4") != null);
        }
    }
    const env = @import("sys").environment;
    try std.testing.expectError(error.InvalidEnvironment, env.parse("XPU_TP", "3"));
    try std.testing.expectEqual(@as(?u64, 4), try env.parse("XPU_TP", "4"));
    try std.testing.expectError(error.InvalidEnvironment, env.parse("XPU_DUMP_H", "bad\npath"));
}
