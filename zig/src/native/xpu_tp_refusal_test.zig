//! XPU startup refusal checks for --vision and unsupported --tp values before device access.
const std = @import("std");
const native = @import("xpu.zig");

test "draft sources: XPU refuses --vision and unsupported --tp before device access" {
    if (comptime @import("builtin").os.tag != .windows) {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        for (native.families) |f| {
            var problem: []const u8 = "";
            try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = f.model_type, .vision = .{} }, &problem) == null);
            try std.testing.expect(std.mem.indexOf(u8, problem, "--vision") != null);
        }
        if (@import("xpu_engine_options").qwen27) {
            for ([_][]const u8{ "qwen3_5", "qwen4_exp" }) |mt| {
                var problem: []const u8 = "";
                try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = mt, .tp = 3 }, &problem) == null);
                try std.testing.expect(std.mem.indexOf(u8, problem, "--tp 1, 2 or 4") != null);
            }
        }
    }
    if (@import("xpu_engine_options").qwen27) {
        const fn_nat = @import("xpu_flashnext.zig");
        const d = try fn_nat.parseChunkOpts(null, null);
        try std.testing.expectEqual(@as(u32, 64), d.busy);
        try std.testing.expectEqual(@as(u32, 2048), d.solo);
        const hi = try fn_nat.parseChunkOpts("128", "1024");
        try std.testing.expectEqual(@as(u32, 128), hi.busy);
        try std.testing.expectEqual(@as(u32, 1024), hi.solo);
        try std.testing.expectError(error.BadChunk, fn_nat.parseChunkOpts("0", null));
        try std.testing.expectError(error.BadChunk, fn_nat.parseChunkOpts("2049", null));
        try std.testing.expectError(error.BadChunkSolo, fn_nat.parseChunkOpts("128", "64"));
        try std.testing.expectError(error.BadChunkSolo, fn_nat.parseChunkOpts(null, "4096"));
    }
    const env = @import("sys").environment;
    try std.testing.expectError(error.InvalidEnvironment, env.parse("XPU_TP", "3"));
    try std.testing.expectEqual(@as(?u64, 4), try env.parse("XPU_TP", "4"));
    try std.testing.expectError(error.InvalidEnvironment, env.parse("XPU_DUMP_H", "bad\npath"));
}
