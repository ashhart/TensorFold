//! XPU startup refusal checks for --vision (text-only families) and unsupported --tp values before device access.
const std = @import("std");
const native = @import("xpu.zig");

test "draft sources: XPU refuses --vision and unsupported --tp before device access" {
    if (comptime @import("builtin").os.tag != .windows) {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        for (native.families) |f| {
            if (std.mem.eql(u8, f.model_type, "lfm3")) continue; // LFM3 reads pictures: checked below
            var problem: []const u8 = "";
            try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = f.model_type, .vision = .{} }, &problem) == null);
            try std.testing.expect(std.mem.indexOf(u8, problem, "--vision") != null);
        }
        // the text-only families' refusal fires before their own open, and names the one family that reads pictures
        for ([_][]const u8{ "nemotron_h", "gemma4", "qwen3_5", "qwen4_exp", "llama" }) |mt| {
            var problem: []const u8 = "";
            try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = mt, .vision = .{} }, &problem) == null);
            try std.testing.expectStringStartsWith(problem, "--vision: the native Intel GPU engine reads pictures for LFM3-VL");
        }
        const nem = @import("xpu_nem.zig");
        try std.testing.expect(nem.readsPictures("lfm3_vl") and nem.readsPictures("lfm3"));
        try std.testing.expect(!nem.readsPictures("nemotron_h") and !nem.readsPictures("gemma4") and !nem.readsPictures("qwen3_5"));
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
