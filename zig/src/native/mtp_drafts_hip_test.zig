//! The `--mtp-drafts` refusal by family and world size, before any load.
const std = @import("std");
const api = @import("engine_api");
const qwen35 = @import("qwen3_5");
const flashnext = @import("flashnext");
const hip = @import("hip.zig");

test "--mtp-drafts: Flash Next takes seven at --tp 1 and three above it, every other HIP family three; past that refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    inline for (.{ .{ flashnext.native, "qwen4_exp", 7, 1, "" }, .{ qwen35.native, "qwen3_5", 3, 1, "" }, .{ flashnext.native, "qwen4_exp", 3, 2, " with --tp above 1" }, .{ qwen35.native, "qwen3_5", 3, 2, "" } }) |c| {
        var o: api.Open = .{ .dir = "d", .model_type = c[1], .mtp_drafts = c[2], .tp = c[3], .master = if (c[3] > 1) "node0" else "" };
        try hip.checkGroup(c[0], a, o, &problem);
        o.mtp_drafts = c[2] + 1;
        try std.testing.expectError(error.Refused, hip.checkGroup(c[0], a, o, &problem));
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "--mtp-drafts {d}: the HIP engine drafts at most {d}{s}", .{ c[2] + 1, c[2], c[4] }), problem);
    }
}
