//! The Qwen3.8-27B startup line: draft stream count, its singular wording and the optional suffixes.
const std = @import("std");
const platform = @import("sys");
const lb = @import("qwen27_lanes.zig");
const startup = @import("qwen27_startup.zig");
const StartupLine = startup.StartupLine;

test "qwen27 startup: draft stream count, singular text and optional suffixes" {
    const limit = try lb.draftSlots();
    const limited = platform.c.getenv("TF_Q27_DRAFT_SLOTS") != null;
    try std.testing.expectEqual(@as(u32, if (limited) 1 else 8), limit);
    // the wording: the smaller of the stream count and the draft-slot setting
    const wording = [_]struct { streams: u32, slots: u32, want: []const u8 }{
        .{ .streams = 1, .slots = 0, .want = "" },
        .{ .streams = 1, .slots = 1, .want = "DFlash2 drafts for the only stream" },
        .{ .streams = 1, .slots = 8, .want = "DFlash2 drafts for the only stream" },
        .{ .streams = 1, .slots = 99, .want = "DFlash2 drafts for the only stream" },
        .{ .streams = 2, .slots = 0, .want = "" },
        .{ .streams = 2, .slots = 1, .want = "DFlash2 drafts for 1 of 2 streams" },
        .{ .streams = 2, .slots = 8, .want = "DFlash2 drafts for both streams" },
        .{ .streams = 2, .slots = 99, .want = "DFlash2 drafts for both streams" },
        .{ .streams = 4, .slots = 1, .want = "DFlash2 drafts for 1 of 4 streams" },
        .{ .streams = 4, .slots = 8, .want = "DFlash2 drafts for all 4 streams" },
        .{ .streams = 4, .slots = 99, .want = "DFlash2 drafts for all 4 streams" },
        .{ .streams = 16, .slots = 1, .want = "DFlash2 drafts for 1 of 16 streams" },
        .{ .streams = 16, .slots = 8, .want = "DFlash2 drafts for 8 of 16 streams" },
        .{ .streams = 16, .slots = 99, .want = "DFlash2 drafts for all 16 streams" },
    };
    var buf: [64]u8 = undefined;
    for (wording) |case| try std.testing.expectEqualStrings(case.want, try startup.draftWording(&buf, case.streams, case.slots));
    // the whole line: the wording, a drafter's absence and the optional suffixes
    const lines = [_]struct { streams: u32, slots: u32, images: bool, want: []const u8 }{
        .{ .streams = 1, .slots = 8, .images = false, .want = "Qwen3.8-27B: 1 stream of 4096 tokens, DFlash2 drafts for the only stream" },
        .{ .streams = 2, .slots = 8, .images = true, .want = "Qwen3.8-27B: 2 streams of 4096 tokens, DFlash2 drafts for both streams, images read" },
        .{ .streams = 4, .slots = 8, .images = false, .want = "Qwen3.8-27B: 4 streams of 4096 tokens, DFlash2 drafts for all 4 streams" },
        .{ .streams = 16, .slots = 8, .images = false, .want = "Qwen3.8-27B: 16 streams of 4096 tokens, DFlash2 drafts for 8 of 16 streams" },
        .{ .streams = 2, .slots = 0, .images = false, .want = "Qwen3.8-27B: 2 streams of 4096 tokens" },
    };
    var actual: [192]u8 = undefined;
    for (lines) |case| {
        const got = try std.fmt.bufPrint(&actual, "{f}", .{StartupLine{ .streams = case.streams, .context = 4096, .draft_slots = case.slots, .images = case.images }});
        try std.testing.expectEqualStrings(case.want, got);
    }
}
