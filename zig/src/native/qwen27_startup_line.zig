//! The 27B's startup line: its streams, context, DFlash2's drafted streams and whether images are read.
const std = @import("std");
const platform = @import("sys");
const lb = @import("qwen27_lanes.zig");

pub const StartupLine = struct {
    streams: u32,
    context: u32,
    draft_slots: u32,
    images: bool,

    pub fn format(line: StartupLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("Qwen3.8-27B: {d} stream{s} of {d} tokens", .{ line.streams, if (line.streams == 1) "" else "s", line.context });
        const drafts = @min(line.streams, line.draft_slots);
        if (drafts > 0) try w.print(", DFlash2 drafts for {d} stream{s}", .{ drafts, if (drafts == 1) "" else "s" });
        if (line.images) try w.writeAll(", images read");
    }
};

test "qwen27 startup: draft stream count, singular text and optional suffixes" {
    const limit = try lb.draftSlots();
    const limited = platform.c.getenv("TF_Q27_DRAFT_SLOTS") != null;
    try std.testing.expectEqual(@as(u32, if (limited) 1 else 8), limit);
    const cases = [_]struct { u32, []const u8, []const u8 }{
        .{ 1, "1 stream", "1 stream" },
        .{ 2, "2 streams", "2 streams" },
        .{ 9, "9 streams", "8 streams" },
        .{ 16, "16 streams", "8 streams" },
    };
    var mismatches: usize = 0;
    for (cases) |case| {
        for ([_]bool{ false, true }) |drafter| {
            for ([_]bool{ false, true }) |with_images| {
                var draft: [64]u8 = undefined;
                const suffix = if (drafter) try std.fmt.bufPrint(&draft, ", DFlash2 drafts for {s}", .{if (limited) "1 stream" else case[2]}) else "";
                var expected: [192]u8 = undefined;
                const want = try std.fmt.bufPrint(&expected, "Qwen3.8-27B: {s} of 4096 tokens{s}{s}", .{ case[1], suffix, if (with_images) ", images read" else "" });
                var actual: [192]u8 = undefined;
                const got = try std.fmt.bufPrint(&actual, "{f}", .{StartupLine{ .streams = case[0], .context = 4096, .draft_slots = if (drafter) limit else 0, .images = with_images }});
                std.testing.expectEqualStrings(want, got) catch {
                    mismatches += 1;
                };
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}
