//! The 27B server's startup line: how many streams draft and the optional suffixes.
const std = @import("std");

/// The startup line's drafting clause: the smaller of `streams` and `draft_slots`, or "" when none draft.
pub fn draftWording(buf: []u8, streams: u32, draft_slots: u32, kind: []const u8) ![]const u8 {
    const drafts = @min(streams, draft_slots);
    if (drafts == 0) return "";
    if (drafts == streams) {
        if (streams == 1) return std.fmt.bufPrint(buf, "{s} drafts for the only stream", .{kind});
        if (streams == 2) return std.fmt.bufPrint(buf, "{s} drafts for both streams", .{kind});
        return std.fmt.bufPrint(buf, "{s} drafts for all {d} streams", .{ kind, streams });
    }
    return std.fmt.bufPrint(buf, "{s} drafts for {d} of {d} streams", .{ kind, drafts, streams });
}

pub const StartupLine = struct {
    streams: u32,
    context: u32,
    draft_slots: u32,
    images: bool,
    draft_kind: []const u8 = "DFlash2",

    pub fn format(line: StartupLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("Qwen3.8-27B: {d} stream{s} of {d} tokens", .{ line.streams, if (line.streams == 1) "" else "s", line.context });
        var wording: [64]u8 = undefined;
        const drafting = draftWording(&wording, line.streams, line.draft_slots, line.draft_kind) catch return error.WriteFailed;
        if (drafting.len > 0) try w.print(", {s}", .{drafting});
        if (line.images) try w.writeAll(", images read");
    }
};

test "qwen27 startup: a v1 drafter's line names DFlash, a DFlash2 drafter's stays DFlash2" {
    var buf: [96]u8 = undefined;
    const v1 = try std.fmt.bufPrint(&buf, "{f}", .{StartupLine{ .streams = 2, .context = 4096, .draft_slots = 8, .images = false, .drafter = "DFlash" }});
    try std.testing.expectEqualStrings("Qwen3.8-27B: 2 streams of 4096 tokens, DFlash drafts for both streams", v1);
    var two: [96]u8 = undefined;
    const stock = try std.fmt.bufPrint(&two, "{f}", .{StartupLine{ .streams = 1, .context = 4096, .draft_slots = 8, .images = false }});
    try std.testing.expectEqualStrings("Qwen3.8-27B: 1 stream of 4096 tokens, DFlash2 drafts for the only stream", stock);
}
