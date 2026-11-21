//! The 27B server's startup line: how many streams draft and the optional suffixes.
const std = @import("std");

/// The startup line's drafting clause: the smaller of `streams` and `draft_slots`, or "" when none draft.
pub fn draftWording(buf: []u8, streams: u32, draft_slots: u32) ![]const u8 {
    const drafts = @min(streams, draft_slots);
    if (drafts == 0) return "";
    if (drafts == streams) {
        if (streams == 1) return "DFlash2 drafts for the only stream";
        if (streams == 2) return "DFlash2 drafts for both streams";
        return std.fmt.bufPrint(buf, "DFlash2 drafts for all {d} streams", .{streams});
    }
    return std.fmt.bufPrint(buf, "DFlash2 drafts for {d} of {d} streams", .{ drafts, streams });
}

pub const StartupLine = struct {
    streams: u32,
    context: u32,
    draft_slots: u32,
    images: bool,

    pub fn format(line: StartupLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("Qwen3.8-27B: {d} stream{s} of {d} tokens", .{ line.streams, if (line.streams == 1) "" else "s", line.context });
        var wording: [64]u8 = undefined;
        const drafting = draftWording(&wording, line.streams, line.draft_slots) catch return error.WriteFailed;
        if (drafting.len > 0) try w.print(", {s}", .{drafting});
        if (line.images) try w.writeAll(", images read");
    }
};
