const std = @import("std");
const errors = @import("errors.zig");
const prompt = @import("drama_prompt");

/// Refuses a prompt past `limit` rows; `text_tokens` is the request text's share of `rows` as the prompt encoded it.
pub fn check(cx: *errors.Cx, text_tokens: usize, refs: []const prompt.Reference, rows: usize, limit: usize) errors.Refused!void {
    if (rows <= limit) return;
    var reference_frames: usize = 0;
    for (refs) |ref| reference_frames = std.math.add(usize, reference_frames, ref.codes.len) catch
        return cx.other("speech reference frame count overflow");
    return cx.fail(.request, @import("drama_prompt.zig").error_prefix ++ " {d} assembled prompt rows exceed the limit of {d} assembled prompt rows; " ++
        "request text: {d} tokens; acoustic references: {d} frames; assembled rows also include transcripts and wrappers", .{ rows, limit, text_tokens, reference_frames });
}
