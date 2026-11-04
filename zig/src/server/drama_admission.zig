const std = @import("std");
const errors = @import("errors.zig");
const text_mod = @import("model_text.zig");
const prompt = @import("drama_prompt");

pub fn check(cx: *errors.Cx, tokenizer: text_mod.Text, input: []const u8, refs: []const prompt.Reference, rows: usize, limit: usize) errors.Refused!void {
    if (rows <= limit) return;
    const ids = tokenizer.encode(cx.a, input, false) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.other("the tokenizer cannot count this speech request's text tokens"),
    };
    defer cx.a.free(ids);
    var reference_frames: usize = 0;
    for (refs) |ref| reference_frames = std.math.add(usize, reference_frames, ref.codes.len) catch
        return cx.other("speech reference frame count overflow");
    return cx.fail(.request, @import("drama_prompt.zig").error_prefix ++ " {d} assembled prompt rows exceed the limit of {d} assembled prompt rows; " ++
        "request text: {d} tokens; acoustic references: {d} frames; assembled rows also include transcripts and wrappers", .{ rows, limit, ids.len, reference_frames });
}
