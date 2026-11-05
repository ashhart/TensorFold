//! The engine's own words for a request it cannot serve, asked once in `chat.prepare` before anything is submitted.
const std = @import("std");
const api = @import("engine_api");
const Cx = @import("errors.zig").Cx;

/// The engine's refusal of a whole request, in the request's arena, cut at `api.refusal_bytes` and a UTF-8 boundary.
pub fn of(engine: api.Engine, cx: *Cx, request: *const api.Request) error{OutOfMemory}!?[]const u8 {
    var out: [api.refusal_bytes]u8 = undefined;
    const written = engine.refusal(request, &out) orelse return null;
    var end = @min(written.len, out.len); // the message's bytes are in `out`: the return value only says how many
    while (end > 0 and !std.unicode.utf8ValidateSlice(out[0..end])) end -= 1;
    if (end == 0) return "this engine refused the request without a reason";
    return try cx.a.dupe(u8, out[0..end]);
}
