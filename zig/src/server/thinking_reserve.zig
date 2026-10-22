//! The thinking reserve extends the total reply limit while keeping the original answer quota.
const std = @import("std");
const api = @import("engine_api");

pub fn apply(request: *api.Request, reserve: i64, room: i64, think_end: ?u32) void {
    if (reserve <= 0 or think_end == null) return;
    request.answer_max = request.max_tokens;
    request.max_tokens = @intCast(@min(@as(i64, request.max_tokens) +| reserve, room, std.math.maxInt(u32)));
    request.think_end = think_end;
}
