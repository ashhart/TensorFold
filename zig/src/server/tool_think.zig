//! A required or named call closes the think block early enough to leave room for the call.
const std = @import("std");
const api = @import("engine_api");

pub fn apply(request: *api.Request, required: bool, think_close: []const u32, think_end: ?u32) void {
    if (!required or think_close.len == 0) return;
    const room = @max(256, request.max_tokens / 4);
    const cap = request.max_tokens -| room;
    request.think_budget = @max(1, if (request.think_budget > 0) @min(request.think_budget, cap) else cap);
    request.think_close = think_close;
    request.think_end = think_end;
}
