//! --loop-guard-stop: the finish reason and the error a reply carries when the loop guard ended it in its reasoning.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

/// OpenAI's finish reason for a reply the server ended before its answer, which OpenAI-compatible clients all parse.
pub const finish_reason = "content_filter";

/// The error's ``code``, which a client keys on.
pub const code = "reasoning_loop";

/// OpenAI's error object for a stopped reply: beside a whole reply's choices, or as a stream's last event.
pub fn errorBody(a: Allocator, period: u32) Allocator.Error!json.Value {
    const o = try json.newObject(a);
    const message = "the loop guard ended the reply in its reasoning, which repeated a {d}-token cycle: no answer or tool calls follow (--loop-guard-stop)";
    try o.put(a, "message", .{ .string = try std.fmt.allocPrint(a, message, .{period}) });
    try o.put(a, "type", .{ .string = "server_error" });
    try o.put(a, "param", .null);
    try o.put(a, "code", .{ .string = code });
    return .{ .object = o };
}
