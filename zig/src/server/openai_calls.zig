//! Finished calls from native parsers or the shared markup parser, under the request's tool policy, with their ids.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const cortheon = @import("cortheon/cortheon.zig");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub fn attach(r: anytype, reply: *chat.Reply) Allocator.Error!?[]Value {
    const tools = r.plan.input.tools;
    if (tools.len == 0) return null;
    if (reply.native_tool_format) {
        const found = reply.native_calls;
        const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse return null;
        reply.content = try r.plan.policy.parsedContent(r.a, reply.content);
        if (!reply.native_cut_calls) reply.finish_reason = finished(reply.finish_reason);
        reply.tool_calls_streamed = false;
        const kept = if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls;
        try complete(r, reply, kept, reply.native_cut_call);
        return if (@import("chat_inkling2.zig").enabled(r.srv)) try numbered(r, kept) else kept;
    }
    const parsed = try tool_parse.parse(r.a, reply.content, tools, r.plan.policy.maxCalls());
    if (parsed.rejected_envelopes > 0) {
        log.line("warning: kept {d} malformed or unoffered tool envelopes as content", .{parsed.rejected_envelopes});
        r.srv.metrics.keptToolCalls(r.srv.io, parsed.kept_unoffered, parsed.kept_malformed);
        if (reply.runtime == .object) try reply.runtime.object.put(r.a, "kept_tool_calls", try keptBlock(r.a, parsed));
    }
    // max_tokens cut a call the stream had begun: the whole reply returns it cut too, as the stream did (#1238)
    const cut = if (!r.plan.policy.single and std.mem.eql(u8, reply.finish_reason, "length")) try tool_stream.cutCalls(r.a, reply.content, tools) else null;
    if (cut != null) reply.content = try @import("reply_text.zig").cutProse(r.a, reply.content);
    const found = cut orelse parsed.calls;
    const cut_call: ?*json.Object = if (cut) |list| if (list.len > 0) list[list.len - 1].object else null else null;
    const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse {
        if (r.plan.policy.single or parsed.calls != null) reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
        return null;
    };
    if (cut != null) {
        const returned = try numbered(r, calls);
        try complete(r, reply, returned, cut_call);
        return returned;
    }
    reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
    reply.finish_reason = finished(reply.finish_reason);
    if (r.plan.policy.single) reply.tool_calls_streamed = false;
    const returned = try numbered(r, if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls);
    try complete(r, reply, returned, null);
    return returned;
}

/// Calls in a reply that ended on its own finish tool_calls; any other end (length, deadline, cancel, loop) stays.
pub fn finished(reason: []const u8) []const u8 {
    return if (std.mem.eql(u8, reason, "stop")) "tool_calls" else reason;
}


/// A capped reply's complete call indices; object identity preserves the cut position after Cortheon filtering.
fn complete(r: anytype, reply: *chat.Reply, calls: []Value, cut: ?*json.Object) Allocator.Error!void {
    if (reply.runtime != .object or !std.mem.eql(u8, reply.finish_reason, "length")) return;
    const returned_cut = cut != null and calls.len > 0 and calls[calls.len - 1].object == cut.?;
    const indices = try r.a.alloc(Value, calls.len - @intFromBool(returned_cut));
    for (indices, 0..) |*slot, i| slot.* = try json.intValue(r.a, i);
    try reply.runtime.object.put(r.a, "complete_tool_calls", .{ .array = indices });
}
/// Each call the client gets takes the request's id for its place in the reply (``ids.Calls``).
fn numbered(r: anytype, calls: []Value) Allocator.Error![]Value {
    for (calls, 0..) |c, i| if (c == .object) try c.object.put(r.a, "id", .{ .string = try r.call_ids.id(r.a, r.plan.input.choice, i) });
    return calls;
}

/// A streamed call's opening delta takes the id ``attach`` gives the call at its index, so both replies agree.
pub fn streamed(r: anytype, delta: Value) Allocator.Error!void {
    const list = delta.get("tool_calls") orelse return;
    if (list != .array) return;
    for (list.array) |one| {
        if (one != .object or !one.object.contains("id")) continue;
        const at = std.math.cast(u64, (one.get("index") orelse continue).int64() orelse continue) orelse continue;
        try one.object.put(r.a, "id", .{ .string = try r.call_ids.id(r.a, r.plan.input.choice, at) }); // one of n: its own
    }
}

/// The reply's `tensorfold.kept_tool_calls`: both counts, and the unoffered names the model wrote.
pub fn keptBlock(a: Allocator, parsed: tool_parse.Parsed) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "unoffered", try json.intValue(a, parsed.kept_unoffered));
    try o.put(a, "malformed", try json.intValue(a, parsed.kept_malformed));
    if (parsed.unoffered_names.len > 0) {
        const list = try a.alloc(Value, parsed.unoffered_names.len);
        for (parsed.unoffered_names, 0..) |name, i| list[i] = .{ .string = name };
        try o.put(a, "unoffered_names", .{ .array = list });
    }
    return .{ .object = o };
}
