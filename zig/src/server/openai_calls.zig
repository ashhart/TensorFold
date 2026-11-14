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
        reply.finish_reason = "tool_calls";
        reply.tool_calls_streamed = false;
        return if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls;
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
    const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse {
        if (r.plan.policy.single or parsed.calls != null) reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
        return null;
    };
    if (cut != null) return try numbered(r, calls);
    reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
    reply.finish_reason = "tool_calls";
    if (r.plan.policy.single) reply.tool_calls_streamed = false;
    return try numbered(r, if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls);
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
fn keptBlock(a: Allocator, parsed: tool_parse.Parsed) Allocator.Error!Value {
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
