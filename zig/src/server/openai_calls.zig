//! Finished calls from native parsers or the shared markup parser, under the request's tool policy.
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
    if (parsed.rejected_envelopes > 0) log.line("warning: kept {d} malformed or unoffered tool envelopes as content", .{parsed.rejected_envelopes});
    // max_tokens cut a call the stream had begun: the whole reply returns it cut too, as the stream did (#1238)
    const cut = if (!r.plan.policy.single and std.mem.eql(u8, reply.finish_reason, "length")) try tool_stream.cutCalls(r.a, reply.content, tools) else null;
    if (cut != null) reply.content = try @import("reply_text.zig").cutProse(r.a, reply.content);
    const found = cut orelse parsed.calls;
    const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse {
        if (r.plan.policy.single or parsed.calls != null) reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
        return null;
    };
    if (cut != null) return calls;
    reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
    reply.finish_reason = "tool_calls";
    if (r.plan.policy.single) reply.tool_calls_streamed = false;
    return if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls;
}
