//! Inkling 2's reply blocks (inkling_reply.zig) as reasoning, content and calls, in whole and streamed replies.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const split = @import("inkling_reply.zig");
const chat = @import("chat.zig");
const Value = json.Value;

/// Inkling 2's replies end their sampling with this marker; a tokenizer that holds it speaks the block format.
pub fn is(text: anytype) bool {
    return text.vtable.token_id(text.ctx, split.markers.end_sampling) != null and text.vtable.token_id(text.ctx, split.markers.model) != null;
}

/// The reply so far (ids past the prompt, specials spelled out) split as vLLM's Inkling parser splits it.
pub fn read(g: anytype, ids: []const u32, finished: bool) std.mem.Allocator.Error!@import("kimi_reply.zig").Result {
    const raw = try g.srv.text.decode(g.a, ids);
    if (!g.chat) return .{ .content = g.stops.visible(raw, !finished), .reasoning = "", .calls = null, .reasoning_tokens = 0 };
    const r = try split.split(g.a, raw, finished);
    var calls: ?[]Value = null; // a raw completion stays unsplit above; a reply a stop string ended keeps no calls
    if (g.tools.len > 0 and r.calls.len > 0 and g.stops.matched(raw) == null) {
        const list = try g.a.alloc(Value, r.calls.len); // no ids: the model writes none (openai_calls.zig keys them)
        for (r.calls, list) |c, *v| {
            const f = try json.newObject(g.a);
            try f.put(g.a, "name", .{ .string = c.name });
            try f.put(g.a, "arguments", .{ .string = c.arguments });
            const o = try json.newObject(g.a);
            try o.put(g.a, "type", .{ .string = "function" });
            try o.put(g.a, "function", .{ .object = f });
            v.* = .{ .object = o };
        }
        calls = list;
    }
    return .{
        .content = g.stops.visible(r.content, !finished),
        .reasoning = g.stops.visible(r.reasoning, !finished),
        .calls = calls,
        .reasoning_tokens = 0,
    };
}

/// A streamed reply's new reasoning and content; calls wait for the finished split.
pub fn stream(g: anytype, sink: chat.Sink) chat.Failure!void {
    const ids = reply_text.stripTrailing(g.collected.items, g.srv.eos);
    const r = try read(g, ids, false);
    try @import("chat_kimi.zig").emitDeltas(g, sink, r, false);
}
