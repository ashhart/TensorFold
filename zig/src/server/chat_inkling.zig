//! Inkling 2's reply blocks (inkling_reply.zig) as reasoning, content and calls, in whole and streamed replies.
const std = @import("std");
const json = @import("json");
const split = @import("inkling_reply.zig");
const chat = @import("chat.zig");
const Result = @import("kimi_reply.zig").Result;
const Value = json.Value;

/// The reply so far (ids past the prompt, specials spelled out) split as vLLM's Inkling parser splits it.
pub fn read(g: anytype, ids: []const u32, finished: bool) std.mem.Allocator.Error!Result {
    const raw = try g.srv.text.decode(g.a, ids);
    if (!g.chat) return .{ .content = g.stops.visible(raw, !finished), .reasoning = "", .calls = null, .reasoning_tokens = 0 };
    const r = try split.split(g.a, raw, finished, g.tools); // the request's tools, narrowed by tool_choice
    var calls: ?[]Value = null; // a raw completion stays unsplit above; a reply a stop string ended keeps no calls
    if (r.calls.len > 0 and g.stops.matched(raw) == null) {
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
        .reasoning_tokens = split.thinkingTokens(g.srv.text, ids),
    };
}

/// A streamed reply's new reasoning and content, split only past the text the last step read; calls wait.
pub fn stream(g: anytype, sink: chat.Sink) chat.Failure!void {
    const all = g.visible.text.items; // the reply's decode so far, append-only (chat_generation.zig decodes it)
    if (!g.chat) return @import("chat_kimi.zig").emitDeltas(g, sink, .{ .content = g.stops.visible(all, true), .reasoning = "", .calls = null, .reasoning_tokens = 0 }, true);
    if (g.inkling == null) g.inkling = .{ .a = g.a, .offered = g.tools };
    const s = &g.inkling.?;
    try s.feed(all, true);
    const r = try s.result(all, false);
    const shown: Result = .{ .content = g.stops.visible(r.content, true), .reasoning = g.stops.visible(r.reasoning, true), .calls = null, .reasoning_tokens = 0 };
    try @import("chat_kimi.zig").emitDeltas(g, sink, shown, s.fed == all.len);
}
