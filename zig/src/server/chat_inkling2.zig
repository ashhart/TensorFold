//! Inkling 2's release block grammar on chat replies; raw completions remain literal.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const native_reply = @import("native_reply.zig");
const parser = @import("inkling2_reply.zig");
const chat = @import("chat.zig");

/// Only HfText's exact config.json model_type sets this; aliases and engine display names do not.
pub fn enabled(srv: anytype) bool {
    return srv.text.native_inkling2;
}

pub fn read(g: anytype, ids: []const u32, finished: bool) std.mem.Allocator.Error!native_reply.Result {
    var result = if (g.chat and g.inkling2 != null)
        try g.inkling2.?.extend(ids, g.tools)
    else if (g.chat)
        try parser.parse(g.a, g.srv.text, ids, g.tools)
    else
        native_reply.Result{
            .content = try g.srv.text.decode(g.a, ids),
            .reasoning = "",
            .calls = null,
            .reasoning_tokens = 0,
        };
    result.content = g.stops.visible(result.content, !finished);
    result.reasoning = g.stops.visible(result.reasoning, !finished);
    if (g.stops.strings.len > 0) {
        const raw = try g.srv.text.decode(g.a, ids);
        if (g.stops.matched(raw) != null) result.calls = null;
    }
    return result;
}

pub fn stream(g: anytype, sink: chat.Sink) chat.Failure!void {
    const ids = reply_text.stripTrailing(g.collected.items, g.srv.eos);
    if (g.chat and g.inkling2 == null) g.inkling2 = try parser.Stream.init(g.a, g.srv.text);
    const result = try read(g, ids, false);
    // The persistent token cursor grows monotonically; calls wait for its final close.
    const reason = g.streamed_reasoning.after(result.reasoning, true);
    if (reason.len > 0) {
        try g.streamed_reasoning.set(g.a, result.reasoning, reason);
        try g.emit(sink, try reply_text.deltaOf(g.a, "reasoning_content", reason));
    }
    const delta = g.streamed.after(result.content, true);
    if (delta.len > 0) {
        try g.streamed.set(g.a, result.content, delta);
        try g.emit(sink, .{ .string = delta });
    }
}
