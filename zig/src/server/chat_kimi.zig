//! Kimi's trusted XTML channels bypass the generic prose-to-tool parser, including in streamed replies.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const native = @import("kimi_reply.zig");
const chat = @import("chat.zig");

pub fn read(g: anytype, ids: []const u32, finished: bool) std.mem.Allocator.Error!native.Result {
    var result = if (g.chat) try native.parse(g.a, g.srv.text, ids, g.thinking, g.tools) else native.Result{ .content = try g.srv.text.decode(g.a, ids), .reasoning = "", .calls = null, .reasoning_tokens = 0 };
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
    const result = try read(g, ids, false);
    const thought = reply_text.pyStrip(result.reasoning);
    const reason = g.streamed_reasoning.after(thought, true);
    if (reason.len > 0) {
        try g.streamed_reasoning.set(g.a, thought, reason);
        try g.emit(sink, try reply_text.deltaOf(g.a, "reasoning_content", reason));
    }
    const delta = g.streamed.after(result.content, true);
    if (delta.len > 0) {
        try g.streamed.set(g.a, result.content, delta);
        try g.emit(sink, .{ .string = delta });
    }
    // Calls wait for the finished native parse; later malformed controls cannot retract an executed call.
}
