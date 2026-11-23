//! Inkling 2's release block grammar on chat replies; raw completions remain literal.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const native_reply = @import("native_reply.zig");
const parser = @import("inkling2_reply.zig");
const chat = @import("chat.zig");
const Allocator = std.mem.Allocator;

/// Only HfText's config.json model_type sets this, from the family's two: Small's and Mini's; aliases do not.
pub fn enabled(srv: anytype) bool {
    return srv.text.native_inkling2;
}

/// Where the first stop string starts in the raw text: the generation stops there and reports it.
fn firstStop(strings: []const []const u8, raw: []const u8) ?usize {
    var first: ?usize = null;
    for (strings) |stop| if (std.mem.indexOf(u8, raw, stop)) |at| {
        if (first == null or at < first.?) first = at;
    };
    return first;
}

/// `text` less a tail that could begin a stop string, held while the reply grows.
fn held(strings: []const []const u8, text: []const u8) []const u8 {
    var keep = text.len;
    for (strings) |stop| {
        var size = @min(stop.len -| 1, text.len);
        while (size > 0) : (size -= 1) if (std.mem.endsWith(u8, text, stop[0..size])) {
            keep = @min(keep, text.len - size);
            break;
        };
    }
    return text[0..keep];
}

/// The reply as its raw text shows it before byte `at`, where a stop starts: whole tokens, then part of one.
fn cut(g: anytype, ids: []const u32, at: usize) Allocator.Error!?parser.Stream {
    var s = try parser.Stream.init(g.a, g.srv.text);
    var offset: usize = 0;
    for (ids, 0..) |id, k| {
        const bytes = (try g.srv.text.tokenBytes(g.a, id)) orelse return null;
        if (offset + bytes.len > at) {
            _ = try s.extend(ids[0..k], g.tools);
            try s.partial(id, bytes[0 .. at - offset]);
            return s;
        }
        offset += bytes.len;
    }
    _ = try s.extend(ids, g.tools);
    return s;
}

pub fn read(g: anytype, ids: []const u32, finished: bool) Allocator.Error!native_reply.Result {
    const stopping = g.stops.strings.len > 0;
    // the raw text: the generation's own incremental decode while it streams, decoded once when it is done
    const raw: []const u8 = if (!g.chat or stopping) (if (finished) try g.srv.text.decode(g.a, ids) else g.visible.text.items) else "";
    if (!g.chat) return .{ .content = g.stops.visible(raw, !finished), .reasoning = "", .calls = null, .reasoning_tokens = 0 };
    const at = if (stopping) firstStop(g.stops.strings, raw) else null;
    // a raw stop cuts the reply where the generation stopped; calls finished before it stay
    var cut_stream = if (at) |offset| try cut(g, ids, offset) else null;
    var whole: parser.Stream = undefined;
    const s: *parser.Stream = if (cut_stream) |*c| c else if (g.inkling2) |*p| p else blk: {
        whole = try parser.Stream.init(g.a, g.srv.text);
        break :blk &whole;
    };
    var result = if (cut_stream != null) try s.result(finished) else try s.extend(ids, g.tools);
    if (finished and cut_stream == null) result = try s.done();
    if (at != null and cut_stream == null) {
        // no token bytes to place the stop: cut the parsed channels instead
        result.content = g.stops.visible(result.content, false);
        result.reasoning = g.stops.visible(result.reasoning, false);
    } else if (at == null and !finished) {
        result.content = held(g.stops.strings, result.content);
        result.reasoning = held(g.stops.strings, result.reasoning);
    }
    // a bare reply has no native calls; a tool grammar writes the generic envelope, which the shared parser reads
    if (finished and s.bare() and g.tools.len > 0) {
        const parsed = try @import("tool_parse.zig").parse(g.a, result.content, g.tools, null);
        if (parsed.calls) |calls| {
            // numbered as the native calls are, so the whole reply and its stream give the same ids
            for (calls, 0..) |call, i| if (call == .object) try call.object.put(g.a, "id", .{ .string = try std.fmt.allocPrint(g.a, "call_{d}", .{i}) });
            result.calls = calls;
            result.content = parsed.content;
        }
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
    // a bare reply offering tools may be a grammar's call envelope: its text waits for the finished reply
    if (g.chat and g.tools.len > 0 and g.inkling2.?.bare()) return;
    const delta = g.streamed.after(result.content, true);
    if (delta.len > 0) {
        try g.streamed.set(g.a, result.content, delta);
        try g.emit(sink, .{ .string = delta });
    }
}
