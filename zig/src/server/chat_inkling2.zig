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

/// Tokens a stop check decodes: the longest stop's bytes plus 8, as a byte token can carry one byte of a character.
pub fn stopWindow(strings: []const []const u8) usize {
    var longest: usize = 0;
    for (strings) |stop| longest = @max(longest, stop.len);
    return longest + 8;
}

/// Whether the newest tokens hold a stop in the reply's lossy decode; the window's leading continuations are skipped.
pub fn stopIn(a: Allocator, text: anytype, tail: []const u32, strings: []const []const u8) Allocator.Error!bool {
    var bytes: std.ArrayList(u8) = .empty;
    for (tail) |id| {
        const b = (try text.tokenBytes(a, id)) orelse return generic(a, text, tail, strings);
        try bytes.appendSlice(a, b);
    }
    var skip: usize = 0;
    while (skip < @min(3, bytes.items.len) and bytes.items[skip] & 0xC0 == 0x80) skip += 1;
    const empty: parser.RawText = .{};
    const view = try empty.view(a, bytes.items[skip..], true);
    return firstStop(strings, view.text.items) != null;
}

fn generic(a: Allocator, text: anytype, tail: []const u32, strings: []const []const u8) Allocator.Error!bool {
    const decoded = try text.decode(a, tail);
    return firstStop(strings, decoded) != null;
}

/// Where the first stop string starts in the raw text: the generation stops there and reports it.
fn firstStop(strings: []const []const u8, raw: []const u8) ?usize {
    var first: ?usize = null;
    for (strings) |stop| if (std.mem.indexOf(u8, raw, stop)) |at| {
        if (first == null or at < first.?) first = at;
    };
    return first;
}

/// Where a tail of `raw` that could begin a stop string starts, held while the reply grows; null: none could.
fn held(strings: []const []const u8, raw: []const u8) ?usize {
    var start: ?usize = null;
    for (strings) |stop| {
        var size = @min(stop.len -| 1, raw.len);
        while (size > 0) : (size -= 1) if (std.mem.endsWith(u8, raw, stop[0..size])) {
            if (start == null or raw.len - size < start.?) start = raw.len - size;
            break;
        };
    }
    return start;
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

/// The reply ended on its token cap: a call it was writing is returned cut, as the shared route returns one.
fn atLength(g: anytype) bool {
    if (comptime !@hasField(@TypeOf(g.*), "box")) {
        return false; // a test's stand-in generation has no engine box
    } else {
        if (g.deadline_hit) return false;
        if (g.reason) |r| return std.mem.eql(u8, r, "length");
        return g.box.reason == .length;
    }
}

/// The request's parallel_tool_calls: false; a test's stand-in generation has none.
fn single(g: anytype) bool {
    return if (comptime @hasField(@TypeOf(g.*), "single")) g.single else false;
}

/// A bare reply's calls in the shared route's order: parsed calls, cut recovery at the cap, then the one-call filter.
fn bareCalls(g: anytype, result: *native_reply.Result) Allocator.Error!void {
    const policy: @import("tool_policy.zig").Policy = .{ .single = single(g) };
    const parsed = try @import("tool_parse.zig").parse(g.a, result.content, g.tools, policy.maxCalls());
    if (parsed.rejected_envelopes > 0) result.kept = parsed;
    const cut_calls = if (!policy.single and atLength(g)) try @import("tool_stream.zig").cutCalls(g.a, result.content, g.tools) else null;
    if (cut_calls != null) {
        result.content = try reply_text.cutProse(g.a, result.content);
        result.cut_calls = true;
    }
    const found = cut_calls orelse parsed.calls orelse {
        if (policy.single or parsed.calls != null) result.content = try policy.parsedContent(g.a, parsed.content);
        return;
    };
    if (cut_calls == null) result.content = try policy.parsedContent(g.a, parsed.content);
    // numbered as the native calls are, so the whole reply and its stream give the same ids
    for (found, 0..) |call, i| if (call == .object) try call.object.put(g.a, "id", .{ .string = try std.fmt.allocPrint(g.a, "call_{d}", .{i}) });
    result.calls = found;
}

pub fn read(g: anytype, ids: []const u32, finished: bool) Allocator.Error!native_reply.Result {
    const stopping = g.stops.strings.len > 0;
    if (!g.chat) {
        // the raw text: the generation's own incremental decode while it streams, decoded once when it is done
        const raw = if (finished) try g.srv.text.decode(g.a, ids) else g.visible.text.items;
        return .{ .content = g.stops.visible(raw, !finished), .reasoning = "", .calls = null, .reasoning_tokens = 0 };
    }
    var whole: parser.Stream = undefined;
    const s: *parser.Stream = if (g.inkling2) |*p| p else blk: {
        whole = try parser.Stream.init(g.a, g.srv.text);
        break :blk &whole;
    };
    if (stopping and s.next == 0) s.track_raw = true;
    var result = try s.extend(ids, g.tools);
    if (finished) result = try s.done();
    if (stopping and s.raw_ok) {
        // one coordinate: a stop is found in the lossy decode the generation checks, then placed in the exact bytes
        const raw = s.raw.items;
        const decoded = try s.decoded.view(g.a, raw, finished);
        const text = decoded.text.items;
        const stop = firstStop(g.stops.strings, text);
        if (stop orelse (if (finished) null else held(g.stops.strings, text))) |found| if (found < text.len) {
            var part = (try cut(g, ids, decoded.exact(raw.len, found))).?;
            // usage counts every generated id: a stop cuts what is shown, never what was made
            const tokens = result.reasoning_tokens;
            result = try part.result(finished);
            result.reasoning_tokens = tokens;
        };
    } else if (stopping) {
        // no token-local bytes: the raw decode places the stop, and the parsed channels are cut by it
        const raw = if (finished) try g.srv.text.decode(g.a, ids) else g.visible.text.items;
        if (firstStop(g.stops.strings, raw) != null) {
            result.content = g.stops.visible(result.content, false);
            result.reasoning = g.stops.visible(result.reasoning, false);
        } else if (!finished) {
            result.content = g.stops.visible(result.content, true);
            result.reasoning = g.stops.visible(result.reasoning, true);
        }
    }
    // a bare reply has no native calls; a tool grammar writes the generic envelope, which the shared parser reads
    if (finished and s.bare() and g.tools.len > 0) try bareCalls(g, &result);
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
