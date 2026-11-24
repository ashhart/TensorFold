//! Inkling 2's full native reply, split by control-token identity rather than text.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const native_reply = @import("native_reply.zig");
const tool_specs = @import("tool_specs.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
pub const Result = native_reply.Result;

const Kind = enum {
    message_model,
    message_other, // a user, system or tool header: never the model's turn
    thinking,
    content_text,
    invoke_tool_text,
    tool_error,
    invoke_tool_json,
    end_message,
    end_sampling,
};

const Control = struct { kind: Kind, spelling: []const u8 };
const controls = [_]Control{
    .{ .kind = .message_model, .spelling = "<|message_model|>" },
    .{ .kind = .message_other, .spelling = "<|message_user|>" },
    .{ .kind = .message_other, .spelling = "<|message_system|>" },
    .{ .kind = .message_other, .spelling = "<|message_tool|>" },
    .{ .kind = .thinking, .spelling = "<|content_thinking|>" },
    .{ .kind = .content_text, .spelling = "<|content_text|>" },
    .{ .kind = .invoke_tool_text, .spelling = "<|content_invoke_tool_text|>" },
    .{ .kind = .tool_error, .spelling = "<|content_tool_error|>" },
    .{ .kind = .invoke_tool_json, .spelling = "<|content_invoke_tool_json|>" },
    .{ .kind = .end_message, .spelling = "<|end_message|>" },
    .{ .kind = .end_sampling, .spelling = "<|content_model_end_sampling|>" },
};
const Resolved = [controls.len]?u32;
const State = enum { none, thinking, content, tool_json, foreign };

fn resolve(a: Allocator, text: model_text.Text) Allocator.Error!Resolved {
    var out: Resolved = undefined;
    for (&out) |*entry| entry.* = null;
    for (controls, 0..) |control, i| {
        const id = text.tokenId(control.spelling) orelse continue;
        if (!text.isAdded(id)) continue;
        const vocabulary_spelling = try text.tokenString(a, id);
        defer a.free(vocabulary_spelling);
        if (std.mem.eql(u8, vocabulary_spelling, control.spelling)) out[i] = id;
    }
    return out;
}

fn controlKind(text: model_text.Text, ids: Resolved, id: u32) ?Kind {
    if (!text.isAdded(id)) return null;
    for (ids, controls) |control_id, control| if (control_id == id) return control.kind;
    return null;
}

const replacement = "\u{fffd}";
const unicode = @import("tokenizer").unicode;

/// `b` as the tokenizer's lossy decode reads it (`unicode.scan`), each piece and its offset into `sink`; returns used.
fn lossy(b: []const u8, done: bool, sink: anytype) Allocator.Error!usize {
    var i: usize = 0;
    while (i < b.len) {
        const step = unicode.scan(b, i);
        if (step.cp != null) {
            try sink.emit(i, b[i..][0..step.len]);
        } else if (!done and i + step.len == b.len and b[i] >= 0xC2 and b[i] <= 0xF4) {
            break; // a character still arriving
        } else try sink.emit(i, replacement);
        i += step.len;
    }
    return i;
}

/// A channel's whole UTF-8 characters, a tail waiting past dropped tokens; one U+FFFD per maximal invalid subpart.
const Channel = struct {
    text: std.ArrayList(u8) = .empty,
    tail: std.ArrayList(u8) = .empty,

    const Sink = struct {
        a: Allocator,
        out: *std.ArrayList(u8),
        fn emit(k: @This(), _: usize, piece: []const u8) Allocator.Error!void {
            try k.out.appendSlice(k.a, piece);
        }
    };

    fn settle(c: *Channel, a: Allocator, b: []const u8, done: bool) Allocator.Error!usize {
        return lossy(b, done, Sink{ .a = a, .out = &c.text });
    }

    fn add(c: *Channel, a: Allocator, bytes: []const u8) Allocator.Error!void {
        try c.tail.appendSlice(a, bytes);
        const b = c.tail.items;
        const i = try c.settle(a, b, false);
        std.mem.copyForwards(u8, b[0 .. b.len - i], b[i..]);
        c.tail.shrinkRetainingCapacity(b.len - i);
    }

    /// Whole characters while the reply grows; once it is done, an unfinished character reads as one replacement.
    fn view(c: *const Channel, a: Allocator, done: bool) Allocator.Error![]const u8 {
        if (!done or c.tail.items.len == 0) return c.text.items;
        var out: Channel = .{};
        try out.text.appendSlice(a, c.text.items);
        _ = try out.settle(a, c.tail.items, true);
        return out.text.items;
    }

    fn clear(c: *Channel) void {
        c.text.clearRetainingCapacity();
        c.tail.clearRetainingCapacity();
    }
};

fn finishTool(a: Allocator, payload: []const u8, tools: []const Value, calls: *std.ArrayList(Value), rejected: *usize) Allocator.Error!void {
    const document = switch (try json.parse(a, payload)) {
        .ok => |value| value,
        .err => {
            rejected.* += 1;
            return;
        },
    };
    if (document != .object) {
        rejected.* += 1;
        return;
    }
    const name_value = document.get("name") orelse {
        rejected.* += 1;
        return;
    };
    const arguments = document.get("args") orelse {
        rejected.* += 1;
        return;
    };
    if (name_value != .string or arguments != .object) {
        rejected.* += 1;
        return;
    }
    const name = try a.dupe(u8, name_value.string); // Calls outlive the reusable payload.
    var offered = false;
    for (tools) |tool| {
        if (std.mem.eql(u8, name, try tool_specs.toolName(a, tool))) {
            offered = true;
            break;
        }
    }
    if (!offered) {
        rejected.* += 1;
        return;
    }

    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    try function.put(a, "arguments", .{ .string = try json.stringify(a, arguments, .{ .ascii = false, .compact = true }) });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try std.fmt.allocPrint(a, "call_{d}", .{calls.items.len}) });
    try call.put(a, "type", .{ .string = "function" });
    try call.put(a, "function", .{ .object = function });
    try calls.append(a, .{ .object = call });
}

/// The reply's raw text as its lossy decode reads it, controls spelled, and where each piece's exact bytes begin.
pub const RawText = struct {
    text: std.ArrayList(u8) = .empty,
    dec: std.ArrayList(usize) = .empty, // each piece's offset in `text`
    src: std.ArrayList(usize) = .empty, // and its exact bytes' offset in the raw bytes
    settled: usize = 0,

    const Sink = struct {
        a: Allocator,
        r: *RawText,
        base: usize,
        fn emit(k: @This(), at: usize, piece: []const u8) Allocator.Error!void {
            try k.r.dec.append(k.a, k.r.text.items.len);
            try k.r.src.append(k.a, k.base + at);
            try k.r.text.appendSlice(k.a, piece);
        }
    };

    fn add(r: *RawText, a: Allocator, raw: []const u8) Allocator.Error!void {
        r.settled += try lossy(raw[r.settled..], false, Sink{ .a = a, .r = r, .base = r.settled });
    }

    /// The decoded text; once the reply is done, an unfinished character at its end reads as replacements.
    pub fn view(r: *const RawText, a: Allocator, raw: []const u8, done: bool) Allocator.Error!RawText {
        if (!done or r.settled == raw.len) return r.*;
        var out: RawText = .{ .settled = r.settled };
        try out.text.appendSlice(a, r.text.items);
        try out.dec.appendSlice(a, r.dec.items);
        try out.src.appendSlice(a, r.src.items);
        _ = try lossy(raw[r.settled..], true, Sink{ .a = a, .r = &out, .base = r.settled });
        return out;
    }

    /// The exact-byte offset of decoded offset `at`, which starts a piece: a stop and its prefixes start on one.
    pub fn exact(r: *const RawText, raw_len: usize, at: usize) usize {
        if (at >= r.text.items.len) return raw_len;
        var lo: usize = 0;
        var hi: usize = r.dec.items.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (r.dec.items[mid] <= at) lo = mid else hi = mid;
        }
        return r.src.items[lo];
    }
};

/// Owns the reply parser between growing token prefixes; results borrow its arena-backed lists.
pub const Stream = struct {
    a: Allocator,
    text: model_text.Text,
    resolved: Resolved,
    content: Channel = .{},
    reasoning: Channel = .{},
    tool_payload: Channel = .{},
    calls: std.ArrayList(Value) = .empty,
    state: State = .none,
    reasoning_tokens: usize = 0, // every id from a thinking block's header through its close
    header: usize = 0, // ids of the current message header so far, which a thinking block counts as its own
    messages: usize = 0, // message headers seen; text before the first is a bare reply, as a grammar holds one
    foreign: bool = false, // the newest header is a user, system or tool turn
    rejected: usize = 0, // tool calls refused: malformed, unoffered, or left open by a new header
    next: usize = 0,
    raw: std.ArrayList(u8) = .empty, // the reply's exact token bytes, controls spelled, kept for stop strings
    track_raw: bool = false,
    raw_ok: bool = true, // every token has token-local bytes; false: the raw text cannot place a stop
    decoded: RawText = .{}, // `raw` as the lossy decode reads it: stops are found there, then placed in `raw`

    pub fn init(a: Allocator, text: model_text.Text) Allocator.Error!Stream {
        return .{ .a = a, .text = text, .resolved = try resolve(a, text) };
    }

    pub fn result(s: *const Stream, finished: bool) Allocator.Error!Result {
        return .{
            .content = try s.content.view(s.a, finished),
            .reasoning = try s.reasoning.view(s.a, finished),
            .calls = if (s.calls.items.len == 0) null else s.calls.items,
            .reasoning_tokens = s.reasoning_tokens,
            .rejected = s.rejected > 0 or s.state == .tool_json,
            .rejected_calls = s.rejected + @intFromBool(s.state == .tool_json),
            .open_think = s.state == .thinking,
        };
    }

    /// The finished reply: an unfinished character at a channel's end reads as replacements.
    pub fn done(s: *const Stream) Allocator.Error!Result {
        return s.result(true);
    }

    /// No message header has arrived: the reply is bare text, as a grammar from its first token writes.
    pub fn bare(s: *const Stream) bool {
        return s.messages == 0;
    }

    /// Neither a listed control nor any other special token of the tokenizer: text a block holds.
    pub fn payload(s: *const Stream, id: u32) bool {
        return controlKind(s.text, s.resolved, id) == null and !s.text.isSpecial(id);
    }

    fn channel(s: *Stream) ?*Channel {
        return switch (s.state) {
            .thinking => &s.reasoning,
            .content => &s.content,
            .tool_json => &s.tool_payload,
            .none => if (s.bare()) &s.content else null,
            .foreign => null,
        };
    }

    /// Payload ids' bytes, token by token, so a character split around a dropped token stays whole.
    fn add(s: *Stream, ch: *Channel, run: []const u32) Allocator.Error!void {
        for (run, 0..) |id, i| {
            const bytes = (try s.text.tokenBytes(s.a, id)) orelse {
                // a decoder whose tokens' bytes depend on their neighbours: the rest of the run decodes as one
                const decoded = try s.text.decode(s.a, run[i..]);
                return ch.add(s.a, decoded);
            };
            try ch.add(s.a, bytes);
        }
    }

    /// The first `len` bytes of payload token `id`, where a stop string cuts the reply inside it.
    pub fn partial(s: *Stream, id: u32, bytes: []const u8) Allocator.Error!void {
        if (!s.payload(id)) return;
        if (s.channel()) |ch| try ch.add(s.a, bytes);
    }

    /// Append `ids`' exact bytes to the raw text; a token without token-local bytes ends it.
    fn keepRaw(s: *Stream, ids: []const u32) Allocator.Error!void {
        for (ids) |id| {
            if (!s.raw_ok) return;
            if (try s.text.tokenBytes(s.a, id)) |bytes| try s.raw.appendSlice(s.a, bytes) else s.raw_ok = false;
        }
        if (s.raw_ok) try s.decoded.add(s.a, s.raw.items);
    }

    /// Consume only the new suffix; ids is a monotonically growing full prefix, and EOS stripping must never shrink it.
    pub fn extend(s: *Stream, ids: []const u32, tools: []const Value) Allocator.Error!Result {
        std.debug.assert(ids.len >= s.next);
        const from = s.next;
        while (s.next < ids.len) {
            const id = ids[s.next];
            const kind = controlKind(s.text, s.resolved, id);
            if (kind == .message_model or kind == .message_other) {
                // every role header is a hard boundary: it closes an open block, and a call in it never finished
                if (s.state == .tool_json) s.rejected += 1;
                s.state = .none;
                s.messages += 1;
                s.header = 1;
                s.foreign = kind == .message_other;
                s.next += 1;
                continue;
            }
            if (s.state == .thinking) s.reasoning_tokens += 1;
            if (kind == .end_sampling) {
                // closes nothing: an open channel stays reported, and with ignore_eos the reply goes on
                s.next += 1;
                continue;
            }
            if (kind == .end_message) {
                if (s.state == .tool_json) try finishTool(s.a, try s.tool_payload.view(s.a, true), tools, &s.calls, &s.rejected);
                s.state = .none;
                s.header = 0;
                s.next += 1;
                continue;
            }
            if (s.state == .none) if (kind) |found| {
                s.state = if (s.foreign) .foreign else switch (found) {
                    .thinking => .thinking,
                    .content_text, .invoke_tool_text, .tool_error => .content,
                    .invoke_tool_json => .tool_json,
                    else => .none,
                };
                if (s.state == .thinking) s.reasoning_tokens += s.header + 1;
                if (s.state == .tool_json) s.tool_payload.clear();
                s.header = 0;
                s.next += 1;
                continue;
            };
            const ch = s.channel() orelse {
                // a header's name, a foreign turn's text, or text between blocks: never the reply's
                if (s.state == .none and s.header > 0) s.header += 1;
                s.next += 1;
                continue;
            };
            if (!s.payload(id)) {
                // a control or any other special token inside a block is never payload
                s.next += 1;
                continue;
            }
            const start = s.next;
            s.next += 1;
            while (s.next < ids.len and s.payload(ids[s.next])) s.next += 1;
            if (s.state == .thinking) s.reasoning_tokens += s.next - start - 1;
            try s.add(ch, ids[start..s.next]);
        }
        if (s.track_raw) try s.keepRaw(ids[from..s.next]);
        return s.result(false);
    }
};

/// Parse a whole reply through the same cursor used by streamed replies.
pub fn parse(a: Allocator, text: model_text.Text, ids: []const u32, tools: []const Value) Allocator.Error!Result {
    var stream = try Stream.init(a, text);
    _ = try stream.extend(ids, tools);
    return stream.done();
}
