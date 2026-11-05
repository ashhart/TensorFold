//! DSML invokes as written: a call goes out once its first parameter is whole in the block's spelling, then streams.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_dsml = @import("tool_dsml.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const string_flag = "\" string=\"true\">";
const json_flag = "\" string=\"false\">";

/// What the Streamer does after the DSML state machine ran: go on, wait for text, or parse the block at its closer.
pub const Next = enum { more, wait, rejected, outside };

/// A DSML block's place: between invokes, in a name, between parameters, in a key, or in a value.
pub const Stage = enum { invokes, name, params, key, value };

/// The DSML block the Streamer is inside; every index is into the reply text, resumed and never rescanned.
pub const Dsml = struct {
    sp: *const tool_dsml.Spelling = &tool_dsml.spellings[0],
    start: usize = 0, // the block opener's first byte
    stage: Stage = .invokes,
    at: usize = 0, // where the stage's reading resumes
    from: usize = 0, // the open name's, key's or value's first byte
    name: []const u8 = "", // the open invoke's offered name
    header: bool = false, // the open invoke's header went out
    params: usize = 0, // the open invoke's whole parameters
    sent: usize = 0, // the block's headers that went out
    typed: bool = false, // the open value is JSON (``string="false"``)
    value_sent: usize = 0, // the open value's next byte to send
    held: std.ArrayList(u8) = .empty, // the open invoke's arguments, written before its header
};

/// The reply's ``text`` so far: the deltas the open DSML block adds, and what the Streamer should do next.
pub fn feed(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    while (true) {
        const next = switch (s.dsml.stage) {
            .invokes => invokes(s, text),
            .name => try name(s, text),
            .params => try params(s, text, out),
            .key => try key(s, text, out),
            .value => try value(s, text, out),
        };
        if (next != .more) return next;
    }
}

/// The cut the reply's end left inside a streamed value: the bytes held back for a closer that never came.
pub fn cut(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
    const d = &s.dsml;
    if (d.stage != .value or !d.header) return;
    const end = text.len - closerTail(text[@min(d.from, text.len)..], d.sp);
    try sendValue(s, text, end, out);
    if (d.typed) try send(s, out, try s.a.dupe(u8, s.buffer.items)); // the value's JSON never closed
    s.pos = end;
}

/// Between invokes: the next whole invoke opener, or the block's closer once it made a call.
fn invokes(s: anytype, text: []const u8) Next {
    const d = &s.dsml;
    const i = word(text, &d.at) orelse return .wait;
    const open = tagAt(text, i, d.sp.invoke_open);
    const close = tagAt(text, i, d.sp.close);
    if (open == .whole) {
        d.from = i + d.sp.invoke_open.len;
        d.at = d.from;
        d.stage = .name;
        return .more;
    }
    if (close == .whole) {
        if (d.sent == 0) return .rejected; // a block without a call
        s.pos = i + d.sp.close.len;
        s.searched = s.pos;
        s.state = .outside;
        return .outside;
    }
    return if (open == .part or close == .part) .wait else .rejected;
}

/// An invoke's name: the bytes to the first quote, which ``>`` must follow, naming an offered tool.
fn name(s: anytype, text: []const u8) Allocator.Error!Next {
    const d = &s.dsml;
    const q = std.mem.indexOfScalarPos(u8, text, d.at, '"');
    if (has(text, d.from, d.at, q orelse text.len, d.sp.close)) return .rejected; // the block closed in the name
    const at = q orelse {
        d.at = text.len;
        return .wait;
    };
    d.at = at;
    if (at + 1 >= text.len) return .wait;
    if (text[at + 1] != '>') return .rejected;
    const written = reply_text.pyStrip(text[d.from..at]);
    d.name = s.known.get(try std.ascii.allocLowerString(s.a, written)) orelse return .rejected;
    d.header = false;
    d.params = 0;
    d.held.clearRetainingCapacity();
    s.args_open = false;
    d.at = at + 2;
    d.stage = .params;
    return .more;
}

/// Between parameters: the next whole parameter opener, or the invoke's closer.
fn params(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const d = &s.dsml;
    const i = word(text, &d.at) orelse return .wait;
    const param = tagAt(text, i, d.sp.param_open);
    const close = tagAt(text, i, d.sp.invoke_close);
    if (param == .whole) {
        if (d.params > 0) try decide(s, out); // a whole parameter, then this spelling's next opener
        d.from = i + d.sp.param_open.len;
        d.at = d.from;
        d.stage = .key;
        return .more;
    }
    if (close == .whole) {
        try decide(s, out);
        try send(s, out, if (s.args_open) "}" else "{}");
        d.at = i + d.sp.invoke_close.len;
        d.stage = .invokes;
        return .more;
    }
    return if (param == .part or close == .part) .wait else .rejected;
}

/// A parameter's key: the bytes to the first quote, then its ``string`` flag; its JSON key goes out.
fn key(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const d = &s.dsml;
    const q = std.mem.indexOfScalarPos(u8, text, d.at, '"');
    const end = q orelse text.len;
    if (has(text, d.from, d.at, end, d.sp.invoke_close) or has(text, d.from, d.at, end, d.sp.close)) return .rejected;
    const at = q orelse {
        d.at = text.len;
        return .wait;
    };
    d.at = at;
    const flag = text[at..];
    const typed = std.mem.startsWith(u8, flag, json_flag);
    if (!typed and !std.mem.startsWith(u8, flag, string_flag)) {
        return if (std.mem.startsWith(u8, string_flag, flag) or std.mem.startsWith(u8, json_flag, flag)) .wait else .rejected;
    }
    const quoted = try json.quote(s.a, text[d.from..at], .{ .ascii = false });
    try send(s, out, try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (typed) ":" else ":\"" }));
    s.args_open = true;
    d.typed = typed;
    d.from = at + (if (typed) json_flag.len else string_flag.len);
    d.at = d.from;
    d.value_sent = d.from;
    s.buffer.clearRetainingCapacity();
    d.stage = .value;
    return .more;
}

/// A parameter's value: a string's JSON bytes as they arrive, or a JSON value whole at its closer.
fn value(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const d = &s.dsml;
    const from = @max(d.from, d.at -| (longest(d.sp) - 1));
    const end = std.mem.indexOfPos(u8, text, from, d.sp.param_close);
    const before = end orelse text.len;
    // the invoke or the block closing before the parameter does: tool_dsml rejects it
    if (has(text, from, from, before, d.sp.invoke_close) or has(text, from, from, before, d.sp.close)) return .rejected;
    const e = end orelse {
        try sendValue(s, text, text.len - closerTail(text[@min(d.from, text.len)..], d.sp), out);
        d.at = text.len;
        return .wait;
    };
    try sendValue(s, text, e, out);
    if (d.typed) {
        const v = switch (try json.parseText(s.a, s.buffer.items)) {
            .ok => |v| v,
            .err => return .rejected,
        };
        try send(s, out, try json.stringify(s.a, v, .{ .ascii = false, .compact = true }));
    } else try send(s, out, "\"");
    d.params += 1;
    d.at = e + d.sp.param_close.len;
    d.stage = .params;
    return .more;
}

/// The open invoke's header, once, then the arguments written before it as one piece.
fn decide(s: anytype, out: *std.ArrayList(Value)) Allocator.Error!void {
    const d = &s.dsml;
    if (d.header) return;
    const args_open = s.args_open;
    try s.begin(d.name, out);
    s.args_open = args_open;
    d.header = true;
    d.sent += 1;
    if (d.held.items.len > 0) try out.append(s.a, try s.args(try s.a.dupe(u8, d.held.items)));
    d.held.clearRetainingCapacity();
}

/// An arguments piece: out under the open header, else held until the header goes out.
fn send(s: anytype, out: *std.ArrayList(Value), piece: []const u8) Allocator.Error!void {
    if (s.dsml.header) try out.append(s.a, try s.args(piece)) else try s.dsml.held.appendSlice(s.a, piece);
}

/// The value's bytes from ``value_sent`` to ``end``: escaped into the arguments, or into the buffer JSON decodes.
fn sendValue(s: anytype, text: []const u8, end: usize, out: *std.ArrayList(Value)) Allocator.Error!void {
    const d = &s.dsml;
    if (end <= d.value_sent) return;
    if (d.typed) try s.buffer.appendSlice(s.a, text[d.value_sent..end]) else try send(s, out, try s.esc(text[d.value_sent..end]));
    d.value_sent = end;
}

/// The first byte at or after ``at`` that is not whitespace, ``at`` moved to it; null while none is whole yet.
fn word(text: []const u8, at: *usize) ?usize {
    var i = at.*;
    defer at.* = i;
    while (i < text.len) {
        const n = reply_text.spaceAt(text, i);
        if (n > 0) {
            i += n;
            continue;
        }
        const full = if (text[i] < 0x80) 1 else std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        return if (i + full > text.len) null else i; // a cut sequence may still be whitespace
    }
    return null;
}

const Tag = enum { whole, part, no };

/// Whether ``text`` at ``i`` is ``tag`` whole, all of the text so far is its start, or neither.
fn tagAt(text: []const u8, i: usize, tag: []const u8) Tag {
    const rest = text[i..];
    if (std.mem.startsWith(u8, rest, tag)) return .whole;
    return if (rest.len < tag.len and std.mem.startsWith(u8, tag, rest)) .part else .no;
}

/// Whether ``tag`` lies whole in ``text[from..end]``, searched from ``at`` back by its length.
fn has(text: []const u8, from: usize, at: usize, end: usize, tag: []const u8) bool {
    const lo = @max(from, at -| (tag.len - 1));
    return lo < end and std.mem.indexOf(u8, text[lo..end], tag) != null;
}

/// The longest closer a value watches for.
fn longest(sp: *const tool_dsml.Spelling) usize {
    return @max(sp.param_close.len, @max(sp.invoke_close.len, sp.close.len));
}

/// How many bytes at the end of a value could begin one of the closers it watches for.
fn closerTail(written: []const u8, sp: *const tool_dsml.Spelling) usize {
    var tail: usize = 0;
    for ([_][]const u8{ sp.param_close, sp.invoke_close, sp.close }) |tag| tail = @max(tail, reply_text.partialTag(written, tag));
    return tail;
}
