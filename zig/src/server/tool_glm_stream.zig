//! GLM's ``<tool_call>NAME<arg_key>K...`` calls as written; a later unclosed or wordless key can undo a sent repair.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const key_open = "<arg_key>";
const key_close = "</arg_key>";
const value_open = "<arg_value>";
const value_close = "</arg_value>";
const call_close = "</tool_call>";
const space = reply_text.isSpaceByte;

/// What the Streamer does after the GLM state machine ran: go on, wait for text, or send the call whole.
pub const Next = enum { more, wait, unstreamable, outside };

/// A GLM call's place: the header's decision, or one of its key-value pairs.
pub const Stage = enum { decide, keys, key, value };

/// The GLM call the Streamer is inside; every field is an index into the reply text, resumed and never rescanned.
pub const Glm = struct {
    base: usize = 0, // the payload's first byte, after ``<tool_call>``
    scan: usize = 0, // where the header's tag scan resumes
    counted: usize = 0, // where the lone closers' tag counts resume
    opens: usize = 0, // ``<arg_key>``s before ``counted``
    closes: usize = 0, // ``</arg_key>``s before ``counted``
    stage: Stage = .decide,
    seg: usize = 0, // the last pair's end: a repaired key's word starts at or after it
    key_text: usize = 0, // the key's first byte
    key_from: usize = 0, // where the ``</arg_key>`` search resumes
    value_from: usize = 0, // the value's first byte
    value_scanned: usize = 0, // no ``</arg_value>`` starts before here
    value_sent: usize = 0, // the value's next byte to send
};

/// The reply's ``text`` so far: the deltas the open GLM call adds, and what the Streamer should do next.
pub fn feed(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    while (true) {
        const next = switch (s.glm.stage) {
            .decide => try decide(s, text, out),
            .keys => try keys(s, text, out),
            .key => try key(s, text, out),
            .value => try value(s, text, out),
        };
        if (next != .more) return next;
    }
}

/// The cut the reply's end left inside a value: the bytes held back for a closer that never came.
pub fn cut(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
    if (s.glm.stage != .value) return;
    const end = text.len - reply_text.partialTag(text[@min(s.glm.value_from, text.len)..], value_close);
    try sendValue(s, text, end, out);
    if (s.typed) try out.append(s.a, try s.args(s.buffer.items)); // the value's JSON never closed
    s.pos = end;
}

/// The value's bytes from ``value_sent`` to ``end``: out escaped, or into the buffer a typed value decodes.
fn sendValue(s: anytype, text: []const u8, end: usize, out: *std.ArrayList(Value)) Allocator.Error!void {
    if (end <= s.glm.value_sent) return;
    if (s.typed) try s.buffer.appendSlice(s.a, text[s.glm.value_sent..end]) else try out.append(s.a, try s.args(try s.esc(text[s.glm.value_sent..end])));
    s.glm.value_sent = end;
}

/// The header's decision: the name before the first ``<arg_key>`` streams when it names an offered tool.
fn decide(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    var at = @max(s.glm.scan, s.glm.base);
    while (true) {
        const i = std.mem.indexOfScalarPos(u8, text, at, '<') orelse {
            s.glm.scan = text.len;
            if (reply_text.charCount(text[@min(s.glm.base, text.len)..]) > 256) return .unstreamable;
            return .wait;
        };
        const rest = text[i..];
        if (std.mem.startsWith(u8, rest, key_open)) {
            const raw = reply_text.pyStrip(text[@min(s.glm.base, text.len)..i]);
            if (raw.len == 0 or std.mem.indexOfAny(u8, raw, "<>\n\r") != null) return .unstreamable;
            for ([_][]const u8{ "{", "[", "call:", ":" }) |lead| if (std.mem.startsWith(u8, raw, lead)) return .unstreamable;
            const name = s.known.get(try std.ascii.allocLowerString(s.a, raw)) orelse return .unstreamable;
            try s.begin(name, out);
            s.state = .glm;
            s.glm.key_text = i + key_open.len;
            s.glm.key_from = i + key_open.len;
            s.pos = i + key_open.len;
            s.glm.stage = .key;
            return .more;
        }
        if (std.mem.startsWith(u8, rest, key_close)) return .unstreamable; // repairGlmKeys' case
        if (std.ascii.startsWithIgnoreCase(rest, call_close)) return .unstreamable; // no argument: its closer parses it
        if (prefix(rest, key_open) or prefix(rest, key_close) or prefixIgnoreCase(rest, call_close)) {
            s.glm.scan = i;
            return .wait;
        }
        at = i + 1;
    }
}

/// Between pairs: the next ``<arg_key>``, a lone ``</arg_key>`` repairGlmKeys reads as a key, or the envelope's end.
fn keys(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    var at = @min(s.pos, text.len);
    while (true) {
        const i = std.mem.indexOfScalarPos(u8, text, at, '<') orelse {
            s.pos = text.len;
            return .wait;
        };
        const rest = text[i..];
        if (std.mem.startsWith(u8, rest, key_open)) {
            s.glm.key_text = i + key_open.len;
            s.glm.key_from = i + key_open.len;
            s.pos = i + key_open.len;
            s.glm.stage = .key;
            return .more;
        }
        if (std.mem.startsWith(u8, rest, key_close)) {
            const word = if (loneKeyRepaired(s, text, i)) repairWord(s, text, i) else null;
            if (word) |w| {
                s.glm.key_text = w;
                s.glm.key_from = w;
                s.pos = w;
                s.glm.stage = .key;
                return .more;
            }
            at = i + key_close.len; // no repair to make, or nothing to put back: neither reply makes a pair of it
            continue;
        }
        if (std.ascii.startsWithIgnoreCase(rest, call_close)) return endCall(s, i, out);
        if (prefix(rest, key_open) or prefix(rest, key_close) or prefixIgnoreCase(rest, call_close)) {
            s.pos = i;
            return .wait;
        }
        at = i + 1;
    }
}

/// Whether ``repairGlmKeys`` repairs the lone ``</arg_key>`` at ``i``: closers outnumber openers through it.
fn loneKeyRepaired(s: anytype, text: []const u8, i: usize) bool {
    if (s.glm.counted < s.glm.base) s.glm.counted = s.glm.base; // nothing is counted before the payload
    s.glm.opens += std.mem.count(u8, text[s.glm.counted..i], key_open);
    s.glm.closes += std.mem.count(u8, text[s.glm.counted..i], key_close);
    s.glm.counted = i;
    return s.glm.closes + 1 > s.glm.opens;
}

/// The envelope's closer ends the call: its arguments' JSON closes, or is ``{}`` when it made no pair.
fn endCall(s: anytype, at: usize, out: *std.ArrayList(Value)) Allocator.Error!Next {
    try out.append(s.a, try s.args(if (s.args_open) "}" else "{}"));
    s.pos = at + call_close.len;
    s.searched = s.pos;
    s.state = .outside;
    return .outside;
}

/// Where the word repairGlmKeys puts back ahead of the lone ``</arg_key>`` at ``i`` starts, or null.
fn repairWord(s: anytype, text: []const u8, i: usize) ?usize {
    var e = i;
    while (e > s.glm.seg and space(text[e - 1])) e -= 1;
    var w = e;
    while (w > s.glm.seg and !space(text[w - 1])) w -= 1;
    return if (w < e) w else null;
}

/// A pair's key: its ``</arg_key>`` is the first one an ``<arg_value>`` follows, as ``glmCall`` finds it.
fn key(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    var from = @min(s.glm.key_from, text.len);
    while (true) {
        const ke: usize = std.mem.indexOfPos(u8, text, from, key_close) orelse {
            const ce = std.ascii.findIgnoreCasePos(text, from, call_close) orelse {
                s.glm.key_from = @max(from, text.len -| (call_close.len - 1));
                return .wait;
            };
            return endCall(s, ce, out); // the envelope ends before this key's value
        };
        if (std.ascii.findIgnoreCasePos(text, from, call_close)) |ce| if (ce < ke) return endCall(s, ce, out);
        var j = ke + key_close.len;
        while (j < text.len and space(text[j])) j += 1;
        if (j == text.len) {
            s.glm.key_from = ke; // its value's opener may still come
            return .wait;
        }
        if (std.mem.startsWith(u8, text[j..], value_open)) {
            const name = reply_text.pyStrip(text[s.glm.key_text..ke]);
            s.value_schema = try tool_params.schemaOf(&s.schemas, s.current, name, s.a);
            s.typed = try tool_params.schemaTakesJson(s.a, s.value_schema);
            s.buffer.clearRetainingCapacity();
            const quoted = try json.quote(s.a, name, .{ .ascii = false });
            try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (s.typed) ":" else ":\"" })));
            s.args_open = true;
            s.glm.value_from = j + value_open.len;
            s.glm.value_scanned = s.glm.value_from;
            s.glm.value_sent = s.glm.value_from;
            s.pos = s.glm.value_from;
            s.glm.stage = .value;
            return .more;
        }
        if (std.mem.startsWith(u8, value_open, text[j..])) {
            s.glm.key_from = ke; // the opener is not whole yet
            return .wait;
        }
        from = ke + 1; // this closer has no value: the key runs on to the next one, as glmCall's does
    }
}

/// A pair's value: a string's JSON bytes as they arrive, or the JSON of a typed value at its closer.
fn value(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const from = @max(@min(s.glm.value_from, text.len), s.glm.value_scanned -| (value_close.len - 1));
    const found = std.mem.indexOfPos(u8, text, from, value_close);
    const end = if (found) |e| e else text.len - reply_text.partialTag(text[@min(s.glm.value_from, text.len)..], value_close);
    try sendValue(s, text, end, out);
    if (found == null) {
        s.glm.value_scanned = text.len;
        s.pos = end;
        return .wait;
    }
    if (s.typed) {
        const v = try tool_params.decode(s.a, s.buffer.items, s.value_schema, false);
        try out.append(s.a, try s.args(try json.stringify(s.a, v, .{ .ascii = false, .compact = true })));
    } else try out.append(s.a, try s.args("\""));
    s.pos = found.? + value_close.len;
    s.glm.seg = s.pos;
    s.glm.stage = .keys;
    return .more;
}

/// ``rest`` is the start of ``tag`` but not all of it.
fn prefix(rest: []const u8, tag: []const u8) bool {
    return rest.len < tag.len and std.mem.startsWith(u8, tag, rest);
}

/// ``prefix`` of a tag whose case does not matter, in any case.
fn prefixIgnoreCase(rest: []const u8, tag: []const u8) bool {
    return rest.len > 0 and rest.len < tag.len and std.ascii.startsWithIgnoreCase(tag, rest);
}
