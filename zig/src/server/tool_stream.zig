//! Tool calls while they are written: XML calls as argument deltas, and the one-call policy's text filter.
const std = @import("std");
const json = @import("json.zig");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const space = reply_text.isSpaceByte;

/// ``\s*<TAG=([^>\n]+)>\n?`` at the start of ``rest``: the name and the bytes matched.
fn opening(rest: []const u8, tag: []const u8) ?struct { name: []const u8, len: usize, newline: bool } {
    var i: usize = 0;
    while (i < rest.len and space(rest[i])) i += 1;
    if (!std.mem.startsWith(u8, rest[i..], tag)) return null;
    i += tag.len;
    const start = i;
    while (i < rest.len and rest[i] != '>' and rest[i] != '\n') i += 1;
    if (i == start or i >= rest.len or rest[i] != '>') return null;
    const name = rest[start..i];
    i += 1;
    const newline = i < rest.len and rest[i] == '\n';
    return .{ .name = name, .len = i + @intFromBool(newline), .newline = newline };
}

/// ``\s*TEXT`` at the start of ``rest``: bytes matched.
fn closing(rest: []const u8, text: []const u8) ?usize {
    var i: usize = 0;
    while (i < rest.len and space(rest[i])) i += 1;
    return if (std.mem.startsWith(u8, rest[i..], text)) i + text.len else null;
}

/// Streams Qwen's XML calls as JSON argument deltas matching the final parser.
pub const Streamer = struct {
    a: Allocator,
    schemas: tool_params.Schemas,
    known: std.StringHashMapUnmanaged([]const u8) = .empty,
    current: []const u8 = "",
    value_schema: Value = .null,
    typed: bool = false,
    buffer: std.ArrayList(u8) = .empty,
    pos: usize = 0,
    state: enum { outside, call, function, value, closing, abort } = .outside,
    index: i64 = -1,
    args_open: bool = false,
    lead: bool = false,
    held: []const u8 = "",
    streamed: bool = false,
    searched: usize = 0, // the answer is only ever extended, so prose already searched holds no opener

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!Streamer {
        var s: Streamer = .{ .a = a, .schemas = try tool_params.schemas(a, tools) };
        for (tools) |tool| {
            if (tool != .object) continue;
            const spec = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
            const name = if (spec.get("name")) |n| (if (n.truthy()) try tool_specs.pyStr(a, n) else "") else "";
            if (name.len == 0) continue;
            try s.known.put(a, try std.ascii.allocLowerString(a, name), name);
        }
        return s;
    }

    fn args(s: *Streamer, fragment: []const u8) Allocator.Error!Value {
        const function = try json.newObject(s.a);
        try function.put(s.a, "arguments", .{ .string = fragment });
        const one = try json.newObject(s.a);
        try one.put(s.a, "index", try json.intValue(s.a, s.index));
        try one.put(s.a, "function", .{ .object = function });
        return wrap(s.a, .{ .object = one });
    }

    fn esc(s: *Streamer, text: []const u8) Allocator.Error![]const u8 {
        const q = try json.quote(s.a, text, .{ .ascii = false });
        return q[1 .. q.len - 1];
    }

    /// The deltas the reply's ``text`` (its whole answer so far) adds.
    pub fn feed(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        while (s.state != .abort) {
            const rest = text[@min(s.pos, text.len)..];
            switch (s.state) {
                .outside => {
                    if (text.len < s.searched) s.searched = s.pos;
                    const from = @min(text.len, @max(s.pos, s.searched -| ("<tool_call>".len - 1)));
                    const j = reply_text.findTag(text, from, "<tool_call>") orelse {
                        s.searched = text.len;
                        return;
                    };
                    s.pos = j + "<tool_call>".len;
                    s.state = .call;
                },
                .call => {
                    const m = opening(rest, "<function=") orelse {
                        const trimmed = reply_text.pyStrip(rest);
                        if (reply_text.charCount(rest) > 256 or (trimmed.len > 0 and trimmed[0] != '<')) s.state = .abort;
                        return;
                    };
                    const lower = try std.ascii.allocLowerString(s.a, reply_text.pyStrip(m.name));
                    const name = s.known.get(lower) orelse {
                        s.state = .abort;
                        return;
                    };
                    s.current = name;
                    s.index += 1;
                    s.streamed = true;
                    const function = try json.newObject(s.a);
                    try function.put(s.a, "name", .{ .string = name });
                    try function.put(s.a, "arguments", .{ .string = "" });
                    const one = try json.newObject(s.a);
                    try one.put(s.a, "index", try json.intValue(s.a, s.index));
                    try one.put(s.a, "id", .{ .string = try ids.make(s.a, "call_", 24) });
                    try one.put(s.a, "type", .{ .string = "function" });
                    try one.put(s.a, "function", .{ .object = function });
                    try out.append(s.a, try wrap(s.a, .{ .object = one }));
                    s.args_open = false;
                    s.pos += m.len;
                    s.state = .function;
                },
                .function => {
                    if (opening(rest, "<parameter=")) |m| {
                        const key = reply_text.pyStrip(m.name);
                        s.value_schema = try tool_params.schemaOf(&s.schemas, s.current, key, s.a);
                        s.typed = typedSchema(s.value_schema);
                        s.buffer.clearRetainingCapacity();
                        const quoted = try json.quote(s.a, key, .{ .ascii = false });
                        try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (s.typed) ":" else ":\"" })));
                        s.args_open = true;
                        s.pos += m.len;
                        s.state = .value;
                        s.lead = !m.newline;
                        s.held = "";
                        continue;
                    }
                    if (closing(rest, "</function>")) |n| {
                        try out.append(s.a, try s.args(if (s.args_open) "}" else "{}"));
                        s.pos += n;
                        s.state = .closing;
                        continue;
                    }
                    return;
                },
                .value => {
                    const tail = "</parameter>";
                    const end = reply_text.findTag(rest, 0, tail);
                    const chunk = if (end) |e| rest[0..e] else rest[0..trimTail(rest, tail.len)];
                    s.pos += chunk.len;
                    var piece = try std.mem.concat(s.a, u8, &.{ s.held, chunk });
                    if (s.lead and piece.len > 0) {
                        if (piece[0] == '\n') piece = piece[1..];
                        s.lead = false;
                    }
                    var keep: []const u8 = undefined;
                    if (end != null) {
                        keep = if (std.mem.endsWith(u8, piece, "\n")) piece[0 .. piece.len - 1] else piece; // the framing newline
                        s.held = "";
                    } else {
                        keep = reply_text.pyRstrip(piece); // its last newline may be the framing one
                        s.held = piece[keep.len..];
                    }
                    if (s.typed) {
                        try s.buffer.appendSlice(s.a, keep);
                    } else if (keep.len > 0) try out.append(s.a, try s.args(try s.esc(keep)));
                    if (end == null) return;
                    if (s.typed) {
                        const v = try tool_params.decode(s.a, s.buffer.items, s.value_schema, true);
                        try out.append(s.a, try s.args(try json.stringify(s.a, v, .{ .ascii = false })));
                    } else try out.append(s.a, try s.args("\""));
                    s.pos += tail.len;
                    s.state = .function;
                },
                .closing => {
                    const n = closing(rest, "</tool_call>") orelse return;
                    s.pos += n;
                    s.state = .outside;
                },
                .abort => return,
            }
        }
    }
};

/// Bytes of ``rest`` that cannot be the start of a ``tail``-byte closer (Python slices characters).
fn trimTail(rest: []const u8, tail: usize) usize {
    const chars = reply_text.charCount(rest);
    if (chars <= tail) return 0;
    return rest.len - reply_text.afterChars(rest, chars - tail).len;
}

fn typedSchema(schema: Value) bool {
    const t = schema.get("type") orelse return false;
    if (t != .string) return false;
    for ([_][]const u8{ "array", "object", "boolean", "integer", "number", "null" }) |k| if (std.mem.eql(u8, k, t.string)) return true;
    return false;
}

const wrap = @import("tool_parse.zig").wrapCalls;

fn nameLike(s: []const u8) bool {
    if (s.len == 0 or !(std.ascii.isAlphabetic(s[0]) or s[0] == '_')) return false;
    for (s[1..]) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-' or ch >= 0x80)) return false;
    return true;
}

/// Hides tool envelopes from streamed prose, holding only a partial tag between deltas.
pub const TextFilter = struct {
    pending: std.ArrayList(u8) = .empty,
    hidden: ?[]const u8 = null,

    pub fn feed(f: *TextFilter, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        try f.pending.appendSlice(a, text);
        var out: std.ArrayList(u8) = .empty;
        while (f.pending.items.len > 0) {
            const p = f.pending.items;
            if (f.hidden) |name| {
                const close = try std.mem.concat(a, u8, &.{ "</", name, ">" });
                const end = std.ascii.findIgnoreCase(p, close) orelse {
                    const keep = reply_text.charCount(close) - 1;
                    const chars = reply_text.charCount(p);
                    const tail = if (chars > keep) reply_text.afterChars(p, chars - keep) else p;
                    try f.reset(a, tail);
                    break;
                };
                try f.reset(a, p[end + close.len ..]);
                f.hidden = null;
                continue;
            }
            const start = std.mem.indexOfScalar(u8, p, '<') orelse {
                try out.appendSlice(a, p);
                f.pending.clearRetainingCapacity();
                break;
            };
            try out.appendSlice(a, p[0..start]);
            const rest = p[start..];
            const end = std.mem.indexOfScalar(u8, rest, '>') orelse {
                const name = rest[1..];
                const colon = std.mem.indexOfScalar(u8, name, ':');
                const possible = name.len == 0 or nameLike(name) or (colon != null and nameLike(name[0..colon.?]) and
                    std.ascii.startsWithIgnoreCase("tool_call", name[colon.? + 1 ..]));
                if (possible) {
                    try f.reset(a, rest);
                    break;
                }
                try out.append(a, '<');
                try f.reset(a, rest[1..]);
                continue;
            };
            const name = rest[1..end];
            if (callTag(name)) {
                f.hidden = try std.ascii.allocLowerString(a, name);
            } else try out.appendSlice(a, rest[0 .. end + 1]);
            try f.reset(a, rest[end + 1 ..]);
        }
        return out.items;
    }

    fn reset(f: *TextFilter, a: Allocator, rest: []const u8) Allocator.Error!void {
        const copy = try a.dupe(u8, rest);
        f.pending.clearRetainingCapacity();
        try f.pending.appendSlice(a, copy);
    }

    pub fn finish(f: *TextFilter) []const u8 {
        const tail = if (f.hidden != null) "" else f.pending.items;
        f.pending = .empty;
        return tail;
    }
};

/// ``(?:NAME:)?tool_call``, any case.
fn callTag(name: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(name, "tool_call")) return true;
    const colon = std.mem.indexOfScalar(u8, name, ':') orelse return false;
    return nameLike(name[0..colon]) and std.ascii.eqlIgnoreCase(name[colon + 1 ..], "tool_call");
}

/// ``parallel_tool_calls: false`` keeps one call and hides call markup from streamed prose.
pub const Policy = struct {
    single: bool = false,
    filter: TextFilter = .{},

    pub fn maxCalls(p: *const Policy) ?usize {
        return if (p.single) 1 else null;
    }

    /// Filters a str delta's text (dict deltas lose tool_calls and filter content), or passes it through.
    pub fn text(p: *Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        return if (p.single) p.filter.feed(a, s) else s;
    }

    pub fn flush(p: *Policy) []const u8 {
        return if (p.single) p.filter.finish() else "";
    }

    pub fn content(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var f: TextFilter = .{};
        const head = try f.feed(a, s);
        return std.mem.concat(a, u8, &.{ head, f.finish() });
    }
};
