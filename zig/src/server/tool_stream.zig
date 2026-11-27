//! Tool calls while they are written: XML, GLM and DSML calls as argument deltas.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const reply_code = @import("reply_code.zig");
const reply_text = @import("reply_text.zig");
const tool_dsml = @import("tool_dsml.zig");
const tool_dsml_stream = @import("tool_dsml_stream.zig");
const tool_glm_cut = @import("tool_glm_cut.zig");
const tool_json_cut = @import("tool_json_cut.zig");
const tool_minimax = @import("tool_minimax.zig");
const tool_json_close = @import("tool_json_close.zig");
const tool_glm_stream = @import("tool_glm_stream.zig");
const tool_params = @import("tool_params.zig");
const tool_policy = @import("tool_policy.zig");
const tool_specs = @import("tool_specs.zig");
const tool_parse = @import("tool_parse.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const TextFilter = tool_policy.TextFilter;
pub const Policy = tool_policy.Policy;

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

/// Streams XML, GLM and DSML argument deltas, else whole envelopes at their closers, matching the final parser.
pub const Streamer = struct {
    a: Allocator,
    schemas: tool_params.Schemas,
    known: std.StringHashMapUnmanaged([]const u8) = .empty,
    current: []const u8 = "",
    value_schema: tool_params.Schema = .{ .value = .null, .root = .null },
    typed: bool = false,
    buffer: std.ArrayList(u8) = .empty,
    pos: usize = 0,
    state: enum { outside, call, buffered, function, value, closing, glm, dsml, minimax, abort } = .outside,
    index: i64 = -1,
    args_open: bool = false,
    lead: bool = false,
    held: []const u8 = "",
    streamed: bool = false,
    searched: usize = 0, // the answer is only ever extended, so prose already searched holds no opener
    tools: []const Value,
    dsml_start: ?usize = null,
    dsml_close: []const u8 = "", // the closer of the open DSML block's spelling
    dsml_searched: usize = 0,
    dsml_seen: bool = false,
    buffered_start: usize = 0,
    buffered_close: []const u8 = "",
    glm: tool_glm_stream.Glm = .{},
    dsml: tool_dsml_stream.Dsml = .{},
    json_closer: tool_json_close.Closer = .{ .at = 0 },
    code: reply_code.Scanner = .{},
    finished: bool = false,

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!Streamer {
        var s: Streamer = .{ .a = a, .schemas = try tool_params.schemas(a, tools), .tools = tools };
        for (tools) |tool| {
            if (tool != .object) continue;
            const name = try tool_specs.toolName(a, tool);
            if (name.len == 0) continue;
            try s.known.put(a, try std.ascii.allocLowerString(a, name), name);
        }
        return s;
    }

    /// One arguments delta under the open call.
    pub fn args(s: *Streamer, fragment: []const u8) Allocator.Error!Value {
        const function = try json.newObject(s.a);
        try function.put(s.a, "arguments", .{ .string = fragment });
        const one = try json.newObject(s.a);
        try one.put(s.a, "index", try json.intValue(s.a, s.index));
        try one.put(s.a, "function", .{ .object = function });
        return wrap(s.a, .{ .object = one });
    }

    /// ``text`` as the bytes of a JSON string, its quotes off.
    pub fn esc(s: *Streamer, text: []const u8) Allocator.Error![]const u8 {
        const q = try json.quote(s.a, text, .{ .ascii = false });
        return q[1 .. q.len - 1];
    }

    /// The delta that opens call ``name``: an XML, GLM or DSML call's header, before its arguments.
    pub fn begin(s: *Streamer, name: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
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
    }

    fn emitCalls(s: *Streamer, calls: []const Value, out: *std.ArrayList(Value)) Allocator.Error!void {
        for (try tool_parse.deltas(s.a, calls)) |delta| {
            const one = delta.get("tool_calls").?.array[0];
            const index = one.get("index").?.int64().?;
            try one.object.put(s.a, "index", try json.intValue(s.a, index + s.index + 1));
            try out.append(s.a, delta);
        }
        s.index += @intCast(calls.len);
        s.streamed = true;
    }

    /// A complete GLM JSON payload through the finished parser, before its streaming envelope closes.
    pub fn jsonCall(s: *Streamer, payload: []const u8, out: *std.ArrayList(Value)) Allocator.Error!bool {
        const text = try std.mem.concat(s.a, u8, &.{ "<tool_call>", payload, "</tool_call>" });
        const parsed = tool_parse.parse(s.a, text, s.tools, null) catch return false;
        const calls = parsed.calls orelse return false;
        if (calls.len != 1) return false;
        try s.emitCalls(calls, out);
        return true;
    }

    /// The deltas the reply's ``text`` (its whole answer so far) adds.
    pub fn feed(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        while (s.state != .abort) {
            const rest = text[@min(s.pos, text.len)..];
            switch (s.state) {
                .outside => {
                    if (s.pos == 0) {
                        const trimmed = reply_text.pyStrip(text);
                        if (trimmed.len > 0 and (trimmed[0] == '{' or trimmed[0] == '[' or
                            std.mem.startsWith(u8, trimmed, "```")))
                        {
                            s.state = .abort;
                            return;
                        }
                    }
                    if (s.dsml_start) |start| {
                        const from = @max(s.pos, s.dsml_searched -| (s.dsml_close.len - 1));
                        const end = reply_text.findTag(text, from, s.dsml_close) orelse {
                            s.dsml_searched = text.len;
                            return;
                        };
                        s.pos = end + s.dsml_close.len;
                        s.searched = s.pos;
                        s.dsml_start = null;
                        const parsed = try tool_parse.parse(s.a, text[start..s.pos], s.tools, null);
                        if (parsed.calls) |calls| try s.emitCalls(calls, out);
                    }
                    s.code.jumpTo(s.pos);
                    if (text.len < s.searched) s.searched = s.pos;
                    const from = @min(text.len, @max(@max(s.pos, s.code.skip_end), s.searched -| (tool_dsml.open_max - 1)));
                    var at = from;
                    while (std.mem.indexOfScalarPos(u8, text, at, '<')) |start| {
                        switch (s.code.check(text, start, s.finished)) {
                            .skip => |to| {
                                at = to;
                                continue;
                            },
                            .pending => {
                                s.searched = start;
                                return;
                            },
                            .outside => {},
                        }
                        if (tool_dsml.openedBy(text[start..])) |sp| {
                            s.dsml_seen = true;
                            s.pos = start + sp.open.len;
                            s.dsml = .{ .sp = sp, .start = start, .at = s.pos, .output_start = out.items.len, .index_start = s.index, .streamed_start = s.streamed };
                            s.state = .dsml;
                            break;
                        }
                        const mm = tool_minimax.open[1..];
                        if (start > 0 and text[start - 1] == ']' and std.mem.startsWith(u8, text[start..], mm)) {
                            s.buffered_start = start - 1;
                            s.pos = start + mm.len;
                            s.state = .minimax;
                            break;
                        }
                        if (start > 0 and text[start - 1] == ']' and text[start..].len < mm.len and std.mem.startsWith(u8, mm, text[start..])) {
                            s.searched = start;
                            return;
                        }
                        if (std.mem.startsWith(u8, text[start..], "<tool_call>")) {
                            s.buffered_start = start;
                            s.pos = start + "<tool_call>".len;
                            s.glm = .{ .base = s.pos, .scan = s.pos };
                            s.state = .call;
                            break;
                        }
                        if (std.mem.startsWith(u8, text[start..], "<|tool_call>")) {
                            s.buffered_start = start;
                            s.buffered_close = "<tool_call|>";
                            s.pos = start + "<|tool_call>".len;
                            s.dsml_searched = s.pos;
                            if (!s.giveUp()) return;
                            break;
                        }
                        const end = std.mem.indexOfAnyPos(u8, text, start + 1, "<>") orelse {
                            s.searched = start;
                            return;
                        };
                        if (text[end] == '<') {
                            at = end;
                            continue;
                        }
                        if (tool_policy.callTag(text[start + 1 .. end])) {
                            s.buffered_start = start;
                            s.buffered_close = try std.mem.concat(s.a, u8, &.{ "</", text[start + 1 .. end], ">" });
                            s.pos = end + 1;
                            s.dsml_searched = s.pos;
                            if (!s.giveUp()) return;
                            break;
                        }
                        at = end + 1;
                    } else {
                        s.searched = text.len;
                        return;
                    }
                },
                .call => {
                    if (opening(rest, "<function=")) |m| {
                        const lower = try std.ascii.allocLowerString(s.a, reply_text.pyStrip(m.name));
                        const name = s.known.get(lower) orelse {
                            s.buffered_close = "</tool_call>";
                            s.dsml_searched = s.pos;
                            if (!s.giveUp()) return;
                            continue;
                        };
                        try s.begin(name, out);
                        s.pos += m.len;
                        s.state = .function;
                        continue;
                    }
                    const trimmed = reply_text.pyStrip(rest);
                    // GLM's ``NAME<arg_key>K</arg_key><arg_value>V</arg_value>`` arguments stream as written
                    if (!s.dsml_seen and !std.mem.startsWith(u8, trimmed, "<function=")) {
                        if (!try s.glmStep(text, out)) return;
                        continue;
                    }
                    if (reply_text.charCount(rest) > 256 or (trimmed.len > 0 and trimmed[0] != '<') or
                        (s.dsml_seen and std.ascii.findIgnoreCase(rest, "</tool_call>") != null))
                    {
                        s.buffered_close = "</tool_call>";
                        s.dsml_searched = s.pos;
                        if (!s.giveUp()) return;
                        continue;
                    }
                    return;
                },
                .glm => {
                    if (!try s.glmStep(text, out)) return;
                    continue;
                },
                .dsml => {
                    if (!try s.dsmlStep(text, out)) return;
                    continue;
                },
                .buffered => {
                    const tail = s.buffered_close;
                    const json_envelope = std.mem.eql(u8, tail, "</tool_call>");
                    const found: ?usize = if (json_envelope) s.json_closer.next(text) else blk: {
                        const from = @max(s.pos, s.dsml_searched -| (tail.len - 1));
                        break :blk std.ascii.findIgnoreCasePos(text, from, tail);
                    };
                    const end = found orelse {
                        if (!json_envelope) s.dsml_searched = text.len;
                        return;
                    };
                    s.pos = end + tail.len;
                    s.searched = s.pos;
                    s.state = .outside;
                    const parsed = try tool_parse.parse(s.a, text[s.buffered_start..s.pos], s.tools, null);
                    if (parsed.calls) |calls| try s.emitCalls(calls, out);
                },
                .minimax => {
                    if (s.buffered_start > text.len or !std.mem.startsWith(u8, text[s.buffered_start..], tool_minimax.open)) {
                        s.pos = @min(s.buffered_start, text.len);
                        s.searched = s.pos;
                        s.code.jumpTo(s.pos);
                        s.state = .outside;
                        continue;
                    }
                    const end = std.mem.indexOfPos(u8, text, s.buffered_start + tool_minimax.open.len, tool_minimax.close) orelse return;
                    s.pos = end + tool_minimax.close.len;
                    s.searched = s.pos;
                    s.state = .outside;
                    const converted = try tool_minimax.dsml(s.a, text[s.buffered_start..s.pos], s.tools);
                    const parsed = try tool_parse.parse(s.a, converted, s.tools, null);
                    if (parsed.calls) |calls| try s.emitCalls(calls, out);
                },
                .function => {
                    while (s.pos < text.len and space(text[s.pos])) s.pos += 1;
                    const between = text[s.pos..];
                    if (opening(between, "<parameter=")) |m| {
                        const key = reply_text.pyStrip(m.name);
                        s.value_schema = try tool_params.schemaOf(&s.schemas, s.current, key, s.a);
                        s.typed = try typedSchema(s.a, s.value_schema);
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
                    if (closing(between, "</function>")) |n| {
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
                        try out.append(s.a, try s.args(try json.stringify(s.a, v, .{ .ascii = false, .compact = true })));
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

    /// A call that the reply's end left open: its name streamed, its envelope not closed.
    pub fn open(s: *const Streamer) bool {
        if (s.state == .dsml) return s.dsml.sent > 0; // the block's whole parse would keep it all as text
        return s.index >= 0 and (s.state == .glm or s.state == .function or s.state == .value or s.state == .closing);
    }

    /// An envelope this stream cannot stream: with nothing sent yet the finish sends every call, else its closer does.
    fn giveUp(s: *Streamer) bool {
        if (!s.dsml_seen and !s.streamed) {
            s.state = .abort;
            return false;
        }
        s.state = .buffered;
        s.json_closer = .{ .at = s.buffered_start + "<tool_call>".len };
        return true;
    }

    /// One GLM state-machine step: whether ``feed`` goes on with the state the step left.
    fn glmStep(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!bool {
        switch (try tool_glm_stream.feed(s, text, out)) {
            .wait => return false,
            .outside => return true,
            .unstreamable => {
                s.buffered_close = "</tool_call>";
                s.dsml_searched = s.pos;
                return s.giveUp();
            },
            .more => unreachable,
        }
    }

    /// One DSML state-machine step: whether ``feed`` goes on; a block it cannot stream parses whole at its closer.
    fn dsmlStep(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!bool {
        switch (try tool_dsml_stream.feed(s, text, out)) {
            .wait => return false,
            .outside => return true,
            .invalid => {
                out.items.len = s.dsml.output_start;
                s.index = s.dsml.index_start;
                s.streamed = s.dsml.streamed_start;
                s.state = .abort;
                return false;
            },
            .rejected => {
                s.dsml_start = s.dsml.start;
                s.dsml_close = s.dsml.sp.close;
                s.pos = s.dsml.start + s.dsml.sp.open.len;
                s.dsml_searched = s.pos;
                s.state = .outside;
                return true;
            },
            .more => unreachable,
        }
    }

    /// ``feed`` at the reply's end: resolves any open backtick ambiguity without cutting open values.
    pub fn finish(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        s.finished = true;
        return s.feed(text, out);
    }

    /// ``feed`` of the reply's final text; a value it cuts sends the bytes held back for a closer that never came.
    pub fn finishCut(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        try s.finish(text, out);
        if (s.state == .glm) return tool_glm_stream.cut(s, text, out); // a GLM value's held bytes
        if (s.state == .dsml) return tool_dsml_stream.cut(s, text, out); // a DSML value's held bytes
        if (s.state != .value) return;
        var cut = try std.mem.concat(s.a, u8, &.{ if (s.typed) s.buffer.items else "", s.held, text[@min(s.pos, text.len)..] });
        if (s.lead and cut.len > 0 and cut[0] == '\n') cut = cut[1..];
        s.lead = false;
        s.pos = text.len;
        s.held = "";
        s.buffer.clearRetainingCapacity();
        if (cut.len > 0) try out.append(s.a, try s.args(if (s.typed) cut else try s.esc(cut)));
    }
};

/// A streamed reply's final prose: ``cutProse`` for a cut call, else ``finishedProse``; one-call mode rstrips it.
pub fn endProse(a: Allocator, content: []const u8, text: []const u8, tools: []const Value, reason: []const u8, single: bool) Allocator.Error![]const u8 {
    if (single) return reply_text.pyRstrip(try reply_text.finishedProse(a, content));
    const cut = std.mem.eql(u8, reason, "length") and (try cutCalls(a, text, tools)) != null;
    return if (cut) reply_text.cutProse(a, content) else reply_text.finishedProse(a, content);
}

/// A reply max_tokens cut inside a call: the calls its stream sent, the cut one included; null when none was left open.
pub fn cutCalls(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error!?[]Value {
    var s: Streamer = try .init(a, tools);
    var deltas: std.ArrayList(Value) = .empty;
    try s.finishCut(text, &deltas);
    // GLM and JSON envelopes stream no call before they close
    if (!s.open()) return (try tool_glm_cut.cutCalls(a, text, tools)) orelse try tool_json_cut.cutCalls(a, text, tools);
    const Call = struct { id: []const u8 = "", name: []const u8 = "", args: std.ArrayList(u8) = .empty };
    const built = try a.alloc(Call, @intCast(s.index + 1));
    for (built) |*c| c.* = .{};
    for (deltas.items) |d| {
        const one = d.get("tool_calls").?.array[0];
        const c = &built[@intCast(one.get("index").?.int64().?)];
        if (one.strField("id")) |id| c.id = id;
        const f = one.get("function").?;
        if (f.strField("name")) |n| c.name = n;
        if (f.strField("arguments")) |piece| try c.args.appendSlice(a, piece);
    }
    const calls = try a.alloc(Value, built.len);
    for (built, calls) |*c, *slot| {
        const function = try json.newObject(a);
        try function.put(a, "name", .{ .string = c.name });
        try function.put(a, "arguments", .{ .string = c.args.items });
        const one = try json.newObject(a);
        try one.put(a, "id", .{ .string = if (c.id.len > 0) c.id else try ids.make(a, "call_", 24) });
        try one.put(a, "type", .{ .string = "function" });
        try one.put(a, "function", .{ .object = function });
        slot.* = .{ .object = one };
    }
    return calls;
}

/// Bytes of ``rest`` that cannot be the start of a ``tail``-byte closer (Python slices characters).
fn trimTail(rest: []const u8, tail: usize) usize {
    const chars = reply_text.charCount(rest);
    if (chars <= tail) return 0;
    return rest.len - reply_text.afterChars(rest, chars - tail).len;
}

fn typedSchema(a: Allocator, schema: tool_params.Schema) Allocator.Error!bool {
    return tool_params.schemaTakesJson(a, schema);
}

const wrap = @import("tool_parse.zig").wrapCalls;

test {
    _ = @import("tool_stream_test.zig");
}
