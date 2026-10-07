//! Tool calls in a finished reply, every family's markup, as OpenAI tool_calls (``tools.parse_tool_calls_from_content``).
const std = @import("std");
const json = @import("json.zig");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const tool_specs = @import("tool_specs.zig");
const tool_streaming = @import("tool_streaming.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const strip = reply_text.pyStrip;

// the streaming half (deltas and the wrapper) is beside this file, in tool_streaming.zig.
pub const deltas = tool_streaming.deltas;
pub const wrapCalls = tool_streaming.wrapCalls;

const Error = error{ Invalid, OutOfMemory };
const Call = struct { name: []const u8, arguments: Value };
const Envelope = struct { start: usize, end: usize, payload: []const u8 };

pub const Parsed = struct { content: []const u8, calls: ?[]Value };

const dsml_open = "<\u{ff5c}DSML\u{ff5c}tool_calls>";
const dsml_close = "</\u{ff5c}DSML\u{ff5c}tool_calls>";
const invoke_open = "<\u{ff5c}DSML\u{ff5c}invoke name=\"";
const invoke_close = "</\u{ff5c}DSML\u{ff5c}invoke>";
const param_open = "<\u{ff5c}DSML\u{ff5c}parameter name=\"";
const param_close = "</\u{ff5c}DSML\u{ff5c}parameter>";

fn findCI(hay: []const u8, needle: []const u8, from: usize) ?usize {
    if (from > hay.len) return null;
    return std.ascii.findIgnoreCasePos(hay, from, needle);
}

fn find(hay: []const u8, needle: []const u8, from: usize) ?usize {
    if (from > hay.len) return null;
    return std.mem.indexOfPos(u8, hay, from, needle);
}

/// Python's ``\w`` on one byte, non-ASCII bytes counting as letters.
fn word(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch >= 0x80;
}

const space = reply_text.isSpaceByte;

fn allSpace(s: []const u8) bool {
    return strip(s).len == 0;
}

/// (start, end, payload) of each tool-call block in order; a block inside an earlier one is part of it.
fn envelopes(a: Allocator, text: []const u8) Allocator.Error![]Envelope {
    var found: std.ArrayList(Envelope) = .empty;
    var pos: usize = 0;
    while (findCI(text, "<tool_call>", pos)) |s| {
        const e = findCI(text, "</tool_call>", s + 11) orelse break;
        try found.append(a, .{ .start = s, .end = e + 12, .payload = strip(text[s + 11 .. e]) });
        pos = e + 12;
    }
    pos = 0;
    while (std.mem.indexOfScalarPos(u8, text, pos, '<')) |s| {
        pos = s + 1;
        var j = s + 1;
        if (j >= text.len or !(std.ascii.isAlphabetic(text[j]) or text[j] == '_')) continue;
        j += 1;
        while (j < text.len and (word(text[j]) or text[j] == '.' or text[j] == '-')) j += 1;
        if (!std.ascii.startsWithIgnoreCase(text[j..], ":tool_call>")) continue;
        const closer = try std.mem.concat(a, u8, &.{ "</", text[s + 1 .. j], ":tool_call>" });
        const e = findCI(text, closer, j + 11) orelse continue;
        try found.append(a, .{ .start = s, .end = e + closer.len, .payload = strip(text[j + 11 .. e]) });
        pos = e + closer.len;
    }
    pos = 0;
    while (find(text, "<|tool_call>", pos)) |s| {
        const e = find(text, "<tool_call|>", s + 12) orelse break;
        try found.append(a, .{ .start = s, .end = e + 12, .payload = strip(text[s + 12 .. e]) });
        pos = e + 12;
    }
    pos = 0;
    while (find(text, dsml_open, pos)) |s| {
        const e = find(text, dsml_close, s + dsml_open.len) orelse break;
        try found.append(a, .{ .start = s, .end = e + dsml_close.len, .payload = text[s .. e + dsml_close.len] });
        pos = e + dsml_close.len;
    }
    std.mem.sort(Envelope, found.items, {}, struct {
        fn less(_: void, x: Envelope, y: Envelope) bool {
            if (x.start != y.start) return x.start < y.start;
            if (x.end != y.end) return x.end < y.end;
            return std.mem.order(u8, x.payload, y.payload) == .lt;
        }
    }.less);
    var kept: std.ArrayList(Envelope) = .empty;
    for (found.items) |env| if (kept.items.len == 0 or env.start >= kept.items[kept.items.len - 1].end) try kept.append(a, env);
    return kept.items;
}

/// ``parse_tool_calls_from_content``: the reply's content and its calls (null when it makes none).
pub fn parse(a: Allocator, text: []const u8, tools: []const Value, max_calls: ?usize) Allocator.Error!Parsed {
    if (tools.len == 0) return .{ .content = text, .calls = null };
    var known: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    for (tools) |tool| {
        const name = try tool_specs.toolName(a, tool);
        try known.put(a, try std.ascii.allocLowerString(a, name), name);
    }
    const schemas = try tool_params.schemas(a, tools);
    const envs = try envelopes(a, text);
    if (envs.len == 0) {
        if (try bareCalls(a, text, &known, max_calls)) |calls| return .{ .content = "", .calls = calls };
        return .{ .content = text, .calls = null };
    }
    var calls: std.ArrayList(Value) = .empty;
    var residue: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    for (envs) |env| {
        try residue.appendSlice(a, text[cursor..env.start]);
        cursor = env.end;
        if (max_calls) |m| if (calls.items.len >= m) continue;
        const parsed: ?[]Call = blk: {
            if (std.mem.startsWith(u8, env.payload, dsml_open)) {
                const inner = env.payload[dsml_open.len .. env.payload.len - dsml_close.len];
                break :blk dsmlCalls(a, inner) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Invalid => null,
                };
            }
            const one = payload(a, env.payload, &schemas, max_calls != null) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Invalid => break :blk null,
            };
            if (one) |c| {
                const list = try a.alloc(Call, 1);
                list[0] = c;
                break :blk list;
            }
            break :blk null;
        };
        const good = if (parsed) |list| list.len > 0 and for (list) |c| {
            if ((try callName(a, c, &known)) == null) break false;
        } else true else false;
        if (!good) {
            // a malformed call stays text in either mode: the reply is content, never an error or a client retry loop
            try residue.appendSlice(a, text[env.start..env.end]);
            continue;
        }
        for (parsed.?) |c| {
            if (max_calls == null or calls.items.len < max_calls.?) try calls.append(a, try openaiCall(a, (try callName(a, c, &known)).?, c.arguments));
        }
    }
    try residue.appendSlice(a, text[cursor..]);
    return .{ .content = strip(residue.items), .calls = if (calls.items.len > 0) calls.items else null };
}

/// What closes the call ``text`` ends inside (the model's end token came before its ``
pub fn closeCall(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error![]const u8 {
    var pos: usize = 0;
    // the opener ``envelopes`` leaves without a closer, so the one appended pairs with it
    const start = while (findCI(text, "<tool_call>", pos)) |s| {
        pos = (findCI(text, "</tool_call>", s + 11) orelse break s) + 12;
    } else return "";
    const block = text[start + 11 ..];
    const tail = try argumentsClose(a, strip(block));
    const closed = try std.mem.concat(a, u8, &.{ "<tool_call>", block, tail, "</tool_call>" });
    // the strict reading: one whole call to an offered tool, nothing but its markup
    if ((try parse(a, closed, tools, 1)).calls == null) return "";
    return std.mem.concat(a, u8, &.{ tail, "</tool_call>" });
}

/// What a call's arguments lack to close: XML's ``
fn argumentsClose(a: Allocator, body: []const u8) Allocator.Error![]const u8 {
    if (std.ascii.startsWithIgnoreCase(body, "<function=")) return if (std.ascii.endsWithIgnoreCase(body, "</function>")) "" else "</function>";
    if (body.len == 0 or (body[0] != '{' and body[0] != '[')) return "";
    const closed = try tool_params.closedJson(a, body) orelse return "";
    return closed[body.len..];
}

/// ``call_name``: the offered tool's spelling; else the name itself when it is 1 to 64 of ``[A-Za-z0-9_-]`` and the arguments a finite object (an unoffered tool is the client's to refuse); else null.
fn callName(a: Allocator, c: Call, known: *const std.StringArrayHashMapUnmanaged([]const u8)) Allocator.Error!?[]const u8 {
    const name = strip(c.name);
    if (known.get(try std.ascii.allocLowerString(a, name))) |offered| return offered;
    if (name.len == 0 or name.len > 64 or c.arguments != .object or !tool_params.finite(c.arguments)) return null;
    for (name) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '-')) return null;
    return name;
}

/// ``_openai_tool_call``: the call under ``name`` (``callName``), arguments as compact JSON.
fn openaiCall(a: Allocator, name: []const u8, arguments: Value) Allocator.Error!Value {
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    try function.put(a, "arguments", .{ .string = try json.stringify(a, arguments, .{ .ascii = false, .compact = true }) });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
    try call.put(a, "type", .{ .string = "function" });
    try call.put(a, "function", .{ .object = function });
    return .{ .object = call };
}

fn bareCalls(a: Allocator, text: []const u8, known: *const std.StringArrayHashMapUnmanaged([]const u8), max_calls: ?usize) Allocator.Error!?[]Value {
    const stripped = stripFence(strip(text));
    if (stripped.len == 0 or (stripped[0] != '[' and stripped[0] != '{')) return null;
    const value = switch (try json.parseText(a, stripped)) {
        .ok => |v| v,
        .err => return null,
    };
    const items: []const Value = if (value == .array) value.array else &.{value};
    var calls: std.ArrayList(Value) = .empty;
    for (items) |item| {
        if (max_calls) |m| if (calls.items.len >= m) break;
        if (item != .object) return null;
        const doc = try json.stringify(a, item, .{ .ascii = false });
        const parsed = payload(a, doc, null, max_calls != null) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Invalid => {
                if (max_calls == null) return null; // JSON that is not a call (a structured answer) is the reply's content
                continue;
            },
        };
        const c = parsed orelse return null;
        // bare JSON naming a tool the request did not offer is a structured answer, the reply's content
        const name = known.get(try std.ascii.allocLowerString(a, c.name)) orelse return null;
        try calls.append(a, try openaiCall(a, name, c.arguments));
    }
    return if (calls.items.len > 0) calls.items else null;
}

/// ``_strip_json_fence``: the inside of a ```json fence, else the text.
fn stripFence(text: []const u8) []const u8 {
    var i: usize = 0;
    while (i < text.len and space(text[i])) i += 1;
    if (!std.mem.startsWith(u8, text[i..], "```")) return text;
    i += 3;
    if (std.ascii.startsWithIgnoreCase(text[i..], "json")) i += 4;
    var at = i;
    while (find(text, "```", at)) |f| : (at = f + 1) {
        if (allSpace(text[f + 3 ..])) return strip(text[i..f]);
    }
    return text;
}

/// ``_parse_tool_call_payload``: one call from a block; null when it is not one, Invalid where Python raises.
fn payload(a: Allocator, block: []const u8, schemas: ?*const tool_params.Schemas, complete: bool) Error!?Call {
    if (std.mem.startsWith(u8, block, "call:") or std.mem.startsWith(u8, block, ":")) {
        if (try gemmaCall(a, block)) |c| return c;
    }
    const decoded: ?Value = switch (try json.parseText(a, block)) {
        .ok => |v| v,
        .err => null,
    };
    if (decoded) |v| {
        if (v == .array) {
            for (v.array) |item| {
                if (try payload(a, try json.stringify(a, item, .{ .ascii = false }), null, complete)) |c| return c;
            }
            return null;
        }
        if (v == .object) {
            const function = v.get("function");
            const holder = if (function != null and function.? == .object) function.? else v;
            const name_keys: []const []const u8 = if (holder.object == v.object) &.{ "name", "tool", "function", "call" } else &.{ "name", "tool", "function" };
            const name = if (holder.firstTruthy(name_keys)) |n| strip(try tool_specs.pyStr(a, n)) else "";
            var arguments: ?Value = null;
            for ([_][]const u8{ "arguments", "args", "parameters" }) |k| if (holder.has(k)) {
                arguments = holder.get(k).?;
                break;
            };
            const args = arguments orelse blk: {
                const loose = try json.newObject(a);
                for (holder.object.keys(), holder.object.values()) |k, item| {
                    const skip = for ([_][]const u8{ "name", "tool", "function", "call", "type" }) |x| {
                        if (std.mem.eql(u8, x, k)) break true;
                    } else false;
                    if (!skip) try loose.put(a, k, item);
                }
                break :blk Value{ .object = loose };
            };
            if (name.len == 0) return error.Invalid;
            const object = try jsonObject(a, args);
            if (complete and !tool_params.finite(object)) return error.Invalid;
            return .{ .name = name, .arguments = object };
        }
    }
    if (try functionBlock(a, block, schemas, complete)) |c| return c;
    if (decoded == null) {
        const lead = std.mem.trimStart(u8, block, " \t\r\n\x0b\x0c");
        if (lead.len == 0 or (lead[0] != '{' and lead[0] != '[' and lead[0] != '<')) return glmCall(a, block, schemas, complete);
    }
    return null;
}

/// ``_tool_json_object``: arguments as an object, Invalid where Python raises.
fn jsonObject(a: Allocator, v: Value) Error!Value {
    switch (v) {
        .object => return v,
        .null => return .{ .object = try json.newObject(a) },
        .string => |s| {
            const doc = strip(s);
            if (doc.len == 0) return .{ .object = try json.newObject(a) };
            const parsed = switch (try json.parseText(a, doc)) {
                .ok => |p| p,
                .err => return error.Invalid,
            };
            if (parsed == .object) return parsed;
            return error.Invalid;
        },
        else => return error.Invalid,
    }
}

/// ``<function=NAME> <parameter=K>V</parameter> </function>`` (Qwen3-Coder's XML).
fn functionBlock(a: Allocator, block: []const u8, schemas: ?*const tool_params.Schemas, complete: bool) Error!?Call {
    var i: usize = 0;
    while (i < block.len and space(block[i])) i += 1;
    if (!std.ascii.startsWithIgnoreCase(block[i..], "<function=")) return null;
    i += 10;
    const name_start = i;
    while (i < block.len and block[i] != '>' and !space(block[i])) i += 1;
    if (i == name_start or i >= block.len or block[i] != '>') return null;
    const name = block[name_start..i];
    const body_start = i + 1;
    var at = body_start;
    const close = while (findCI(block, "</function>", at)) |f| : (at = f + 1) {
        if (allSpace(block[f + 11 ..])) break f;
    } else return null;
    const body = strip(block[body_start..close]);
    const params = try parameterBlocks(a, body);
    if (complete and !allSpace(params.residue)) return null;
    const arguments = try json.newObject(a);
    for (params.items) |p| {
        const key = strip(p.key);
        const schema: Value = if (schemas) |s| try tool_params.schemaOf(s, name, key, a) else .{ .object = try json.newObject(a) };
        try arguments.put(a, key, try tool_params.decode(a, p.value, schema, true));
    }
    return .{ .name = name, .arguments = .{ .object = arguments } };
}

const Param = struct { key: []const u8, value: []const u8 };
const Params = struct { items: []Param, residue: []const u8 };

/// ``<parameter=K>\n?(.*?)\n?</parameter>`` matches, and the body without them.
fn parameterBlocks(a: Allocator, body: []const u8) Allocator.Error!Params {
    var items: std.ArrayList(Param) = .empty;
    var residue: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    var kept: usize = 0;
    while (findCI(body, "<parameter=", pos)) |s| {
        pos = s + 1;
        var i = s + 11;
        const key_start = i;
        while (i < body.len and body[i] != '>' and !space(body[i])) i += 1;
        if (i == key_start or i >= body.len or body[i] != '>') continue;
        const key = body[key_start..i];
        i += 1;
        if (i < body.len and body[i] == '\n') i += 1;
        const e = findCI(body, "</parameter>", i) orelse break;
        var value = body[i..e];
        if (std.mem.endsWith(u8, value, "\n")) value = value[0 .. value.len - 1];
        try items.append(a, .{ .key = key, .value = value });
        try residue.appendSlice(a, body[kept..s]);
        kept = e + 12;
        pos = kept;
    }
    try residue.appendSlice(a, body[kept..]);
    return .{ .items = items.items, .residue = residue.items };
}

/// GLM's ``NAME<arg_key>K</arg_key><arg_value>V</arg_value>...``, values decoded as Qwen's (without Python spellings).
fn glmCall(a: Allocator, block: []const u8, schemas: ?*const tool_params.Schemas, complete: bool) Error!?Call {
    const cut = std.mem.indexOf(u8, block, "<arg_key>") orelse block.len;
    const name = strip(block[0..cut]);
    if (name.len == 0) return null;
    for (name) |ch| if (!(word(ch) or ch == '.' or ch == ':' or ch == '-')) return null;
    const rest = block[cut..];
    var pairs: std.ArrayList(Param) = .empty;
    var residue: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    var kept: usize = 0;
    outer: while (find(rest, "<arg_key>", pos)) |s| {
        pos = s + 1;
        var key_end_at = s + 9;
        while (find(rest, "</arg_key>", key_end_at)) |ke| : (key_end_at = ke + 1) {
            var j = ke + 10;
            while (j < rest.len and space(rest[j])) j += 1;
            if (!std.mem.startsWith(u8, rest[j..], "<arg_value>")) continue;
            const v_start = j + 11;
            const ve = find(rest, "</arg_value>", v_start) orelse continue :outer;
            try pairs.append(a, .{ .key = rest[s + 9 .. ke], .value = rest[v_start..ve] });
            try residue.appendSlice(a, rest[kept..s]);
            kept = ve + 12;
            pos = kept;
            continue :outer;
        }
    }
    try residue.appendSlice(a, rest[kept..]);
    if (complete and !allSpace(residue.items)) return null;
    const arguments = try json.newObject(a);
    for (pairs.items) |p| {
        const key = strip(p.key);
        const schema: Value = if (schemas) |s| try tool_params.schemaOf(s, name, key, a) else .{ .object = try json.newObject(a) };
        try arguments.put(a, key, try tool_params.decode(a, p.value, schema, false));
    }
    return .{ .name = name, .arguments = .{ .object = arguments } };
}

/// Gemma 4's ``call:NAME{key:value,...}`` (or bare ``:NAME{...}``): strings between ``<|"|>``, keys bare.
fn gemmaCall(a: Allocator, block: []const u8) Error!?Call {
    var i: usize = 0;
    if (std.mem.startsWith(u8, block, "call:")) i = 4;
    if (i >= block.len or block[i] != ':') return null;
    i += 1;
    const name_start = i;
    while (i < block.len and (word(block[i]) or block[i] == '.' or block[i] == '-')) i += 1;
    if (i == name_start) return null;
    const name = block[name_start..i];
    while (i < block.len and space(block[i])) i += 1;
    if (i >= block.len or block[i] != '{') return null;
    var end = block.len;
    if (std.mem.endsWith(u8, block, "}\n")) end -= 1;
    if (end <= i or block[end - 1] != '}') return null;
    const inner = block[i..end];
    var strings: std.ArrayList([]const u8) = .empty;
    var marked: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (find(inner, "<|\"|>", pos)) |s| {
        const e = find(inner, "<|\"|>", s + 5) orelse break;
        try marked.appendSlice(a, inner[pos..s]);
        try marked.print(a, "\x00{d}\x00", .{strings.items.len});
        try strings.append(a, inner[s + 5 .. e]);
        pos = e + 5;
    }
    try marked.appendSlice(a, inner[pos..]);
    const src = marked.items;
    var keyed: std.ArrayList(u8) = .empty;
    var p: usize = 0;
    var copied: usize = 0;
    while (p < src.len) : (p += 1) {
        if (p == 0 or (src[p - 1] != '{' and src[p - 1] != ',')) continue;
        var j = p;
        while (j < src.len and space(src[j])) j += 1;
        if (j >= src.len or !(std.ascii.isAlphabetic(src[j]) or src[j] == '_')) continue;
        const k_start = j;
        while (j < src.len and (word(src[j]) or src[j] == '-')) j += 1;
        const key = src[k_start..j];
        while (j < src.len and space(src[j])) j += 1;
        if (j >= src.len or src[j] != ':') continue;
        try keyed.appendSlice(a, src[copied..p]);
        try keyed.print(a, "\"{s}\":", .{key});
        copied = j + 1;
        p = j;
    }
    try keyed.appendSlice(a, src[copied..]);
    var text: []const u8 = keyed.items;
    for (strings.items, 0..) |value, idx| {
        const mark = try std.fmt.allocPrint(a, "\x00{d}\x00", .{idx});
        text = try std.mem.replaceOwned(u8, a, text, mark, try json.quote(a, value, .{ .ascii = false }));
    }
    return .{ .name = name, .arguments = try jsonObject(a, .{ .string = text }) };
}

/// Every invoke of a DSML block, or Invalid when anything in it is not a well-formed invoke.
fn dsmlCalls(a: Allocator, block: []const u8) Error![]Call {
    var calls: std.ArrayList(Call) = .empty;
    var at: usize = 0;
    var pos: usize = 0;
    while (find(block, invoke_open, pos)) |s| {
        pos = s + 1;
        const name_start = s + invoke_open.len;
        const q = std.mem.indexOfScalarPos(u8, block, name_start, '"') orelse continue;
        if (!std.mem.startsWith(u8, block[q..], "\">")) continue;
        const e = find(block, invoke_close, q + 2) orelse continue;
        if (!allSpace(block[at..s])) return error.Invalid;
        at = e + invoke_close.len;
        pos = at;
        const body = block[q + 2 .. e];
        const arguments = try json.newObject(a);
        var p: usize = 0;
        var kept: usize = 0;
        var residue: std.ArrayList(u8) = .empty;
        while (find(body, param_open, p)) |ps| {
            p = ps + 1;
            const ns = ps + param_open.len;
            const nq = std.mem.indexOfScalarPos(u8, body, ns, '"') orelse continue;
            const flag = body[nq..];
            const is_string = std.mem.startsWith(u8, flag, "\" string=\"true\">");
            if (!is_string and !std.mem.startsWith(u8, flag, "\" string=\"false\">")) continue;
            const v_start = nq + (if (is_string) "\" string=\"true\">".len else "\" string=\"false\">".len);
            const pe = find(body, param_close, v_start) orelse continue;
            const value = body[v_start..pe];
            try residue.appendSlice(a, body[kept..ps]);
            kept = pe + param_close.len;
            p = kept;
            if (is_string) {
                try arguments.put(a, body[ns..nq], .{ .string = value });
            } else switch (try json.parseText(a, value)) {
                .ok => |v| try arguments.put(a, body[ns..nq], v),
                .err => return error.Invalid,
            }
        }
        try residue.appendSlice(a, body[kept..]);
        if (!allSpace(residue.items)) return error.Invalid;
        try calls.append(a, .{ .name = strip(block[name_start..q]), .arguments = .{ .object = arguments } });
    }
    if (calls.items.len == 0 or !allSpace(block[at..])) return error.Invalid;
    return calls.items;
}
