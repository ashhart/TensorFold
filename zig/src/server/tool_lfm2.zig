//! LFM2's Pythonic tool calls (`<|tool_call_start|>[name(key=value)]<|tool_call_end|>`) rewritten as DSML.
const std = @import("std");
const json = @import("json");
const tool_params = @import("tool_params.zig");
const tool_kinds = @import("tool_kinds.zig");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const open = reply_text.calls[3][0];
pub const close = reply_text.calls[3][1];
const dsml_open = reply_text.calls[2][0];
const dsml_close = reply_text.calls[2][1];

/// `text` with each complete block of offered calls as DSML; while streaming (not `finished`) each call as it ends.
pub fn dsml(a: Allocator, text: []const u8, tools: []const Value, finished: bool) Allocator.Error![]const u8 {
    if (tools.len == 0) return text;
    var at = reply_text.findTag(text, 0, open) orelse return text;
    const schemas = try tool_params.schemas(a, tools);
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    var end = text.len;
    var rewrote = false;
    while (true) {
        const e = reply_text.findTag(text, at + open.len, close) orelse {
            if (!finished) { // the stream's parser resumes where it stopped: it must meet the calls so far as DSML
                end = at;
                const early = try calls(a, text[at + open.len ..], &schemas);
                if (early.list.items.len > 0) {
                    try out.appendSlice(a, text[pos..at]);
                    try each(a, &out, early.list.items);
                    pos = at;
                    rewrote = true;
                }
            }
            break;
        };
        const found = try calls(a, text[at + open.len .. e], &schemas);
        if (found.whole or (!finished and found.list.items.len > 0)) {
            try out.appendSlice(a, text[pos..at]);
            if (finished) try all(a, &out, found.list.items) else try each(a, &out, found.list.items);
            pos = e + close.len;
            rewrote = true;
        }
        at = reply_text.findTag(text, e + close.len, open) orelse break;
    }
    if (!rewrote) return text[0..end];
    try out.appendSlice(a, text[pos..end]);
    return out.items;
}

/// The invokes as one DSML envelope.
fn all(a: Allocator, out: *std.ArrayList(u8), invokes: []const []const u8) Allocator.Error!void {
    try out.appendSlice(a, dsml_open);
    for (invokes) |one| try out.appendSlice(a, one);
    try out.appendSlice(a, "\n" ++ dsml_close);
}

/// Each invoke as an envelope of its own, so a call is whole markup as soon as it ends.
fn each(a: Allocator, out: *std.ArrayList(u8), invokes: []const []const u8) Allocator.Error!void {
    for (invokes) |one| try all(a, out, &.{one});
}

/// A block's calls as DSML invokes: those complete so far, and whether the whole body is a valid list of calls.
const Calls = struct { list: std.ArrayList([]const u8) = .empty, whole: bool = false };

/// The offered calls of a block's body that are complete, in order, however much of the body has been written.
fn calls(a: Allocator, body: []const u8, schemas: *const tool_params.Schemas) Allocator.Error!Calls {
    var found: Calls = .{};
    // DSML markup in a value would end the block early, so no call is made from where it begins
    const markup = std.mem.indexOf(u8, body, "\u{ff5c}DSML\u{ff5c}") orelse body.len + 1;
    var p: Scanner = .{ .s = body };
    if (!p.eat('[')) return found;
    p.space();
    const listed = if (p.i < p.s.len and p.s[p.i] == '{')
        try jsonCalls(a, &found.list, &p, schemas, markup)
    else
        try pythonCalls(a, &found.list, &p, schemas, markup);
    found.whole = listed and markup > body.len;
    return found;
}

/// A Pythonic list's calls up to the first that is not one; true when the list ends the body.
fn pythonCalls(
    a: Allocator,
    list: *std.ArrayList([]const u8),
    p: *Scanner,
    schemas: *const tool_params.Schemas,
    markup: usize,
) Allocator.Error!bool {
    while (true) {
        const name = p.name();
        if (name.len == 0 or !p.eat('(')) return false;
        const tool = schemas.get(try std.ascii.allocLowerString(a, name)) orelse return false;
        const args = (try arguments(a, p, schemas, name, tool.properties.keys())) orelse return false;
        if (p.i <= markup) try push(a, list, name, args);
        if (p.eat(',')) continue;
        if (p.eat(']')) break;
        return false;
    }
    p.space();
    return p.i == p.s.len;
}

/// One call's invoke, added to `list`.
fn push(a: Allocator, list: *std.ArrayList([]const u8), name: []const u8, args: *json.Object) Allocator.Error!void {
    var out: std.ArrayList(u8) = .empty;
    try invoke(a, &out, name, args);
    try list.append(a, out.items);
}

/// One call as a DSML invoke, its arguments in order.
fn invoke(a: Allocator, out: *std.ArrayList(u8), name: []const u8, args: *json.Object) Allocator.Error!void {
    try out.print(a, "\n<\u{ff5c}DSML\u{ff5c}invoke name=\"{s}\">", .{name});
    for (args.keys(), args.values()) |k, v| {
        if (v == .string) {
            try out.print(a, "<\u{ff5c}DSML\u{ff5c}parameter name=\"{s}\" string=\"true\">{s}</\u{ff5c}DSML\u{ff5c}parameter>", .{ k, v.string });
        } else {
            const spelled = json.stringify(a, v, .{ .ascii = false }) catch return error.OutOfMemory;
            try out.print(a, "<\u{ff5c}DSML\u{ff5c}parameter name=\"{s}\" string=\"false\">{s}</\u{ff5c}DSML\u{ff5c}parameter>", .{ k, spelled });
        }
    }
    try out.appendSlice(a, "</\u{ff5c}DSML\u{ff5c}invoke>");
}

/// A JSON list of `{"name", "arguments"}` calls (`parameters` for `arguments`, or a JSON string of them) as invokes.
fn jsonCalls(
    a: Allocator,
    list: *std.ArrayList([]const u8),
    p: *Scanner,
    schemas: *const tool_params.Schemas,
    markup: usize,
) Allocator.Error!bool {
    for (p.s[0..p.i]) |c| if (c != '[' and !Scanner.jsonBlank(c)) return false; // only the blanks JSON allows
    while (true) {
        p.jsonSpace();
        const call = switch (try json.parseText(a, p.object() orelse return false)) {
            .ok => |v| v,
            .err => return false,
        };
        const name = call.get("name") orelse return false;
        if (name != .string or name.string.len == 0) return false;
        if (schemas.get(try std.ascii.allocLowerString(a, name.string)) == null) return false;
        var given = call.get("arguments") orelse call.get("parameters") orelse Value{ .object = try json.newObject(a) };
        if (given == .string) given = switch (try json.parseText(a, given.string)) {
            .ok => |v| v,
            .err => return false,
        };
        if (given != .object) return false;
        const args = try json.newObject(a);
        for (given.object.keys(), given.object.values()) |k, v| {
            if (std.mem.indexOfScalar(u8, k, '"') != null or !tool_params.finite(v)) return false;
            const raw = json.stringify(a, v, .{ .ascii = false }) catch return error.OutOfMemory;
            try args.put(a, k, try coerce(a, v, raw, try tool_params.schemaOf(schemas, name.string, k, a)));
        }
        if (p.i <= markup) try push(a, list, name.string, args);
        p.jsonSpace();
        if (p.take(',')) continue;
        if (p.take(']')) break;
        return false;
    }
    p.jsonSpace();
    return p.i == p.s.len;
}

/// A call's arguments up to its `)`, positionals named in schema property order; null if Python would refuse them.
fn arguments(
    a: Allocator,
    p: *Scanner,
    schemas: *const tool_params.Schemas,
    name: []const u8,
    order: []const []const u8,
) Allocator.Error!?*json.Object {
    const args = try json.newObject(a);
    var keyword = false;
    var positional: usize = 0;
    if (p.eat(')')) return args;
    while (true) {
        const named = p.keyword();
        if (named == null and (keyword or positional >= order.len)) return null; // positional after a keyword, or past the properties
        keyword = named != null;
        const key = named orelse blk: {
            positional += 1;
            break :blk order[positional - 1];
        };
        if (std.mem.indexOfScalar(u8, key, '"') != null or args.contains(key)) return null;
        const raw = p.value() orelse return null;
        const v = (try literal(a, raw)) orelse return null;
        try args.put(a, key, try coerce(a, v, raw, try tool_params.schemaOf(schemas, name, key, a)));
        if (p.eat(',')) {
            if (p.eat(')')) return args;
            continue;
        }
        if (p.eat(')')) return args;
        return null;
    }
}

/// A value as JSON spells it, else as Python does.
fn literal(a: Allocator, raw: []const u8) Allocator.Error!?Value {
    const v = switch (try json.parseText(a, raw)) {
        .ok => |v| v,
        .err => (try tool_params.pythonLiteral(a, raw)) orelse return null,
    };
    return if (tool_params.finite(v)) v else null;
}

/// A value the schema does not take, read as the type it names: "4" as 4, 4.0 as 4, 1 as true, 123 as "123".
fn coerce(a: Allocator, v: Value, raw: []const u8, schema: tool_params.Schema) Allocator.Error!Value {
    const kinds = (try tool_kinds.schemaKinds(a, schema.value, schema.root)) orelse return v;
    switch (v) {
        .string => return if (kinds.contains(.string)) v else try tool_params.decode(a, v.string, schema, true),
        .int, .float, .bool => {
            if (tool_kinds.accepts(kinds, v)) return v;
            if (kinds.contains(.string)) return .{ .string = raw };
            if (v == .float and kinds.contains(.integer) and @trunc(v.float) == v.float and @abs(v.float) < 1e15) {
                return .{ .int = try std.fmt.allocPrint(a, "{d}", .{@as(i64, @intFromFloat(v.float))}) };
            }
            if (v == .int and kinds.contains(.boolean) and (std.mem.eql(u8, v.int, "0") or std.mem.eql(u8, v.int, "1"))) {
                return .{ .bool = v.int[0] == '1' };
            }
            return v;
        },
        else => return v,
    }
}

/// The block's text, read left to right; every reader skips the blanks before what it reads.
const Scanner = struct {
    s: []const u8,
    i: usize = 0,

    fn space(p: *Scanner) void {
        while (p.i < p.s.len and reply_text.isSpaceByte(p.s[p.i])) p.i += 1;
    }

    fn jsonBlank(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    fn jsonSpace(p: *Scanner) void {
        while (p.i < p.s.len and jsonBlank(p.s[p.i])) p.i += 1;
    }

    /// `c` at the cursor, consumed, with no blanks skipped.
    fn take(p: *Scanner, c: u8) bool {
        if (p.i >= p.s.len or p.s[p.i] != c) return false;
        p.i += 1;
        return true;
    }

    /// The `{...}` at the cursor, braces matched outside strings, consumed; null when absent or not yet closed.
    fn object(p: *Scanner) ?[]const u8 {
        const start = p.i;
        if (start >= p.s.len or p.s[start] != '{') return null;
        var depth: usize = 0;
        var j = start;
        while (j < p.s.len) : (j += 1) switch (p.s[j]) {
            '"' => {
                j += 1;
                while (j < p.s.len and p.s[j] != '"') j += if (p.s[j] == '\\') 2 else 1;
            },
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if (depth == 0) {
                    p.i = j + 1;
                    return p.s[start..p.i];
                }
            },
            else => {},
        };
        return null;
    }

    fn eat(p: *Scanner, c: u8) bool {
        p.space();
        if (p.i >= p.s.len or p.s[p.i] != c) return false;
        p.i += 1;
        return true;
    }

    /// A tool's name: letters, digits, `_`, `-` and `.`.
    fn name(p: *Scanner) []const u8 {
        p.space();
        const start = p.i;
        while (p.i < p.s.len and (std.ascii.isAlphanumeric(p.s[p.i]) or std.mem.indexOfScalar(u8, "_-.", p.s[p.i]) != null)) p.i += 1;
        return p.s[start..p.i];
    }

    /// `key=` at the cursor, consumed; null and the cursor kept when the argument is positional.
    fn keyword(p: *Scanner) ?[]const u8 {
        p.space();
        const start = p.i;
        var j = p.i;
        while (j < p.s.len and (std.ascii.isAlphanumeric(p.s[j]) or p.s[j] == '_')) j += 1;
        if (j == start or std.ascii.isDigit(p.s[start])) return null;
        var k = j;
        while (k < p.s.len and reply_text.isSpaceByte(p.s[k])) k += 1;
        if (k >= p.s.len or p.s[k] != '=' or (k + 1 < p.s.len and p.s[k + 1] == '=')) return null;
        p.i = k + 1;
        return p.s[start..j];
    }

    /// One argument's text up to the `,` or `)` that ends it outside strings and brackets.
    fn value(p: *Scanner) ?[]const u8 {
        p.space();
        const start = p.i;
        var depth: usize = 0;
        while (p.i < p.s.len) {
            const c = p.s[p.i];
            switch (c) {
                '"', '\'' => {
                    p.i += 1;
                    while (p.i < p.s.len and p.s[p.i] != c) p.i += if (p.s[p.i] == '\\') 2 else 1;
                    if (p.i >= p.s.len) return null;
                },
                '[', '(', '{' => depth += 1,
                ']', ')', '}' => {
                    if (depth == 0) return if (c == ')' and p.i > start) reply_text.pyRstrip(p.s[start..p.i]) else null;
                    depth -= 1;
                },
                ',' => if (depth == 0) return if (p.i > start) reply_text.pyRstrip(p.s[start..p.i]) else null,
                else => {},
            }
            p.i += 1;
        }
        return null;
    }
};

const tool_parse = @import("tool_parse.zig");

const test_tools =
    \\[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"filters":{"type":"object"}}}}},
    \\ {"type":"function","function":{"name":"web-search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
;

/// Each call of the whole reply as `name arguments`, after the rewrite.
fn whole(a: Allocator, text: []const u8, tools: []const Value) ![]const []const u8 {
    const r = try tool_parse.parse(a, try dsml(a, text, tools, true), tools, null);
    var out: std.ArrayList([]const u8) = .empty;
    for (r.calls orelse &.{}) |c| {
        const f = c.get("function").?;
        try out.append(a, try std.fmt.allocPrint(a, "{s} {s}", .{ f.get("name").?.string, f.get("arguments").?.string }));
    }
    return out.items;
}

test "tool_lfm2: a block of one call is the call, and no content is left" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const reply = "<|tool_call_start|>[lookup(query=\"fast\", limit=4)]<|tool_call_end|>";
    const r = try tool_parse.parse(a, try dsml(a, reply, tools, true), tools, null);
    try std.testing.expectEqualStrings("", r.content);
    try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
    const function = r.calls.?[0].get("function").?;
    try std.testing.expectEqualStrings("lookup", function.get("name").?.string);
    try std.testing.expectEqualStrings("{\"query\":\"fast\",\"limit\":4}", function.get("arguments").?.string);
    // the stream hides the block as LFM2 wrote it, while it streams and once it ends
    try std.testing.expectEqualStrings("", try reply_text.hideToolCalls(a, reply, false));
    try std.testing.expectEqualStrings("", try reply_text.finishedProse(a, reply));
}

test "tool_lfm2: positional and keyword arguments, nested values in either spelling, two calls in one list" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const cases = [_]struct { []const u8, []const []const u8 }{
        .{ "[lookup(\"fast\", limit=4)]", &.{"lookup {\"query\":\"fast\",\"limit\":4}"} },
        .{ "[lookup('fast', 4,)]", &.{"lookup {\"query\":\"fast\",\"limit\":4}"} },
        .{ "[lookup(query='a, b)', filters={'tags': [1, True, None], 'next': {'y': (False,)}})]", &.{"lookup {\"query\":\"a, b)\",\"filters\":{\"tags\":[1,true,null],\"next\":{\"y\":[false]}}}"} },
        .{ "[lookup(filters={\"tags\": [1.5, true, null], \"next\": {\"y\": []}})]", &.{"lookup {\"filters\":{\"tags\":[1.5,true,null],\"next\":{\"y\":[]}}}"} },
        .{ "[lookup(query=\"say \\\"hi\\\"\"), web-search(q=\"zig\")]", &.{ "lookup {\"query\":\"say \\\"hi\\\"\"}", "web-search {\"q\":\"zig\"}" } },
        .{ " [ Lookup ( ) ] ", &.{"lookup {}"} },
    };
    for (cases) |c| {
        const got = try whole(a, try std.mem.concat(a, u8, &.{ open, c[0], close }), tools);
        try std.testing.expectEqual(c[1].len, got.len);
        for (c[1], got) |want, have| try std.testing.expectEqualStrings(want, have);
    }
}

test "tool_lfm2: a block Python or the request would refuse stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const bodies = [_][]const u8{
        "[other(q=1)]", // a tool the request did not offer
        "[lookup(limit=4, \"fast\")]", // positional after keyword
        "[lookup(\"a\", 1, {}, 2)]", // more positional arguments than properties
        "[lookup(query=\"a\", query=\"b\")]",
        "[lookup(query=fast)]", // a name is no literal
        "[lookup(query=\"open)]",
        "lookup(query=\"x\")",
        "[lookup(query=\"x\")] trailing",
    };
    for (bodies) |body| {
        const text = try std.mem.concat(a, u8, &.{ "Hi ", open, body, close });
        try std.testing.expectEqualStrings(text, try dsml(a, text, tools, true));
        const r = try tool_parse.parse(a, try dsml(a, text, tools, true), tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
    // a refused block keeps its text, and the good block beside it still becomes a call
    const mixed = open ++ "[other()]" ++ close ++ " and " ++ open ++ "[lookup(4)]" ++ close;
    const r = try tool_parse.parse(a, try dsml(a, mixed, tools, true), tools, null);
    try std.testing.expectEqualStrings(open ++ "[other()]" ++ close ++ " and", r.content);
    try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
}

test "tool_lfm2: a block still being written is held back while streaming and stays text once the reply ends" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const cut = "Checking. <|tool_call_start|>[lookup(query=\"fa";
    try std.testing.expectEqualStrings("Checking. ", try dsml(a, cut, tools, false));
    try std.testing.expectEqualStrings("Checking. ", try reply_text.hideToolCalls(a, cut, false));
    // as an unclosed DSML block does: the reply's content keeps it, the stream's prose does not show it
    try std.testing.expectEqualStrings(cut, try dsml(a, cut, tools, true));
    const r = try tool_parse.parse(a, try dsml(a, cut, tools, true), tools, null);
    try std.testing.expect(r.calls == null);
    try std.testing.expectEqualStrings(cut, r.content);
    // a finished reply's unclosed envelope is text (#1243), as the content above keeps it
    try std.testing.expectEqualStrings(cut, try reply_text.finishedProse(a, cut));
}

/// Each call a stream sends when `reply` arrives a byte at a time, as `name arguments`.
fn streamed(a: Allocator, reply: []const u8, tools: []const Value) ![]const []const u8 {
    var s = try @import("tool_stream.zig").Streamer.init(a, tools);
    var deltas: std.ArrayList(Value) = .empty;
    for (1..reply.len + 1) |n| try s.feed(try dsml(a, reply[0..n], tools, n == reply.len), &deltas);
    var names: std.ArrayList([]const u8) = .empty;
    var args: std.ArrayList(std.ArrayList(u8)) = .empty;
    for (deltas.items) |d| for (d.get("tool_calls").?.array) |one| {
        const index: usize = @intCast(one.get("index").?.int64().?);
        const f = one.get("function").?;
        if (f.get("name")) |n| {
            try std.testing.expectEqual(names.items.len, index);
            try names.append(a, n.string);
            try args.append(a, .empty);
        }
        try args.items[index].appendSlice(a, f.get("arguments").?.string);
    };
    var out: std.ArrayList([]const u8) = .empty;
    for (names.items, args.items) |n, arg| try out.append(a, try std.fmt.allocPrint(a, "{s} {s}", .{ n, arg.items }));
    return out.items;
}

test "tool_lfm2: the calls a stream sends at every byte boundary are the whole reply's calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const reply = "Checking. <|tool_call_start|>[lookup(query=\"fast\", limit=4), web-search('zig')]<|tool_call_end|>" ++
        " then <|tool_call_start|>[lookup(filters={\"a\": [True, None]})]<|tool_call_end|>" ++
        " and <|tool_call_start|>[{\"name\":\"lookup\",\"arguments\":{\"limit\":\"2\"}}, {\"name\":\"web-search\"}]<|tool_call_end|>" ++
        " last <|tool_call_start|>[lookup(query=\"a\"), web-search(q=\"b\"), lookup(4)]<|tool_call_end|>";
    const want = try whole(a, reply, tools);
    try std.testing.expectEqual(@as(usize, 8), want.len);
    const got = try streamed(a, reply, tools);
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
}

test "tool_lfm2: a value is read as the type the schema names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"typed","parameters":{"type":"object","properties":{
        \\"s":{"type":"string"},"n":{"type":"integer"},"b":{"type":"boolean"},"r":{"type":"number"},"u":{"anyOf":[{"type":"string"},{"type":"integer"}]}}}}}]
    )).ok.array;
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "[typed(s=123, n=4, b=True)]", "typed {\"s\":\"123\",\"n\":4,\"b\":true}" },
        .{ "[typed(s=True, n='4', b='true')]", "typed {\"s\":\"True\",\"n\":4,\"b\":true}" },
        .{ "[typed(n=4.0, b=1, r=2)]", "typed {\"n\":4,\"b\":true,\"r\":2}" },
        .{ "[typed(n=4.5, b=0, r='1.5')]", "typed {\"n\":4.5,\"b\":false,\"r\":1.5}" },
        .{ "[typed(s='12', u='7', n='x')]", "typed {\"s\":\"12\",\"u\":\"7\",\"n\":\"x\"}" },
    };
    for (cases) |c| {
        const got = try whole(a, try std.mem.concat(a, u8, &.{ open, c[0], close }), tools);
        try std.testing.expectEqual(@as(usize, 1), got.len);
        try std.testing.expectEqualStrings(c[1], got[0]);
    }
}

test "tool_lfm2: a JSON list of calls is the same calls, with the schema's types" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const cases = [_]struct { []const u8, []const []const u8 }{
        .{ "[{\"name\":\"lookup\",\"arguments\":{\"query\":\"x\",\"limit\":4}}]", &.{"lookup {\"query\":\"x\",\"limit\":4}"} },
        .{ " [ {\"name\": \"Lookup\", \"parameters\": {\"query\": \"x\"}} , {\"name\": \"web-search\"} ] ", &.{ "lookup {\"query\":\"x\"}", "web-search {}" } },
        .{ "[{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"a, b\\\",\\\"filters\\\":{\\\"t\\\":[1,null]}}\"}]", &.{"lookup {\"query\":\"a, b\",\"filters\":{\"t\":[1,null]}}"} },
        .{ "[{\"name\":\"lookup\",\"arguments\":{\"query\":123,\"limit\":\"4\"}}]", &.{"lookup {\"query\":\"123\",\"limit\":4}"} },
        .{ "[{\"name\":\"lookup\",\"arguments\":{\"limit\":4.0}},{\"name\":\"lookup\",\"arguments\":{\"limit\":true}}]", &.{ "lookup {\"limit\":4}", "lookup {\"limit\":true}" } },
    };
    for (cases) |c| {
        const got = try whole(a, try std.mem.concat(a, u8, &.{ open, c[0], close }), tools);
        try std.testing.expectEqual(c[1].len, got.len);
        for (c[1], got) |want, have| try std.testing.expectEqualStrings(want, have);
    }
}

test "tool_lfm2: a JSON block the request would refuse stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const bodies = [_][]const u8{
        "[{\"name\":\"other\",\"arguments\":{}}]", // a tool the request did not offer
        "[{\"name\":\"lookup\",\"arguments\":{}}, {\"name\":\"other\"}]", // one bad call spoils the list
        "[{\"arguments\":{\"query\":\"x\"}}]",
        "[{\"name\":4,\"arguments\":{}}]",
        "[{\"name\":\"lookup\",\"arguments\":[1]}]",
        "[{\"name\":\"lookup\",\"arguments\":\"not json\"}]",
        "[{\"name\":\"lookup\",\"arguments\":\"[1]\"}]",
        "[{\"name\":\"lookup\",\"arguments\":{\"query\":\"x\"}}", // no closing bracket
        "[{\"name\":\"lookup\",\"arguments\":{\"query\":\"x\"}}] trailing",
        "[{\"name\":\"lookup\",\"arguments\":{\"query\":\"x\"}},]",
        "[{\"name\":\"lookup\"}, lookup()]", // the two forms do not mix
        "{\"name\":\"lookup\",\"arguments\":{}}", // not a list
        "[]",
    };
    for (bodies) |body| {
        const text = try std.mem.concat(a, u8, &.{ "Hi ", open, body, close });
        try std.testing.expectEqualStrings(text, try dsml(a, text, tools, true));
        const r = try tool_parse.parse(a, try dsml(a, text, tools, true), tools, null);
        try std.testing.expect(r.calls == null);
    }
}

test "tool_lfm2: a JSON block still being written is held back, and its calls stream the same as the whole reply's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, test_tools)).ok.array;
    const cut = "Checking. <|tool_call_start|>[{\"name\":\"lookup\",\"arguments\":{\"query\":\"fa";
    try std.testing.expectEqualStrings("Checking. ", try dsml(a, cut, tools, false));
    try std.testing.expectEqualStrings("Checking. ", try reply_text.hideToolCalls(a, cut, false));
    const reply = "Checking. <|tool_call_start|>[{\"name\":\"lookup\",\"arguments\":{\"query\":\"fast\",\"limit\":\"4\"}}," ++
        " {\"name\":\"web-search\",\"parameters\":{\"q\":\"zig\"}}]<|tool_call_end|>";
    const want = try whole(a, reply, tools);
    try std.testing.expectEqual(@as(usize, 2), want.len);
    const got = try streamed(a, reply, tools);
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
    try std.testing.expectEqualStrings("lookup {\"query\":\"fast\",\"limit\":4}", got[0]);
}
