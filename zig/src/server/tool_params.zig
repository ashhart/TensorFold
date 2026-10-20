//! Typed tool parameters: XML parameter text decoded by the offered schema (``tool_parameters.py``).
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Valid JSON object arguments, before any legacy markup conversion can rewrite their string contents.
pub fn parseObject(a: Allocator, text: []const u8) Allocator.Error!?Value {
    const v = switch (try json.parseText(a, text)) {
        .ok => |v| v,
        .err => return null,
    };
    return if (v == .object) v else null;
}

/// One offered tool's parameter schemas: its properties, and the root its references point into.
pub const Tool = struct {
    properties: *const json.Object,
    root: Value,
};

/// Each offered tool's parameter schemas by lowercase name.
pub const Schemas = std.StringHashMapUnmanaged(Tool);

pub fn schemas(a: Allocator, tools: []const Value) Allocator.Error!Schemas {
    var out: Schemas = .empty;
    const empty = try json.newObject(a);
    for (tools) |tool| {
        if (tool != .object) continue;
        const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
        var parameters: Value = .{ .object = empty };
        if (function.get("parameters")) |p| if (p.truthy()) {
            parameters = p;
        };
        if (!parameters.truthy()) if (function.get("input_schema")) |p| if (p.truthy()) {
            parameters = p;
        };
        var properties: Value = .{ .object = empty };
        if (parameters == .object) if (parameters.get("properties")) |p| if (p.truthy()) {
            properties = p;
        };
        const kept = try json.newObject(a);
        if (properties == .object) for (properties.object.keys(), properties.object.values()) |k, v| {
            if (v == .object) try kept.put(a, k, v);
        };
        const name = if (function.get("name")) |n| try tool_specs.pyStr(a, n) else "";
        try out.put(a, try std.ascii.allocLowerString(a, name), .{ .properties = kept, .root = parameters });
    }
    return out;
}

/// One parameter's schema, with the root its references point into.
pub const Schema = struct {
    value: Value,
    root: Value,
};

/// The schema of ``key`` in tool ``name``, or an empty one.
pub fn schemaOf(s: *const Schemas, name: []const u8, key: []const u8, a: Allocator) Allocator.Error!Schema {
    const lower = try std.ascii.allocLowerString(a, name);
    const tool = s.get(lower) orelse return try emptySchema(a);
    return .{ .value = tool.properties.get(key) orelse .{ .object = try json.newObject(a) }, .root = tool.root };
}

/// The schema of a parameter no tool declares.
pub fn emptySchema(a: Allocator) Allocator.Error!Schema {
    const empty = try json.newObject(a);
    return .{ .value = .{ .object = empty }, .root = .{ .object = empty } };
}

/// The JSON kinds a schema can name.
const Kind = enum { array, object, boolean, integer, number, null, string };

/// The kinds a schema accepts.
const Kinds = std.EnumSet(Kind);

/// The kind a JSON value is.
fn kindOfValue(v: Value) ?Kind {
    return switch (v) {
        .array => .array,
        .object => .object,
        .bool => .boolean,
        .int => .integer,
        .float => .number,
        .null => .null,
        .string => .string,
    };
}

/// How deep a chain of references may go before the schema is left alone.
const max_ref_depth = 32;

/// The kinds a schema accepts, with the references it points into resolved.
const Resolver = struct {
    a: Allocator,
    root: Value,
    /// The references on the path, so a cycle stops.
    path: [max_ref_depth][]const u8 = @splat(""),
    depth: usize = 0,

    /// The kinds the schema accepts: its own kinds and its combiners' kinds.
    fn kinds(self: *Resolver, schema: Value) Allocator.Error!?Kinds {
        const own = try self.ownKinds(schema);
        const combined = try self.combinedKinds(schema);
        if (own) |o| {
            if (combined) |c| return o.intersectWith(c);
            return o;
        }
        return combined;
    }

    /// The kinds the schema names itself: its type, its enum or const, and what its reference points at.
    fn ownKinds(self: *Resolver, schema: Value) Allocator.Error!?Kinds {
        if (schema != .object) return null;
        var named: ?Kinds = null;
        named = addKinds(named, typeKinds(schema.get("type")));
        if (schema.get("enum")) |e| {
            if (e == .array) named = addKinds(named, valueKinds(e.array));
        }
        if (schema.get("const")) |c| named = addKinds(named, valueKinds(&.{c}));
        if (schema.get("$ref")) |r| {
            if (r == .string) named = addKinds(named, try self.refKinds(r.string));
        }
        return named;
    }

    /// The kinds the schema's combiners name: oneOf and anyOf take the union, allOf the intersection.
    fn combinedKinds(self: *Resolver, schema: Value) Allocator.Error!?Kinds {
        if (schema != .object) return null;
        var either: ?Kinds = null;
        for ([_][]const u8{ "oneOf", "anyOf" }) |key| {
            const branches = schema.get(key) orelse continue;
            if (branches != .array) continue;
            for (branches.array) |branch| {
                const branch_kinds = (try self.kinds(branch)) orelse continue;
                if (branch_kinds.count() == 0) continue;
                either = if (either) |e| e.unionWith(branch_kinds) else branch_kinds;
            }
        }
        var all: ?Kinds = null;
        if (schema.get("allOf")) |branches| {
            if (branches == .array) for (branches.array) |branch| {
                const branch_kinds = (try self.kinds(branch)) orelse continue;
                if (branch_kinds.count() == 0) continue;
                all = if (all) |a| a.intersectWith(branch_kinds) else branch_kinds;
            };
        }
        if (either) |e| {
            if (all) |a| return e.intersectWith(a);
            return e;
        }
        return all;
    }

    /// The kinds the reference points at, or null when it points nowhere.
    fn refKinds(self: *Resolver, ref: []const u8) Allocator.Error!?Kinds {
        if (self.depth >= max_ref_depth) return null;
        for (self.path[0..self.depth]) |seen| if (std.mem.eql(u8, seen, ref)) return null;
        const target = (try resolvePointer(self.a, self.root, ref)) orelse return null;
        self.path[self.depth] = ref;
        self.depth += 1;
        defer self.depth -= 1;
        return try self.kinds(target);
    }
};

/// The kinds a schema's type names: one name, or a list of them.
fn typeKinds(t: ?Value) ?Kinds {
    const v = t orelse return null;
    var kinds: Kinds = .empty;
    switch (v) {
        .string => if (std.meta.stringToEnum(Kind, v.string)) |k| kinds.insert(k),
        .array => for (v.array) |one| if (one == .string) if (std.meta.stringToEnum(Kind, one.string)) |k| kinds.insert(k),
        else => {},
    }
    return if (kinds.count() > 0) kinds else null;
}

/// The kinds the values of an enum or a const are.
fn valueKinds(values: []const Value) ?Kinds {
    var kinds: Kinds = .empty;
    for (values) |v| if (kindOfValue(v)) |k| kinds.insert(k);
    return if (kinds.count() > 0) kinds else null;
}

/// The kinds the two sets share, with a set that names none adding nothing.
fn addKinds(kinds: ?Kinds, more: ?Kinds) ?Kinds {
    const m = more orelse return kinds;
    if (m.count() == 0) return kinds;
    return if (kinds) |k| k.unionWith(m) else m;
}

/// The value a local reference points at, or null when it points nowhere.
fn resolvePointer(a: Allocator, root: Value, ref: []const u8) Allocator.Error!?Value {
    if (ref.len == 0 or ref[0] != '#') return null;
    var current = root;
    const rest = ref[1..];
    if (rest.len == 0) return current;
    if (rest[0] != '/') return null;
    var tokens = std.mem.splitScalar(u8, rest[1..], '/');
    while (tokens.next()) |token| {
        const name = try unescapePointer(a, token);
        switch (current) {
            .object => current = current.object.get(name) orelse return null,
            .array => {
                const index = std.fmt.parseInt(usize, name, 10) catch return null;
                if (index >= current.array.len) return null;
                current = current.array[index];
            },
            else => return null,
        }
    }
    return current;
}

/// The JSON pointer token with its `~1` and `~0` unescaped.
fn unescapePointer(a: Allocator, token: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, token, '~') == null) return token;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < token.len) : (i += 1) {
        if (token[i] != '~' or i + 1 >= token.len) {
            try out.append(a, token[i]);
            continue;
        }
        i += 1;
        try out.append(a, if (token[i] == '1') '/' else if (token[i] == '0') '~' else '~');
    }
    return out.items;
}

/// The schema takes a value that is not a string.
fn takesJson(kinds: Kinds) bool {
    var without_string = kinds;
    without_string.remove(.string);
    return without_string.count() > 0;
}

/// The parsed value is one the schema accepts; a string keeps the text the model wrote.
fn accepts(kinds: Kinds, v: Value) bool {
    const kind = kindOfValue(v) orelse return false;
    if (kind == .string) return false;
    if (kinds.contains(kind)) return true;
    return kind == .integer and kinds.contains(.number);
}

/// The schema takes a value that is not a string: a union of them, a list of types, or a plain typed schema.
pub fn schemaTakesJson(a: Allocator, schema: Schema) Allocator.Error!bool {
    var resolver: Resolver = .{ .a = a, .root = schema.root };
    const kinds = (try resolver.kinds(schema.value)) orelse return false;
    return takesJson(kinds);
}

/// ``json.dumps(v, allow_nan=False)`` succeeds: no NaN or infinity anywhere.
pub fn finite(v: Value) bool {
    return switch (v) {
        .float => |f| std.math.isFinite(f),
        .array => |items| for (items) |i| {
            if (!finite(i)) break false;
        } else true,
        .object => |o| for (o.values()) |i| {
            if (!finite(i)) break false;
        } else true,
        else => true,
    };
}

/// ``closed_json``: the arrays and objects ``text`` left open, closed; null unless it only stops short of them.
pub fn closedJson(a: Allocator, text: []const u8) Allocator.Error!?[]const u8 {
    var closers: std.ArrayList(u8) = .empty;
    var in_string = false;
    var escaped = false;
    for (text) |ch| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else {
                escaped = ch == '\\';
                in_string = ch != '"';
            }
        } else if (ch == '"') {
            in_string = true;
        } else if (ch == '[' or ch == '{') {
            try closers.append(a, if (ch == '[') ']' else '}');
        } else if (ch == ']' or ch == '}') {
            if (closers.items.len == 0 or closers.pop().? != ch) return null;
        }
    }
    if (closers.items.len == 0 or in_string) return null;
    std.mem.reverse(u8, closers.items);
    return try std.mem.concat(a, u8, &.{ reply_text.pyRstrip(text), closers.items });
}

/// ``decode_parameter``: the schema's type when the text spells one, else the text; ``python`` also reads Python's spelling.
pub fn decode(a: Allocator, value: []const u8, schema: Schema, python: bool) Allocator.Error!Value {
    var resolver: Resolver = .{ .a = a, .root = schema.root };
    const kinds = (try resolver.kinds(schema.value)) orelse return .{ .string = value };
    if (!takesJson(kinds)) return .{ .string = value };
    const closed: ?[]const u8 = if (kinds.contains(.array) or kinds.contains(.object)) try closedJson(a, value) else null;
    for ([_]?[]const u8{ value, closed }) |candidate| {
        const text = candidate orelse continue;
        const parsed = switch (try json.parseText(a, text)) {
            .ok => |v| v,
            .err => continue,
        };
        if (!finite(parsed)) continue;
        return if (accepts(kinds, parsed)) parsed else .{ .string = value };
    }
    if (!python) return .{ .string = value };
    const parsed = (try pythonLiteral(a, value)) orelse return .{ .string = value };
    if (!finite(parsed)) return .{ .string = value };
    return if (accepts(kinds, parsed)) parsed else .{ .string = value };
}

/// ``_python_literal``: True, None, quoted strings, numbers, lists, tuples and dicts; null if ``text`` is not one.
pub fn pythonLiteral(a: Allocator, text: []const u8) Allocator.Error!?Value {
    const stripped = reply_text.pyStrip(text);
    var lower_buf: [8]u8 = undefined;
    if (stripped.len <= 5) {
        const lower = std.ascii.lowerString(&lower_buf, stripped);
        if (std.mem.eql(u8, lower, "true")) return .{ .bool = true };
        if (std.mem.eql(u8, lower, "false")) return .{ .bool = false };
        if (std.mem.eql(u8, lower, "none") or std.mem.eql(u8, lower, "null")) return .null;
    }
    var p: Literal = .{ .a = a, .s = stripped };
    const v = p.expr(0) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Invalid => return null,
    };
    p.space();
    return if (p.i == p.s.len) v else null;
}

const Literal = struct {
    a: Allocator,
    s: []const u8,
    i: usize = 0,

    const Error = error{ Invalid, OutOfMemory };

    fn space(p: *Literal) void {
        while (p.i < p.s.len and (p.s[p.i] == ' ' or p.s[p.i] == '\t' or p.s[p.i] == '\n' or p.s[p.i] == '\r')) p.i += 1;
    }

    fn peek(p: *Literal) ?u8 {
        p.space();
        return if (p.i < p.s.len) p.s[p.i] else null;
    }

    fn expr(p: *Literal, depth: usize) Error!Value {
        if (depth > 200) return error.Invalid;
        const c = p.peek() orelse return error.Invalid;
        switch (c) {
            '[' => return p.sequence(']', depth),
            '(' => return p.sequence(')', depth),
            '{' => return p.dict(depth),
            '\'', '"' => return p.strings(),
            'r', 'R', 'u', 'U' => if (p.i + 1 < p.s.len and (p.s[p.i + 1] == '\'' or p.s[p.i + 1] == '"')) return p.strings(),
            else => {},
        }
        if (std.mem.startsWith(u8, p.s[p.i..], "None") and !p.wordAt(p.i + 4)) return p.word(4, .null);
        if (std.mem.startsWith(u8, p.s[p.i..], "True") and !p.wordAt(p.i + 4)) return p.word(4, .{ .bool = true });
        if (std.mem.startsWith(u8, p.s[p.i..], "False") and !p.wordAt(p.i + 5)) return p.word(5, .{ .bool = false });
        return p.number();
    }

    fn wordAt(p: *Literal, at: usize) bool {
        return at < p.s.len and (std.ascii.isAlphanumeric(p.s[at]) or p.s[at] == '_');
    }

    fn word(p: *Literal, n: usize, v: Value) Value {
        p.i += n;
        return v;
    }

    fn sequence(p: *Literal, close: u8, depth: usize) Error!Value {
        p.i += 1;
        var items: std.ArrayList(Value) = .empty;
        var commas: usize = 0;
        while (true) {
            if (p.peek() == close) {
                p.i += 1;
                if (close == ')' and items.items.len == 1 and commas == 0) return items.items[0]; // (x) is x
                return .{ .array = items.items };
            }
            try items.append(p.a, try p.expr(depth + 1));
            const next = p.peek() orelse return error.Invalid;
            if (next == ',') {
                p.i += 1;
                commas += 1;
            } else if (next != close) return error.Invalid;
        }
    }

    fn dict(p: *Literal, depth: usize) Error!Value {
        p.i += 1;
        const o = try json.newObject(p.a);
        if (p.peek() == '}') {
            p.i += 1;
            return .{ .object = o };
        }
        while (true) {
            const key = try p.expr(depth + 1);
            if (p.peek() != ':') return error.Invalid; // a set is no JSON value
            p.i += 1;
            const v = try p.expr(depth + 1);
            const name: []const u8 = switch (key) {
                .string => |s| s,
                .int => |t| t,
                .bool => |b| if (b) "true" else "false",
                .null => "null",
                .float => |f| blk: {
                    var buf: [40]u8 = undefined;
                    break :blk try p.a.dupe(u8, json.floatRepr(&buf, f));
                },
                else => return error.Invalid,
            };
            try o.put(p.a, name, v);
            const next = p.peek() orelse return error.Invalid;
            if (next == ',') {
                p.i += 1;
                if (p.peek() == '}') {
                    p.i += 1;
                    return .{ .object = o };
                }
            } else if (next == '}') {
                p.i += 1;
                return .{ .object = o };
            } else return error.Invalid;
        }
    }

    /// One or more adjacent string literals, joined as Python's parser joins them.
    fn strings(p: *Literal) Error!Value {
        var out: std.ArrayList(u8) = .empty;
        var any = false;
        while (p.peek()) |c| {
            var raw = false;
            if (c == 'r' or c == 'R' or c == 'u' or c == 'U') {
                if (p.i + 1 >= p.s.len or (p.s[p.i + 1] != '\'' and p.s[p.i + 1] != '"')) break;
                raw = c == 'r' or c == 'R';
                p.i += 1;
            } else if (c != '\'' and c != '"') break;
            try p.string(&out, raw);
            any = true;
        }
        if (!any) return error.Invalid;
        return .{ .string = out.items };
    }

    fn string(p: *Literal, out: *std.ArrayList(u8), raw: bool) Error!void {
        const q = p.s[p.i];
        const triple = p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q;
        p.i += if (triple) 3 else 1;
        while (true) {
            if (p.i >= p.s.len) return error.Invalid;
            const c = p.s[p.i];
            if (c == q and (!triple or (p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q))) {
                p.i += if (triple) 3 else 1;
                return;
            }
            if (c == '\n' and !triple) return error.Invalid;
            if (c != '\\') {
                try out.append(p.a, c);
                p.i += 1;
                continue;
            }
            if (p.i + 1 >= p.s.len) return error.Invalid;
            const e = p.s[p.i + 1];
            if (raw) {
                try out.appendSlice(p.a, p.s[p.i .. p.i + 2]);
                p.i += 2;
                continue;
            }
            p.i += 2;
            const simple: ?u8 = switch (e) {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                'a' => 7,
                'b' => 8,
                'f' => 12,
                'v' => 11,
                '0'...'7' => blk: {
                    var v: u21 = e - '0';
                    var n: usize = 1;
                    while (n < 3 and p.i < p.s.len and p.s[p.i] >= '0' and p.s[p.i] <= '7') : (n += 1) {
                        v = v * 8 + (p.s[p.i] - '0');
                        p.i += 1;
                    }
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                'x', 'u', 'U' => blk: {
                    const n: usize = if (e == 'x') 2 else if (e == 'u') 4 else 8;
                    if (p.i + n > p.s.len) return error.Invalid;
                    const v = std.fmt.parseInt(u21, p.s[p.i .. p.i + n], 16) catch return error.Invalid;
                    p.i += n;
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                '\n' => null,
                else => blk: {
                    try out.appendSlice(p.a, &.{ '\\', e });
                    break :blk null;
                },
            };
            if (simple) |ch| try out.append(p.a, ch);
        }
    }

    fn appendCode(a: Allocator, out: *std.ArrayList(u8), v: u21) Error!void {
        if (v > 0x10ffff) return error.Invalid;
        var buf: [4]u8 = undefined;
        const n = @import("json").wtf8Encode(v, &buf);
        try out.appendSlice(a, buf[0..n]);
    }

    fn number(p: *Literal) Error!Value {
        var negative = false;
        while (p.peek()) |c| {
            if (c == '-') negative = !negative else if (c != '+') break;
            p.i += 1;
        }
        const start = p.i;
        while (p.i < p.s.len and (std.ascii.isAlphanumeric(p.s[p.i]) or p.s[p.i] == '_' or p.s[p.i] == '.' or
            ((p.s[p.i] == '+' or p.s[p.i] == '-') and p.i > start and (p.s[p.i - 1] == 'e' or p.s[p.i - 1] == 'E')))) p.i += 1;
        const text = p.s[start..p.i];
        if (text.len == 0) return error.Invalid;
        const sign: []const u8 = if (negative) "-" else "";
        if (text.len > 2 and text[0] == '0' and std.ascii.findIgnoreCase("xob", text[1..2]) != null) {
            const base: u8 = switch (std.ascii.toLower(text[1])) {
                'x' => 16,
                'o' => 8,
                else => 2,
            };
            const digits = std.mem.replaceOwned(u8, p.a, text[2..], "_", "") catch return error.OutOfMemory;
            const v = std.fmt.parseInt(i128, digits, base) catch return error.Invalid;
            return .{ .int = try std.fmt.allocPrint(p.a, "{s}{d}", .{ sign, v }) };
        }
        if (std.mem.indexOfAny(u8, text, ".eE") == null) {
            const digits = std.mem.replaceOwned(u8, p.a, text, "_", "") catch return error.OutOfMemory;
            for (digits) |ch| if (!std.ascii.isDigit(ch)) return error.Invalid;
            if (digits.len > 1 and digits[0] == '0' and std.mem.trimStart(u8, digits, "0").len > 0) return error.Invalid;
            const trimmed = std.mem.trimStart(u8, digits, "0");
            if (trimmed.len == 0) return .{ .int = "0" };
            return .{ .int = try std.mem.concat(p.a, u8, &.{ sign, trimmed }) };
        }
        if (std.mem.indexOfAny(u8, text, "jJ") != null) return error.Invalid; // complex
        const f = @import("fields.zig").pyFloat(text) orelse return error.Invalid;
        if (std.mem.indexOfAny(u8, text, "infaINFA") != null and !std.ascii.isDigit(text[0]) and text[0] != '.') return error.Invalid;
        return .{ .float = if (negative) -f else f };
    }
};

/// The schema of a parameter, with the root its references point into.
fn parseSchema(a: Allocator, text: []const u8) Allocator.Error!Schema {
    const value = (try json.parse(a, text)).ok;
    return .{ .value = value, .root = value };
}

test "decode parameters" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const int_schema = try parseSchema(a, "{\"type\": \"integer\"}");
    try std.testing.expectEqualStrings("5", (try decode(a, "5", int_schema, true)).int);
    try std.testing.expectEqualStrings("5.5", (try decode(a, "5.5", int_schema, true)).string);
    const arr = try parseSchema(a, "{\"type\": \"array\"}");
    const closed = try decode(a, "[1, [2]", arr, true);
    try std.testing.expectEqual(@as(usize, 2), closed.array.len);
    const py = try decode(a, "['a', None, True]", arr, true);
    try std.testing.expectEqualStrings("[\"a\", null, true]", try json.stringify(a, py, .{}));
    try std.testing.expectEqual(Value.null, (try decode(a, " None ", try parseSchema(a, "{\"type\": \"null\"}"), true)));
}
