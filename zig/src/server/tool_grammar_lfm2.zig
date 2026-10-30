//! LFM2's calls as a grammar: `<|tool_call_start|>[name(key=value, ...)]<|tool_call_end|>`, Python literals.
const std = @import("std");
const json = @import("json");
const tool_specs = @import("tool_specs.zig");
const tool_params = @import("tool_params.zig");
const tool_kinds = @import("tool_kinds.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Rules shared by every tool: blanks, Python's scalars, and lists and dicts of any literal.
const prelude =
    \\blank ::= [ \t\r\n]{0,8}
    \\ws ::= [ \t\n]{0,4}
    \\py_str ::= "'" ([^'\\\x00-\x1f] | "\\" ["'\\nrt])* "'" | "\"" ([^"\\\x00-\x1f] | "\\" ["'\\nrt])* "\""
    \\py_int ::= "-"? ("0" | [1-9] [0-9]*)
    \\py_num ::= py_int ("." [0-9]+)? ([eE] [+\-]? [0-9]+)?
    \\py_bool ::= "True" | "False"
    \\py_none ::= "None"
    \\py_value ::= py_str | py_num | py_bool | py_none | py_list | py_dict
    \\py_list ::= "[" ws (py_value (ws "," ws py_value)* ws)? "]"
    \\py_dict ::= "{" ws (py_str ws ":" ws py_value (ws "," ws py_str ws ":" ws py_value)* ws)? "}"
    \\
;

/// Whether the probe call `rendered` is in LFM2's form: `[probe(key='v')]` between its call markers.
pub fn isStyle(rendered: []const u8, comptime probe: []const u8, comptime key: []const u8) bool {
    return std.mem.indexOf(u8, rendered, "<|tool_call_start|>[" ++ probe ++ "(" ++ key ++ "=") != null;
}

/// The structural tag (xgrammar's JSON) whose one format is `ebnf`'s grammar.
pub fn build(a: Allocator, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const format = try json.newObject(a);
    try format.put(a, "type", .{ .string = "grammar" });
    try format.put(a, "grammar", .{ .string = try ebnf(a, tools, parallel) });
    const tag = try json.newObject(a);
    try tag.put(a, "type", .{ .string = "structural_tag" });
    try tag.put(a, "format", .{ .object = format });
    return json.stringify(a, .{ .object = tag }, .{ .ascii = false, .compact = true });
}

/// The grammar for one block of calls to `tools`: one call, or (`parallel`) one or more in the same list.
pub fn ebnf(a: Allocator, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const schemas = try tool_params.schemas(a, tools);
    var rules: std.ArrayList(u8) = .empty;
    var names: std.ArrayList(u8) = .empty;
    for (tools, 0..) |t, i| {
        const name = try tool_specs.toolName(a, t);
        try names.print(a, "{s}call_{d}", .{ if (i > 0) " | " else "", i });
        try rules.print(a, "call_{d} ::= {s} a_{d}_0_0 \")\"\n", .{ i, try literal(a, try std.fmt.allocPrint(a, "{s}(", .{name})), i });
        const tool = schemas.get(try std.ascii.allocLowerString(a, name)).?;
        try properties(a, &rules, i, tool);
    }
    const more = if (parallel) " (\", \" call)*" else "";
    return std.fmt.allocPrint(a, "root ::= blank \"<|tool_call_start|>[\" call{s} \"]<|tool_call_end|>\" blank\ncall ::= {s}\n{s}{s}", .{ more, names.items, prelude, rules.items });
}

/// Rules a_I_J_S: tool I's arguments from property J on, S = 1 when one is written already; optional ones may go.
fn properties(a: Allocator, rules: *std.ArrayList(u8), tool: usize, schema: tool_params.Tool) Allocator.Error!void {
    var required: std.StringHashMapUnmanaged(void) = .empty;
    if (schema.root == .object) if (schema.root.get("required")) |r| if (r == .array) for (r.array) |k| if (k == .string) try required.put(a, k.string, {});
    var usable: std.ArrayList(usize) = .empty;
    for (schema.properties.keys(), 0..) |k, j| if (keyword(k)) try usable.append(a, j);
    for (usable.items, 0..) |j, n| {
        const key = schema.properties.keys()[j];
        const value = try alternatives(a, schema.properties.values()[j], schema.root);
        for ([_]u8{ 0, 1 }) |state| {
            const item = try std.fmt.allocPrint(a, "{s}\"{s}=\" ({s}) a_{d}_{d}_1", .{ if (state == 1) "\", \" " else "", key, value, tool, n + 1 });
            const skip = if (required.contains(key)) "" else try std.fmt.allocPrint(a, " | a_{d}_{d}_{d}", .{ tool, n + 1, state });
            try rules.print(a, "a_{d}_{d}_{d} ::= {s}{s}\n", .{ tool, n, state, item, skip });
        }
    }
    for ([_]u8{ 0, 1 }) |state| try rules.print(a, "a_{d}_{d}_{d} ::= \"\"\n", .{ tool, usable.items.len, state });
}

/// A keyword argument's name: Python's identifier, as the parser reads it.
fn keyword(k: []const u8) bool {
    if (k.len == 0 or std.ascii.isDigit(k[0])) return false;
    for (k) |c| if (!std.ascii.isAlphanumeric(c) and c != '_') return false;
    return true;
}

/// What a value of this schema may be: its enum or const as literals, else a rule per kind it names.
fn alternatives(a: Allocator, schema: Value, root: Value) Allocator.Error![]const u8 {
    if (try fixed(a, schema)) |text| return text;
    const kinds = (try tool_kinds.schemaKinds(a, schema, root)) orelse return "py_value";
    var out: std.ArrayList(u8) = .empty;
    const rules = [_]struct { tool_kinds.Kind, []const u8 }{
        .{ .string, "py_str" }, .{ .integer, "py_int" }, .{ .number, "py_num" },  .{ .boolean, "py_bool" },
        .{ .null, "py_none" },  .{ .array, "py_list" },  .{ .object, "py_dict" },
    };
    for (rules) |r| {
        if (!kinds.contains(r[0]) or (r[0] == .integer and kinds.contains(.number))) continue;
        try out.print(a, "{s}{s}", .{ if (out.items.len > 0) " | " else "", r[1] });
    }
    return if (out.items.len == 0) "py_value" else out.items;
}

/// A schema's enum or const of scalars as the literals Python spells them; null when it has neither.
fn fixed(a: Allocator, schema: Value) Allocator.Error!?[]const u8 {
    if (schema != .object) return null;
    var one = [_]Value{.null};
    const values: []const Value = if (schema.get("enum")) |e| (if (e == .array) e.array else return null) else if (schema.get("const")) |c| blk: {
        one[0] = c;
        break :blk &one;
    } else return null;
    if (values.len == 0) return null;
    var out: std.ArrayList(u8) = .empty;
    for (values) |v| {
        if (v == .string and plainText(v.string)) {
            const both = try std.fmt.allocPrint(a, "\"{s}\"", .{v.string});
            const single = try std.fmt.allocPrint(a, "'{s}'", .{v.string});
            try out.print(a, "{s}{s} | {s}", .{ if (out.items.len > 0) " | " else "", try literal(a, both), try literal(a, single) });
            continue;
        }
        const text = switch (v) {
            .string => try json.stringify(a, v, .{ .ascii = false }),
            .bool => |b| if (b) "True" else "False",
            .null => "None",
            .int, .float => try json.stringify(a, v, .{}),
            else => return null,
        };
        try out.print(a, "{s}{s}", .{ if (out.items.len > 0) " | " else "", try literal(a, text) });
    }
    return out.items;
}

/// Text that reads the same between single or double quotes: no quote, backslash or control character.
fn plainText(text: []const u8) bool {
    for (text) |c| if (c < 0x20 or c == 0x7f or c == '"' or c == '\'' or c == '\\') return false;
    return true;
}

/// `text` as a quoted grammar literal.
fn literal(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, '"');
    for (text) |c| switch (c) {
        '"', '\\' => try out.print(a, "\\{c}", .{c}),
        '\n' => try out.appendSlice(a, "\\n"),
        '\r' => try out.appendSlice(a, "\\r"),
        '\t' => try out.appendSlice(a, "\\t"),
        0...8, 11, 12, 14...31, 127 => try out.print(a, "\\x{x:0>2}", .{c}),
        else => try out.append(a, c),
    };
    try out.append(a, '"');
    return out.items;
}
