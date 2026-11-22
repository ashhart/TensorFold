//! MiniMax-M3's tool calls (`]<]minimax[>[`-prefixed XML) rewritten as DeepSeek-V4's DSML, which the parser reads.
const std = @import("std");
const json = @import("json");
const tool_params = @import("tool_params.zig");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const tool_kinds = @import("tool_kinds.zig");

pub const ns = "]<]minimax[>[";
pub const open = ns ++ "<tool_call>";
pub const close = ns ++ "</tool_call>";
const dsml_open = reply_text.calls[2][0];
const dsml_close = reply_text.calls[2][1];

/// `text` with each complete, well-formed MiniMax call block as a DSML block typed by the tools' schemas.
pub fn dsml(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error![]const u8 {
    if (tools.len == 0 or std.mem.indexOf(u8, text, open) == null) return text;
    const schemas = try tool_params.schemas(a, tools);
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, open)) |s| {
        const e = std.mem.indexOfPos(u8, text, s + open.len, close) orelse break;
        try out.appendSlice(a, text[pos..s]);
        if (try block(a, text[s + open.len .. e], &schemas)) |b| try out.appendSlice(a, b) else try out.appendSlice(a, text[s .. e + close.len]);
        pos = e + close.len;
    }
    try out.appendSlice(a, text[pos..]);
    return out.items;
}

/// The close token's byte offset after complete MiniMax call blocks before it have been rewritten.
pub fn closeAfterRewrite(a: Allocator, raw: []const u8, tools: []const Value, where: reply_text.Close) Allocator.Error!reply_text.Close {
    return switch (where) {
        .token => |at| if (at <= raw.len and std.mem.indexOf(u8, raw[0..at], open) != null)
            .{ .token = (try dsml(a, raw[0..at], tools)).len }
        else
            where,
        else => where,
    };
}

/// An element's opener: its name and the first byte of its contents.
const Opener = struct { tag: []const u8, content: usize };

fn opener(text: []const u8, at: usize) ?Opener {
    if (at > text.len or !std.mem.startsWith(u8, text[at..], ns ++ "<")) return null;
    const start = at + ns.len + 1;
    const gt = std.mem.indexOfScalarPos(u8, text, start, '>') orelse return null;
    const tag = text[start..gt];
    if (tag.len == 0 or tag[0] == '/') return null;
    return .{ .tag = tag, .content = gt + 1 };
}

const Parsed = struct { tag: []const u8, value: Value, end: usize };
const max_depth = 64;

/// One element and its matching closer, read without rescanning any descendant.
fn element(a: Allocator, text: []const u8, at: *usize, schema: tool_params.Schema, depth: usize) Allocator.Error!?Parsed {
    if (depth >= max_depth) return null;
    const opened = opener(text, at.*) orelse return null;
    const current = (try tool_kinds.resolved(a, schema.value, schema.root)) orelse return null;
    const kind = if (current == .object) current.get("type") else null;
    const structured = current == .object and (current.get("properties") != null or current.get("items") != null or
        (kind != null and kind.? == .string and (std.mem.eql(u8, kind.?.string, "object") or std.mem.eql(u8, kind.?.string, "array"))));
    var closer_buf: [256]u8 = undefined;
    const closer = std.fmt.bufPrint(&closer_buf, ns ++ "</{s}>", .{opened.tag}) catch return null;
    var p = opened.content;
    const lead = p + (text[p..].len - std.mem.trimStart(u8, text[p..], " \t\r\n").len);
    if (!structured or !std.mem.startsWith(u8, text[lead..], ns ++ "<")) {
        const end = std.mem.indexOfPos(u8, text, p, closer) orelse return null;
        // a literal NaN or Infinity refuses a value its schema reads only as JSON; a string's text is kept as written
        if (!tool_params.finiteJson(text[p..end]) and try tool_params.jsonOnly(a, .{ .value = current, .root = schema.root })) return null;
        const decoded = try tool_params.decode(a, text[p..end], .{ .value = current, .root = schema.root }, false);
        at.* = end + closer.len;
        return .{ .tag = opened.tag, .value = decoded, .end = at.* };
    }
    var items: std.ArrayList(Value) = .empty;
    const obj = try json.newObject(a);
    var all_items = true;
    while (p < text.len) {
        p += text[p..].len - std.mem.trimStart(u8, text[p..], " \t\r\n").len;
        if (std.mem.startsWith(u8, text[p..], closer)) {
            at.* = p + closer.len;
            return .{ .tag = opened.tag, .value = if (all_items and items.items.len > 0) .{ .array = items.items } else .{ .object = obj }, .end = at.* };
        }
        const child_open = opener(text, p) orelse return null;
        const is_item = std.mem.eql(u8, child_open.tag, "item");
        const sub = try childSchema(a, current, schema.root, child_open.tag, is_item);
        const child = (try element(a, text, &p, .{ .value = sub, .root = schema.root }, depth + 1)) orelse return null;
        all_items = all_items and is_item;
        if (is_item) try items.append(a, child.value);
        try obj.put(a, child.tag, child.value);
    }
    return null;
}

/// The schema for one nested name, resolving references and selecting a supported combiner branch.
fn childSchema(a: Allocator, schema: Value, root: Value, tag: []const u8, is_item: bool) Allocator.Error!Value {
    if (schema == .object) {
        if (is_item) if (schema.get("items")) |items| return items;
        if (schema.get("properties")) |properties| if (properties == .object) if (properties.get(tag)) |child| return child;
        for ([_][]const u8{ "anyOf", "oneOf", "allOf" }) |key| if (schema.get(key)) |branches| if (branches == .array) {
            for (branches.array) |branch| {
                const current = (try tool_kinds.resolved(a, branch, root)) orelse continue;
                if (current != .object) continue;
                if (is_item and current.get("items") != null) return childSchema(a, current, root, tag, is_item);
                if (current.get("properties")) |properties| if (properties == .object and properties.get(tag) != null)
                    return childSchema(a, current, root, tag, is_item);
            }
        };
    }
    return .{ .object = try json.newObject(a) };
}

/// Text safe to interpolate in a DSML tag or attribute.
fn safeName(text: []const u8) bool {
    return std.mem.indexOfScalar(u8, text, '"') == null and std.mem.indexOf(u8, text, "\u{ff5c}DSML\u{ff5c}") == null;
}

fn safeValue(v: Value) bool {
    return switch (v) {
        .float => |f| !std.math.isNan(f),
        .string => true,
        .array => |items| for (items) |item| {
            if (!safeValue(item)) break false;
        } else true,
        .object => |obj| for (obj.keys(), obj.values()) |key, item| {
            if (!safeName(key) or !safeValue(item)) break false;
        } else true,
        else => true,
    };
}

/// One block's invokes as DSML, or null when anything in it is not a well-formed invoke.
fn block(a: Allocator, body: []const u8, schemas: *const tool_params.Schemas) Allocator.Error!?[]const u8 {
    const invoke = ns ++ "<invoke name=\"";
    const invoke_end = ns ++ "</invoke>";
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, dsml_open);
    var p: usize = 0;
    var calls: usize = 0;
    while (true) {
        const rest = std.mem.trimStart(u8, body[p..], " \t\r\n");
        p = body.len - rest.len;
        if (p == body.len) break;
        if (!std.mem.startsWith(u8, rest, invoke)) return null;
        const n0 = p + invoke.len;
        const q = std.mem.indexOfPos(u8, body, n0, "\">") orelse return null;
        const name = body[n0..q];
        if (!safeName(name)) return null;
        try out.print(a, "\n<\u{ff5c}DSML\u{ff5c}invoke name=\"{s}\">", .{name});
        var k = q + 2;
        while (true) {
            const r = std.mem.trimStart(u8, body[k..], " \t\r\n");
            k = body.len - r.len;
            if (std.mem.startsWith(u8, body[k..], invoke_end)) break;
            const child_open = opener(body, k) orelse return null;
            var cut = k;
            const el = (try element(a, body, &cut, try tool_params.schemaOf(schemas, name, child_open.tag, a), 0)) orelse return null;
            if (!safeName(el.tag) or !safeValue(el.value)) return null;
            k = el.end;
            const spelled = json.stringify(a, el.value, .{ .ascii = true }) catch return error.OutOfMemory;
            try out.print(a, "<\u{ff5c}DSML\u{ff5c}parameter name=\"{s}\" string=\"false\">{s}</\u{ff5c}DSML\u{ff5c}parameter>", .{ el.tag, spelled });
        }
        try out.appendSlice(a, "</\u{ff5c}DSML\u{ff5c}invoke>");
        p = k + invoke_end.len;
        calls += 1;
    }
    if (calls == 0) return null;
    try out.appendSlice(a, "\n" ++ dsml_close);
    return out.items;
}

test "tool_minimax: the template's calls as DSML, typed by the schema, nested items as arrays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}}}}}]
    )).ok;
    const text = "Checking." ++ open ++ "\n" ++ ns ++ "<invoke name=\"get_weather\">" ++ ns ++ "<city>Paris" ++ ns ++ "</city>" ++ ns ++ "<days>3" ++ ns ++ "</days>" ++ ns ++ "<tags>" ++ ns ++ "<item>a" ++ ns ++ "</item>" ++ ns ++ "<item>b" ++ ns ++ "</item>" ++ ns ++ "</tags>" ++ ns ++ "</invoke>\n" ++ close;
    const got = try dsml(a, text, tools.array);
    const parsed = try @import("tool_parse.zig").parse(a, got, tools.array, null);
    try std.testing.expectEqualStrings("Checking.", parsed.content);
    const call = parsed.calls.?[0];
    const args = (try json.parse(a, call.get("function").?.get("arguments").?.string)).ok;
    try std.testing.expectEqualStrings("Paris", args.get("city").?.string);
    try std.testing.expectEqualStrings("3", args.get("days").?.int);
    try std.testing.expectEqualStrings("b", args.get("tags").?.array[1].string);
    // a block still being written stays as written
    const partial = "x" ++ open ++ ns ++ "<invoke name=\"get_weather\">";
    try std.testing.expectEqualStrings(partial, try dsml(a, partial, tools.array));
}
