//! MiniMax-M3's tool calls (`]<]minimax[>[`-prefixed XML) rewritten as DeepSeek-V4's DSML, which the parser reads.
const std = @import("std");
const json = @import("json");
const tool_params = @import("tool_params.zig");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

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

/// An element `ns<tag>inner ns</tag>` at `at` (same-named children nest): its tag, inner text and end.
const Element = struct { tag: []const u8, inner: []const u8, end: usize };

fn element(text: []const u8, at: usize) ?Element {
    if (!std.mem.startsWith(u8, text[at..], ns ++ "<")) return null;
    const t0 = at + ns.len + 1;
    const gt = std.mem.indexOfScalarPos(u8, text, t0, '>') orelse return null;
    const tag = text[t0..gt];
    if (tag.len == 0 or tag[0] == '/') return null;
    var opener_buf: [256]u8 = undefined;
    var closer_buf: [256]u8 = undefined;
    const opener = std.fmt.bufPrint(&opener_buf, ns ++ "<{s}>", .{tag}) catch return null;
    const closer = std.fmt.bufPrint(&closer_buf, ns ++ "</{s}>", .{tag}) catch return null;
    var depth: usize = 1;
    var p = gt + 1;
    while (true) {
        const c = std.mem.indexOfPos(u8, text, p, closer) orelse return null;
        const o = std.mem.indexOfPos(u8, text, p, opener);
        if (o != null and o.? < c) {
            depth += 1;
            p = o.? + opener.len;
            continue;
        }
        depth -= 1;
        if (depth == 0) return .{ .tag = tag, .inner = text[gt + 1 .. c], .end = c + closer.len };
        p = c + closer.len;
    }
}

/// An element's value: nested `item`s an array, other nested tags an object, else a scalar by its schema.
fn value(a: Allocator, inner: []const u8, schema: tool_params.Schema) Allocator.Error!?Value {
    const lead = inner.len - std.mem.trimStart(u8, inner, " \t\r\n").len;
    if (!std.mem.startsWith(u8, inner[lead..], ns ++ "<")) return try tool_params.decode(a, inner, schema, false);
    var items: std.ArrayList(Value) = .empty;
    const obj = try json.newObject(a);
    var all_items = true;
    var p = lead;
    while (p < inner.len) {
        const rest = std.mem.trimStart(u8, inner[p..], " \t\r\n");
        p = inner.len - rest.len;
        if (p == inner.len) break;
        const el = element(inner, p) orelse return null;
        p = el.end;
        const is_item = std.mem.eql(u8, el.tag, "item");
        all_items = all_items and is_item;
        const sub: Value = if (schema.value == .object) (if (is_item) schema.value.get("items") else if (schema.value.get("properties")) |props| (if (props == .object) props.get(el.tag) else null) else null) orelse .{ .object = try json.newObject(a) } else .{ .object = try json.newObject(a) };
        const v = (try value(a, el.inner, .{ .value = sub, .root = schema.root })) orelse return null;
        if (is_item) try items.append(a, v);
        try obj.put(a, el.tag, v);
    }
    return if (all_items and items.items.len > 0) .{ .array = items.items } else .{ .object = obj };
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
        const e = std.mem.indexOfPos(u8, body, q + 2, invoke_end) orelse return null;
        const name = body[n0..q];
        try out.print(a, "\n<\u{ff5c}DSML\u{ff5c}invoke name=\"{s}\">", .{name});
        var k = q + 2;
        while (true) {
            const r = std.mem.trimStart(u8, body[k..e], " \t\r\n");
            k = e - r.len;
            if (k == e) break;
            const el = element(body[0..e], k) orelse return null;
            k = el.end;
            const v = (try value(a, el.inner, try tool_params.schemaOf(schemas, name, el.tag, a))) orelse return null;
            if (v == .string) {
                try out.print(a, "<\u{ff5c}DSML\u{ff5c}parameter name=\"{s}\" string=\"true\">{s}</\u{ff5c}DSML\u{ff5c}parameter>", .{ el.tag, v.string });
            } else {
                const spelled = json.stringify(a, v, .{}) catch return error.OutOfMemory;
                try out.print(a, "<\u{ff5c}DSML\u{ff5c}parameter name=\"{s}\" string=\"false\">{s}</\u{ff5c}DSML\u{ff5c}parameter>", .{ el.tag, spelled });
            }
        }
        try out.appendSlice(a, "</\u{ff5c}DSML\u{ff5c}invoke>");
        p = e + invoke_end.len;
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
