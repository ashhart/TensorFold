//! A JSON call max_tokens cut (``<tool_call>\n{"name": "f", "arguments": {...``, Ripple's form): the call, cut.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// A reply max_tokens cut inside a JSON call: the whole calls before it, then it, cut; null when it ends in none.
pub fn cutCalls(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error!?[]Value {
    const start = tool_parse.openCall(text) orelse return null;
    const cut = try cutCall(a, text[start + "<tool_call>".len ..], tools) orelse return null;
    const before = (try tool_parse.parse(a, text[0..start], tools, null)).calls orelse &.{};
    return try std.mem.concat(a, Value, &.{ before, &.{cut} });
}

/// The JSON string at `s[0]` (a quote): its end past the closing quote; null when the cut came first.
fn stringEnd(s: []const u8) ?usize {
    var i: usize = 1;
    while (i < s.len) : (i += 1) switch (s[i]) {
        '\\' => i += 1,
        '"' => return i + 1,
        else => {},
    };
    return null;
}

/// The end of the JSON value `s` begins with when it is whole (balanced brackets outside strings); null if cut.
fn valueEnd(s: []const u8) ?usize {
    var depth: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        switch (s[i]) {
            '"' => {
                i += stringEnd(s[i..]) orelse return null;
                if (depth == 0) return i;
                continue;
            },
            '{', '[' => depth += 1,
            '}', ']' => {
                if (depth == 0) return i;
                depth -= 1;
                if (depth == 0) return i + 1;
            },
            else => if (depth == 0 and (s[i] == ',' or std.ascii.isWhitespace(s[i]))) return i,
        }
        i += 1;
    }
    return null;
}

/// `key`'s value in the written object `body` (after its quoted key and colon); null when the cut came first.
fn after(body: []const u8, key: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, body, key) orelse return null;
    const rest = reply_text.pyLstrip(body[at + key.len ..]);
    if (rest.len == 0 or rest[0] != ':') return null;
    return reply_text.pyLstrip(rest[1..]);
}

/// The call an open envelope's `written` payload begins, arguments as far as written; null unless it names a tool.
fn cutCall(a: Allocator, written: []const u8, tools: []const Value) Allocator.Error!?Value {
    const body = reply_text.pyLstrip(written);
    if (body.len == 0 or body[0] != '{') return null;
    const quoted = after(body, "\"name\"") orelse return null;
    if (quoted.len == 0 or quoted[0] != '"') return null;
    const name_end = stringEnd(quoted) orelse return null; // the name is whole once its closing quote came
    const name = switch (try json.parseText(a, quoted[0..name_end])) {
        .ok => |v| if (v == .string) (try offered(a, v.string, tools)) orelse return null else return null,
        .err => return null,
    };
    var args: []const u8 = "";
    if (after(body, "\"arguments\"")) |v| args = v[0 .. valueEnd(v) orelse v.len];
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    try function.put(a, "arguments", .{ .string = args });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
    try call.put(a, "type", .{ .string = "function" });
    try call.put(a, "function", .{ .object = function });
    return .{ .object = call };
}

/// The offered tool's spelling of `name`, as ``tool_parse.parse`` matches it; null if none.
fn offered(a: Allocator, name: []const u8, tools: []const Value) Allocator.Error!?[]const u8 {
    for (tools) |tool| {
        const spelled = try tool_specs.toolName(a, tool);
        if (spelled.len > 0 and std.mem.eql(u8, spelled, name)) return spelled;
    }
    return null;
}

test "a JSON call cut inside its arguments is the call, cut; one cut before its name is whole stays text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parseText(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{}}}]")).ok.array;
    const head = "<think>\nok\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ";
    const mid = (try cutCalls(a, head ++ "{\"city\": \"Syd", tools)).?;
    try std.testing.expectEqual(@as(usize, 1), mid.len);
    const f = mid[0].get("function").?;
    try std.testing.expectEqualStrings("get_weather", f.strField("name").?);
    try std.testing.expectEqualStrings("{\"city\": \"Syd", f.strField("arguments").?);
    const whole = (try cutCalls(a, head ++ "{\"city\": \"Sydney\"}}\n</", tools)).?;
    try std.testing.expectEqualStrings("{\"city\": \"Sydney\"}", whole[0].get("function").?.strField("arguments").?);
    try std.testing.expect(try cutCalls(a, "<tool_call>\n{\"na", tools) == null);
    try std.testing.expect(try cutCalls(a, "<tool_call>\n{\"name\": \"other\", \"arguments\": {", tools) == null);
    try std.testing.expect(try cutCalls(a, "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>", tools) == null);
}
