//! A JSON call max_tokens cut (``<tool_call>\n{"name": "f", "arguments": {...``, Ripple's form): the call, cut.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The keys ``tool_parse.payload`` reads a call's name and arguments from (a nested `function` names by the first 3).
const name_keys = [_][]const u8{ "name", "tool", "function", "call" };
const argument_keys = [_][]const u8{ "arguments", "args", "parameters" };

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

/// A member of a written object: its key, its value as far as written, and whether the cut came inside the value.
const Member = struct { key: []const u8, value: []const u8, cut: bool };

/// The members the object text `body` (from its `{`) writes before the cut, in order.
fn members(a: Allocator, body: []const u8) Allocator.Error![]const Member {
    var out: std.ArrayList(Member) = .empty;
    var i = skip(body, 1);
    while (i < body.len and body[i] == '"') {
        const key_end = i + (stringEnd(body[i..]) orelse break);
        const key = switch (try json.parseText(a, body[i..key_end])) {
            .ok => |v| if (v == .string) v.string else break,
            .err => break,
        };
        i = skip(body, key_end);
        if (i >= body.len or body[i] != ':') break;
        i = skip(body, i + 1);
        if (i >= body.len) break;
        const end = valueEnd(body[i..]) orelse {
            try out.append(a, .{ .key = key, .value = body[i..], .cut = true });
            break;
        };
        try out.append(a, .{ .key = key, .value = body[i..][0..end], .cut = false });
        i = skip(body, i + end);
        if (i >= body.len or body[i] != ',') break;
        i = skip(body, i + 1);
    }
    return out.items;
}

fn skip(s: []const u8, from: usize) usize {
    var i = from;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    return i;
}

/// The first written member named `key`.
fn find(ms: []const Member, key: []const u8) ?Member {
    for (ms) |m| if (std.mem.eql(u8, m.key, key)) return m;
    return null;
}

/// The call the whole payload `doc` makes in a closed envelope, as ``tool_parse.parse`` reads it; null if none.
fn closed(a: Allocator, doc: []const u8, tools: []const Value) Allocator.Error!?Value {
    const text = try std.mem.concat(a, u8, &.{ "<tool_call>", doc, "</tool_call>" });
    const calls = (try tool_parse.parse(a, text, tools, null)).calls orelse return null;
    return calls[0];
}

/// The call an open envelope's `written` payload begins, keyed as a closed one; null unless it names a tool.
fn cutCall(a: Allocator, written: []const u8, tools: []const Value) Allocator.Error!?Value {
    const body = reply_text.pyLstrip(written);
    if (body.len == 0 or body[0] != '{') return null;
    if (valueEnd(body)) |end| return closed(a, body[0..end], tools);
    const top = try members(a, body);
    const nested = find(top, "function");
    const inner = nested != null and nested.?.value[0] == '{';
    const holder = if (inner) try members(a, nested.?.value) else top;
    const quoted = for (if (inner) name_keys[0..3] else &name_keys) |k| {
        const m = find(holder, k) orelse continue;
        if (m.cut) return null; // the name is whole once its closing quote came
        if (m.value[0] == '"' and m.value.len > 2) break m.value;
    } else return null;
    const args = for (argument_keys) |k| {
        if (find(holder, k)) |m| break m;
    } else return null; // no arguments written yet: the text stays
    if (!args.cut) {
        const doc = try std.mem.concat(a, u8, &.{ "{\"name\":", quoted, ",\"arguments\":", args.value, "}" });
        return closed(a, doc, tools);
    }
    const name = switch (try json.parseText(a, quoted)) {
        .ok => |v| if (v == .string) (try offered(a, v.string, tools)) orelse return null else return null,
        .err => return null,
    };
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    const written_args = if (args.value[0] == '"') try cutString(a, args.value) else args.value;
    try function.put(a, "arguments", .{ .string = written_args });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
    try call.put(a, "type", .{ .string = "function" });
    try call.put(a, "function", .{ .object = function });
    return .{ .object = call };
}

/// A cut JSON string's text as far as written; an escape the cut split is dropped.
fn cutString(a: Allocator, s: []const u8) Allocator.Error![]const u8 {
    var end = s.len;
    while (end > 1 and end + 6 > s.len) : (end -= 1) {
        switch (try json.parseText(a, try std.mem.concat(a, u8, &.{ s[0..end], "\"" }))) {
            .ok => |v| if (v == .string) return v.string,
            .err => {},
        }
    }
    return "";
}

/// The offered tool's spelling of `name`, as ``tool_parse.parse`` matches it; null if none.
fn offered(a: Allocator, name: []const u8, tools: []const Value) Allocator.Error!?[]const u8 {
    for (tools) |tool| {
        const spelled = try tool_specs.toolName(a, tool);
        if (spelled.len > 0 and std.mem.eql(u8, spelled, name)) return spelled;
    }
    return null;
}

const weather = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{}}}]";

/// The arguments of the call `payload` begins in an open envelope; null when it stays text.
fn cutArguments(a: Allocator, payload: []const u8) !?[]const u8 {
    const tools = (try json.parseText(a, weather)).ok.array;
    const calls = try cutCalls(a, try std.mem.concat(a, u8, &.{ "<tool_call>\n", payload }), tools) orelse return null;
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("get_weather", calls[0].get("function").?.strField("name").?);
    return calls[0].get("function").?.strField("arguments").?;
}

test "a JSON call cut inside its arguments is the call, cut; one cut before its name is whole stays text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parseText(a, weather)).ok.array;
    const head = "<think>\nok\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ";
    const mid = (try cutCalls(a, head ++ "{\"city\": \"Syd", tools)).?;
    try std.testing.expectEqual(@as(usize, 1), mid.len);
    const f = mid[0].get("function").?;
    try std.testing.expectEqualStrings("get_weather", f.strField("name").?);
    try std.testing.expectEqualStrings("{\"city\": \"Syd", f.strField("arguments").?);
    const whole = (try cutCalls(a, head ++ "{\"city\": \"Sydney\"}}\n</", tools)).?;
    try std.testing.expectEqualStrings("{\"city\":\"Sydney\"}", whole[0].get("function").?.strField("arguments").?);
    try std.testing.expect(try cutCalls(a, "<tool_call>\n{\"na", tools) == null);
    try std.testing.expect(try cutCalls(a, "<tool_call>\n{\"name\": \"other\", \"arguments\": {", tools) == null);
    try std.testing.expect(try cutCalls(a, "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>", tools) == null);
}

test "a cut JSON call reads the keys a closed one does, and whole arguments spell as the closed call's" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const city = "{\"city\":\"Sydney\"}";
    const named = "{\"name\": \"get_weather\", ";
    try expectArguments(a, city, named ++ "\"arguments\": {\"city\": \"Sydney\"}");
    try expectArguments(a, city, "{\"arguments\": {\"city\": \"Sydney\"}, \"name\": \"get_weather\"");
    try expectArguments(a, "{\"city\": \"Sy", named ++ "\"parameters\": {\"city\": \"Sy");
    try expectArguments(a, "{\"city\": \"Sy", "{\"tool\": \"get_weather\", \"args\": {\"city\": \"Sy");
    const nested = "{\"type\": \"function\", \"function\": " ++ named ++ "\"arguments\": ";
    try expectArguments(a, "{\"city\": \"Sy", nested ++ "\"{\\\"city\\\": \\\"Sy");
    try expectArguments(a, "{\"city\": \"Sy", nested ++ "\"{\\\"city\\\": \\\"Sy\\");
    try expectArguments(a, city, nested ++ "{\"city\": \"Sydney\"}");
    // the arguments' own "name" is not the call's, and a call with no arguments key yet stays text
    try std.testing.expect(try cutArguments(a, "{\"arguments\": {\"name\": \"get_weather\", \"city\": \"Sy") == null);
    try std.testing.expect(try cutArguments(a, named ++ "\"argu") == null);
}

fn expectArguments(a: Allocator, want: []const u8, payload: []const u8) !void {
    try std.testing.expectEqualStrings(want, (try cutArguments(a, payload)) orelse return error.NoCall);
}
