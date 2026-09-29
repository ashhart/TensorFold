const std = @import("std");
const V = std.json.Value;
const chat = @import("chat.zig");
const whitespace = " \r\n\t";
pub const Call = struct { id: []const u8, type: []const u8 = "function", function: struct { name: []const u8, arguments: []const u8 } };
pub const Result = struct { content: []const u8, calls: []const Call = &.{} };
const Parsed = struct { name: []const u8, arguments: V };
const Envelope = struct { begin: usize, end: usize, payload: []const u8, dsml: bool = false };

fn trim(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, whitespace);
}
fn standardTag(name: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(name, "tool_call")) return true;
    const colon = std.mem.indexOfScalar(u8, name, ':') orelse return false;
    if (colon == 0 or !std.ascii.eqlIgnoreCase(name[colon + 1 ..], "tool_call")) return false;
    if (!std.ascii.isAlphabetic(name[0]) and name[0] != '_') return false;
    for (name[1..colon]) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '.' and c != '-') return false;
    return true;
}
pub fn singleContent(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, cursor, '<')) |begin| {
        try out.appendSlice(a, text[cursor..begin]);
        const end = std.mem.indexOfScalarPos(u8, text, begin + 1, '>') orelse {
            try out.appendSlice(a, text[begin..]);
            return out.items;
        };
        const name = text[begin + 1 .. end];
        if (standardTag(name)) {
            const close = try std.fmt.allocPrint(a, "</{s}>", .{name});
            const close_at = std.ascii.findIgnoreCasePos(text, end + 1, close) orelse return out.items;
            cursor = close_at + close.len;
        } else {
            try out.appendSlice(a, text[begin .. end + 1]);
            cursor = end + 1;
        }
    }
    try out.appendSlice(a, text[cursor..]);
    return out.items;
}
fn object(a: std.mem.Allocator, value: V) !V {
    var result = value;
    if (result == .null) return .{ .object = .empty };
    if (result == .string) {
        if (trim(result.string).len == 0) return .{ .object = .empty };
        result = (try std.json.parseFromSlice(V, a, result.string, .{})).value;
    }
    if (result != .object) return error.InvalidToolArguments;
    return result;
}
fn tool(tools: V, name: []const u8) ?V {
    if (tools != .array) return null;
    for (tools.array.items) |spec| {
        const known = chat.toolName(spec) catch continue;
        if (std.ascii.eqlIgnoreCase(known, name)) return spec.object.get("function") orelse spec;
    }
    return null;
}
pub fn canonicalName(tools: V, name: []const u8) ?[]const u8 {
    return chat.toolName(tool(tools, name) orelse return null) catch null;
}
pub fn typedParameter(tools: V, name: []const u8, key: []const u8) bool {
    const function = tool(tools, name) orelse return false;
    const parameters = function.object.get("parameters") orelse function.object.get("input_schema") orelse return false;
    if (parameters != .object) return false;
    const properties = parameters.object.get("properties") orelse return false;
    if (properties != .object) return false;
    const schema = properties.object.get(key) orelse return false;
    if (schema != .object) return false;
    const kind = schema.object.get("type") orelse return false;
    if (kind != .string) return false;
    for ([_][]const u8{ "array", "object", "boolean", "integer", "number", "null" }) |expected| if (std.mem.eql(u8, kind.string, expected)) return true;
    return false;
}
pub fn parameter(a: std.mem.Allocator, tools: V, name: []const u8, key: []const u8, text: []const u8) !V {
    const fallback = V{ .string = text };
    const function = tool(tools, name) orelse return fallback;
    const parameters = function.object.get("parameters") orelse function.object.get("input_schema") orelse return fallback;
    if (parameters != .object) return fallback;
    const properties = parameters.object.get("properties") orelse return fallback;
    if (properties != .object) return fallback;
    const schema = properties.object.get(key) orelse return fallback;
    if (schema != .object) return fallback;
    const kind = schema.object.get("type") orelse return fallback;
    if (kind != .string) return fallback;
    const parsed = std.json.parseFromSlice(V, a, text, .{}) catch return fallback;
    const valid = switch (parsed.value) {
        .null => std.mem.eql(u8, kind.string, "null"),
        .bool => std.mem.eql(u8, kind.string, "boolean"),
        .integer => std.mem.eql(u8, kind.string, "integer") or std.mem.eql(u8, kind.string, "number"),
        .float => |value| std.math.isFinite(value) and std.mem.eql(u8, kind.string, "number"),
        .array => std.mem.eql(u8, kind.string, "array"),
        .object => std.mem.eql(u8, kind.string, "object"),
        else => false,
    };
    return if (valid) parsed.value else fallback;
}
fn jsonPayload(a: std.mem.Allocator, value: V) anyerror!Parsed {
    if (value == .array) {
        for (value.array.items) |item| if (jsonPayload(a, item)) |parsed| return parsed else |_| {};
        return error.InvalidToolCall;
    }
    if (value != .object) return error.InvalidToolCall;
    const nested = value.object.get("function") orelse .null;
    const function = if (nested == .object) nested else value;
    const name = function.object.get("name") orelse function.object.get("tool") orelse function.object.get("function") orelse function.object.get("call") orelse return error.InvalidToolName;
    if (name != .string or trim(name.string).len == 0) return error.InvalidToolName;
    var arguments = function.object.get("arguments") orelse function.object.get("args") orelse function.object.get("parameters") orelse blk: {
        var loose = V{ .object = try function.object.clone(a) };
        for ([_][]const u8{ "name", "tool", "function", "call", "type" }) |key| _ = loose.object.swapRemove(key);
        break :blk loose;
    };
    arguments = try object(a, arguments);
    return .{ .name = trim(name.string), .arguments = arguments };
}

fn gemma(a: std.mem.Allocator, input: []const u8) !Parsed {
    const brace = std.mem.indexOfScalar(u8, input, '{') orelse return error.InvalidToolCall;
    const name = trim(input[5..brace]);
    if (name.len == 0) return error.InvalidToolName;
    var json: std.ArrayList(u8) = .empty;
    const text = input[brace..];
    var at: usize = 0;
    const marker = "<|\"|>";
    var key = false;
    while (at < text.len) {
        if (std.mem.startsWith(u8, text[at..], marker)) {
            const end = std.mem.indexOfPos(u8, text, at + marker.len, marker) orelse return error.InvalidToolCall;
            try json.appendSlice(a, try std.json.Stringify.valueAlloc(a, text[at + marker.len .. end], .{}));
            at = end + marker.len;
            key = false;
            continue;
        }
        if (key and (std.ascii.isAlphabetic(text[at]) or text[at] == '_')) {
            var end = at + 1;
            while (end < text.len and (std.ascii.isAlphanumeric(text[end]) or text[end] == '_' or text[end] == '-')) : (end += 1) {}
            var colon = end;
            while (colon < text.len and std.ascii.isWhitespace(text[colon])) : (colon += 1) {}
            if (colon < text.len and text[colon] == ':') {
                try json.appendSlice(a, try std.json.Stringify.valueAlloc(a, text[at..end], .{}));
                at = end;
                key = false;
                continue;
            }
        }
        if (!std.ascii.isWhitespace(text[at])) key = text[at] == '{' or text[at] == ',';
        try json.append(a, text[at]);
        at += 1;
    }
    return .{ .name = name, .arguments = try object(a, .{ .string = json.items }) };
}

fn tagged(a: std.mem.Allocator, input: []const u8, tools: V, complete: bool) !Parsed {
    const qwen = input.len >= 10 and std.ascii.eqlIgnoreCase(input[0..10], "<function=");
    var name: []const u8 = undefined;
    var body: []const u8 = undefined;
    if (qwen) {
        const end = std.mem.indexOfScalar(u8, input, '>') orelse return error.InvalidToolCall;
        if (input.len < 11 or !std.ascii.eqlIgnoreCase(input[input.len - 11 ..], "</function>")) return error.InvalidToolCall;
        if (end < 10 or end + 1 > input.len - "</function>".len) return error.InvalidToolCall;
        name = trim(input[10..end]);
        body = input[end + 1 .. input.len - "</function>".len];
    } else {
        const end = std.mem.indexOf(u8, input, "<arg_key>") orelse input.len;
        name = trim(input[0..end]);
        body = input[end..];
    }
    if (name.len == 0) return error.InvalidToolName;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '.' and c != ':' and c != '-') return error.InvalidToolName;
    var args = V{ .object = .empty };
    const open = if (qwen) "<parameter=" else "<arg_key>";
    var cursor: usize = 0;
    while (if (qwen) std.ascii.findIgnoreCasePos(body, cursor, open) else std.mem.indexOfPos(u8, body, cursor, open)) |begin| {
        if (complete and trim(body[cursor..begin]).len > 0) return error.InvalidToolCall;
        const key_begin = begin + open.len;
        const key_end = if (qwen) std.mem.indexOfScalarPos(u8, body, key_begin, '>') orelse return error.InvalidToolCall else std.mem.indexOfPos(u8, body, key_begin, "</arg_key>") orelse return error.InvalidToolCall;
        const key_name = trim(body[key_begin..key_end]);
        var value_begin = key_end + (if (qwen) @as(usize, 1) else "</arg_key>".len);
        if (!qwen) {
            while (value_begin < body.len and std.ascii.isWhitespace(body[value_begin])) : (value_begin += 1) {}
            if (!std.mem.startsWith(u8, body[value_begin..], "<arg_value>")) return error.InvalidToolCall;
            value_begin += "<arg_value>".len;
        }
        const close = if (qwen) "</parameter>" else "</arg_value>";
        const value_end = (if (qwen) std.ascii.findIgnoreCasePos(body, value_begin, close) else std.mem.indexOfPos(u8, body, value_begin, close)) orelse return error.InvalidToolCall;
        var value = body[value_begin..value_end];
        if (qwen) {
            if (std.mem.startsWith(u8, value, "\n")) value = value[1..];
            if (std.mem.endsWith(u8, value, "\n")) value = value[0 .. value.len - 1];
        }
        try args.object.put(a, key_name, try parameter(a, tools, name, key_name, value));
        cursor = value_end + close.len;
    }
    if (complete and trim(body[cursor..]).len > 0) return error.InvalidToolCall;
    return .{ .name = name, .arguments = args };
}

fn payload(a: std.mem.Allocator, text: []const u8, tools: V, complete: bool) !Parsed {
    const input = trim(text);
    if (std.mem.startsWith(u8, input, "call:")) return gemma(a, input);
    if (std.json.parseFromSlice(V, a, input, .{})) |parsed| return jsonPayload(a, parsed.value) else |_| {}
    return tagged(a, input, tools, complete);
}
fn envelopes(a: std.mem.Allocator, text: []const u8) ![]Envelope {
    var found: std.ArrayList(Envelope) = .empty;
    var cursor: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, cursor, '<')) |begin| {
        cursor = begin + 1;
        const tag_end = std.mem.indexOfScalarPos(u8, text, cursor, '>') orelse break;
        const name = text[cursor..tag_end];
        const is_dsml = std.mem.eql(u8, name, "｜DSML｜tool_calls");
        const is_gemma = std.mem.eql(u8, name, "|tool_call");
        if (!is_dsml and !is_gemma and !standardTag(name)) continue;
        const close = if (is_gemma) "<tool_call|>" else try std.fmt.allocPrint(a, "</{s}>", .{name});
        const end = std.ascii.findIgnoreCasePos(text, tag_end + 1, close) orelse continue;
        try found.append(a, .{ .begin = begin, .end = end + close.len, .payload = trim(text[tag_end + 1 .. end]), .dsml = is_dsml });
        cursor = end + close.len;
    }
    return found.toOwnedSlice(a);
}
fn dsml(a: std.mem.Allocator, text: []const u8) ![]Parsed {
    var found: std.ArrayList(Parsed) = .empty;
    var remaining = trim(text);
    const invoke = "<｜DSML｜invoke name=\"";
    const param = "<｜DSML｜parameter name=\"";
    while (remaining.len > 0) {
        if (!std.mem.startsWith(u8, remaining, invoke)) return error.InvalidToolCall;
        const name_end = std.mem.indexOfPos(u8, remaining, invoke.len, "\">") orelse return error.InvalidToolCall;
        const name = trim(remaining[invoke.len..name_end]);
        const close = "</｜DSML｜invoke>";
        const end = std.mem.indexOf(u8, remaining, close) orelse return error.InvalidToolCall;
        if (end < name_end + 2) return error.InvalidToolCall;
        var body = trim(remaining[name_end + 2 .. end]);
        var args = V{ .object = .empty };
        while (body.len > 0) {
            if (!std.mem.startsWith(u8, body, param)) return error.InvalidToolCall;
            const key_end = std.mem.indexOfPos(u8, body, param.len, "\" string=\"") orelse return error.InvalidToolCall;
            const key = body[param.len..key_end];
            const flag_begin = key_end + "\" string=\"".len;
            const flag_end = std.mem.indexOfPos(u8, body, flag_begin, "\">") orelse return error.InvalidToolCall;
            const flag = body[flag_begin..flag_end];
            const param_close = "</｜DSML｜parameter>";
            const value_end = std.mem.indexOf(u8, body, param_close) orelse return error.InvalidToolCall;
            if (value_end < flag_end + 2) return error.InvalidToolCall;
            const value = body[flag_end + 2 .. value_end];
            const decoded: V = if (std.mem.eql(u8, flag, "true")) .{ .string = value } else if (std.mem.eql(u8, flag, "false")) (try std.json.parseFromSlice(V, a, value, .{})).value else return error.InvalidToolCall;
            try args.object.put(a, key, decoded);
            body = trim(body[value_end + param_close.len ..]);
        }
        try found.append(a, .{ .name = name, .arguments = args });
        remaining = trim(remaining[end + close.len ..]);
    }
    if (found.items.len == 0) return error.InvalidToolCall;
    return found.toOwnedSlice(a);
}
fn append(a: std.mem.Allocator, calls: *std.ArrayList(Call), parsed: Parsed, tools: V, prefix: []const u8) !void {
    const spec = tool(tools, parsed.name) orelse return error.UnknownTool;
    try calls.append(a, .{ .id = try std.fmt.allocPrint(a, "call_{s}_{d}", .{ prefix, calls.items.len }), .function = .{ .name = try chat.toolName(spec), .arguments = try std.json.Stringify.valueAlloc(a, parsed.arguments, .{}) } });
}

pub fn parse(a: std.mem.Allocator, text: []const u8, tools: V, max_calls: ?usize, prefix: []const u8) !Result {
    if (tools != .array or tools.array.items.len == 0) return .{ .content = text };
    const blocks = try envelopes(a, text);
    var calls: std.ArrayList(Call) = .empty;
    if (blocks.len == 0) {
        var bare = trim(text);
        if (std.mem.startsWith(u8, bare, "```") and std.mem.endsWith(u8, bare, "```") and bare.len >= 6) {
            bare = trim(bare[3 .. bare.len - 3]);
            if (std.mem.startsWith(u8, bare, "json")) bare = trim(bare[4..]);
        }
        const parsed = std.json.parseFromSlice(V, a, bare, .{}) catch return .{ .content = text };
        const values = if (parsed.value == .array) parsed.value.array.items else &.{parsed.value};
        for (values) |value| {
            if (max_calls != null and calls.items.len >= max_calls.?) break;
            const one = jsonPayload(a, value) catch {
                if (max_calls == null) return .{ .content = text };
                continue;
            };
            if (tool(tools, one.name) == null) return .{ .content = text };
            try append(a, &calls, one, tools, prefix);
        }
        return .{ .content = if (calls.items.len > 0) "" else text, .calls = calls.items };
    }
    var residue: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    for (blocks) |block| {
        try residue.appendSlice(a, text[cursor..block.begin]);
        cursor = block.end;
        if (max_calls != null and calls.items.len >= max_calls.?) continue;
        const parsed = if (block.dsml) dsml(a, block.payload) catch &.{} else blk: {
            const one = payload(a, block.payload, tools, max_calls != null) catch break :blk &.{};
            const list = try a.alloc(Parsed, 1);
            list[0] = one;
            break :blk list;
        };
        var valid = parsed.len > 0;
        for (parsed) |one| valid = valid and tool(tools, one.name) != null;
        if (!valid) {
            if (max_calls == null) try residue.appendSlice(a, text[block.begin..block.end]);
            continue;
        }
        for (parsed) |one| {
            if (max_calls != null and calls.items.len >= max_calls.?) break;
            try append(a, &calls, one, tools, prefix);
        }
    }
    try residue.appendSlice(a, text[cursor..]);
    return .{ .content = trim(residue.items), .calls = calls.items };
}

pub fn preview(a: std.mem.Allocator, text: []const u8, tools: V, max_calls: ?usize) ![]const u8 {
    // Bare JSON can turn into a call; an unfinished envelope can turn back into
    // prose. Neither is safe to publish as content before its meaning is known.
    const bare = trim(text);
    if (bare.len > 0 and (bare[0] == '{' or bare[0] == '[' or bare[0] == '`')) return "";
    var end = text.len;
    var cursor: usize = 0;
    var completed = false;
    while (std.mem.indexOfScalarPos(u8, text, cursor, '<')) |begin| {
        const tag_end = std.mem.indexOfScalarPos(u8, text, begin + 1, '>') orelse {
            end = begin;
            break;
        };
        const name = text[begin + 1 .. tag_end];
        if (standardTag(name) or std.mem.eql(u8, name, "｜DSML｜tool_calls") or std.mem.eql(u8, name, "|tool_call")) {
            const close = if (std.mem.eql(u8, name, "|tool_call")) "<tool_call|>" else try std.fmt.allocPrint(a, "</{s}>", .{name});
            const at = std.ascii.findIgnoreCasePos(text, tag_end + 1, close) orelse {
                end = begin;
                break;
            };
            cursor = at + close.len;
            completed = true;
        } else cursor = begin + 1;
    }
    if (!completed and text.len > 0 and std.mem.indexOfScalar(u8, whitespace, text[0]) != null) return "";
    var result = (try parse(a, text[0..end], tools, max_calls, "preview")).content;
    if (max_calls != null) result = try singleContent(a, result);
    return std.mem.trimEnd(u8, result, whitespace);
}

fn equal(left: V, right: V) bool {
    if (std.meta.activeTag(left) != std.meta.activeTag(right)) return false;
    return switch (left) {
        .null => true,
        .bool => |v| v == right.bool,
        .integer => |v| v == right.integer,
        .float => |v| v == right.float,
        .string, .number_string => |v| std.mem.eql(u8, v, if (right == .string) right.string else right.number_string),
        .array => |v| blk: {
            if (v.items.len != right.array.items.len) break :blk false;
            for (v.items, right.array.items) |x, y| if (!equal(x, y)) break :blk false;
            break :blk true;
        },
        .object => |v| blk: {
            if (v.count() != right.object.count()) break :blk false;
            var entries = v.iterator();
            while (entries.next()) |entry| if (!equal(entry.value_ptr.*, right.object.get(entry.key_ptr.*) orelse break :blk false)) break :blk false;
            break :blk true;
        },
    };
}

pub fn check(io: std.Io, path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    const fixtures = try std.json.parseFromSlice(V, a, bytes, .{});
    var prefixes: usize = 0;
    for (fixtures.value.array.items, 0..) |fixture, index| {
        const text = fixture.object.get("text").?.string;
        const limit = fixture.object.get("max_calls").?;
        const got = try parse(a, text, fixture.object.get("tools").?, if (limit == .null) null else @intCast(limit.integer), "fixture");
        const want = fixture.object.get("calls").?.array.items;
        var same = std.mem.eql(u8, got.content, fixture.object.get("content").?.string) and got.calls.len == want.len;
        if (fixture.object.get("single_content")) |expected_content| same = same and std.mem.eql(u8, try singleContent(a, got.content), expected_content.string);
        if (same) for (got.calls, want) |call, expected| {
            const args = try std.json.parseFromSlice(V, a, call.function.arguments, .{});
            const expected_args = try std.json.parseFromSlice(V, a, expected.object.get("arguments").?.string, .{});
            same = same and std.mem.eql(u8, call.function.name, expected.object.get("name").?.string) and equal(args.value, expected_args.value);
        };
        if (!same) {
            std.debug.print("Tool fixture {d}, limit {any}, input: {s}\nExpected: {s}\nGot: {s}\n", .{ index, limit, text, try std.json.Stringify.valueAlloc(a, fixture, .{}), try std.json.Stringify.valueAlloc(a, got, .{}) });
            return error.ToolParserMismatch;
        }
        const final_content = if (limit == .null) got.content else try singleContent(a, got.content);
        var sent: usize = 0;
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        for (0..text.len + 1) |end| {
            if (end < text.len and text[end] & 0xc0 == 0x80) continue;
            _ = scratch.reset(.retain_capacity);
            const content = try preview(scratch.allocator(), text[0..end], fixture.object.get("tools").?, if (limit == .null) null else @intCast(limit.integer));
            if (content.len < sent or !std.mem.startsWith(u8, final_content, content)) {
                std.debug.print("Tool content fixture {d}, byte {d}: {s}\nExpected final: {s}\nPreview: {s}\n", .{ index, end, text, final_content, content });
                return error.ToolContentPrefixMismatch;
            }
            sent = content.len;
            prefixes += 1;
        }
    }
    std.debug.print("PASS: {d} tool parser fixtures match upstream; {d} content prefixes remain stable\n", .{ fixtures.value.array.items.len, prefixes });
}

test "tool grammars preserve schema types and malformed calls stay content" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try std.json.parseFromSlice(V, a,
        \\[{"type":"function","function":{"name":"weather","parameters":{"properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]
    , .{})).value;
    for ([_][]const u8{
        "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\",\"days\":2}}</tool_call>",
        "<x:tool_call><function=weather><parameter=city>Paris</parameter><parameter=days>2</parameter></function></x:tool_call>",
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
        "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|>",
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter><｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
    }) |input| {
        const result = try parse(a, input, tools, null, "test");
        try std.testing.expectEqual(@as(usize, 1), result.calls.len);
        try std.testing.expectEqualStrings("", result.content);
        try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":2}", result.calls[0].function.arguments);
    }
    const invalid = "<tool_call>{\"name\":\"missing\"}</tool_call>";
    try std.testing.expectEqualStrings(invalid, (try parse(a, invalid, tools, null, "test")).content);
    try std.testing.expectEqualStrings("", (try parse(a, invalid, tools, 1, "test")).content);
}
