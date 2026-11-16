//! ``tools`` of a Responses request as the chat tools that run them: one level of namespaces, flattened.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const responses = @import("responses_translate.zig");
const tool_specs = @import("tool_specs.zig");
const a = std.testing.allocator;

const History = struct {
    pub fn conversation(_: @This(), c: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return c.refuse("this fixture has no stored response");
    }
};

/// A request with these tools as the chat completion that runs them.
fn chat(cx: *errors.Cx, tools: []const u8, choice: ?[]const u8) !json.Value {
    const b = try json.newObject(cx.a);
    try b.put(cx.a, "input", .{ .string = "hi" });
    try b.put(cx.a, "tools", (try json.parse(cx.a, tools)).ok);
    if (choice) |ch| try b.put(cx.a, "tool_choice", (try json.parse(cx.a, ch)).ok);
    return (try responses.translate(cx, .{ .object = b }, History{})).chat;
}

/// The chat tool_choice name a request with these tools and this choice carries.
fn chosen(cx: *errors.Cx, tools: []const u8, choice: []const u8) ![]const u8 {
    const c = (try chat(cx, tools, choice)).object.get("tool_choice").?;
    return c.get("function").?.strField("name").?;
}

/// The chat tools a request with these tools offers.
fn offered(cx: *errors.Cx, tools: []const u8) ![]json.Value {
    return (try chat(cx, tools, null)).object.get("tools").?.array;
}

test "namespace tools: a function child is flattened as <namespace>__<tool> with both descriptions" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const list = try offered(&cx, "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"current conditions\",\"parameters\":{\"type\":\"object\",\"properties\":{}},\"strict\":true}]}]");
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("function", list[0].strField("type").?);
    const function = list[0].get("function").?;
    try std.testing.expectEqualStrings("weather__get_weather", function.strField("name").?);
    try std.testing.expectEqualStrings("weather tools: current conditions", function.strField("description").?);
    try std.testing.expect(function.get("parameters") != null);
    try std.testing.expectEqual(true, function.get("strict").?.bool);
}

test "namespace tools: a request without one keeps the tools bytes it had before namespacing" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const list = try offered(&cx, "[{\"type\":\"function\",\"name\":\"lookup\",\"description\":\"look up a word\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]},\"strict\":true},{\"type\":\"function\",\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]");
    try std.testing.expectEqualStrings("[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"description\":\"look up a word\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]},\"strict\":true}},{\"type\":\"function\",\"function\":{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]", try json.stringify(cx.a, .{ .array = list }, .{ .compact = true }));
}

test "namespace tools: two children that flatten to one name are refused by that name" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"the forecast\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"}]},{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"the radar\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "weather__get_weather") != null);
}

test "namespace tools: a child that asks to be loaded later is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"defer_loading\":true}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "defer_loading") != null);
}

test "namespace tools: a namespace inside a namespace is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"outer\",\"description\":\"outer tools\",\"tools\":[{\"type\":\"namespace\",\"name\":\"inner\",\"description\":\"inner tools\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"}]}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "another namespace") != null);
}

test "namespace tools: a hosted tool is still refused by the type it names" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"web_search\"}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "'web_search'") != null);
}

const weather_ns = "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"now\"}]}]";
const mixed_ns = "[{\"type\":\"function\",\"name\":\"now\"},{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"now\"}]}]";

test "tool_choice: a named namespace child reaches the engine under its flattened name" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectEqualStrings("weather__get_weather", try chosen(&cx, weather_ns, "{\"type\":\"function\",\"name\":\"get_weather\"}"));
    const list = (try chat(&cx, weather_ns, "{\"type\":\"function\",\"name\":\"get_weather\"}")).object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("weather__get_weather", list[0].get("function").?.strField("name").?);
}

test "tool_choice: a child named with its namespace spelled out resolves to that group's child" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectEqualStrings("weather__get_weather", try chosen(&cx, weather_ns, "{\"type\":\"function\",\"name\":\"get_weather\",\"namespace\":\"weather\"}"));
}

test "tool_choice: the flattened spelling is the name it keeps" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectEqualStrings("weather__get_weather", try chosen(&cx, weather_ns, "{\"type\":\"function\",\"name\":\"weather__get_weather\"}"));
}

test "tool_choice: allowed_tools keeps the namespace's selected child and drops the rest" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const tools =
        \\[{"type":"function","name":"lookup"},{"type":"namespace","name":"weather","description":"weather tools","tools":[{"type":"function","name":"get_weather"}]}]
    ;
    const ch = try chat(&cx, tools, "{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"}]}");
    const list = ch.object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("weather__get_weather", list[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("auto", ch.object.get("tool_choice").?.string);
}

test "tool_choice: an unqualified child name two namespaces hold is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const tools =
        \\[{"type":"namespace","name":"weather","description":"w","tools":[{"type":"function","name":"now"}]},{"type":"namespace","name":"radar","description":"r","tools":[{"type":"function","name":"now"}]}]
    ;
    try std.testing.expectError(error.Refused, chat(&cx, tools, "{\"type\":\"function\",\"name\":\"now\"}"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "now") != null);
}

test "tool_choice: an unresolved name keeps its spelling, which the chat route then refuses by" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const flat = "[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\",\"name\":\"now\"}]";
    for ([_][]const u8{ flat, weather_ns }) |listed| for ([_][]const u8{ "absent", "nope" }) |name| {
        const choice = try std.fmt.allocPrint(cx.a, "{{\"type\":\"function\",\"name\":\"{s}\"}}", .{name});
        const ch = try chat(&cx, listed, choice);
        try std.testing.expectEqualStrings(name, ch.object.get("tool_choice").?.get("function").?.strField("name").?);
        var spec: errors.Cx = .{ .a = cx.a };
        try std.testing.expectError(error.Refused, tool_specs.active(&spec, ch.object.get("tools"), ch.object.get("tool_choice")));
        try std.testing.expectEqual(errors.Kind.other, spec.kind);
        const words = try std.fmt.allocPrint(cx.a, "tool_choice requested unknown tool '{s}'", .{name});
        try std.testing.expectEqualStrings(words, spec.message);
    };
}

test "tool_choice: a flat-only request passes an unknown name through to the chat route's own refusal" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const flat = "[{\"type\":\"function\",\"name\":\"lookup\"}]";
    const ch = try chat(&cx, flat, "{\"type\":\"function\",\"name\":\"nope\"}");
    try std.testing.expectEqual(@as(usize, 1), ch.object.get("tools").?.array.len);
    try std.testing.expectEqualStrings("nope", ch.object.get("tool_choice").?.get("function").?.strField("name").?);
    var spec: errors.Cx = .{ .a = cx.a };
    try std.testing.expectError(error.Refused, tool_specs.active(&spec, ch.object.get("tools"), ch.object.get("tool_choice")));
    try std.testing.expectEqual(errors.Kind.other, spec.kind);
    try std.testing.expectEqualStrings("tool_choice requested unknown tool 'nope'", spec.message);
}

test "tool_choice: allowed_tools drops a name no tool carries instead of refusing" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const flat = "[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\",\"name\":\"now\"}]";
    const ch = try chat(&cx, flat, "{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\",\"name\":\"absent\"}]}");
    const list = ch.object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("lookup", list[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("auto", ch.object.get("tool_choice").?.string);
    const none = try chat(&cx, flat, "{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"absent\"}]}");
    try std.testing.expectEqual(@as(usize, 0), none.object.get("tools").?.array.len);
    const stray = try chat(&cx, flat, "{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"now\",\"namespace\":\"anything\"}]}");
    try std.testing.expectEqual(@as(usize, 0), stray.object.get("tools").?.array.len); // a namespace no group carries names nothing
}

test "tool_choice: allowed_tools may spell the namespace on an entry" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const ch = try chat(&cx, weather_ns, "{\"type\":\"allowed_tools\",\"mode\":\"required\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"namespace\":\"weather\"},{\"type\":\"function\",\"name\":\"nope\"}]}");
    const list = ch.object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("weather__get_weather", list[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("required", ch.object.get("tool_choice").?.string);
}

test "tool_choice: a top-level tool keeps its own name, and its own name wins over a child's" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const tools =
        \\[{"type":"function","name":"now"},{"type":"namespace","name":"weather","description":"weather tools","tools":[{"type":"function","name":"now"}]}]
    ;
    try std.testing.expectEqualStrings("now", try chosen(&cx, tools, "{\"type\":\"function\",\"name\":\"now\"}"));
}

test "tool_choice: a literal flat name with the namespace separator keeps its own spelling" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const tools = "[{\"type\":\"function\",\"name\":\"weather__get_weather\"}]";
    try std.testing.expectEqualStrings("weather__get_weather", try chosen(&cx, tools, "{\"type\":\"function\",\"name\":\"weather__get_weather\"}"));
    const list = (try chat(&cx, tools, "{\"type\":\"function\",\"name\":\"weather__get_weather\"}")).object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("weather__get_weather", list[0].get("function").?.strField("name").?);
}

test "a flat name that spells a flattened child collides with it, in either order" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const first =
        \\[{"type":"namespace","name":"weather","description":"w","tools":[{"type":"function","name":"get_weather"}]},{"type":"function","name":"weather__get_weather"}]
    ;
    const second =
        \\[{"type":"function","name":"weather__get_weather"},{"type":"namespace","name":"weather","description":"w","tools":[{"type":"function","name":"get_weather"}]}]
    ;
    try std.testing.expectError(error.Refused, offered(&cx, first));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "weather__get_weather") != null);
    try std.testing.expectError(error.Refused, offered(&cx, second));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "weather__get_weather") != null);
}

test "two flat tools with one name are refused without any namespace wording" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\",\"name\":\"lookup\"}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "lookup") != null);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "namespace") == null);
}

test "a flat function tool that asks to be loaded later is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"function\",\"name\":\"lookup\",\"defer_loading\":true}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "defer_loading") != null);
}

test "a namespace group missing its name, its description or its list is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"description\":\"d\",\"tools\":[{\"type\":\"function\",\"name\":\"x\"}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "name") != null);
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"w\",\"description\":\"\",\"tools\":[{\"type\":\"function\",\"name\":\"x\"}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "description") != null);
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"w\",\"description\":\"d\"}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "list") != null);
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"w\",\"description\":\"d\",\"tools\":[]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "function tool") != null);
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"w\",\"description\":\"d\",\"tools\":\"x\"}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "list") != null);
}

test "a namespace child without a name, or of another type, is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"w\",\"tools\":[{\"type\":\"function\"}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "name") != null);
    try std.testing.expectError(error.Refused, offered(&cx, "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"w\",\"tools\":[{\"type\":\"custom\",\"name\":\"c\"}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "'custom'") != null);
}

test "a namespace child's description: absent or null keeps the namespace's, another type is refused" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const base = "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"";
    var list = try offered(&cx, base ++ "}]}]");
    try std.testing.expectEqualStrings("weather tools", list[0].get("function").?.strField("description").?);
    list = try offered(&cx, base ++ ",\"description\":null}]}]");
    try std.testing.expectEqualStrings("weather tools", list[0].get("function").?.strField("description").?);
    try std.testing.expectError(error.Refused, offered(&cx, base ++ ",\"description\":5}]}]"));
}

test "tool_choice: a selector equal to a child's composed name is that child, not ambiguous" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const tools =
        \\[{"type":"namespace","name":"a","description":"ad","tools":[{"type":"function","name":"b"},{"type":"function","name":"a__b"}]}]
    ;
    try std.testing.expectEqualStrings("a__b", try chosen(&cx, tools, "{\"type\":\"function\",\"name\":\"a__b\"}"));
}

test "tool_choice: an explicit namespace never picks a top-level tool with the same name" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectEqualStrings("weather__now", try chosen(&cx, mixed_ns, "{\"type\":\"function\",\"name\":\"now\",\"namespace\":\"weather\"}"));
}
