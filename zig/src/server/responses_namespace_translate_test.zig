//! ``tools`` of a Responses request as the chat tools that run them: one level of namespaces, flattened.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const responses = @import("responses_translate.zig");
const tool_specs = @import("tool_specs.zig");
const f = @import("control_tokens_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const prompt = @import("prompt.zig");
const Server = @import("server.zig").Server;
const a = std.testing.allocator;

/// A host engine with nothing to answer but the template's own ids.
const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

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

/// A request with these tools and these input items as the chat completion that runs them.
fn chatWith(cx: *errors.Cx, tools: []const u8, items: []const u8) !json.Value {
    const b = try json.newObject(cx.a);
    try b.put(cx.a, "input", (try json.parse(cx.a, items)).ok);
    try b.put(cx.a, "tools", (try json.parse(cx.a, tools)).ok);
    return (try responses.translate(cx, .{ .object = b }, History{})).chat;
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
    try std.testing.expectEqual(@as(usize, 1), stray.object.get("tools").?.array.len); // a flat-only entry is its name: the extra key goes unread
    try std.testing.expectEqualStrings("now", stray.object.get("tools").?.array[0].get("function").?.strField("name").?);
}

test "tool_choice: an allowed_tools entry with no name is dropped flat-only, refused with a namespace" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const flat = "[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\",\"name\":\"now\"}]";
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const ch = try chat(&cx, flat, "{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\"}]}");
    const list = ch.object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("lookup", list[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("auto", ch.object.get("tool_choice").?.string);
    var mixed: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, chat(&mixed, weather_ns, "{\"type\":\"allowed_tools\",\"mode\":\"auto\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"},{\"type\":\"function\"}]}"));
    try std.testing.expect(std.mem.indexOf(u8, mixed.message, "allowed_tools") != null);
}

test "namespace parity: a namespaced request's chat body is its flat twin's, in bytes and in prompt ids" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const ns_items =
        \\[{"role":"user","content":"weather"},{"type":"function_call","call_id":"c1","name":"get_weather","namespace":"weather","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":"sunny"}]
    ;
    const flat_items =
        \\[{"role":"user","content":"weather"},{"type":"function_call","call_id":"c1","name":"weather__get_weather","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":"sunny"}]
    ;
    const flat_tools = "[{\"type\":\"function\",\"name\":\"weather__get_weather\",\"description\":\"weather tools: now\"}]";
    var first: errors.Cx = .{ .a = al };
    var second: errors.Cx = .{ .a = al };
    const ns = try chatWith(&first, weather_ns, ns_items);
    const flat = try chatWith(&second, flat_tools, flat_items);
    try std.testing.expectEqualStrings(try json.stringify(al, flat, .{ .compact = true }), try json.stringify(al, ns, .{ .compact = true }));
    const given = ns.object.get("tools").?.array;
    try std.testing.expectEqual(@as(usize, 1), given.len);
    try std.testing.expectEqualStrings("weather__get_weather", given[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("weather tools: now", given[0].get("function").?.strField("description").?);
    // the same prompt, rendered by the template for both spellings
    var fixture = try f.Fixture.init(al, 0);
    defer fixture.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(al, std.testing.io, fixture.path, null, al, &problem);
    defer hf.deinit();
    var script: Script = .{};
    const srv = try Server.init(al, std.testing.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const ns_ids = try prompt.renderIds(srv, &first, ns.object.get("messages").?, ns.object.get("tools").?.array, false, null, true);
    const flat_ids = try prompt.renderIds(srv, &second, flat.object.get("messages").?, flat.object.get("tools").?.array, false, null, true);
    try std.testing.expectEqualSlices(u32, flat_ids, ns_ids);
    try std.testing.expect(ns_ids.len > 0);
}

test "input: a function_call item's namespace is an offered group, or left off" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const ns_items =
        \\[{"role":"user","content":"weather"},{"type":"function_call","call_id":"c1","name":"get_weather","namespace":"weather","arguments":"{}"}]
    ;
    const calls = (try chatWith(&cx, weather_ns, ns_items)).object.get("messages").?.array[1].get("tool_calls").?.array;
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("weather__get_weather", calls[0].get("function").?.strField("name").?);
    // a null or absent namespace is the old answer: the name exactly as it was sent
    const flat_tools = "[{\"type\":\"function\",\"name\":\"get_weather\"},{\"type\":\"function\",\"name\":\"now\"}]";
    const plain_items =
        \\[{"role":"user","content":"weather"},{"type":"function_call","call_id":"c1","name":"get_weather","namespace":null,"arguments":"{}"},{"type":"function_call","call_id":"c2","name":"now","arguments":"{}"}]
    ;
    const kept = (try chatWith(&cx, flat_tools, plain_items)).object.get("messages").?.array[1].get("tool_calls").?.array;
    try std.testing.expectEqual(@as(usize, 2), kept.len);
    try std.testing.expectEqualStrings("get_weather", kept[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("now", kept[1].get("function").?.strField("name").?);
}

test "input: a function_call item's namespace no offered group carries is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const items =
        \\[{"role":"user","content":"weather"},{"type":"function_call","call_id":"c1","name":"get_weather","namespace":"ops","arguments":"{}"}]
    ;
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, chatWith(&cx, weather_ns, items));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "'ops'") != null);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "not among offered") != null);
    // with no tools declared there is no group to name either
    var none: errors.Cx = .{ .a = arena.allocator() };
    const body = try json.newObject(none.a);
    try body.put(none.a, "input", (try json.parse(none.a, items)).ok);
    try std.testing.expectError(error.Refused, responses.translate(&none, .{ .object = body }, History{}));
    try std.testing.expect(std.mem.indexOf(u8, none.message, "'ops'") != null);
    try std.testing.expect(std.mem.indexOf(u8, none.message, "not among offered") != null);
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

test "flat-only tools keep main's answers: one name twice is no refusal, no namespace wording" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const list = try offered(&cx, "[{\"type\":\"function\",\"name\":\"lookup\"},{\"type\":\"function\",\"name\":\"lookup\"}]");
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expectEqualStrings("lookup", list[0].get("function").?.strField("name").?);
    try std.testing.expectEqualStrings("lookup", list[1].get("function").?.strField("name").?);
    // once a namespace is offered, the same pair is refused by the name they share
    var mixed: errors.Cx = .{ .a = arena.allocator() };
    const with_ns =
        \\[{"type":"function","name":"lookup"},{"type":"namespace","name":"weather","description":"w","tools":[{"type":"function","name":"get_weather"}]},{"type":"function","name":"lookup"}]
    ;
    try std.testing.expectError(error.Refused, offered(&mixed, with_ns));
    try std.testing.expect(std.mem.indexOf(u8, mixed.message, "lookup") != null);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "namespace") == null);
}

test "flat-only tools ignore defer_loading, which a namespaced request still refuses" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const list = try offered(&cx, "[{\"type\":\"function\",\"name\":\"lookup\",\"defer_loading\":true}]");
    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("lookup", list[0].get("function").?.strField("name").?);
    var mixed: errors.Cx = .{ .a = arena.allocator() };
    try std.testing.expectError(error.Refused, offered(&mixed, "[{\"type\":\"function\",\"name\":\"lookup\",\"defer_loading\":true},{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"w\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"}]}]"));
    try std.testing.expect(std.mem.indexOf(u8, mixed.message, "defer_loading") != null);
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
