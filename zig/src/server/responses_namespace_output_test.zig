//! Responses output for namespace tools: call items keep the child's name plus its namespace; history flattens back.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const ids = @import("ids.zig");
const responses = @import("responses.zig");
const translate = @import("responses_translate.zig");
const store_mod = @import("responses_store.zig");
const Sink = @import("responses_sink.zig").Sink;
const Value = json.Value;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const weather_tools = "[{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"now\"}]}]";
const flat_tools = "[{\"type\":\"function\",\"name\":\"weather__get_weather\"}]";

/// A turn's base with the request's raw tools echoed, as ``newBase`` writes it.
fn baseOf(a: Allocator, tools: []const u8) !*json.Object {
    const base = try json.newObject(a);
    try base.put(a, "id", .{ .string = "resp_test" });
    try base.put(a, "object", .{ .string = "response" });
    try base.put(a, "output", .{ .array = &.{} });
    try base.put(a, "tools", (try json.parse(a, tools)).ok);
    return base;
}

/// A collecting transport, so a streamed turn's events can be read back.
const Collect = struct {
    list: std.ArrayList(Value) = .empty,

    fn start(_: *anyopaque) error{Closed}!void {}
    fn note(ctx: *anyopaque, a: Allocator, _: []const u8, payload: *json.Object) error{Closed}!void {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.list.append(a, .{ .object = payload }) catch return error.Closed;
    }
    fn beat(_: *anyopaque) error{Closed}!void {}

    const vt: Sink.VTable = .{ .open = start, .event = note, .keepalive = beat };
};

/// A turn's reply over ``base``, on ``sink``'s transport (``undefined``: no events asked for).
fn replyOf(a: Allocator, base: *json.Object, cx: *errors.Cx, key: *const ids.Key, streaming: bool, sink: Sink) responses.Reply {
    return .{
        .srv = undefined,
        .a = a,
        .sink = sink,
        .base = base,
        .streaming = streaming,
        .owner = null,
        .keep = false,
        .added = &.{},
        .cx = cx,
        .call_ids = .{ .key = key, .body = .null },
    };
}

const call_delta = "{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"weather__get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Rome\\\"}\"}}]}";
const now_delta = "{\"tool_calls\":[{\"index\":0,\"id\":\"call_2\",\"function\":{\"name\":\"now\",\"arguments\":\"{}\"}}]}";
const mixed_tools = "[{\"type\":\"function\",\"name\":\"now\"},{\"type\":\"namespace\",\"name\":\"weather\",\"description\":\"weather tools\",\"tools\":[{\"type\":\"function\",\"name\":\"now\"}]}]";

test "output: a top-level tool's call keeps no namespace even when a namespace holds the same name" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: ids.Key = .of(ids.test_key_text);
    var cx: errors.Cx = .{ .a = a };
    var reply = replyOf(a, try baseOf(a, mixed_tools), &cx, &key, false, undefined);
    try reply.delta((try json.parse(a, now_delta)).ok, null);
    const item: Value = .{ .object = reply.items.items[0] };
    try testing.expectEqualStrings("now", item.strField("name").?);
    try testing.expect(item.get("namespace") == null);
}

test "output: a namespaced request's call item carries the child's name and its namespace" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: ids.Key = .of(ids.test_key_text);
    var cx: errors.Cx = .{ .a = a };
    var reply = replyOf(a, try baseOf(a, weather_tools), &cx, &key, false, undefined);
    try reply.delta((try json.parse(a, call_delta)).ok, null);
    try testing.expectEqual(@as(usize, 1), reply.items.items.len);
    const item: Value = .{ .object = reply.items.items[0] };
    try testing.expectEqualStrings("function_call", item.strField("type").?);
    try testing.expectEqualStrings("call_1", item.strField("call_id").?);
    try testing.expectEqualStrings("get_weather", item.strField("name").?);
    try testing.expectEqualStrings("weather", item.strField("namespace").?);
    try testing.expectEqualStrings("{\"city\":\"Rome\"}", item.strField("arguments").?);
    try testing.expectEqualStrings("in_progress", item.strField("status").?);
}

test "output: a flat-only request's call item keeps exactly the fields it had before namespacing" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: ids.Key = .of(ids.test_key_text);
    var cx: errors.Cx = .{ .a = a };
    var reply = replyOf(a, try baseOf(a, flat_tools), &cx, &key, false, undefined);
    try reply.delta((try json.parse(a, call_delta)).ok, null);
    const made = reply.items.items[0];
    const item: Value = .{ .object = made };
    try testing.expect(item.get("namespace") == null);
    try testing.expect(std.mem.startsWith(u8, item.strField("id").?, "fc_"));
    _ = made.orderedRemove("id");
    try testing.expectEqualStrings(
        "{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"weather__get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Rome\\\"}\",\"status\":\"in_progress\"}",
        try json.stringify(a, item, .{ .compact = true }),
    );
}

test "output: streamed events carry the same namespaced item, and the close event repeats it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: ids.Key = .of(ids.test_key_text);
    var cx: errors.Cx = .{ .a = a };
    var seen: Collect = .{};
    const sink: Sink = .{ .ctx = @ptrCast(&seen), .vt = &Collect.vt };
    var reply = replyOf(a, try baseOf(a, weather_tools), &cx, &key, true, sink);
    try reply.delta((try json.parse(a, call_delta)).ok, null);
    try reply.close("completed");
    const added = seen.list.items[0];
    try testing.expectEqualStrings("response.output_item.added", added.strField("type").?);
    try testing.expectEqualStrings("weather", added.get("item").?.strField("namespace").?);
    try testing.expectEqualStrings("get_weather", added.get("item").?.strField("name").?);
    const done = seen.list.items[seen.list.items.len - 1];
    try testing.expectEqualStrings("response.output_item.done", done.strField("type").?);
    try testing.expectEqualStrings("weather", done.get("item").?.strField("namespace").?);
}

test "history: a stored call item flattens to the chat tool call the flat spelling gives" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const namespaced =
        \\[{"type":"function_call","call_id":"c1","name":"get_weather","namespace":"weather","arguments":"{}"}]
    ;
    const flat =
        \\[{"type":"function_call","call_id":"c1","name":"weather__get_weather","arguments":"{}"}]
    ;
    const null_ns =
        \\[{"type":"function_call","call_id":"c1","name":"weather__get_weather","namespace":null,"arguments":"{}"}]
    ;
    var first: errors.Cx = .{ .a = a };
    var second: errors.Cx = .{ .a = a };
    var third: errors.Cx = .{ .a = a };
    const made = try translate.messages(&first, (try json.parse(a, namespaced)).ok.array);
    const kept = try translate.messages(&second, (try json.parse(a, flat)).ok.array);
    const nulled = try translate.messages(&third, (try json.parse(a, null_ns)).ok.array);
    const want = try json.stringify(a, .{ .array = kept }, .{ .compact = true });
    try testing.expectEqualStrings(want, try json.stringify(a, .{ .array = made }, .{ .compact = true }));
    try testing.expectEqualStrings(want, try json.stringify(a, .{ .array = nulled }, .{ .compact = true }));
    try testing.expect(std.mem.indexOf(u8, want, "\"name\":\"weather__get_weather\"") != null);
}

test "history: a call item with a namespace but no name, or a namespace that is not a string, is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var no_name: errors.Cx = .{ .a = a };
    try testing.expectError(error.Refused, translate.messages(&no_name, (try json.parse(a, "[{\"type\":\"function_call\",\"call_id\":\"c1\",\"namespace\":\"weather\"}]")).ok.array));
    try testing.expect(std.mem.indexOf(u8, no_name.message, "call_id and a name") != null);
    var number: errors.Cx = .{ .a = a };
    try testing.expectError(error.Refused, translate.messages(&number, (try json.parse(a, "[{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"get_weather\",\"namespace\":5}]")).ok.array));
    try testing.expect(std.mem.indexOf(u8, number.message, "namespace") != null);
    var object: errors.Cx = .{ .a = a };
    try testing.expectError(error.Refused, translate.messages(&object, (try json.parse(a, "[{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"get_weather\",\"namespace\":{\"a\":1}}]")).ok.array));
    try testing.expect(std.mem.indexOf(u8, object.message, "namespace") != null);
}

test "store: a namespaced call in a stored output replays as the flattened tool call" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var cx: errors.Cx = .{ .a = a };
    var s: store_mod.Store = .{ .gpa = testing.allocator };
    defer s.deinit();
    const h: store_mod.History = .{ .store = &s, .io = io, .owner = null };
    const user = (try json.parse(a, "{\"role\":\"user\",\"content\":\"hi\"}")).ok;
    var one = [_]Value{user};
    const final = (try json.parse(a, "{\"id\":\"resp_n\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"get_weather\",\"namespace\":\"weather\",\"arguments\":\"{}\"}]}")).ok;
    try s.put(io, &cx, final, &one);
    const messages = h.lookup(&cx, "resp_n") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 2), messages.len);
    const call = messages[1].get("tool_calls").?.array[0];
    try testing.expectEqualStrings("weather__get_weather", call.get("function").?.strField("name").?);
    try testing.expectEqualStrings("c1", call.strField("id").?);
}

test "history: an already flattened name with its namespace is not prefixed twice" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const items = (try json.parse(a,
        \\[{"type":"function_call","call_id":"c1","name":"weather__get_weather","namespace":"weather","arguments":"{}"}]
    )).ok.array;
    const made = try translate.messages(&cx, items);
    const text = try json.stringify(a, .{ .array = made }, .{ .compact = true });
    try testing.expect(std.mem.indexOf(u8, text, "\"name\":\"weather__get_weather\"") != null);
    try testing.expect(std.mem.indexOf(u8, text, "weather__weather__") == null);
}
