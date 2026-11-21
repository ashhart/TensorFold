//! Default Responses contracts measured against the dependency base and the strict-mode candidate.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const translate = @import("responses_translate.zig");
const responses = @import("responses.zig");
const ids = @import("ids.zig");
const Server = @import("server.zig").Server;
const Sink = @import("responses_sink.zig").Sink;
const t = std.testing;
const Value = json.Value;
const NoHistory = struct {
    pub fn conversation(_: @This(), _: *errors.Cx, _: Value) errors.Refused![]Value {
        return &.{};
    }
};
fn parsed(a: std.mem.Allocator, raw: []const u8) !Value {
    return (try json.parse(a, raw)).ok;
}
fn turn(cx: *errors.Cx, raw: []const u8) !translate.Request {
    return translate.translate(cx, try parsed(cx.a, raw), NoHistory{});
}
const tools = "[{\"type\":\"function\",\"name\":\"beta\"},{\"type\":\"function\",\"name\":\"alpha\"}]";
fn names(want: []const []const u8, request: translate.Request) !void {
    const offered = request.chat.get("tools").?.array;
    try t.expectEqual(want.len, offered.len);
    for (want, offered) |name, tool| try t.expectEqualStrings(name, tool.get("function").?.strField("name").?);
}
test "Responses strict off: item 1 allowed_tools keeps the declared prompt tool order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const request = try turn(&cx, "{\"input\":\"x\",\"tools\":" ++ tools ++ ",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\",\"name\":\"alpha\"},{\"type\":\"function\",\"name\":\"beta\"}]}}");
    try names(&.{ "beta", "alpha" }, request);
}
test "Responses strict off: items 2 and 6 ignore flat defer_loading without changing prompt tools" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const plain = try turn(&cx, "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"beta\"}]}");
    for ([_][]const u8{ "true", "false", "null", "1", "\"later\"", "{}" }) |value| {
        const raw = try std.mem.concat(cx.a, u8, &.{ "{\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"beta\",\"defer_loading\":", value, "}]}" });
        const got = try turn(&cx, raw);
        try t.expect(json.equal(plain.chat, got.chat));
    }
}
test "Responses strict off: item 3 truthy non-string function arguments pass through unchanged" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    for ([_][]const u8{ "true", "1", "[1]", "{\"x\":1}" }) |value| {
        const raw = try std.mem.concat(cx.a, u8, &.{ "{\"input\":[{\"type\":\"function_call\",\"call_id\":\"c\",\"name\":\"f\",\"arguments\":", value, "}]}" });
        const got = try turn(&cx, raw);
        const arguments = got.added[0].get("tool_calls").?.array[0].get("function").?.get("arguments").?;
        try t.expect(json.equal(try parsed(cx.a, value), arguments));
    }
}
test "Responses strict off: item 4 missing or false allowed_tools lists keep an empty set" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const missing = try turn(&cx, "{\"input\":\"x\",\"tools\":" ++ tools ++ ",\"tool_choice\":{\"type\":\"allowed_tools\"}}");
    try names(&.{}, missing);
    for ([_][]const u8{ "null", "false", "0", "\"\"", "[]" }) |value| {
        const raw = try std.mem.concat(cx.a, u8, &.{ "{\"input\":\"x\",\"tools\":", tools, ",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":", value, "}}" });
        try names(&.{}, try turn(&cx, raw));
    }
}
test "Responses strict off: item 5 unknown and nameless flat allowed entries are ignored" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const raw = "{\"input\":\"x\",\"tools\":" ++ tools ++ ",\"tool_choice\":{\"type\":\"allowed_tools\",\"tools\":[{\"type\":\"function\",\"name\":\"missing\"},{\"type\":\"function\"},{\"type\":\"function\",\"name\":\"alpha\"}]}}";
    try names(&.{"alpha"}, try turn(&cx, raw));
}
const Capture = struct {
    events: std.ArrayList(Value) = .empty,
    fn opened(_: *anyopaque) error{Closed}!void {}
    fn keepalive(_: *anyopaque) error{Closed}!void {}
    fn event(ctx: *anyopaque, a: std.mem.Allocator, _: []const u8, object: *json.Object) error{Closed}!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        const snapshot = json.stringify(a, .{ .object = object }, .{ .compact = true }) catch return error.Closed;
        const value = (json.parse(a, snapshot) catch return error.Closed).ok;
        self.events.append(a, value) catch return error.Closed;
    }
    fn sink(self: *Capture) Sink {
        return .{ .ctx = self, .vt = &.{ .open = opened, .event = event, .keepalive = keepalive } };
    }
    fn count(self: Capture, kind: []const u8) usize {
        var n: usize = 0;
        for (self.events.items) |v| if (v.typeIs(kind)) {
            n += 1;
        };
        return n;
    }
};
fn reply(a: std.mem.Allocator, cx: *errors.Cx, key: *const ids.Key, capture: *Capture) !responses.Reply {
    return .{ .srv = undefined, .a = a, .sink = capture.sink(), .base = try json.newObject(a), .streaming = true, .keep = false, .added = &.{}, .cx = cx, .owner = null, .call_ids = .{ .key = key, .body = .null } };
}
const first = "{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"alpha\",\"arguments\":\"{}\"}}]}";
const second = "{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"beta\",\"arguments\":\"{}\"}}]}";
test "Responses strict off: item 10 a new call closes the previous streamed call immediately" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const key: ids.Key = .of(ids.test_key_text);
    var capture: Capture = .{};
    var r = try reply(a, &cx, &key, &capture);
    try r.delta(try parsed(a, first), null);
    try t.expectEqual(@as(usize, 0), capture.count("response.function_call_arguments.done"));
    try r.delta(try parsed(a, second), null);
    try t.expectEqual(@as(usize, 6), capture.events.items.len);
    try t.expect(capture.events.items[2].typeIs("response.function_call_arguments.done"));
    try t.expect(capture.events.items[3].typeIs("response.output_item.done"));
    try t.expect(capture.events.items[4].typeIs("response.output_item.added"));
    try t.expectEqualStrings("completed", capture.events.items[3].get("item").?.strField("status").?);
    try r.close("completed");
    try t.expectEqual(@as(usize, 2), capture.count("response.function_call_arguments.done"));
}
test "Responses strict off: item 12 existing function calls keep fc IDs and arguments" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const key: ids.Key = .of(ids.test_key_text);
    var capture: Capture = .{};
    var r = try reply(a, &cx, &key, &capture);
    try r.delta(try parsed(a, first), null);
    const item = r.items.items[0];
    try t.expectEqualStrings("function_call", item.get("type").?.string);
    try t.expect(std.mem.startsWith(u8, item.get("id").?.string, "fc_"));
    try t.expectEqualStrings("{}", item.get("arguments").?.string);
    try t.expect(item.get("input") == null);
}
const Engine = struct {
    fn engine(self: *Engine) @import("engine_api").Engine {
        return .{ .ctx = self, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) @import("engine_api").Info {
        return .{ .context_window = 1 };
    }
    fn submit(_: *anyopaque, _: @import("engine_api").Id, _: *const @import("engine_api").Request, _: @import("engine_api").Sink) @import("engine_api").SubmitError!void {
        return error.Busy;
    }
    fn cancel(_: *anyopaque, _: @import("engine_api").Id) void {}
    fn status(_: *anyopaque, out: *@import("engine_api").Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?@import("engine_api").Memory {
        return null;
    }
};
test "Responses strict off: item 11 failed streamed turns are not stored" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backend: Engine = .{};
    var text: @import("stream_preflight_test.zig").TestText = .{};
    const srv = try Server.init(t.allocator, t.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    var conn: @import("http_conn.zig").Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var capture: Capture = .{};
    const body = try parsed(a, "{\"input\":\"too long\",\"stream\":true,\"store\":true}");
    const request = try translate.translate(&cx, body, NoHistory{});
    const outcome = responses.turn(srv, a, &conn, &cx, body, request, capture.sink(), .{});
    const final = outcome.streamed.?;
    try t.expectEqualStrings("failed", final.strField("status").?);
    try t.expect((try srv.store.get(t.io, a, final.strField("id").?, null)) == null);
    try t.expectEqual(@as(usize, 1), capture.count("response.failed"));
}
