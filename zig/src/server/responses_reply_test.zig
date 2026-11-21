const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const errors = @import("errors.zig");
const responses = @import("responses.zig");
const tool_parse = @import("tool_parse.zig");
const Server = @import("server.zig").Server;
const t = std.testing;

const blank_key: ids.Key = key: {
    @setEvalBranchQuota(100_000);
    break :key .of(ids.test_key_text);
};

fn blankReply(a: std.mem.Allocator, cx: *errors.Cx, srv: *Server, streaming: bool, sink: @import("responses_sink.zig").Sink, chat: json.Value) !responses.Reply {
    return .{ .srv = srv, .a = a, .sink = sink, .base = try json.newObject(a), .streaming = streaming, .keep = false, .added = &.{}, .cx = cx, .owner = null, .call_ids = .{ .key = &blank_key, .body = chat } };
}

test "a call delta that names no id takes the chat run's id at its index, and one that names its id keeps it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: ids.Key = .of(ids.test_key_text);
    const chat = (try json.parse(a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}")).ok;
    var cx: errors.Cx = .{ .a = a };
    var reply: responses.Reply = .{ .srv = undefined, .a = a, .sink = undefined, .base = try json.newObject(a), .streaming = false, .keep = false, .added = &.{}, .cx = &cx, .owner = null, .call_ids = .{ .key = &key, .body = chat } };
    try reply.delta((try json.parse(a, "{\"tool_calls\":[{\"index\":1,\"function\":{\"name\":\"f\"}},{\"index\":2,\"id\":\"call_sent\",\"function\":{\"name\":\"f\"}}]}")).ok, null);
    var calls: ids.Calls = .{ .key = &key, .body = chat };
    try t.expectEqualStrings(try calls.id(a, 0, 1), reply.items.items[0].get("call_id").?.string);
    try t.expectEqualStrings("call_sent", reply.items.items[1].get("call_id").?.string);
}

test "Server recovery: Responses grows reasoning, text and interleaved call arguments in bounded buffers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    var srv: Server = undefined;
    const key: ids.Key = .of(ids.test_key_text);
    var r: responses.Reply = .{ .srv = &srv, .a = a, .sink = undefined, .base = try json.newObject(a), .streaming = false, .keep = false, .added = &.{}, .cx = &cx, .owner = null, .call_ids = .{ .key = &key, .body = .null } };
    const chunk = "abcdefgh";
    for ([_][]const u8{ "reasoning", "message" }) |kind| {
        for (0..3000) |_| try r.write(kind, chunk, null);
        const text = r.current.?.get("content").?.array[0].get("text").?.string;
        try t.expectEqual(@as(usize, 3000 * chunk.len), text.len);
        for (0..3000) |i| try t.expectEqualStrings(chunk, text[i * chunk.len ..][0..chunk.len]);
    }
    const first = (try json.parse(a, "{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"f\",\"arguments\":\"abcdefgh\"}}]}")).ok;
    const second = (try json.parse(a, "{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"g\",\"arguments\":\"abcdefgh\"}}]}")).ok;
    for (0..3000) |_| {
        try r.delta(first, null);
        try r.delta(second, null);
    }
    for ([_]i64{ 0, 1 }) |i| {
        const args = r.calls.get(i).?.get("arguments").?.string;
        try t.expectEqual(@as(usize, 3000 * chunk.len), args.len);
        for (0..3000) |j| try t.expectEqualStrings(chunk, args[j * chunk.len ..][0..chunk.len]);
    }
    try t.expectEqual(@as(usize, 4), r.items.items.len);
    try t.expect(arena.queryCapacity() < 4 << 20);
}

test "Responses custom calls preserve raw input only for offered custom tools" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"custom\",\"name\":\"freeform\"}]")).ok.array;
    const parsed = try tool_parse.parse(a, "<tool_call>{\"name\":\"freeform\",\"arguments\":\"raw  + π\\n{\"}</tool_call>", tools, null);
    const call = parsed.calls.?[0];
    try t.expectEqualStrings("custom", call.get("type").?.string);
    try t.expectEqualStrings("raw  + π\n{", call.get("custom").?.get("input").?.string);
    try t.expectEqualStrings(call.get("custom").?.get("input").?.string, call.get("function").?.get("arguments").?.string);

    const functions = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"ordinary\",\"parameters\":{\"type\":\"object\"}}}]")).ok.array;
    const malformed = try tool_parse.parse(a, "<tool_call>{\"name\":\"ordinary\",\"arguments\":\"not JSON\"}</tool_call>", functions, null);
    try t.expect(malformed.calls == null);
    try t.expectEqual(@as(usize, 1), malformed.kept_malformed);
}

const Capture = struct {
    events: std.ArrayList(Event) = .empty,

    const Event = struct { kind: []const u8, payload: *json.Object };

    fn sink(c: *Capture) @import("responses_sink.zig").Sink {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .keepalive = keepalive } };
    }
    fn open(_: *anyopaque) error{Closed}!void {}
    fn keepalive(_: *anyopaque) error{Closed}!void {}
    fn event(ctx: *anyopaque, _: std.mem.Allocator, kind: []const u8, payload: *json.Object) error{Closed}!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.events.append(t.allocator, .{ .kind = kind, .payload = payload }) catch return error.Closed;
    }
    fn deinit(c: *Capture) void {
        c.events.deinit(t.allocator);
    }
    fn count(c: Capture, kind: []const u8) usize {
        var n: usize = 0;
        for (c.events.items) |one| if (std.mem.eql(u8, one.kind, kind)) {
            n += 1;
        };
        return n;
    }
};

test "Responses custom calls keep IDs and indices, and a partial second call ends incomplete once" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv: Server = undefined;
    var capture: Capture = .{};
    defer capture.deinit();
    var cx: errors.Cx = .{ .a = a };
    var r = try blankReply(a, &cx, &srv, true, capture.sink(), .null);
    const deltas = (try json.parse(a, "{\"content\":\"working\",\"tool_calls\":[{\"index\":0,\"id\":\"call-one\",\"custom\":{\"name\":\"first\",\"input\":\"raw \"}},{\"index\":1,\"id\":\"call-two\",\"custom\":{\"name\":\"second\",\"input\":\"{\\\"partial\"}}]}")).ok;
    try r.delta(deltas, null);
    const final = try r.finish("length", null, null);
    try t.expectEqualStrings("incomplete", final.get("status").?.string);
    try t.expectEqual(@as(usize, 1), capture.count("response.incomplete"));
    try t.expectEqual(@as(usize, 2), capture.count("response.custom_tool_call_input.done"));
    try t.expectEqual(@as(usize, 0), capture.count("response.function_call_arguments.done"));
    const output = final.get("output").?.array;
    try t.expectEqual(@as(usize, 3), output.len); // commentary, then two calls
    try t.expectEqualStrings("call-one", output[1].get("call_id").?.string);
    try t.expectEqualStrings("raw ", output[1].get("input").?.string);
    try t.expectEqualStrings("completed", output[1].get("status").?.string);
    try t.expectEqualStrings("call-two", output[2].get("call_id").?.string);
    try t.expectEqualStrings("{\"partial", output[2].get("input").?.string);
    try t.expectEqualStrings("incomplete", output[2].get("status").?.string);
    try t.expect(std.mem.startsWith(u8, output[1].get("id").?.string, "ctc_"));
    for (capture.events.items, 0..) |event, i| try t.expectEqual(@as(i64, @intCast(i)), event.payload.get("sequence_number").?.int64() orelse 0);
}

test "Responses interleaved same-name calls finish after their last deltas and in call order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv: Server = undefined;
    var capture: Capture = .{};
    defer capture.deinit();
    var cx: errors.Cx = .{ .a = a };
    var r = try blankReply(a, &cx, &srv, true, capture.sink(), .null);
    for ([_][]const u8{
        \\{"tool_calls":[{"index":0,"id":"custom-id","custom":{"name":"shared","input":"raw "}}]}
        ,
        \\{"tool_calls":[{"index":1,"id":"function-id","function":{"name":"shared","arguments":"{\"x\":"}}]}
        ,
        \\{"tool_calls":[{"index":0,"custom":{"input":"+ tail"}}]}
        ,
        \\{"tool_calls":[{"index":1,"function":{"arguments":"1}"}}]}
        ,
    }) |text| try r.delta((try json.parse(a, text)).ok, null);
    const final = try r.finish("stop", null, null);
    const output = final.get("output").?.array;
    try t.expectEqual(@as(usize, 2), output.len);
    try t.expectEqualStrings("custom_tool_call", output[0].strField("type").?);
    try t.expectEqualStrings("raw + tail", output[0].strField("input").?);
    try t.expectEqualStrings("function_call", output[1].strField("type").?);
    try t.expectEqualStrings("{\"x\":1}", output[1].strField("arguments").?);
    var custom_done = false;
    var function_done = false;
    var added: usize = 0;
    var closed: usize = 0;
    for (capture.events.items) |one| {
        if (std.mem.eql(u8, one.kind, "response.custom_tool_call_input.delta")) try t.expect(!custom_done);
        if (std.mem.eql(u8, one.kind, "response.function_call_arguments.delta")) try t.expect(!function_done);
        if (std.mem.eql(u8, one.kind, "response.custom_tool_call_input.done")) custom_done = true;
        if (std.mem.eql(u8, one.kind, "response.function_call_arguments.done")) function_done = true;
        if (std.mem.eql(u8, one.kind, "response.output_item.added")) {
            try t.expectEqual(@as(i64, @intCast(added)), one.payload.get("output_index").?.int64().?);
            added += 1;
        }
        if (std.mem.eql(u8, one.kind, "response.output_item.done")) {
            try t.expectEqual(@as(i64, @intCast(closed)), one.payload.get("output_index").?.int64().?);
            closed += 1;
        }
    }
    try t.expect(custom_done and function_done);
    try t.expectEqual(@as(usize, 1), capture.count("response.custom_tool_call_input.done"));
    try t.expectEqual(@as(usize, 1), capture.count("response.function_call_arguments.done"));
    try t.expectEqual(@as(usize, 2), added);
    try t.expectEqual(@as(usize, 2), closed);
}

test "Responses foreground streamed failures persist exactly when store is true" {
    const Engine = struct {
        fn engine(self: *const @This()) @import("engine_api").Engine {
            return .{ .ctx = @constCast(self), .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
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
    const History = struct {
        pub fn conversation(_: @This(), _: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
            return &.{};
        }
    };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture: Engine = .{};
    var text: @import("stream_preflight_test.zig").TestText = .{};
    const srv = try Server.init(t.allocator, t.io, fixture.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    var conn: @import("http_conn.zig").Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var capture: Capture = .{};
    defer capture.deinit();
    for ([_]bool{ true, false }) |keep| {
        const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"input\":\"too long\",\"stream\":true,\"store\":{}}}", .{keep}))).ok;
        const request = try @import("responses_translate.zig").translate(&cx, body, History{});
        const outcome = responses.turn(srv, a, &conn, &cx, body, request, capture.sink(), .{});
        const final = outcome.streamed.?;
        try t.expectEqualStrings("failed", final.strField("status").?);
        const kept = try srv.store.get(t.io, a, final.strField("id").?, null);
        if (keep) try t.expect(json.equal(final, kept.?)) else try t.expect(kept == null);
    }
    try t.expectEqual(@as(usize, 2), capture.count("response.failed"));
}
