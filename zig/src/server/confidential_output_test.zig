//! --confidential's output rule: no request text in a log line, an HTTP error body, or a stream or socket error event.
const std = @import("std");
const builtin = @import("builtin");
const log = @import("log.zig");
const output = @import("confidential_output.zig");
const fixture = @import("witness_fixture.zig");
const t = std.testing;

const marker = "MARKER-7f3a";

test "confidential: a log line keeps its fixed words, numbers, enums and server ids, and shows `-` for any text" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const Point = struct { x: u8 };
    log.withheld(&w, "a {s} b {d} c {} {{x}} {t} {f} {?d} {any} end", .{ marker, 5, true, std.log.Level.warn, log.Safe{ .text = "chatcmpl-1" }, @as(?u32, 9), Point{ .x = 1 } });
    try t.expectEqualStrings("a - b 5 c true {x} warn chatcmpl-1 9 - end", w.buffered());
}

const Lines = struct {
    seen: std.ArrayList(u8) = .empty,
    fn call(ctx: *anyopaque, text: []const u8) void {
        const l: *Lines = @ptrCast(@alignCast(ctx));
        l.seen.appendSlice(t.allocator, text) catch {};
    }
};

test "confidential: the refused, ended, stream-error and access lines never carry the request's words" {
    var lines: Lines = .{};
    defer lines.seen.deinit(t.allocator);
    log.Test.sink = .{ .ctx = &lines, .call = Lines.call };
    defer log.Test.sink = null;
    // the control: without the flag, each line quotes what it was given
    log.refusedLine("chatcmpl-1", "unknown role '" ++ marker ++ "'");
    log.endedLine("chatcmpl-1", "tool text " ++ marker, 3, 1, 0.5);
    log.line("stream error: {s}", .{marker});
    try t.expectEqual(@as(usize, 3), std.mem.count(u8, lines.seen.items, marker));
    lines.seen.clearRetainingCapacity();
    output.confine();
    defer output.release();
    log.refusedLine("chatcmpl-1", "unknown role '" ++ marker ++ "'");
    log.endedLine("chatcmpl-1", "tool text " ++ marker, 3, 1, 0.5);
    log.line("stream error: {s}", .{marker});
    log.confidentialAccess("chatcmpl-1", 400, "invalid_request_error", null);
    try t.expect(std.mem.indexOf(u8, lines.seen.items, marker) == null);
    try t.expectEqualStrings(
        "[tensorfold] ended chatcmpl-1 reason=error (error) prompt=3 tokens=1 after=0.50s\n" ++
            "[tensorfold] stream error: -\n" ++
            "[tensorfold] chatcmpl-1 400 invalid_request_error -\n",
        lines.seen.items,
    );
}

test "confidential: a server-made id stays in its log line; only the request's own text is withheld" {
    var lines: Lines = .{};
    defer lines.seen.deinit(t.allocator);
    log.Test.sink = .{ .ctx = &lines, .call = Lines.call };
    defer log.Test.sink = null;
    output.confine();
    defer output.release();
    const empty = [_]f32{};
    _ = @import("speech_register_log.zig").jump(t.allocator, "speech-7", &empty, 24000, .{ .st = 1 });
    log.endedLine("chatcmpl-9", null, 1, 1, 0.1);
    try t.expect(std.mem.indexOf(u8, lines.seen.items, "speech speech-7 register unmeasured") != null);
    try t.expect(std.mem.indexOf(u8, lines.seen.items, "ended chatcmpl-9 ") != null);
}

test "confidential: an error event's message, type and code are fixed parts; another event is as given" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json = @import("json");
    const chunk = (try json.parse(a, "{\"choices\":[{\"delta\":{\"content\":\"" ++ marker ++ "\"}}]}")).ok;
    const failed = (try json.parse(a, "{\"error\":{\"message\":\"bad " ++ marker ++ "\",\"type\":\"server_error\",\"code\":\"" ++ marker ++ "\",\"param\":\"" ++ marker ++ "\"}}")).ok;
    const bare = (try json.parse(a, "{\"type\":\"error\",\"code\":\"context_length_exceeded\",\"message\":\"" ++ marker ++ "\",\"param\":\"x\"}")).ok;
    const response = (try json.parse(a, "{\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"error\":{\"message\":\"" ++ marker ++ "\",\"code\":\"server_error\"}}}")).ok;
    // the control: without the flag every event is as given
    try t.expectEqual(failed.object, (try output.event(a, failed, "req-1")).object);
    output.confine();
    defer output.release();
    try t.expectEqual(chunk.object, (try output.event(a, chunk, "req-1")).object);
    for ([_]json.Value{ failed, bare, response }) |v| {
        const text = try json.stringify(a, try output.event(a, v, "req-1"), .{});
        try t.expect(std.mem.indexOf(u8, text, marker) == null);
        try t.expect(std.mem.indexOf(u8, text, "\"request_id\": \"req-1\"") != null);
    }
    const kept = try json.stringify(a, try output.event(a, bare, "req-1"), .{});
    try t.expect(std.mem.indexOf(u8, kept, "\"code\": \"context_length_exceeded\"") != null);
    try t.expect(std.mem.indexOf(u8, kept, @import("errors.zig").context_limit) != null);
    // a refusal that names no status is still a refusal, and the event keeps none of its echoed fields
    const lane = (try json.parse(a, "{\"type\":\"error\",\"stream_id\":\"" ++ marker ++ "\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"" ++ marker ++ "\"}}")).ok;
    const shown = try output.event(a, lane, "req-1");
    try t.expect(std.mem.indexOf(u8, try json.stringify(a, shown, .{}), marker) == null);
    try t.expectEqualStrings(output.refused_message, shown.get("error").?.strField("message").?);
    // the caller's own value is left as it was
    try t.expect(std.mem.indexOf(u8, try json.stringify(a, failed, .{}), marker) != null);
}

fn chat(a: std.mem.Allocator, h: *fixture.Harness, body: []const u8) ![]const u8 {
    return h.ask(a, "POST", "/v1/chat/completions", "", "localhost", body);
}

test "confidential: HTTP error bodies and a failed stream's error event name the request id, never its words" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    const Bad = struct { method: []const u8, path: []const u8, body: []const u8, status: u16 };
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
    const bad = [_]Bad{
        .{ .method = "POST", .path = "/v1/chat/completions", .status = 400, .body = "{\"model\":\"m\"," ++ user ++ ",\"response_format\":{\"type\":\"" ++ marker ++ "\"}}" },
        .{ .method = "POST", .path = "/v1/messages", .status = 400, .body = "{\"model\":\"m\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"" ++ marker ++ "\"}]}]}" },
        .{ .method = "POST", .path = "/v1/responses", .status = 400, .body = "{\"model\":\"m\",\"input\":[{\"type\":\"" ++ marker ++ "\"}]}" },
        .{ .method = "GET", .path = "/v1/responses/" ++ marker, .status = 404, .body = "" },
    };
    // the control: each of these error replies quotes the marker without the flag
    for (bad) |b| {
        const reply = try h.ask(a, b.method, b.path, "", "localhost", b.body);
        try fixture.expectStatus(reply, b.status);
        if (std.mem.indexOf(u8, reply, marker) == null) {
            std.debug.print("this refusal quotes nothing, so it controls nothing: {s}\n", .{reply});
            return error.ControlQuotesNothing;
        }
    }
    engine.fail_message = "the tool call " ++ marker ++ " could not be parsed";
    const stream = "{\"model\":\"m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    try t.expect(std.mem.indexOf(u8, try chat(a, &h, stream), marker) != null);
    output.confine();
    defer output.release();
    for (bad) |b| {
        const reply = try h.ask(a, b.method, b.path, "", "localhost", b.body);
        try fixture.expectStatus(reply, b.status);
        try t.expect(std.mem.indexOf(u8, reply, marker) == null);
        const e = (try fixture.payload(a, reply)).get("error").?;
        try t.expectEqualStrings(output.refused_message, e.strField("message").?);
        try t.expect(std.mem.startsWith(u8, e.strField("request_id").?, "req-") or std.mem.startsWith(u8, e.strField("request_id").?, "chatcmpl-"));
    }
    const failed = try chat(a, &h, stream);
    try fixture.expectStatus(failed, 200);
    try t.expect(std.mem.indexOf(u8, failed, marker) == null);
    try t.expect(std.mem.indexOf(u8, failed, output.failed_message) != null);
}

test "confidential: the same request gets the same reply with the output rule on and off" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    const body = "{\"model\":\"m\",\"seed\":7,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    const off = try fixture.payload(a, try chat(a, &h, body));
    output.confine();
    defer output.release();
    const on = try fixture.payload(a, try chat(a, &h, body));
    const json = @import("json");
    const reply = struct {
        fn of(v: json.Value) []const u8 {
            return v.get("choices").?.items().?[0].get("message").?.strField("content").?;
        }
    };
    try t.expectEqualStrings(reply.of(off), reply.of(on));
    try t.expectEqualStrings(try json.stringify(a, off.get("usage").?, .{}), try json.stringify(a, on.get("usage").?, .{}));
}
