//! Whole replies and every streamed frame against a byte oracle recorded before the buffer changes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const scan = @import("stream_scan_text.zig");
const A = std.mem.Allocator;
const V = json.Value;

const Script = struct {
    ids: []const u32,
    stride: usize,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768, .reasoning = true };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var at: usize = 0;
        while (at < s.ids.len) : (at += s.stride) sink.event(sink.ctx, id, &.{ .tokens = s.ids[at..@min(at + s.stride, s.ids.len)] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Capture = struct {
    a: A,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ptr: *anyopaque, bytes: []const u8) error{Closed}!void {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        c.bytes.appendSlice(c.a, bytes) catch return error.Closed;
    }
};

const Oracle = struct {
    a: A,
    hash: std.crypto.hash.sha2.Sha256 = .init(.{}),
    ids: std.StringHashMapUnmanaged([]const u8) = .empty,
    cases: usize = 0,
    frames: usize = 0,

    // Only generated identifiers, wall clocks and performance counters vary independently of the reply.
    fn normalize(o: *Oracle, v: *V) !void {
        switch (v.*) {
            .object => |obj| for (obj.keys(), obj.values()) |k, *value| {
                if (std.mem.eql(u8, k, "tensorfold")) {
                    value.* = .null;
                } else if (std.mem.eql(u8, k, "created") or std.mem.eql(u8, k, "created_at") or std.mem.eql(u8, k, "completed_at")) {
                    value.* = .{ .int = "0" };
                } else try o.normalize(value);
            },
            .array => |list| for (list) |*value| try o.normalize(value),
            .string => |s| {
                for ([_][]const u8{ "chatcmpl-", "resp_", "msg_", "rs_", "fc_", "call_", "cmpl-" }) |prefix| {
                    if (!std.mem.startsWith(u8, s, prefix) or (s.len != prefix.len + 32 and s.len != prefix.len + 24)) continue;
                    for (s[prefix.len..]) |b| if (!std.ascii.isHex(b)) return;
                    const entry = try o.ids.getOrPut(o.a, s);
                    if (!entry.found_existing) entry.value_ptr.* = try std.fmt.allocPrint(o.a, "{s}{d}", .{ prefix, o.ids.count() });
                    v.* = .{ .string = entry.value_ptr.* };
                    return;
                }
            },
            else => {},
        }
    }
    fn frame(o: *Oracle, bytes: []const u8) !void {
        var v = (try json.parseText(o.a, bytes)).ok;
        try std.testing.expectEqualStrings(bytes, try json.stringify(o.a, v, .{}));
        try o.normalize(&v);
        const canonical = try json.stringify(o.a, v, .{});
        o.hash.update(canonical);
        o.hash.update("\n");
        o.frames += 1;
    }
    fn run(o: *Oracle, text: @import("model_text.zig").Text, tokens: []const u32, route: scan.Family, tools: V, stop: ?[]const u8) !void {
        for ([_]usize{ 1, 2, 7 }) |stride| for ([_]bool{ false, true }) |stream| {
            o.ids.clearRetainingCapacity();
            var script: Script = .{ .ids = tokens, .stride = stride };
            const srv = try Server.init(o.a, std.testing.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
            defer srv.deinit();
            const responses = route == .qwen;
            var body = (try json.parse(o.a, if (responses) "{\"model\":\"m\",\"input\":\"x\",\"store\":false,\"max_output_tokens\":4096}" else "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4096}")).ok;
            try body.object.put(o.a, "stream", .{ .bool = stream });
            try body.object.put(o.a, "tools", tools);
            if (stop) |s| try body.object.put(o.a, "stop", .{ .string = s });
            const request = try json.stringify(o.a, body, .{});
            var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(o.a, "{d}", .{request.len}) }};
            var capture: Capture = .{ .a = o.a };
            var conn: http.Conn = .{ .fd = -1, .peer = "", .buf = try o.a.dupe(u8, request), .end = request.len, .gpa = o.a, .version = "HTTP/1.1", .method = "POST", .path = if (responses) "/v1/responses" else "/v1/chat/completions", .headers = &headers, .write_hooks = .{ .ctx = &capture, .write = Capture.write } };
            if (responses) @import("responses.zig").post(srv, &conn, o.a) else @import("openai.zig").post(srv, &conn, o.a, true);
            const wire = capture.bytes.items;
            try std.testing.expect(std.mem.startsWith(u8, wire, "HTTP/1.1 200"));
            const payload = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoHeaders) + 4 ..];
            if (stream) {
                var lines = std.mem.splitScalar(u8, payload, '\n');
                while (lines.next()) |line| {
                    if (std.mem.startsWith(u8, line, "data: ")) {
                        o.hash.update("data: ");
                        if (std.mem.eql(u8, line[6..], "[DONE]")) o.hash.update("[DONE]\n") else try o.frame(line[6..]);
                    } else {
                        o.hash.update(line);
                        o.hash.update("\n");
                    }
                }
            } else try o.frame(payload);
            o.cases += 1;
        };
    }
};

test "Server recovery: wire bytes for Responses and Kimi match the original handlers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var oracle: Oracle = .{ .a = a };
    var plain: scan.Text = .{};
    const q = try scan.reply(a, .qwen, .{ .reasoning = 300, .content = 40, .value = 40 }, null);
    const qtools = (try json.parse(a, "[{\"type\":\"function\",\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"}}]")).ok;
    try oracle.run(plain.text(), q.ids, .qwen, qtools, null);
    const k = try @import("kimi_text_test.zig").synthetic(std.testing.allocator);
    defer k.deinit();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]")).ok;
    const fixtures = (try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok;
    for (fixtures.get("cases").?.array) |c| {
        const id = c.strField("id").?;
        if (!std.mem.eql(u8, id, "history-reasoning") and !std.mem.eql(u8, id, "tool-roundtrip") and !std.mem.eql(u8, id, "raw-json-tool-argument")) continue;
        const tokens = try @import("kimi_reply_test.zig").assistant(a, k.text(), c);
        for ([_]?[]const u8{ null, "carefully", "lookup", "nested" }) |stop| try oracle.run(k.text(), tokens, .kimi, tools, stop);
    }
    const literal = try k.text().encodePlain(a, " \u{3000}a\u{1f600}<|open|> literal\n");
    try oracle.run(k.text(), literal, .kimi, tools, null);
    const digest = oracle.hash.finalResult();
    try std.testing.expectEqualStrings("e5f3b0118e64c2d69dff00ff5fa62d907b28b65cac0a1d5c724615b11892942b", &std.fmt.bytesToHex(digest, .lower));
    try std.testing.expectEqual(@as(usize, 84), oracle.cases);
    try std.testing.expectEqual(@as(usize, 995), oracle.frames);
}
