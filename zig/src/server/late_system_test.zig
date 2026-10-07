//! A later system message is re-roled to user on Messages when the template cannot keep it, matching chat.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

const openai_body =
    \\{"model":"m","max_tokens":4,"messages":[{"role":"system","content":"lead"},{"role":"user","content":"hi"},{"role":"system","content":"be brief"},{"role":"assistant","content":"ok"},{"role":"user","content":"go"}]}
;
const messages_body =
    \\{"model":"m","max_tokens":4,"system":"lead","messages":[{"role":"user","content":"hi"},{"role":"system","content":"be brief"},{"role":"assistant","content":"ok"},{"role":"user","content":"go"}]}
;
const count_body =
    \\{"model":"m","system":"lead","messages":[{"role":"user","content":"hi"},{"role":"system","content":"be brief"},{"role":"assistant","content":"ok"},{"role":"user","content":"go"}]}
;
const misplaced_body =
    \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"},{"role":"system","content":"be brief"}]}
;

/// Drops a system message after the first, so the late-system probe comes back as user.
const DropLate = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        if (messages != .array) return out.items;
        var first = true;
        for (messages.array) |m| {
            const role = textOf(m, "role");
            const drop = !first and std.mem.eql(u8, role, "system");
            first = false;
            if (drop) continue;
            try out.print(a, "[{s}]{s}\n", .{ role, textOf(m, "content") });
        }
        return out.items;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn textOf(v: json.Value, key: []const u8) []const u8 {
    const x = v.get(key) orelse return "";
    return if (x == .string) x.string else "";
}

/// Copies the prompt the server submits, then finishes at once.
const Capture = struct {
    prompt: []u32 = &.{},

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.testing.allocator.free(e.prompt);
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
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

const Reply = struct { status: u16, body: []u8 };

fn exchange(srv: *server.Server, path: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(std.testing.allocator, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ path, body.len });
    defer std.testing.allocator.free(head);
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, a);
    const raw = try readReady(std.testing.allocator, pair[0]);
    errdefer std.testing.allocator.free(raw);
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const status = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    const payload = try std.testing.allocator.dupe(u8, raw[split + 4 ..]);
    std.testing.allocator.free(raw);
    return .{ .status = status, .body = payload };
}

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn readReady(a: std.mem.Allocator, fd: std.c.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&pollfd, 200) == 0) break;
        const n = std.posix.read(fd, &tmp) catch |e| return e;
        if (n == 0) break;
        try out.appendSlice(a, tmp[0..n]);
    }
    return out.toOwnedSlice(a);
}

fn promptText(ids: []const u32) ![]u8 {
    const text = try std.testing.allocator.alloc(u8, ids.len);
    for (ids, text) |id, *byte| byte.* = @intCast(id);
    return text;
}

test "a mid-conversation system message renders the same prompt on Messages and on chat" {
    var text: DropLate = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    try std.testing.expectEqualStrings("user", srv.late_system);

    const chat = try exchange(srv, "/v1/chat/completions", openai_body);
    defer std.testing.allocator.free(chat.body);
    try std.testing.expectEqual(@as(u16, 200), chat.status);
    const chat_ids = try std.testing.allocator.dupe(u32, backend.prompt);
    defer std.testing.allocator.free(chat_ids);

    const messages = try exchange(srv, "/v1/messages", messages_body);
    defer std.testing.allocator.free(messages.body);
    try std.testing.expectEqual(@as(u16, 200), messages.status);
    try std.testing.expectEqualSlices(u32, chat_ids, backend.prompt);

    const rendered = try promptText(backend.prompt);
    defer std.testing.allocator.free(rendered);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "[user]be brief\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "[system]be brief") == null);

    const counted = try exchange(srv, "/v1/messages/count_tokens", count_body);
    defer std.testing.allocator.free(counted.body);
    try std.testing.expectEqual(@as(u16, 200), counted.status);
    const want = try std.fmt.allocPrint(std.testing.allocator, "\"input_tokens\": {d}", .{backend.prompt.len});
    defer std.testing.allocator.free(want);
    try std.testing.expect(std.mem.indexOf(u8, counted.body, want) != null);

    const misplaced = try exchange(srv, "/v1/messages", misplaced_body);
    defer std.testing.allocator.free(misplaced.body);
    try std.testing.expectEqual(@as(u16, 400), misplaced.status);
    try std.testing.expect(std.mem.indexOf(u8, misplaced.body, "mid-conversation system messages must follow a user turn and precede an assistant or end") != null);
}
