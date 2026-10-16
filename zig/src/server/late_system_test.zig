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
            if (m.get("tool_calls")) |calls| {
                try out.appendSlice(a, try json.stringify(a, calls, .{ .compact = true }));
                try out.append(a, '\n');
            }
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
    background_prompt: []u32 = &.{},
    background_born: u32 = 0,
    reply: []const u8 = "hi",
    warm_turns: bool = false,
    telemetry: []const u8 = "", // the finished event's drafter counters

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        return .{ .warm_turns = e.warm_turns };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        if (request.background) {
            std.testing.allocator.free(e.background_prompt);
            e.background_prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
            e.background_born = request.born;
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
            return;
        }
        std.testing.allocator.free(e.prompt);
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply) |byte| {
            const token: u32 = byte;
            sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
        }
        var counters: [128]u8 = undefined; // as an engine's own buffer: valid only during the event
        @memcpy(counters[0..e.telemetry.len], e.telemetry);
        const stats: api.Stats = .{ .telemetry_json = counters[0..e.telemetry.len] };
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = stats } });
        @memset(&counters, 0xaa);
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
    return exchangeAs(srv, "POST", path, body);
}

fn exchangeAs(srv: *server.Server, method: []const u8, path: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const request_line = "{s} {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n";
    const head = try std.fmt.allocPrint(std.testing.allocator, request_line, .{ method, path, body.len });
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

test "/v1/imprint: without --learn-conversations each route refuses and names the flag; with it, the engine answers" {
    var text: DropLate = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
    }, null);
    defer srv.deinit();
    const go = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"go\"}]}";
    const off = "saved conversations are off: start the server with --learn-conversations";
    const routes_ = [_][2][]const u8{
        .{ "POST", "/v1/imprint/resume" },
        .{ "POST", "/v1/imprint/load" },
        .{ "DELETE", "/v1/imprint" },
    };
    for ([_]bool{ false, true }) |on| {
        srv.config.saved_conversations = on;
        for (routes_) |r| {
            const got = try exchangeAs(srv, r[0], r[1], if (r[0][0] == 'P') go else "");
            defer std.testing.allocator.free(got.body);
            const refused = std.mem.indexOf(u8, got.body, off) != null;
            // on: the route reaches the engine, and this one (no imprint hook) answers 501 as before
            try std.testing.expectEqual(@as(u16, if (on) 501 else 404), got.status);
            try std.testing.expectEqual(!on, refused);
        }
    }
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

const prose_call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}</tool_call>";
const prose_reply = "Before.\n" ++ prose_call ++ "\nAfter.";
const retained_prose = "Before.\n\nAfter.";
const chat_tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
const prose_chat = "{\"model\":\"m\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"go\"}]," ++ chat_tools;

// Route adapters return prose and one call; warming submits both to the engine.
test "tool reply prose survives whole chat, Messages and Responses and their warm prompts" {
    var text: DropLate = .{};
    var backend: Capture = .{ .reply = prose_reply, .warm_turns = true };
    defer std.testing.allocator.free(backend.prompt);
    defer std.testing.allocator.free(backend.background_prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const requests = [_]struct { path: []const u8, body: []const u8 }{
        .{ .path = "/v1/chat/completions", .body = prose_chat ++ "}" },
        .{ .path = "/v1/messages", .body = "{\"model\":\"m\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"go\"}],\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]}" },
        .{ .path = "/v1/responses", .body = "{\"model\":\"m\",\"max_output_tokens\":256,\"input\":\"go\",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]}" },
    };
    for (requests, 0..) |request, route| {
        const got = try exchange(srv, request.path, request.body);
        defer std.testing.allocator.free(got.body);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        const body = (try json.parse(a, got.body)).ok;
        var call_id: []const u8 = undefined;
        switch (route) {
            0 => {
                const choice = body.get("choices").?.array[0];
                try std.testing.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
                const message = choice.get("message").?;
                try std.testing.expectEqualStrings(retained_prose, message.get("content").?.string);
                const calls = message.get("tool_calls").?.array;
                try std.testing.expectEqual(@as(usize, 1), calls.len);
                call_id = calls[0].get("id").?.string;
                try std.testing.expectEqualStrings("{\"q\":\"a\"}", calls[0].get("function").?.get("arguments").?.string);
            },
            1 => {
                try std.testing.expectEqualStrings("tool_use", body.get("stop_reason").?.string);
                const blocks = body.get("content").?.array;
                try std.testing.expectEqual(@as(usize, 2), blocks.len);
                try std.testing.expectEqualStrings("text", blocks[0].get("type").?.string);
                try std.testing.expectEqualStrings(retained_prose, blocks[0].get("text").?.string);
                try std.testing.expectEqualStrings("tool_use", blocks[1].get("type").?.string);
                try std.testing.expectEqualStrings("a", blocks[1].get("input").?.get("q").?.string);
                call_id = blocks[1].get("id").?.string;
            },
            2 => {
                const items = body.get("output").?.array;
                try std.testing.expectEqual(@as(usize, 2), items.len);
                try std.testing.expectEqualStrings("message", items[0].get("type").?.string);
                try std.testing.expectEqualStrings(retained_prose, items[0].get("content").?.array[0].get("text").?.string);
                try std.testing.expectEqualStrings("function_call", items[1].get("type").?.string);
                try std.testing.expectEqualStrings("{\"q\":\"a\"}", items[1].get("arguments").?.string);
                call_id = items[1].get("call_id").?.string;
                const stored = (try srv.store.get(srv.io, a, body.get("id").?.string)).?;
                try std.testing.expectEqualStrings(retained_prose, stored.get("output").?.array[0].get("content").?.array[0].get("text").?.string);
            },
            else => unreachable,
        }
        const warmed = try promptText(backend.background_prompt);
        defer std.testing.allocator.free(warmed);
        try std.testing.expect(std.mem.indexOf(u8, warmed, "[assistant]" ++ retained_prose ++ "\n") != null);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, warmed, call_id));
        try std.testing.expect(std.mem.indexOf(u8, warmed, prose_call) == null);
        try std.testing.expectEqual(backend.prompt.len, backend.background_born); // its states count as the reply's request's
    }
}

test "legacy tool stream sends prose before and after exactly one call and warms the same history" {
    var text: DropLate = .{};
    var backend: Capture = .{ .reply = prose_reply, .warm_turns = true };
    defer std.testing.allocator.free(backend.prompt);
    defer std.testing.allocator.free(backend.background_prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const got = try exchange(srv, "/v1/chat/completions", prose_chat ++ ",\"stream\":true}");
    defer std.testing.allocator.free(got.body);
    try std.testing.expectEqual(@as(u16, 200), got.status);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prose: std.ArrayList(u8) = .empty;
    var arguments: std.ArrayList(u8) = .empty;
    var calls: usize = 0;
    var lines = std.mem.splitScalar(u8, got.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = std.mem.trimEnd(u8, line["data: ".len..], "\r");
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        const payload = (try json.parse(a, data)).ok;
        const choices = payload.get("choices").?.array;
        if (choices.len == 0) continue;
        const delta = choices[0].get("delta").?;
        if (delta.get("content")) |content| try prose.appendSlice(a, content.string);
        if (delta.get("tool_calls")) |tc| for (tc.array) |call| {
            if (call.get("id")) |_| calls += 1;
            if (call.get("function").?.get("arguments")) |args| try arguments.appendSlice(a, args.string);
        };
    }
    try std.testing.expectEqualStrings(retained_prose, prose.items);
    try std.testing.expectEqual(@as(usize, 1), calls);
    try std.testing.expectEqualStrings("{\"q\":\"a\"}", arguments.items);
    const warmed = try promptText(backend.background_prompt);
    defer std.testing.allocator.free(warmed);
    try std.testing.expect(std.mem.indexOf(u8, warmed, "[assistant]" ++ retained_prose ++ "\n") != null);
    try std.testing.expectEqual(backend.prompt.len, backend.background_born);
}

test "single-call whole replies retain prose and tool-only replies still have null content" {
    var text: DropLate = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ prose_reply, prose_call }, 0..) |reply, i| {
        backend.reply = reply;
        const got = try exchange(srv, "/v1/chat/completions", prose_chat ++ ",\"parallel_tool_calls\":false}");
        defer std.testing.allocator.free(got.body);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        const message = (try json.parse(a, got.body)).ok.get("choices").?.array[0].get("message").?;
        if (i == 0) {
            try std.testing.expectEqualStrings(retained_prose, message.get("content").?.string);
        } else {
            try std.testing.expect(message.get("content").? == .null);
        }
        try std.testing.expectEqual(@as(usize, 1), message.get("tool_calls").?.array.len);
    }
}

test "a reply's drafter counters are written from the request's own copy, after the engine's and mailbox's are gone" {
    var text: DropLate = .{};
    var backend: Capture = .{ .telemetry = "{\"recall\": {\"lookups\": 3, \"accepted\": 2}}" };
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = true,
    }, null);
    defer srv.deinit();
    const go = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"go\"}]}";
    const got = try exchange(srv, "/v1/chat/completions", go);
    defer std.testing.allocator.free(got.body);
    try std.testing.expectEqual(@as(u16, 200), got.status);
    try std.testing.expect(std.unicode.utf8ValidateSlice(got.body));
    const proposer = "\"proposer\": {\"recall\": {\"lookups\": 3, \"accepted\": 2}}";
    try std.testing.expect(std.mem.indexOf(u8, got.body, proposer) != null);
}

test "GET and DELETE /v1/recall: 404 without a record, its state, emptied, and 409 for a reader" {
    var text: DropLate = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
    }, null);
    defer srv.deinit();
    const Store = api.recall_store.Store;
    const off = try exchangeAs(srv, "GET", "/v1/recall", "");
    defer std.testing.allocator.free(off.body);
    try std.testing.expectEqual(@as(u16, 404), off.status);
    try std.testing.expect(std.mem.indexOf(u8, off.body, "Recall is off: serve with --recall") != null);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}/recall", .{tmp.sub_path});
    var problem: []const u8 = "";
    const id: api.recall_store.Identity = .{ .digest = @splat(5), .vocab = 1000, .name = "m" };
    const opts: api.recall_store.Options = .{ .dir = dir, .cap_bytes = 64 << 20 };
    const w = (try Store.open(std.testing.allocator, std.testing.io, id, opts, &problem, arena.allocator())).?;
    defer w.close();
    var prompt: [200]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 13) % 1000);
    w.readPrompt(&prompt, 0);
    srv.recall = w;
    const on = try exchangeAs(srv, "GET", "/v1/recall", "");
    defer std.testing.allocator.free(on.body);
    try std.testing.expectEqual(@as(u16, 200), on.status);
    try std.testing.expect(std.mem.indexOf(u8, on.body, "\"state\": \"on\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, on.body, "\"ids\": 201") != null);
    const emptied = try exchangeAs(srv, "DELETE", "/v1/recall", "");
    defer std.testing.allocator.free(emptied.body);
    try std.testing.expectEqual(@as(u16, 200), emptied.status);
    try std.testing.expectEqualStrings("{\"emptied\": true}", emptied.body);
    try std.testing.expectEqual(@as(u32, 0), w.shared.record.held());
    var ro_opts = opts;
    ro_opts.read_only = true;
    const r = (try Store.open(std.testing.allocator, std.testing.io, id, ro_opts, &problem, arena.allocator())).?;
    defer r.close();
    srv.recall = r;
    const refused = try exchangeAs(srv, "DELETE", "/v1/recall", "");
    defer std.testing.allocator.free(refused.body);
    try std.testing.expectEqual(@as(u16, 409), refused.status);
    try std.testing.expect(std.mem.indexOf(u8, refused.body, "reads the record only") != null);
}
