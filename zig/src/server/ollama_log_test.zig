//! Ollama HTTP request-log records: one line a request, and each reply paired with its request by id.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const auth = @import("auth.zig");
const e2e = @import("ollama_e2e_test.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

/// The request log's records, split by the type each line names.
const Log = struct { requests: []json.Value, replies: []json.Value };

fn logRecords(a: Allocator, path: []const u8) !Log {
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 24));
    var requests: std.ArrayList(json.Value) = .empty;
    var replies: std.ArrayList(json.Value) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const parsed = json.parse(a, line) catch return error.BadJson;
        if (parsed != .ok) return error.BadJson;
        const kind = parsed.ok.strField("type") orelse return error.BadJson;
        if (std.mem.eql(u8, kind, "request")) try requests.append(a, parsed.ok) else if (std.mem.eql(u8, kind, "reply")) try replies.append(a, parsed.ok) else return error.BadJson;
    }
    return .{ .requests = requests.items, .replies = replies.items };
}

/// The user prompt text a request record's redacted body holds.
fn promptOf(record: json.Value) ?[]const u8 {
    const body = record.get("body") orelse return null;
    const messages = body.get("messages") orelse return null;
    if (messages != .array) return null;
    for (messages.array) |m| if (m.strField("content")) |text| return text;
    return null;
}

/// One client's request on its own thread: its arena, socket and body belong to that thread alone.
const Client = struct {
    h: *e2e.Harness,
    key: []const u8,
    body: []const u8,
    want: []const u8,
    ok: bool = false,
    fn run(c: *Client) void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const reply = c.h.ask(arena.allocator(), "/api/chat", c.key, c.body) catch return;
        c.ok = e2e.statusIs(reply, 200) and std.mem.indexOf(u8, e2e.bodyOf(reply), c.want) != null;
    }
};

test "ollama HTTP: one request writes one request-log line" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the log stays off on Windows
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    var engine: e2e.Engine = .{};
    var h: e2e.Harness = .{ .log_path = path };
    try h.start(&engine, null);
    defer h.deinit();
    try t.expect(e2e.statusIs(try h.ask(a, "/api/chat", null, e2e.chat_body), 200));
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 16));
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| if (line.len > 0) {
        count += 1;
    };
    try t.expectEqual(@as(usize, 1), count);
}

test "ollama HTTP: replies pair with their request by id" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the log stays off on Windows
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/paired.jsonl", .{tmp.sub_path});
    var engine: e2e.Engine = .{};
    var h: e2e.Harness = .{ .log_path = path, .replies = true };
    try h.start(&engine, null);
    defer h.deinit();
    try t.expect(e2e.statusIs(try h.ask(a, "/api/chat", null, e2e.chat_body), 200));
    const log = try logRecords(a, path);
    try t.expectEqual(@as(usize, 1), log.requests.len);
    try t.expectEqual(@as(usize, 1), log.replies.len);
    try t.expectEqualStrings(log.requests[0].strField("id").?, log.replies[0].strField("id").?);
}

test "request log replies: a request refused in plan still pairs with its reply" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the log stays off on Windows
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/refused.jsonl", .{tmp.sub_path});
    var engine: e2e.Engine = .{};
    var h: e2e.Harness = .{ .log_path = path, .replies = true };
    try h.start(&engine, null);
    defer h.deinit();
    // choiceCount refuses n = 0: the request line is written before the checks, so the 400 reply pairs with it
    const refused = try h.ask(a, "/v1/chat/completions", null, "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"n\":0}");
    try t.expect(e2e.statusIs(refused, 400));
    const log = try logRecords(a, path);
    try t.expectEqual(@as(usize, 1), log.requests.len);
    try t.expectEqual(@as(usize, 1), log.replies.len);
    try t.expectEqualStrings(log.requests[0].strField("id").?, log.replies[0].strField("id").?);
}

test "ollama HTTP: two clients' replies pair with their own request id" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the log stays off on Windows
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/isolated.jsonl", .{tmp.sub_path});
    var keys_file = try e2e.KeyFile.init("alice[inference]: alice-secret\nbob[inference]: bob-secret\n");
    defer keys_file.deinit();
    var keys: auth.Store = undefined;
    var problem: []const u8 = "";
    keys = try auth.Store.init(t.allocator, &.{}, "", keys_file.path, false, &problem);
    defer keys.deinit();
    var engine: e2e.Engine = .{};
    var h: e2e.Harness = .{ .log_path = path, .replies = true };
    try h.start(&engine, &keys);
    defer h.deinit();
    const alice = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"alice asks\"}],\"stream\":false}";
    const bob = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"bob asks\"}],\"stream\":false}";
    var clients = [_]Client{
        .{ .h = &h, .key = "alice-secret", .body = alice, .want = "alice" },
        .{ .h = &h, .key = "bob-secret", .body = bob, .want = "bob" },
    };
    var threads: [clients.len]std.Thread = undefined;
    for (&clients, 0..) |*c, i| threads[i] = try std.Thread.spawn(.{}, Client.run, .{c});
    for (threads) |th| th.join();
    try t.expect(clients[0].ok and clients[1].ok);
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 24));
    try t.expect(std.mem.indexOf(u8, text, "alice-secret") == null);
    try t.expect(std.mem.indexOf(u8, text, "bob-secret") == null);
    const log = try logRecords(a, path);
    try t.expectEqual(@as(usize, 2), log.requests.len);
    try t.expectEqual(@as(usize, 2), log.replies.len);
    var alice_id: ?[]const u8 = null;
    var bob_id: ?[]const u8 = null;
    for (log.requests) |record| {
        const prompt = promptOf(record) orelse return error.MissingPrompt;
        if (std.mem.indexOf(u8, prompt, "alice") != null) {
            alice_id = record.strField("id");
        } else if (std.mem.indexOf(u8, prompt, "bob") != null) {
            bob_id = record.strField("id");
        } else return error.ForeignRequest;
    }
    try t.expect(alice_id != null and bob_id != null);
    try t.expect(!std.mem.eql(u8, alice_id.?, bob_id.?));
    var seen_alice = false;
    var seen_bob = false;
    for (log.replies) |record| {
        const id = record.strField("id").?;
        const reply = record.strField("text") orelse return error.MissingText;
        if (std.mem.eql(u8, id, alice_id.?)) {
            try t.expectEqualStrings("alice", reply);
            seen_alice = true;
        } else if (std.mem.eql(u8, id, bob_id.?)) {
            try t.expectEqualStrings("bob", reply);
            seen_bob = true;
        } else return error.ForeignReply;
    }
    try t.expect(seen_alice and seen_bob);
}
