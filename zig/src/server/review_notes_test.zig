//! Adversarial erase regressions, including the independent review and crash recovery.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const routes = @import("routes.zig");
const Storage = @import("notes/storage.zig").Storage;
const erase = @import("notes/erase_storage.zig");
const places = @import("notes_erase_places.zig");
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const testing = std.testing;
const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");

/// An engine that answers every forget with `out` and records each span it was asked about.
const Recorder = struct {
    out: api.prompt_cache.Forgotten = .{},
    speech: bool = false,
    spans: std.ArrayList([]u32) = .empty,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .forget_prompts = forget } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const rec: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .prompt_cache = true, .speech = if (rec.speech) .{ .sample_rate = 24000, .frame_length = 1920, .max_frames = 1024, .max_prompt_tokens = 8192 } else null };
    }
    fn forget(ctx: *anyopaque, span: []const u32) api.ForgetError!api.prompt_cache.Forgotten {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.spans.append(testing.allocator, testing.allocator.dupe(u32, span) catch return error.Busy) catch return error.Busy;
        return e.out;
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn deinit(e: *@This()) void {
        for (e.spans.items) |s| testing.allocator.free(s);
        e.spans.deinit(testing.allocator);
    }
};

const Reader = struct {
    fd: platform.posix.socket_t,
    buf: [65536]u8 = undefined,
    len: usize = 0,
    fn run(r: *Reader) void {
        while (r.len < r.buf.len) {
            const n = platform.posix.system.read(r.fd, r.buf[r.len..].ptr, r.buf.len - r.len);
            if (n <= 0) return;
            r.len += @intCast(n);
        }
    }
};

fn exchange(srv: *Server, a: std.mem.Allocator, method: []const u8, path: []const u8, body: []const u8) !struct { code: u16, value: json.Value } {
    var pair: [2]platform.posix.socket_t = undefined;
    if (platform.windows) try platform.posix.socketpair(&pair) else if (platform.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = platform.posix.system.close(pair[0]);
    defer _ = platform.posix.system.close(pair[1]);
    const wire = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: test\r\nContent-Length: {d}\r\n\r\n{s}", .{ method, path, body.len, body });
    _ = platform.posix.system.write(pair[0], wire.ptr, wire.len);
    var conn = try Conn.init(testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    if (try conn.readRequest(a, false) != .ready) return error.Request;
    var reader: Reader = .{ .fd = pair[0] };
    const thread = try std.Thread.spawn(.{}, Reader.run, .{&reader});
    routes.dispatch(srv, &conn, a);
    _ = platform.posix.system.shutdown(pair[1], 1);
    thread.join();
    const response = reader.buf[0..reader.len];
    const sep = std.mem.indexOf(u8, response, "\r\n\r\n") orelse return error.Response;
    const code = try std.fmt.parseInt(u16, response[9..12], 10);
    const parsed = try json.parse(a, try a.dupe(u8, response[sep + 4 ..])); // own copy: the reader buffer dies with this frame
    return .{ .code = code, .value = switch (parsed) {
        .ok => |v| v,
        .err => return error.Response,
    } };
}

fn server(a: std.mem.Allocator, bytes: *Bytes, rec: *Recorder, dir: []const u8) !*Server {
    return serverLearn(a, bytes, rec, dir, false);
}

fn serverLearn(a: std.mem.Allocator, bytes: *Bytes, rec: *Recorder, dir: []const u8, learn: bool) !*Server {
    const srv = try Server.init(testing.allocator, testing.io, rec.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .learn = learn }, null);
    errdefer srv.deinit();
    var problem: []const u8 = "";
    srv.notes = try Storage.load(testing.allocator, snapshot, bytes.text(), try std.fs.path.join(a, &.{ dir, "notes.journal" }), &problem);
    try srv.notes.?.beginRun(a, .{ .learn = learn });
    return srv;
}

test "review: a failed removal is never reported complete" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{ .out = .{ .failed = 1 } };
    defer rec.deinit();
    const srv = try server(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a));
    defer srv.deinit();
    const added = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{added.value.get("id").?.int64().?});
    const first = try exchange(srv, a, "DELETE", path, "");
    try testing.expect(!first.value.get("complete").?.truthy());
}

test "review: an incomplete erase can be retried (the erased id must not 404)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{ .out = .{ .failed = 1, .conversation_disk = true } }; // a saved conversation's unlink failed
    defer rec.deinit();
    const srv = try serverLearn(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a), true);
    defer srv.deinit();
    const added = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{added.value.get("id").?.int64().?});
    const first = try exchange(srv, a, "DELETE", path, "");
    try testing.expect(!first.value.get("complete").?.truthy());
    const again = try exchange(srv, a, "DELETE", path, "");
    try testing.expectEqual(@as(u16, 200), again.code);
}

test "review: a short note's raw chat turn is searched in the engine's states" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    const srv = try server(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a));
    defer srv.deinit();
    const note = "I am Ana.";
    _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"I am Ana.\"}");
    const id = srv.notes.?.model.notes.items[srv.notes.?.model.notes.items.len - 1].id;
    const erased = try exchange(srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), "");
    try testing.expectEqual(@as(u16, 200), erased.code);
    // what a saved conversation holds: the user's own turn, rendered and encoded as a chat request is
    var problem: []const u8 = "";
    const turn = try bytes.text().renderIds(a, (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"I am Ana.\"}]")).ok, .{}, &problem);
    _ = note;
    var reached = false;
    for (rec.spans.items) |s| reached = reached or std.mem.indexOf(u32, turn, s) != null;
    try testing.expect(reached);
}

test "review: the raw span of one note does not erase an unrelated conversation" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: Bytes = .{};
    const span = (try places.rawSpanOf(a, bytes.text(), "My favourite colour is green.")).?;
    const other = try bytes.text().encode(a, "[user]My favourite colour is blue.\n[assistant]", false);
    try testing.expect(std.mem.indexOf(u32, other, span) == null);
}

test "review: erasing a take-back keeps (or reports) a later note with the taken-back words" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var problem: []const u8 = "";
    const path = try std.fs.path.join(a, &.{ try tmp.dir.realPathFileAlloc(testing.io, ".", a), "notes.journal" });
    const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer s.deinit();
    try s.beginRun(a, .{});
    _ = try s.apply(a, .{ .op = .add, .source = "w1", .told = 739900, .subject = "me", .attribute = "note:w1", .text = "My colour is green." });
    _ = try s.apply(a, .{ .op = .take_back, .source = "w2", .told = 739901, .subject = "me", .attribute = "note:w1", .target = .{ .source = "w1" }, .text = "I take that back." });
    _ = try s.apply(a, .{ .op = .add, .source = "w3", .told = 739902, .subject = "me", .attribute = "note:w3", .text = "My colour is green." });
    const w3 = s.model.notes.items[s.model.placeOfSource("w3").?].id;
    const takeback = s.model.notes.items[s.model.placeOfSource("w2").?].id;
    var p = try erase.prepare(s, a, takeback);
    defer p.deinit();
    try p.commit(a);
    var named = false;
    for (p.removed) |src| named = named or std.mem.eql(u8, src, "w3");
    for (p.moved) |m| named = named or m[0] == w3;

    try testing.expect(s.model.placeOfSource("w3") != null or named);
}

test "review: a retried erase (lost reply) does not erase a different note" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    const srv = try server(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a));
    defer srv.deinit();
    const first = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
    const second = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My locker code is under the plant.\"}");
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{first.value.get("id").?.int64().?});
    try testing.expectEqual(@as(u16, 200), (try exchange(srv, a, "DELETE", path, "")).code);
    const retry = try exchange(srv, a, "DELETE", path, ""); // the client's retry of the same request
    try testing.expect(retry.code != 200);
    const other_path = try std.fmt.allocPrint(a, "/v1/notes/{d}", .{second.value.get("id").?.int64().?});
    try testing.expectEqual(@as(u16, 200), (try exchange(srv, a, "GET", other_path, "")).code);
}

test "review: an erase query the server does not understand is refused, not taken back" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    const srv = try server(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a));
    defer srv.deinit();
    const added = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=yes", .{added.value.get("id").?.int64().?});
    const got = try exchange(srv, a, "DELETE", path, "");
    try testing.expect(got.code >= 400);
}

test "review: a second server on a snapshot without a journal yet is refused at startup" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var problem: []const u8 = "";
    const path = try std.fs.path.join(a, &.{ try tmp.dir.realPathFileAlloc(testing.io, ".", a), "notes.journal" });
    const one = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer one.deinit();
    const two = Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem) catch null;
    defer if (two) |t| t.deinit();
    try testing.expect(two == null);
}

test "review: pending cleanup survives restart, clears on retry and never recycles an id" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var rec: Recorder = .{ .out = .{ .failed = 1 } };
    defer rec.deinit();
    var id: u32 = 0;
    {
        const srv = try server(a, &bytes, &rec, dir);
        defer srv.deinit();
        const note = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"I am Ana.\"}");
        id = @intCast(note.value.get("id").?.int64().?);
        const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id});
        const first = try exchange(srv, a, "DELETE", path, "");
        try testing.expectEqual(@as(u16, 200), first.code);
        try testing.expect(!first.value.get("complete").?.truthy());
        try testing.expectEqual(@as(usize, 1), srv.notes.?.journal.?.erases.items.len);
        const journal_bytes = try tmp.dir.readFileAlloc(testing.io, "notes.journal", a, .unlimited);
        try testing.expect(std.mem.indexOf(u8, journal_bytes, "I am Ana.") == null);
        try testing.expect(std.mem.indexOf(u8, journal_bytes, "erase_intents") != null);
    }
    rec.out = .{};
    {
        const srv = try server(a, &bytes, &rec, dir);
        defer srv.deinit();
        try testing.expectEqual(@as(usize, 1), srv.notes.?.journal.?.erases.items.len);
        const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id});
        const retry = try exchange(srv, a, "DELETE", path, "");
        try testing.expectEqual(@as(u16, 200), retry.code);
        try testing.expect(retry.value.get("complete").?.truthy());
        try testing.expectEqual(@as(usize, 0), srv.notes.?.journal.?.erases.items.len);
        const other = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My dog is Otto.\"}");
        try testing.expect(other.value.get("id").?.int64().? > id);
        try testing.expectEqual(@as(u16, 404), (try exchange(srv, a, "DELETE", path, "")).code);
        try testing.expectEqual(@as(usize, 1), srv.notes.?.log.items.items.len);
    }
    const srv = try server(a, &bytes, &rec, dir);
    defer srv.deinit();
    try testing.expectEqual(@as(usize, 0), srv.notes.?.journal.?.erases.items.len);
    try testing.expect(srv.notes.?.model.placeOf(id) == null);
    try testing.expect(srv.notes.?.model.nextId() > id);
}

test "review: crash after journal replacement keeps the erase work and the original identity" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    var id: u32 = 0;
    {
        const srv = try server(a, &bytes, &rec, dir);
        defer srv.deinit();
        const note = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
        id = @intCast(note.value.get("id").?.int64().?);
        var p = try erase.prepare(srv.notes.?, a, id);
        defer p.deinit();
        try p.commit(a); // stop at the crash window: no engine forget or reply follows
        try testing.expectEqual(@as(usize, 0), rec.spans.items.len);
    }
    const srv = try server(a, &bytes, &rec, dir);
    defer srv.deinit();
    const retry = try exchange(srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), "");
    try testing.expectEqual(@as(u16, 200), retry.code);
    try testing.expect(retry.value.get("complete").?.truthy());
    try testing.expect(rec.spans.items.len > 0);
}

test "review: historical Recall records are named when today's writer is off" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    var id: u32 = 0;
    {
        const srv = try server(a, &bytes, &rec, dir);
        defer srv.deinit();
        // Add a historical start with Recall enabled before the first source was written.
        srv.notes.?.journal.?.addRun(a, .{ .recall = true });
        const note = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"I am Ana.\"}");
        id = @intCast(note.value.get("id").?.int64().?);
    }
    const srv = try server(a, &bytes, &rec, dir);
    defer srv.deinit();
    const got = try exchange(srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), "");
    try testing.expectEqualStrings("may_hold", got.value.get("places").?.strField("recall").?);
    try testing.expect(got.value.get("cleanup_complete").?.truthy());
    try testing.expectEqual(@as(usize, 0), srv.notes.?.journal.?.erases.items.len);
    try testing.expect(!got.value.get("complete").?.truthy());
}

test "review: upgrading a legacy journal preserves existing ids before an erase and restart" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    const j = @import("notes/journal.zig");
    const w = @import("notes/write.zig");
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(snapshot, &sha, .{});
    var wire: std.ArrayList(u8) = .empty;
    try wire.print(a, "{{\"format\":\"{s}\",\"snapshot_sha256\":\"{s}\",\"next_source\":4}}\n", .{ j.format_v2, std.fmt.bytesToHex(sha, .lower) });
    for ([_]w.Record{
        .{ .op = .add, .source = "w1", .told = 100, .subject = "me", .attribute = "note:w1", .text = "I am Ana." },
        .{ .op = .add, .source = "w2", .told = 101, .subject = "me", .attribute = "note:w2", .text = "I am Ana." },
        .{ .op = .add, .source = "w3", .told = 102, .subject = "me", .attribute = "note:w3", .text = "My dog is Otto." },
    }) |r| {
        try wire.appendSlice(a, try j.encode(a, r));
        try wire.appendSlice(a, try j.commitLine(a, r.source));
    }
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "notes.journal", .data = wire.items });
    var problem: []const u8 = "";
    {
        const srv = try server(a, &bytes, &rec, dir);
        defer srv.deinit();
        try testing.expectEqual(@as(u32, 30), srv.notes.?.model.notes.items[srv.notes.?.model.placeOfSource("w1").?].id);
        try testing.expectEqual(@as(u32, 31), srv.notes.?.model.notes.items[srv.notes.?.model.placeOfSource("w3").?].id);
        const erased = try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "");
        try testing.expectEqual(@as(u16, 200), erased.code);
    }
    const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer s.deinit();
    try testing.expectEqual(@as(u32, 31), s.model.notes.items[s.model.placeOfSource("w3").?].id);
    try testing.expect(s.model.placeOf(30) == null);
    const stored = try tmp.dir.readFileAlloc(testing.io, "notes.journal", a, .unlimited);
    try testing.expect(std.mem.indexOf(u8, stored, j.format) != null);
}

test "review: cleanup-record retirement failure stays pending and retries without harming another note" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{};
    defer rec.deinit();
    const srv = try server(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a));
    defer srv.deinit();
    _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"I am Ana.\"}");
    _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My dog is Otto.\"}");
    var p = try erase.prepare(srv.notes.?, a, 30);
    defer p.deinit();
    const Reply = @import("notes_erase_reply.zig").Reply;
    const reply = try Reply.prepare(a, 30, &p);
    try p.commit(a);
    var faults: @import("notes/journal.zig").Faults = .{ .sync = 0 };
    srv.notes.?.journal.?.faults = &faults;
    const body = (try json.parse(a, try reply.finish(srv, a, &p))).ok;
    try testing.expect(p.retire_failed);
    try testing.expect(!body.get("complete").?.truthy());
    try testing.expectEqualStrings("may_hold", body.get("places").?.strField("journal").?);
    try testing.expectEqual(@as(usize, 1), srv.notes.?.journal.?.erases.items.len);
    srv.notes.?.journal.?.faults = null;
    const retry = try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "");
    try testing.expectEqual(@as(u16, 200), retry.code);
    try testing.expect(retry.value.get("complete").?.truthy());
    const other = try exchange(srv, a, "GET", "/v1/notes/31", "");
    try testing.expectEqual(@as(u16, 200), other.code);
    try testing.expectEqualStrings("My dog is Otto.", other.value.strField("text").?);
}

test "review: speech engine state without a selective forget proof remains named" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: Bytes = .{};
    var rec: Recorder = .{ .speech = true };
    defer rec.deinit();
    const srv = try server(a, &bytes, &rec, try tmp.dir.realPathFileAlloc(testing.io, ".", a));
    defer srv.deinit();
    _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"I am Ana.\"}");
    const got = try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "");
    try testing.expect(!got.value.get("complete").?.truthy());
    try testing.expectEqualStrings("may_hold", got.value.get("places").?.strField("engine_records").?);
}

test "review: a legacy nonnumeric source keeps retry work with unknown provenance" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    var bytes: Bytes = .{};
    var why: []const u8 = "";
    {
        const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &why);
        defer s.deinit();
        try s.beginRun(a, .{});
        const note = try s.apply(a, .{ .op = .add, .source = "legacy-A", .subject = "me", .attribute = "note:A", .told = 100, .text = "I am Ana." });
        var p = try erase.prepare(s, a, note.note);
        defer p.deinit();
        try p.commit(a);
    }
    const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &why);
    defer s.deinit();
    try s.beginRun(a, .{});
    try testing.expectEqual(@as(u64, 0), s.journal.?.erases.items[0].source_num);
    var retry = try erase.prepare(s, a, 30);
    defer retry.deinit();
    try testing.expect(retry.retry);
    try testing.expectEqualStrings("legacy-A", retry.removed[0]);
}
