//! Erase retirement, discoverability and regressions from the second independent review.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const routes = @import("routes.zig");
const Storage = @import("notes/storage.zig").Storage;
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const testing = std.testing;
const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");

const Clear = struct {
    out: api.prompt_cache.Forgotten = .{ .conversation_disk = true },
    speech: bool = false,
    unsupported: bool = false,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .forget_prompts = forget } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .prompt_cache = !e.speech, .speech = if (e.speech) .{ .sample_rate = 24000, .frame_length = 1920, .max_frames = 1024, .max_prompt_tokens = 8192 } else null };
    }
    fn forget(ctx: *anyopaque, _: []const u32) api.ForgetError!api.prompt_cache.Forgotten {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        if (e.unsupported) return error.Unsupported;
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
    const parsed = try json.parse(a, try a.dupe(u8, response[sep + 4 ..]));
    return .{ .code = code, .value = switch (parsed) {
        .ok => |v| v,
        .err => return error.Response,
    } };
}

fn open(a: std.mem.Allocator, bytes: *Bytes, e: *Clear, dir: []const u8, learn: bool) !*Server {
    const srv = try Server.init(testing.allocator, testing.io, e.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .learn = learn }, null);
    errdefer srv.deinit();
    var problem: []const u8 = "";
    srv.notes = try Storage.load(testing.allocator, snapshot, bytes.text(), try std.fs.path.join(a, &.{ dir, "notes.journal" }), &problem);
    try srv.notes.?.beginRun(a, .{ .learn = learn });
    return srv;
}

test "new: with conversation learning on, a fully cleared engine still retires the erase record" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var e: Clear = .{};
    const srv = try open(a, &bytes, &e, dir, true);
    defer srv.deinit();
    const added = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My locker code is 8174.\"}");
    const path = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{added.value.get("id").?.int64().?});
    for (0..3) |_| _ = try exchange(srv, a, "DELETE", path, "");
    const journal = try tmp.dir.readFileAlloc(testing.io, "notes.journal", a, .unlimited);
    // the code's digits as token ids ("8174" is 56,49,55,52 in the byte tokenizer)
    const leaked = std.mem.indexOf(u8, journal, "56, 49, 55, 52") != null or std.mem.indexOf(u8, journal, "56,49,55,52") != null;
    try testing.expect(srv.notes.?.journal.?.erases.items.len == 0 and !leaked);
}

test "new: erasing the newest note never lets its id return after restart" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var e: Clear = .{};
    var last: i64 = 0;
    {
        const srv = try open(a, &bytes, &e, dir, false);
        defer srv.deinit();
        _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"I live in Oslo.\"}");
        const b = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My dog is Otto.\"}");
        // a third write that learns nothing still takes an id
        _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"my dog is otto\"}");
        last = b.value.get("id").?.int64().?;
        const erased = try exchange(srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{last}), "");
        try testing.expect(erased.value.get("complete").?.truthy());
    }
    for (0..2) |_| {
        const srv = try open(a, &bytes, &e, dir, false);
        defer srv.deinit();
        const c = try exchange(srv, a, "POST", "/v1/notes", try std.fmt.allocPrint(a, "{{\"text\":\"New fact number {d}.\"}}", .{last}));
        try testing.expect(c.value.get("id").?.int64().? > last);
    }
}

test "new: permanent learned-root, speech and trace warnings do not retain recoverable spans" {
    for ([_]struct { learn: bool = false, engine: Clear }{
        .{ .learn = true, .engine = .{ .out = .{ .disk = true } } },
        .{ .engine = .{ .speech = true, .unsupported = true } },
        .{ .engine = .{ .out = .{ .unmanaged = 1 } } },
    }) |scenario| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
        var bytes: Bytes = .{};
        var e = scenario.engine;
        const srv = try open(a, &bytes, &e, dir, scenario.learn);
        defer srv.deinit();
        _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My locker code is 8174.\"}");
        const erased = try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "");
        try testing.expect(erased.value.get("cleanup_complete").?.truthy());
        try testing.expect(!erased.value.get("complete").?.truthy());
        try testing.expect(!erased.value.get("retryable").?.truthy());
        try testing.expectEqualStrings("unresolved", erased.value.strField("state").?);
        try testing.expectEqual(@as(usize, 0), srv.notes.?.journal.?.erases.items.len);
        const data = try tmp.dir.readFileAlloc(testing.io, "notes.journal", a, .unlimited);
        try testing.expect(std.mem.indexOf(u8, data, "\"spans\"") == null);
        try testing.expectEqual(@as(u16, 404), (try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "")).code);
    }
}

test "new: pending erases are listed after restart and disappear after engine cleanup" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    var bytes: Bytes = .{};
    var e: Clear = .{ .out = .{ .failed = 1 } };
    {
        const srv = try open(a, &bytes, &e, dir, false);
        defer srv.deinit();
        _ = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My locker code is 8174.\"}");
        const erased = try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "");
        try testing.expect(erased.value.get("retryable").?.truthy());
        try testing.expect(!erased.value.get("cleanup_complete").?.truthy());
    }
    const srv = try open(a, &bytes, &e, dir, false);
    defer srv.deinit();
    const list = try exchange(srv, a, "GET", "/v1/notes", "");
    const pending = list.value.get("pending_erases").?.items().?;
    try testing.expectEqual(@as(usize, 1), pending.len);
    try testing.expectEqual(@as(i64, 30), pending[0].int64().?);
    e.out = .{};
    const erased = try exchange(srv, a, "DELETE", "/v1/notes/30?erase=true", "");
    try testing.expect(erased.value.get("cleanup_complete").?.truthy());
    const after = try exchange(srv, a, "GET", "/v1/notes", "");
    try testing.expectEqual(@as(usize, 0), after.value.get("pending_erases").?.items().?.len);
}
