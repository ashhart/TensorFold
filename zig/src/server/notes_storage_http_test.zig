//! Socket-level Notes storage routing without search, lookup or live-fetch modules.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const routes = @import("routes.zig");
const Storage = @import("notes/storage.zig").Storage;
const testing = std.testing;
pub const Echo = struct {
    cache: bool = false,
    supports_forget: bool = false,
    forgotten: api.prompt_cache.Forgotten = .{},
    pub fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .forget_prompts = forget } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        return .{ .prompt_cache = e.cache };
    }
    fn forget(ctx: *anyopaque, _: []const u32) api.ForgetError!api.prompt_cache.Forgotten {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        if (!e.supports_forget) return error.Unsupported;
        return e.forgotten;
    }
    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'o'} });
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

const Reader = struct {
    fd: std.c.fd_t,
    buf: [65536]u8 = undefined,
    len: usize = 0,
    failed: bool = false,

    fn run(r: *Reader) void {
        while (r.len < r.buf.len) {
            const n = std.c.read(r.fd, r.buf[r.len..].ptr, r.buf.len - r.len);
            if (n == 0) return;
            if (n < 0) {
                r.failed = true;
                return;
            }
            r.len += @intCast(n);
        }
        r.failed = true;
        _ = std.c.shutdown(r.fd, 0);
    }
};

fn exchange(srv: *Server, a: std.mem.Allocator, method: []const u8, path: []const u8, body: []const u8) !struct { code: u16, value: json.Value } {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const wire = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: test\r\nContent-Length: {d}\r\n\r\n{s}", .{ method, path, body.len, body });
    var sent: usize = 0;
    while (sent < wire.len) {
        const n = std.c.write(pair[0], wire[sent..].ptr, wire.len - sent);
        if (n <= 0) return error.Write;
        sent += @intCast(n);
    }
    var conn = try Conn.init(testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    if (try conn.readRequest(a, false) != .ready) return error.Request;
    var reader: Reader = .{ .fd = pair[0] };
    const thread = try std.Thread.spawn(.{}, Reader.run, .{&reader});
    routes.dispatch(srv, &conn, a);
    _ = std.c.shutdown(pair[1], 1);
    thread.join();
    if (reader.failed) return error.Read;
    const response = reader.buf[0..reader.len];
    const sep = std.mem.indexOf(u8, response, "\r\n\r\n") orelse return error.Response;
    const code = try std.fmt.parseInt(u16, response[9..12], 10);
    const parsed = try json.parse(a, response[sep + 4 ..]);
    return .{ .code = code, .value = switch (parsed) {
        .ok => |v| v,
        .err => return error.Response,
    } };
}

test "storage HTTP erase names unsupported caches, active requests and configured learned roots" {
    const scenarios = [_]struct { engine: Echo, learn: bool, pending: []const u8 }{
        .{ .engine = .{ .cache = true }, .learn = false, .pending = "prompt_cache" },
        .{ .engine = .{ .cache = true, .supports_forget = true, .forgotten = .{ .in_flight = 1 } }, .learn = false, .pending = "requests" },
        .{ .engine = .{ .supports_forget = true, .forgotten = .{ .conversation_disk = true, .conversations = 2 } }, .learn = true, .pending = "learn_files" },
    };
    for (scenarios) |scenario| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
        var echo = scenario.engine;
        const srv = try Server.init(testing.allocator, testing.io, echo.engine(), bytes.text(), .{
            .served_name = "m",
            .model_ids = &.{"m"},
            .enable_thinking = false,
            .learn = scenario.learn,
        }, null);
        defer srv.deinit();
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
        const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
        var problem: []const u8 = "";
        srv.notes = try Storage.load(testing.allocator, @embedFile("notes/testdata/l6_bytes_snapshot.json"), bytes.text(), path, &problem);
        try srv.notes.?.beginRun(a, .{ .learn = scenario.learn });
        const added = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
        const id = added.value.get("id").?.int64().?;
        const target = try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id});
        const erased = try exchange(srv, a, "DELETE", target, "");
        try testing.expectEqual(@as(u16, 200), erased.code);
        try testing.expectEqual(id, erased.value.get("erased").?.int64().?);
        try testing.expect(!erased.value.get("complete").?.truthy());
        const status_value = erased.value.get("places").?.get(scenario.pending).?;
        try testing.expectEqualStrings("may_hold", status_value.str().?);
        if (scenario.learn) {
            try testing.expectEqualStrings("current_configured_root_only", erased.value.get("learned_folder_scope").?.str().?);
            try testing.expect(erased.value.get("conversation_states_removed").?.int64().? > 0);
        }
        try testing.expectEqual(@as(usize, 0), bytes.encodes_since_forget.load(.monotonic));
    }
}

test "storage HTTP: real Conn writes, lists, changes and takes back without an index" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    var echo: Echo = .{};
    const srv = try Server.init(testing.allocator, testing.io, echo.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    const off = try exchange(srv, a, "GET", "/v1/notes", "");
    try testing.expectEqual(@as(u16, 404), off.code);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    var problem: []const u8 = "";
    srv.notes = try Storage.load(testing.allocator, @embedFile("notes/testdata/l6_bytes_snapshot.json"), bytes.text(), path, &problem);
    try srv.notes.?.beginRun(a, .{});
    const add = try exchange(srv, a, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}");
    try testing.expectEqual(@as(u16, 201), add.code);
    const id = add.value.get("id").?.int64().?;
    const note_path = try std.fmt.allocPrint(a, "/v1/notes/{d}", .{id});
    const get = try exchange(srv, a, "GET", note_path, "");
    try testing.expectEqual(@as(u16, 200), get.code);
    try testing.expectEqualStrings("My favourite colour is green.", get.value.get("text").?.str().?);
    const change = try exchange(srv, a, "PUT", note_path, "{\"text\":\"My favourite colour is blue.\"}");
    try testing.expectEqual(@as(u16, 201), change.code);
    const newer = try std.fmt.allocPrint(a, "/v1/notes/{d}", .{change.value.get("id").?.int64().?});
    const take = try exchange(srv, a, "DELETE", newer, "");
    try testing.expectEqual(@as(u16, 201), take.code);
    const list = try exchange(srv, a, "GET", "/v1/notes?history=true", "");
    try testing.expectEqual(@as(u16, 200), list.code);
    try testing.expect(list.value.get("history").?.truthy());
    const place = srv.notes.?.model.placeOf(@intCast(id)).?;
    const source = try a.dupe(u8, srv.notes.?.model.notes.items[place].source().?);
    const erase_path = try std.fmt.allocPrint(a, "{s}?erase=true", .{note_path});
    const refused = try exchange(srv, a, "DELETE", erase_path, "{}");
    try testing.expectEqual(@as(u16, 400), refused.code);
    const before = try exchange(srv, a, "GET", note_path, "");
    try testing.expectEqual(@as(u16, 200), before.code);
    const erased = try exchange(srv, a, "DELETE", erase_path, "");
    try testing.expectEqual(@as(u16, 200), erased.code);
    try testing.expect(erased.value.get("erased").?.truthy());
    try testing.expect(erased.value.get("complete").?.truthy());
    try testing.expectEqual(id, erased.value.get("erased").?.int64().?);
    try testing.expect(erased.value.get("renumbered") != null);
    try testing.expect(bytes.forgot > 0);
    try testing.expectEqual(@as(usize, 0), bytes.encodes_since_forget.load(.monotonic));
    try testing.expect(srv.notes.?.model.placeOfSource(source) == null);
    srv.notes.?.deinit();
    srv.notes = null;
    srv.notes = try Storage.load(testing.allocator, @embedFile("notes/testdata/l6_bytes_snapshot.json"), bytes.text(), path, &problem);
    try srv.notes.?.beginRun(a, .{});
    try testing.expect(srv.notes.?.model.placeOfSource(source) == null);
}
