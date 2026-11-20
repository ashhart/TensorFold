//! Durable image jobs through production HTTP dispatch and image execution, without sockets or devices.
const std = @import("std");
const h = @import("image_test_harness.zig");
const json = @import("json");
const root = @import("root.zig");
const Bytes = @import("speech_stream_test.zig").Bytes;
const Conn = @import("http_conn.zig").Conn;
const Server = root.Server;
const Allocator = std.mem.Allocator;
const t = std.testing;
var bytes: Bytes = .{};

const Reply = struct {
    status: u16,
    body: []const u8,
    value: ?json.Value,
};

fn ask(a: Allocator, srv: *Server, stub: *h.Stub, method: []const u8, route_path: []const u8, body: []const u8, principal: ?[32]u8) !Reply {
    var wire: h.Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{
        .fd = -1,
        .peer = "",
        .buf = try a.dupe(u8, body),
        .end = body.len,
        .gpa = a,
        .method = method,
        .path = route_path,
        .version = "HTTP/1.1",
        .headers = &headers,
        .key_digest = principal,
        .write_hooks = .{ .ctx = &wire, .write = h.Wire.write },
        .gone_hook = .{ .ctx = stub, .call = h.Stub.gone },
    };
    defer conn.deinit();
    @import("routes.zig").dispatch(srv, &conn, a);
    const split = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.NoReply;
    const payload = wire.bytes.items[split + 4 ..];
    const parsed = json.parse(a, payload) catch return .{ .status = try std.fmt.parseInt(u16, wire.bytes.items[9..12], 10), .body = payload, .value = null };
    return .{
        .status = try std.fmt.parseInt(u16, wire.bytes.items[9..12], 10),
        .body = payload,
        .value = switch (parsed) {
            .ok => |value| value,
            .err => null,
        },
    };
}

fn makeServer(stub: *h.Stub, directory: ?[]const u8) !*Server {
    const srv = try Server.init(t.allocator, t.io, stub.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    errdefer srv.deinit();
    if (directory) |path_name| try srv.media_jobs.init(srv, path_name, .{ .running = 1 });
    return srv;
}

fn path(a: Allocator, id: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "/v1/images/jobs/{s}{s}", .{ id, suffix });
}

fn waitFor(a: Allocator, srv: *Server, stub: *h.Stub, id: []const u8, principal: ?[32]u8, wanted: []const u8) !Reply {
    for (0..2000) |_| {
        const reply = try ask(a, srv, stub, "GET", try path(a, id, ""), "", principal);
        if (reply.value) |value| if (value.get("status")) |status| if (status == .string and std.mem.eql(u8, status.string, wanted)) return reply;
        std.Thread.yield() catch {};
    }
    return error.MediaJobTimeout;
}

fn owner(comptime byte: u8) ?[32]u8 {
    return @splat(byte);
}

fn tempRoot(a: Allocator, tmp: *std.testing.TmpDir) ![]const u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", a);
}

test "media jobs: routed execution returns exact artifacts and fences every other owner's operation" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    defer srv.deinit();
    const body = "{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"fog\",\"size\":\"512x768\",\"steps\":8,\"guidance\":2,\"n\":2,\"seed\":42}";
    const submitted = try ask(a, srv, &stub, "POST", "/v1/images/jobs", body, owner(1));
    try t.expectEqual(@as(u16, 202), submitted.status);
    const id = submitted.value.?.get("id").?.string;
    const done = try waitFor(a, srv, &stub, id, owner(1), "succeeded");
    try t.expectEqual(@as(usize, 2), done.value.?.get("artifacts").?.array.len);
    try t.expectEqual(@as(usize, 2), stub.calls);
    try t.expectEqual(@as(usize, 2), stub.finished);

    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "GET", try path(a, id, ""), "", owner(2))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "GET", try path(a, id, "/content/0"), "", owner(2))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "POST", try path(a, id, "/cancel"), "", owner(2))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "DELETE", try path(a, id, ""), "", owner(2))).status);

    const content = try ask(a, srv, &stub, "GET", try path(a, id, "/content/0"), "", owner(1));
    try t.expectEqual(@as(u16, 200), content.status);
    try t.expect(content.value == null);
    try t.expectEqual(@as(usize, 40), content.body.len);
    try t.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", content.body[0..8]);
    try t.expectEqual(@as(u16, 204), (try ask(a, srv, &stub, "DELETE", try path(a, id, ""), "", owner(1))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "GET", try path(a, id, ""), "", owner(1))).status);
}

test "media jobs: cancellation reaches the held engine and settles before server teardown" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{ .hold_at = 1 };
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    var open = true;
    defer if (open) srv.deinit();
    const submitted = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hold\"}", owner(3));
    const id = submitted.value.?.get("id").?.string;
    _ = try waitFor(a, srv, &stub, id, owner(3), "running");
    _ = try ask(a, srv, &stub, "POST", try path(a, id, "/cancel"), "", owner(3));
    _ = try waitFor(a, srv, &stub, id, owner(3), "cancelled");
    srv.deinit();
    open = false;
    try t.expectEqual(@as(usize, 1), stub.calls);
    try t.expectEqual(@as(usize, 1), stub.cancels);
    try t.expectEqual(@as(usize, 1), stub.finished);
    try t.expect(stub.pending == null);
}

test "media jobs: invalid input persists no execution and an unconfigured server keeps the route closed" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    var open = true;
    defer if (open) srv.deinit();
    const invalid = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"\",\"n\":1}", owner(1));
    try t.expectEqual(@as(u16, 400), invalid.status);
    try t.expectEqual(@as(usize, 0), stub.calls);
    srv.deinit();
    open = false;

    var closed_stub: h.Stub = .{};
    const closed = try makeServer(&closed_stub, null);
    defer closed.deinit();
    const missing = try ask(a, closed, &closed_stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hi\"}", owner(1));
    try t.expectEqual(@as(u16, 404), missing.status);
    try t.expectEqual(@as(usize, 0), closed_stub.calls);
}

test "media jobs: scoped keys reach only their authorized job operations" {
    const scope = @import("auth_scope.zig");
    const inference: scope.Scopes = .{ .inference = true };
    const read: scope.Scopes = .{ .read = true };
    const id = "media-0123456789abcdef0123456789abcdef";
    const job = "/v1/images/jobs/" ++ id;
    try t.expect(scope.allows(
        inference,
        "POST",
        "/v1/images/jobs",
    ));
    try t.expect(scope.allows(inference, "POST", job ++ "/cancel"));
    try t.expect(scope.allows(inference, "GET", job));
    try t.expect(scope.allows(read, "GET", job ++ "/content/0"));
    try t.expect(!scope.allows(read, "POST", "/v1/images/jobs"));
    try t.expect(!scope.allows(inference, "DELETE", job));
    try t.expect(scope.allows(scope.full, "DELETE", job));
}

test "media jobs: serve CLI carries the opt-in directory and publishes its flag" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cli = @import("cli.zig");
    var usage: cli.Usage = .{};
    const args = try cli.parse(a, &.{ "m", "--media-jobs-dir", "jobs" }, &usage);
    try t.expectEqualStrings("jobs", args.media_jobs_dir.?);
    var output: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&output.writer, .{ .version = "test" });
    try t.expect(std.mem.indexOf(u8, output.written(), "\"--media-jobs-dir\"") != null);
}
