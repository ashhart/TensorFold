//! GET /v1/models/{id} through the dispatcher: the listing's entry by id or alias, escaped or raw, else a 404.
const std = @import("std");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const routes = @import("routes.zig");
const discovery = @import("models_input_types_test.zig");
const t = std.testing;

const Reply = struct { status: []const u8, body: json.Value };

/// A GET of `path` as the dispatcher answers it: the status line and the JSON body.
fn get(a: std.mem.Allocator, srv: *Server, path: []const u8) !Reply {
    var wire: discovery.Wire = .{ .out = .init(a) };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "GET", .path = path, .write_hooks = .{ .ctx = &wire, .write = discovery.Wire.write } };
    routes.dispatch(srv, &conn, a);
    const bytes = wire.out.written();
    const split = std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.NoReply;
    return .{ .status = bytes[0..std.mem.indexOf(u8, bytes, "\r\n").?], .body = (try json.parse(a, bytes[split + 4 ..])).ok };
}

/// An entry's JSON with its created time zeroed: two reads may fall in different seconds.
fn timeless(a: std.mem.Allocator, entry: json.Value) ![]const u8 {
    try entry.object.put(a, "created", .{ .int = "0" });
    return json.stringify(a, entry, .{});
}

fn served() Server {
    var srv = discovery.server(null, true);
    srv.config.model_ids = &.{ "model", "alias", "example-org/model", "50%off" };
    return srv;
}

test "model discovery by id: the served name and every alias return the listing's own entry" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv = served();
    const listed = try discovery.listed(a, &srv);
    for (srv.config.model_ids, listed) |id, entry| {
        for ([_][]const u8{ "/v1/models/", "/models/" }) |head| {
            const got = try get(a, &srv, try std.mem.concat(a, u8, &.{ head, id }));
            try t.expectEqualStrings("HTTP/1.1 200 OK", got.status);
            try t.expectEqualStrings(try timeless(a, entry), try timeless(a, got.body));
        }
    }
}

test "model discovery by id: an org/name id is found raw, escaped with %2F or %2f, and past a query or slash" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv = served();
    const paths = [_][]const u8{
        "/v1/models/example-org/model",   "/v1/models/example-org%2Fmodel",    "/models/example-org%2fmodel",
        "/v1/models/%65xample-org/model", "/v1/models/example-org/model/?x=1",
    };
    for (paths) |path| {
        const got = try get(a, &srv, path);
        try t.expectEqualStrings("HTTP/1.1 200 OK", got.status);
        try t.expectEqualStrings("example-org/model", got.body.strField("id").?);
        try t.expectEqualStrings("model", got.body.strField("object").?);
    }
    // a % that escapes no hex pair leaves the id as written; %25 is the escaped percent sign
    for ([_][]const u8{ "/v1/models/50%off", "/v1/models/50%25off" }) |path|
        try t.expectEqualStrings("50%off", (try get(a, &srv, path)).body.strField("id").?);
}

test "model discovery by id: any other id is a 404 model_not_found naming the decoded id" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv = served();
    const cases = [_][2][]const u8{
        .{ "/v1/models/missing", "model 'missing' is not served here" },
        .{ "/v1/models/example-org%2Fmissing", "model 'example-org/missing' is not served here" },
        .{ "/models/MODEL", "model 'MODEL' is not served here" },
        .{ "/v1/models/example-org", "model 'example-org' is not served here" },
    };
    for (cases) |c| {
        const got = try get(a, &srv, c[0]);
        try t.expectEqualStrings("HTTP/1.1 404 Not Found", got.status);
        const err = got.body.get("error").?;
        try t.expectEqualStrings(c[1], err.strField("message").?);
        try t.expectEqualStrings("invalid_request_error", err.strField("type").?);
        try t.expectEqualStrings("model", err.strField("param").?);
        try t.expectEqualStrings("model_not_found", err.strField("code").?);
    }
}

test "model discovery by id: a path ending in /models still lists every model" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv = served();
    for ([_][]const u8{ "/v1/models", "/models/", "/v1/models/model/models", "/models/models" }) |path| {
        const got = try get(a, &srv, path);
        try t.expectEqualStrings("HTTP/1.1 200 OK", got.status);
        try t.expectEqualStrings("list", got.body.strField("object").?);
        try t.expectEqual(srv.config.model_ids.len, got.body.get("data").?.array.len);
    }
}
