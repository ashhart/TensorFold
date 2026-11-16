//! `tensorfold mcp` on the wire: one message a line, JSON-RPC errors, the handshake and per-request revisions.
const std = @import("std");
const json = @import("json");
const platform = @import("sys");
const mcp = @import("mcp.zig");
const rpc = @import("mcp_rpc.zig");
const stop = @import("benchmark_stop.zig");
const fake = @import("benchmark_fake.zig");
const fixture = @import("mcp_fixture.zig");
const Value = json.Value;

const modern = "\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\",\"io.modelcontextprotocol/clientCapabilities\":{}}";
const initialize = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"example-client\",\"version\":\"1.0\"}}}\n" ++
    "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";

fn result(v: Value) Value {
    return v.get("result").?;
}

test "mcp wire: initialize answers the handshake revision asked for, else the newest it serves, and names the server" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const got = try fixture.exchange(a, &ctx,
        \\{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"example-client","version":"1.0"}}}
        \\{"jsonrpc":"2.0","method":"notifications/initialized"}
        \\{"jsonrpc":"2.0","id":"two","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}
        \\{"jsonrpc":"2.0","id":3,"method":"initialize","params":{"protocolVersion":"2026-07-28"}}
        \\{"jsonrpc":"2.0","id":4,"method":"initialize","params":{}}
        \\{"jsonrpc":"2.0","id":5,"method":"ping"}
        \\{"jsonrpc":"2.0","id":6,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}
        \\
    );
    try std.testing.expectEqual(@as(usize, 6), got.len); // the notification gets no answer
    const first = result(got[0]);
    try std.testing.expectEqualStrings("2025-06-18", first.strField("protocolVersion").?);
    try std.testing.expectEqualStrings("tensorfold", first.get("serverInfo").?.strField("name").?);
    try std.testing.expectEqualStrings(fixture.version, first.get("serverInfo").?.strField("version").?);
    try std.testing.expect(first.get("capabilities").?.get("tools").? == .object);
    try std.testing.expect(fixture.has(first.strField("instructions").?, "serve"));
    try std.testing.expectEqualStrings("two", got[1].strField("id").?);
    for (got[1..4]) |v| try std.testing.expectEqualStrings("2025-11-25", result(v).strField("protocolVersion").?);
    try std.testing.expectEqual(@as(?i64, 5), got[4].get("id").?.int64());
    try std.testing.expectEqual(@as(usize, 0), result(got[4]).object.count()); // ping's empty result
    try std.testing.expectEqualStrings("2025-11-25", result(got[5]).strField("protocolVersion").?);
    try std.testing.expect(result(got[0]).get("resultType") == null); // handshake revisions carry no resultType
}

test "mcp wire: bad JSON, batches, non-objects and bad ids get JSON-RPC errors; replies and blank lines get nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const got = try fixture.exchange(a, &ctx, "{bad json\n" ++
        \\[{"jsonrpc":"2.0","id":1,"method":"ping"}]
        \\[]
        \\42
        \\{"jsonrpc":"1.0","id":3,"method":"ping"}
        \\{"jsonrpc":"2.0","id":null,"method":"ping"}
        \\{"jsonrpc":"2.0","id":1.5,"method":"ping"}
        \\{"jsonrpc":"2.0","id":4,"method":"ping","params":[1]}
        \\{"jsonrpc":"2.0","id":5}
        \\
        \\
    ++ " \t \r\n" ++
        \\{"jsonrpc":"2.0","id":6,"result":{}}
        \\{"jsonrpc":"2.0","method":"notifications/progress","params":7}
        \\{"jsonrpc":"2.0","id":8,"method":"ping"}
        \\{"jsonrpc":"2.0","id":9,"method":7}
    );
    const want = [_]struct { id: ?i64, code: ?i64 }{
        .{ .id = null, .code = rpc.parse_error },
        .{ .id = null, .code = rpc.invalid_request },
        .{ .id = null, .code = rpc.invalid_request },
        .{ .id = null, .code = rpc.invalid_request },
        .{ .id = 3, .code = rpc.invalid_request },
        .{ .id = null, .code = rpc.invalid_request },
        .{ .id = null, .code = rpc.invalid_request },
        .{ .id = 4, .code = rpc.invalid_params },
        .{ .id = 5, .code = rpc.invalid_request },
        .{ .id = 8, .code = null },
        .{ .id = 9, .code = rpc.invalid_request },
    };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, v| {
        try std.testing.expectEqualStrings("2.0", v.strField("jsonrpc").?);
        try std.testing.expectEqual(w.id, if (v.get("id")) |id| id.int64() else return error.TestUnexpectedResult);
        try std.testing.expectEqual(w.code, fixture.code(v));
        if (w.code != null) try std.testing.expect(v.get("error").?.strField("message").?.len > 0);
    }
    try std.testing.expect(fixture.has(got[0].get("error").?.strField("message").?, "Parse error"));
    try std.testing.expect(fixture.has(got[1].get("error").?.strField("message").?, "batches are not supported"));
}

test "mcp wire: a line past 4 MiB is refused and the next one is still served" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const pad = try a.alloc(u8, rpc.max_line + 16);
    @memset(pad, 'x');
    const input = try std.mem.concat(a, u8, &.{ "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\",\"params\":{\"pad\":\"", pad, "\"}}\n", "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n" });
    const got = try fixture.exchange(a, &ctx, input);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqual(@as(?i64, rpc.invalid_request), fixture.code(got[0]));
    try std.testing.expect(got[0].get("id").? == .null);
    try std.testing.expectEqual(@as(?i64, 2), got[1].get("id").?.int64());
    try std.testing.expectEqual(@as(?i64, null), fixture.code(got[1]));
}

test "mcp wire: a stop signal while stdin stays open ends the session as its end would, with the signal's exit code" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    var fds: [2]std.posix.fd_t = undefined;
    if (platform.c.pipe(&fds) != 0) return error.SkipZigTest;
    defer for (fds) |fd| {
        _ = platform.c.close(fd);
    };
    const greeting = initialize ++ "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n";
    try std.testing.expectEqual(@as(isize, greeting.len), platform.c.write(fds[1], greeting, greeting.len));
    var stdin = (std.Io.File{ .handle = fds[0], .flags = .{ .nonblocking = false } }).readerStreaming(std.testing.io, try a.alloc(u8, rpc.max_line));
    var out: std.Io.Writer.Allocating = .init(a);
    stop.reset();
    defer stop.reset();
    const later = try std.Thread.spawn(.{}, struct {
        fn go() void {
            fake.nap(300);
            stop.request(.TERM);
        }
    }.go, .{});
    defer later.join();
    try std.testing.expectEqual(stop.exitCode(.TERM), try mcp.run(&ctx, &stdin.interface, &out.writer));
    const got = try fixture.lines(a, out.written());
    try std.testing.expectEqual(@as(usize, 2), got.len); // both lines answered, then the signal, with stdin still open
    try std.testing.expectEqual(@as(?i64, 2), got[1].get("id").?.int64());
}

test "mcp wire: 2026-07-28 requests name their revision in _meta; others before initialize are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const got = try fixture.exchange(a, &ctx, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"server/discover\",\"params\":{" ++ modern ++ "}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\",\"params\":{" ++ modern ++ "}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/list\",\"params\":{\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2099-01-01\",\"io.modelcontextprotocol/clientCapabilities\":{}}}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/list\",\"params\":{\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\"}}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"ping\",\"params\":{" ++ modern ++ "}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"resources/list\",\"params\":{" ++ modern ++ "}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"server/discover\"}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/call\",\"params\":{\"name\":\"status\",\"arguments\":{}," ++ modern ++ "}}\n");
    try std.testing.expectEqual(@as(usize, 9), got.len);
    try std.testing.expectEqual(@as(?i64, rpc.invalid_params), fixture.code(fixture.find(got, 1).?));
    const found = result(fixture.find(got, 2).?);
    try std.testing.expectEqualStrings("complete", found.strField("resultType").?);
    try std.testing.expectEqual(@as(usize, 3), found.get("supportedVersions").?.array.len);
    try std.testing.expectEqualStrings("2026-07-28", found.get("supportedVersions").?.array[0].string);
    try std.testing.expect(found.get("capabilities").?.get("tools").? == .object);
    try std.testing.expectEqualStrings("tensorfold", found.get("_meta").?.get(rpc.meta_server).?.strField("name").?);
    const listed = result(fixture.find(got, 3).?);
    try std.testing.expectEqualStrings("complete", listed.strField("resultType").?);
    try std.testing.expectEqual(@as(usize, 7), listed.get("tools").?.array.len);
    try std.testing.expectEqual(@as(?i64, 3_600_000), listed.get("ttlMs").?.int64());
    try std.testing.expectEqualStrings("public", listed.strField("cacheScope").?);
    const unsupported = fixture.find(got, 4).?;
    try std.testing.expectEqual(@as(?i64, rpc.unsupported_version), fixture.code(unsupported));
    const data = unsupported.get("error").?.get("data").?;
    try std.testing.expectEqualStrings("2099-01-01", data.strField("requested").?);
    try std.testing.expectEqual(@as(usize, 3), data.get("supported").?.array.len);
    try std.testing.expectEqual(@as(?i64, rpc.invalid_params), fixture.code(fixture.find(got, 5).?));
    for ([_]i64{ 6, 7 }) |id| try std.testing.expectEqual(@as(?i64, rpc.method_not_found), fixture.code(fixture.find(got, id).?));
    try std.testing.expectEqual(@as(?i64, rpc.invalid_params), fixture.code(fixture.find(got, 8).?)); // a dual-era probe falls back
    const called = result(fixture.find(got, 9).?);
    try std.testing.expectEqualStrings("complete", called.strField("resultType").?);
    try std.testing.expect(called.get("isError").?.bool == false);
}

test "mcp tools/list: seven tools in a fixed order, each an object schema with no extra properties and its hints" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const got = try fixture.exchange(a, &ctx, initialize ++ "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n");
    const listed = result(fixture.find(got, 2).?);
    try std.testing.expect(listed.get("resultType") == null and listed.get("ttlMs") == null);
    const tools = listed.get("tools").?.array;
    const names = [_][]const u8{ "machine", "models", "info", "pull", "serve", "stop", "status" };
    const required = [_]?[]const u8{ null, null, "model", "repo", "model", "id", null };
    const read_only = [_]bool{ true, true, true, false, false, false, true };
    try std.testing.expectEqual(names.len, tools.len);
    for (tools, names, required, read_only) |t, name, need, ro| {
        try std.testing.expectEqualStrings(name, t.strField("name").?);
        try std.testing.expect(t.strField("title").?.len > 0 and t.strField("description").?.len > 0);
        const schema = t.get("inputSchema").?;
        try std.testing.expectEqualStrings("object", schema.strField("type").?);
        try std.testing.expect(schema.get("properties").? == .object);
        try std.testing.expect(schema.get("additionalProperties").?.bool == false);
        if (need) |n| try std.testing.expectEqualStrings(n, schema.get("required").?.array[0].string) else try std.testing.expect(schema.get("required") == null);
        try std.testing.expectEqual(ro, t.get("annotations").?.get("readOnlyHint").?.bool);
    }
    const serve_props = tools[4].get("inputSchema").?.get("properties").?;
    try std.testing.expectEqualStrings("integer", serve_props.get("context").?.strField("type").?);
    try std.testing.expectEqual(@as(?i64, 1), serve_props.get("context").?.get("minimum").?.int64());
    try std.testing.expect(tools[5].get("annotations").?.get("destructiveHint").?.bool);
    try std.testing.expect(tools[3].get("annotations").?.get("openWorldHint").?.bool);
}

test "mcp tools/call: an unknown tool or a malformed call is a protocol error; bad arguments are tool errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const got = try fixture.exchange(a, &ctx, initialize ++
        \\{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nope"}}
        \\{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{}}
        \\{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"info","arguments":[1]}}
        \\{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"info"}}
        \\{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"status","arguments":{"verbose":true}}}
        \\{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"serve","arguments":{"model":"Org/Flash","context":0}}}
        \\{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"info","arguments":{"model":7}}}
        \\{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"status","arguments":{}}}
        \\{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"stop","arguments":{"id":"9"}}}
        \\{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"pull","arguments":{"repo":"Org/Flash","revision":null,"extra":1}}}
        \\
    );
    try std.testing.expectEqual(@as(usize, 11), got.len);
    for ([_]i64{ 2, 3, 4 }) |id| try std.testing.expectEqual(@as(?i64, rpc.invalid_params), fixture.code(fixture.find(got, id).?));
    try std.testing.expect(fixture.has(fixture.find(got, 2).?.get("error").?.strField("message").?, "Unknown tool: nope"));
    const refusals = [_]struct { id: i64, says: []const u8 }{
        .{ .id = 5, .says = "info needs model" },
        .{ .id = 6, .says = "status takes no argument named verbose; it takes no arguments" },
        .{ .id = 7, .says = "context must be a whole number from 1" },
        .{ .id = 8, .says = "model must be a non-empty string" },
        .{ .id = 10, .says = "started no server 9" },
        .{ .id = 11, .says = "pull takes no argument named extra; it takes repo, revision" },
    };
    for (refusals) |r| {
        const v = fixture.find(got, r.id).?;
        try std.testing.expect(fixture.isError(v));
        try std.testing.expect(fixture.has(fixture.text(v), r.says));
        try std.testing.expect(result(v).get("structuredContent") == null);
    }
    const status = fixture.find(got, 9).?;
    try std.testing.expect(!fixture.isError(status));
    try std.testing.expect(fixture.has(fixture.text(status), "started no servers"));
    try std.testing.expectEqual(@as(usize, 0), result(status).get("structuredContent").?.get("servers").?.array.len);
}
