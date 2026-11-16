//! The serve, stop and status tools with a stand-in serve child, and every server stopped at the session's end.
const std = @import("std");
const json = @import("json");
const fixture = @import("mcp_fixture.zig");
const fake = @import("benchmark_fake.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const posix = std.posix;

const initialize = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{}}}\n";

fn call(a: Allocator, id: i64, tool: []const u8, args: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"jsonrpc\":\"2.0\",\"id\":{d},\"method\":\"tools/call\",\"params\":{{\"name\":\"{s}\",\"arguments\":{s}}}}}\n", .{ id, tool, args });
}

fn structured(v: Value) Value {
    return v.get("result").?.get("structuredContent").?;
}

const How = enum { serves, serves_then_dies, exits, hangs };

/// A stand-in for this executable: it records its pid, arguments and key, then serves, exits or hangs as told.
const World = struct {
    home: fixture.Home,
    f: *fake.Fake,
    pidfile: []const u8,
    argsfile: []const u8,
    keyfile: []const u8,
    ctx: @import("mcp.zig").Context,

    fn init(w: *World, a: Allocator, how: How) !void {
        w.home = try fixture.Home.init(a);
        w.pidfile = try w.home.path(a, "pid");
        w.argsfile = try w.home.path(a, "args");
        w.keyfile = try w.home.path(a, "key");
        const routes = try a.dupe(fake.Route, &.{
            .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"status\": \"ok\"}")}) },
            .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"data\": [{\"id\": \"Lightning\"}]}")}) },
        });
        w.f = try fake.Fake.startKeyed(std.testing.allocator, routes, w.keyfile); // wants the key the child was given
        const serving = try std.fmt.allocPrint(a, "printf '[tensorfold] serving %s at http://127.0.0.1:{d}/v1 (x)\\n' \"$name\"\n", .{w.f.port});
        const then = switch (how) {
            .serves => try std.mem.concat(a, u8, &.{ serving, "exec sleep 30\n" }),
            .serves_then_dies => try std.mem.concat(a, u8, &.{ serving, "sleep 2\nprintf 'error: the GPU ran out of memory\\n'\nexit 1\n" }),
            .exits => "printf 'loading\\nerror: cannot read tokenizer.json\\n'\nexit 3\n",
            .hangs => "exec sleep 30\n",
        };
        const script = try std.fmt.allocPrint(a, "#!/bin/sh\necho $$ > '{s}'\nprintf '%s\\n' \"$@\" > '{s}'\nprintf '%s' \"$TENSORFOLD_API_KEY\" > '{s}'\n" ++
            "while [ $# -gt 0 ]; do if [ \"$1\" = --name ]; then name=$2; fi; shift; done\n{s}", .{ w.pidfile, w.argsfile, w.keyfile, then });
        const exe = try w.home.path(a, "tensorfold-stand-in");
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = exe, .data = script, .flags = .{ .permissions = .fromMode(0o755) } });
        _ = try w.home.checkpoint(a, "Org/Lightning", fixture.lightning);
        w.ctx = w.home.context();
        w.ctx.exe = exe;
    }

    fn deinit(w: *World) void {
        w.f.stop();
        w.home.deinit();
    }

    /// The stand-in's pid, once it has written it.
    fn pid(w: *const World, a: Allocator) !posix.pid_t {
        try fixture.appears(w.pidfile);
        var waited: u32 = 0;
        while (waited < 10_000) : (waited += 20) {
            const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, w.pidfile, a, .limited(64));
            if (std.mem.indexOfScalar(u8, text, '\n') != null) return std.fmt.parseInt(posix.pid_t, std.mem.trim(u8, text, " \n"), 10);
            std.testing.io.sleep(.fromMilliseconds(20), .awake) catch {};
        }
        return error.TestTimedOut;
    }
};

extern "c" fn getpgid(pid: posix.pid_t) posix.pid_t;

fn gone(pid: posix.pid_t) bool {
    posix.kill(pid, @fromBackingInt(0)) catch |e| return e == error.ProcessNotFound;
    return false;
}

test "mcp serve: a cached model served on loopback with its own key, listed by status and stopped by stop" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: World = undefined;
    try w.init(a, .serves);
    defer w.deinit();
    var live: fixture.Live = undefined;
    live.setUp(&w.ctx);
    defer live.tearDown();
    try live.send(initialize);
    try live.send("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"serve\",\"arguments\":{\"model\":\"Org/Lightning\",\"context\":4096},\"_meta\":{\"progressToken\":7}}}\n");
    const served = try live.answer(a, 2);
    try std.testing.expect(!fixture.isError(served));
    const s = structured(served);
    const base = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1", .{w.f.port});
    try std.testing.expectEqualStrings(base, s.strField("base_url").?);
    try std.testing.expectEqualStrings("Lightning", s.strField("name").?);
    try std.testing.expectEqualStrings("1", s.strField("id").?);
    const key = s.strField("api_key").?;
    try std.testing.expectEqual(@as(usize, 64), key.len);
    for (key) |ch| try std.testing.expect(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'));
    try std.testing.expect(fixture.has(fixture.text(served), key) and fixture.has(fixture.text(served), base));
    try std.testing.expectEqualStrings(key, try std.Io.Dir.cwd().readFileAlloc(std.testing.io, w.keyfile, a, .limited(256)));
    const args = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, w.argsfile, a, .limited(64 << 10));
    try std.testing.expect(fixture.has(args, "serve\n"));
    try std.testing.expect(fixture.has(args, "/hub/models--Org--Lightning/snapshots/" ++ fixture.sha ++ "\n--host\n127.0.0.1\n--port\n0\n--name\nLightning\n--no-update-check\n--context\n4096\n"));
    var probed = false; // the readiness probe asked the models route with the child's own key
    for (0..w.f.count()) |i| if (std.mem.eql(u8, w.f.request(i).path, "/v1/models")) {
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "Bearer {s}", .{key}), w.f.header(i, "authorization").?);
        probed = true;
    };
    try std.testing.expect(probed and w.f.refusals() == 0);
    const note = (try live.written(a))[1]; // the progress the token asked for, before the answer
    try std.testing.expectEqualStrings("notifications/progress", note.strField("method").?);
    try std.testing.expectEqual(@as(?i64, 7), note.get("params").?.get("progressToken").?.int64());
    const pid = try w.pid(a);
    try std.testing.expectEqual(pid, getpgid(pid)); // a process group of its own: a terminal's Ctrl-C reaches only the session

    try live.send(try call(a, 3, "serve", "{\"model\":\"Org/Lightning\",\"context\":4096}"));
    const again = structured(try live.answer(a, 3));
    try std.testing.expect(again.get("already_running").?.bool);
    try std.testing.expectEqualStrings(key, again.strField("api_key").?);
    try live.send(try call(a, 4, "status", "{}"));
    const listed = structured(try live.answer(a, 4)).get("servers").?.array;
    try std.testing.expectEqual(@as(usize, 1), listed.len);
    try std.testing.expect(listed[0].get("running").?.bool);
    try std.testing.expectEqualStrings(base, listed[0].strField("base_url").?);
    try std.testing.expect(listed[0].get("api_key") == null); // status names no key

    try live.send(try call(a, 5, "stop", "{\"id\":\"1\"}"));
    const stopped = try live.answer(a, 5);
    try std.testing.expectEqualStrings("stopped", structured(stopped).strField("state").?);
    try std.testing.expect(gone(pid));
    try live.send(try call(a, 6, "status", "{}"));
    try std.testing.expect(!structured(try live.answer(a, 6)).get("servers").?.array[0].get("running").?.bool);
    try live.send(try call(a, 7, "stop", "{\"id\":\"1\"}"));
    try std.testing.expect(fixture.has(fixture.text(try live.answer(a, 7)), "was already stopped"));
}

test "mcp serve: the session's end, as stdin's end brings it, stops every server it started" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: World = undefined;
    try w.init(a, .serves);
    defer w.deinit();
    var live: fixture.Live = undefined;
    live.setUp(&w.ctx);
    var ended = false;
    defer if (!ended) live.tearDown();
    try live.send(initialize);
    try live.send(try call(a, 2, "serve", "{\"model\":\"Org/Lightning\"}"));
    try std.testing.expect(!fixture.isError(try live.answer(a, 2)));
    const pid = try w.pid(a);
    try std.testing.expect(!gone(pid));
    ended = true;
    live.tearDown();
    try std.testing.expect(gone(pid));
}

test "mcp serve: a server that exits on its own shows as exited in status, with its last log line" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: World = undefined;
    try w.init(a, .serves_then_dies);
    defer w.deinit();
    var live: fixture.Live = undefined;
    live.setUp(&w.ctx);
    defer live.tearDown();
    try live.send(initialize);
    try live.send(try call(a, 2, "serve", "{\"model\":\"Org/Lightning\"}"));
    try std.testing.expect(!fixture.isError(try live.answer(a, 2)));
    var id: i64 = 3;
    const exited = while (id < 503) : (id += 1) { // status reaps the child once it has exited
        try live.send(try call(a, id, "status", "{}"));
        const servers = structured(try live.answer(a, id)).get("servers").?.array;
        if (std.mem.eql(u8, servers[0].strField("state").?, "exited")) break servers[0];
        std.testing.io.sleep(.fromMilliseconds(20), .awake) catch {};
    } else return error.TestTimedOut;
    try std.testing.expect(!exited.get("running").?.bool);
    try std.testing.expectEqualStrings("error: the GPU ran out of memory", exited.get("last_log").?.array[0].string);
    try live.send(try call(a, 600, "stop", "{\"id\":\"1\"}"));
    try std.testing.expect(fixture.has(fixture.text(try live.answer(a, 600)), "had already exited"));
}

test "mcp serve: a child that exits while loading is a tool error with its last log line, and leaves no entry" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: World = undefined;
    try w.init(a, .exits);
    defer w.deinit();
    var live: fixture.Live = undefined;
    live.setUp(&w.ctx);
    defer live.tearDown();
    try live.send(initialize);
    try live.send(try call(a, 2, "serve", "{\"model\":\"Org/Lightning\"}"));
    const v = try live.answer(a, 2);
    try std.testing.expect(fixture.isError(v));
    try std.testing.expect(fixture.has(fixture.text(v), "tensorfold serve exited while loading Org/Lightning; its last log lines: error: cannot read tokenizer.json"));
    try live.send(try call(a, 3, "status", "{}"));
    try std.testing.expectEqual(@as(usize, 0), structured(try live.answer(a, 3)).get("servers").?.array.len);
}

test "mcp serve: a cancelled serve stops its loading child and answers nothing; stop ends a loading one too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: World = undefined;
    try w.init(a, .hangs);
    defer w.deinit();
    var live: fixture.Live = undefined;
    live.setUp(&w.ctx);
    defer live.tearDown();
    try live.send(initialize);
    try live.send(try call(a, 2, "serve", "{\"model\":\"Org/Lightning\"}"));
    const first = try w.pid(a);
    try live.send("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":2,\"reason\":\"example\"}}\n");
    try live.idle();
    try std.testing.expect(gone(first));
    try std.testing.expect(fixture.find(try live.written(a), 2) == null);

    try std.Io.Dir.cwd().deleteFile(std.testing.io, w.pidfile);
    try live.send(try call(a, 3, "serve", "{\"model\":\"Org/Lightning\"}"));
    const second = try w.pid(a);
    try live.send(try call(a, 4, "status", "{}"));
    const loading = structured(try live.answer(a, 4)).get("servers").?.array;
    try std.testing.expectEqualStrings("starting", loading[0].strField("state").?);
    try std.testing.expect(loading[0].get("base_url").? == .null);
    try live.send(try call(a, 5, "stop", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\"}}", .{loading[0].strField("id").?})));
    try std.testing.expect(fixture.has(fixture.text(try live.answer(a, 5)), "was still loading"));
    const ended = try live.answer(a, 3);
    try std.testing.expect(fixture.isError(ended));
    try std.testing.expect(fixture.has(fixture.text(ended), "was stopped while it loaded"));
    try std.testing.expect(gone(second));
}

test "mcp serve: models it cannot serve, paths it refuses and settings serve refuses are tool errors; nothing starts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: World = undefined;
    try w.init(a, .serves);
    defer w.deinit();
    _ = try w.home.checkpoint(a, "Org/Draft", "{\"model_type\": \"gemma3n\"}");
    const cases = [_]struct { args: []const u8, says: []const u8 }{
        .{ .args = "{\"model\":\"Org/Missing\"}", .says = "not in the Hugging Face cache" },
        .{ .args = "{\"model\":\"Org/Draft\"}", .says = "no TensorFold family in this build serves model_type gemma3n" },
        .{ .args = "{\"model\":\"plain-name\"}", .says = "relative paths are refused" },
        .{ .args = "{\"model\":\"Org/..\"}", .says = "neither a Hugging Face repo id" },
        .{ .args = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\"}}", .{w.home.dir}), .says = "is not a checkpoint directory" },
        .{ .args = "{\"model\":\"Org/Lightning\",\"drafter\":\"Org/Missing\"}", .says = "drafter: Org/Missing is not in the Hugging Face cache" },
    };
    var input: std.ArrayList(u8) = .empty;
    try input.appendSlice(a, initialize);
    for (cases, 2..) |c, id| try input.appendSlice(a, try call(a, @intCast(id), "serve", c.args));
    const got = try fixture.exchange(a, &w.ctx, input.items);
    for (cases, 2..) |c, id| {
        const v = fixture.find(got, @intCast(id)).?;
        try std.testing.expect(fixture.isError(v));
        try std.testing.expect(fixture.has(fixture.text(v), c.says));
    }
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(std.testing.io, w.pidfile, .{}));
}
