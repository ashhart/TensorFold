//! The router over real serve children: fake_serve as each model's server, its own key, replies and exits.
const std = @import("std");
const listener_mod = @import("listener.zig");
const catalog = @import("router/catalog.zig");
const front = @import("router/front.zig");
const tc = @import("router/test_child.zig");
const t = std.testing;
const posix = std.posix;

/// The scripted engine's checkpoint: the build runs this test from the repository's root.
const fixtures = "zig/tests/server/fixtures";

fn entry(a: std.mem.Allocator, id: []const u8, aliases: []const []const u8) !catalog.Entry {
    const argv = try catalog.childArgv(a, @import("fake_serve_exe").path, .{ .id = id, .aliases = aliases, .model = fixtures, .flags = &.{"--no-thinking"} });
    return .{ .id = id, .aliases = aliases, .argv = argv, .startup_s = 30 };
}

fn chat(a: std.mem.Allocator, model: []const u8, stream: bool) ![]const u8 {
    const body = try std.fmt.allocPrint(a, "{{\"model\": \"{s}\", \"messages\": [{{\"role\": \"user\", \"content\": \"hi\"}}], \"stream\": {}}}", .{ model, stream });
    return tc.post(a, "/v1/chat/completions", body, "");
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}

test "serve --router: real serve children answer under their own keys, give way under --max-loaded and restart" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const entries = [_]catalog.Entry{ try entry(a, "alpha", &.{"a1"}), try entry(a, "beta", &.{}) };
    const pool = try tc.pool(a, &tmp, &entries, .{ .max_loaded = 1, .log_dir = "" });
    defer pool.deinit();
    defer pool.stopAll();
    const f = try a.create(front.Front);
    f.* = .{ .gpa = t.allocator, .io = t.io, .pool = pool, .entries = &entries };
    const lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) });
    defer lis.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, front.Front.serve, .{ f, lis, &stop });
    defer {
        stop.store(true, .release);
        accept.join();
        var waited: u32 = 0;
        while (f.open.load(.acquire) > 0 and waited < 500) : (waited += 1) @import("benchmark_fake.zig").nap(10);
    }
    const port = lis.port();
    // an alias reaches its model, whose own server answers to the name that was asked
    const first = try tc.exchangeWith(a, port, try chat(a, "a1", false));
    try t.expectEqual(@as(u16, 200), tc.status(first));
    try t.expect(has(tc.bodyOf(first), "\"model\": \"a1\"") and has(tc.bodyOf(first), "Hello"));
    const child = (try pool.loaded(a))[0];
    const direct = try tc.exchangeWith(a, child.port, try chat(a, "alpha", false));
    try t.expectEqual(@as(u16, 401), tc.status(direct));
    const keyed = try tc.exchangeWith(a, child.port, try tc.get(a, "/v1/models", try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{&child.key})));
    try t.expectEqual(@as(u16, 200), tc.status(keyed));
    const streamed = try tc.exchangeWith(a, port, try chat(a, "alpha", true));
    try t.expect(has(streamed, "text/event-stream") and std.mem.endsWith(u8, streamed, "data: [DONE]\n\n"));
    const listed = tc.bodyOf(try tc.exchangeWith(a, port, try tc.get(a, "/v1/models", "")));
    try t.expect(has(listed, "\"id\": \"alpha\", \"object\": \"model\"") and has(listed, "\"capabilities\": {") and has(listed, "\"loaded\": true"));
    // --max-loaded 1: beta's request stops alpha, and a beta that dies is started again by the next request
    const alpha = tc.pid(pool, 0).?;
    try t.expectEqual(@as(u16, 200), tc.status(try tc.exchangeWith(a, port, try chat(a, "beta", false))));
    try t.expect(tc.gone(alpha) and tc.pid(pool, 0) == null);
    const beta = tc.pid(pool, 1).?;
    try posix.kill(beta, .KILL);
    const Exited = struct {
        fn check(pid: posix.pid_t) bool {
            return @import("benchmark_server.zig").ended(pid);
        }
    };
    try t.expect(tc.within(5_000, beta, Exited.check)); // a request racing the kill would rightly get 502
    try t.expectEqual(@as(u16, 200), tc.status(try tc.exchangeWith(a, port, try chat(a, "beta", false))));
    try t.expect(tc.pid(pool, 1).? != beta);
}
