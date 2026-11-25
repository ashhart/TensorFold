//! KR69 regressions at terminal, discovery, retained-history and serialized-POST boundaries.
const std = @import("std");
const chat = @import("chat_cli.zig");
const fake = @import("benchmark_fake.zig");
const fixtures = @import("chat_cli_test.zig");
const stop = @import("benchmark_stop.zig");
const platform = @import("sys");
const Allocator = std.mem.Allocator;

fn terminalCommand(r: *fixtures.Run, argv: []const []const u8) !u8 {
    r.out.clearRetainingCapacity();
    r.err.clearRetainingCapacity();
    r.stdin = .fixed(r.input);
    const ctx: chat.Context = .{
        .a = r.a,
        .io = std.testing.io,
        .environ = &r.env,
        .out = &r.out.writer,
        .err = &r.err.writer,
        .input = r.reader orelse &r.stdin,
        .interactive = r.interactive,
        .terminal = true,
        .self_exe = r.self_exe,
    };
    return chat.command(&ctx, argv);
}

fn absent(text: []const u8, secrets: []const []const u8) !void {
    for (secrets) |secret| try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}

fn postCount(f: *fake.Fake) usize {
    var count: usize = 0;
    for (0..f.count()) |i| count += @intFromBool(std.mem.eql(u8, f.request(i).method, "POST"));
    return count;
}

fn stream(a: Allocator, parts: []const []const u8) ![]const fake.Write {
    var bytes: std.ArrayList(u8) = .empty;
    try bytes.appendSlice(a, try fake.headOf(a, "200 OK", "text/event-stream"));
    for (parts) |part| try bytes.appendSlice(a, part);
    return a.dupe(fake.Write, &.{.{ .bytes = bytes.items }});
}

fn delta(a: Allocator, content: []const u8) ![]const u8 {
    return fake.event(a, try std.fmt.allocPrint(a, "{{\"choices\": [{{\"index\": 0, \"delta\": {{\"content\": \"{s}\"}}, \"finish_reason\": null}}]}}", .{content}));
}

fn successful(a: Allocator, content: []const u8) ![]const fake.Write {
    return stream(a, try a.dupe([]const u8, &.{
        try delta(a, content),
        try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
        try fake.event(a, "[DONE]"),
    }));
}

fn httpError(a: Allocator, message: []const u8) ![]const fake.Write {
    return fake.jsonReply(a, "500 Internal Server Error", try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{message}));
}

fn sseError(a: Allocator, message: []const u8) ![]const fake.Write {
    return stream(a, try a.dupe([]const u8, &.{
        try fake.event(a, try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{message})),
        try fake.event(a, "[DONE]"),
    }));
}

fn longKey(a: Allocator) ![]u8 {
    const key = try a.alloc(u8, 256);
    for (key, 0..) |*c, i| c.* = 'a' + @as(u8, @intCast(i % 26));
    return key;
}

test "chat CLI: KR69 held raw and nested prefixes never reach output or a following POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try stream(a, try a.dupe([]const u8, &.{
            try delta(a, "fixture-secre"),
            try delta(a, "X"),
            try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
            try fake.event(a, "[DONE]"),
        })),
        try successful(a, "fixture-secreX"),
        try stream(a, try a.dupe([]const u8, &.{
            try delta(a, "%66ixture-secre%"),
            try delta(a, "7X"),
            try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
            try fake.event(a, "[DONE]"),
        })),
        try successful(a, "done"),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "fixture-secret");
    r.interactive = true;
    r.input = "one\ntwo\nthree\nlast\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    const attacks = [_][]const u8{ "fixture-secre", "%66ixture-secre" };
    try absent(r.out.written(), &attacks);
    try absent(r.err.written(), &attacks);
    try std.testing.expectEqual(@as(usize, 4), postCount(f));
    for (0..f.count()) |i| if (std.mem.eql(u8, f.request(i).method, "POST")) try absent(f.request(i).body, &attacks);
}

test "chat CLI: KR69 already-truncated raw and nested credentials stay out of HTTP and SSE errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const key = try longKey(a);
    const raw = key[0..255];
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try httpError(a, raw),
        try sseError(a, "%66ixture-secre"),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", try std.mem.concat(a, u8, &.{ key, ",fixture-secret" }));
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    const url = try f.url(a, "");
    for (0..2) |_| {
        try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", url, "safe-model" }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try absent(shown.items, &.{ raw, "%66ixture-secre" });
}

test "chat CLI: KR69 an already-truncated discovered key suppresses the model before any POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const key = try longKey(a);
    const raw = key[0..255];
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{raw});
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "unused")}));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", key);
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try absent(r.out.written(), &.{raw});
    try absent(r.err.written(), &.{raw});
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
}

fn discoveryFailure(a: Allocator, status: []const u8) !*fake.Fake {
    return fake.Fake.startKeyed(std.testing.allocator, try a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, status, "{}")}) },
    }), null);
}

test "chat CLI: KR69 discovery 401 contains its credential-valued hint" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try discoveryFailure(a, "401 Unauthorized");
    defer f.stop();
    const secret = "TENSORFOLD_API_KEY";
    try r.env.put("TENSORFOLD_API_KEY", secret);
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try absent(r.out.written(), &.{secret});
    try absent(r.err.written(), &.{secret});
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
}

test "chat CLI: KR69 generic discovery errors contain encoded credential URLs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try discoveryFailure(a, "500 Internal Server Error");
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "credential-source");
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, "/credential%2Dsource") }));
    try absent(r.out.written(), &.{ "credential-source", "credential%2Dsource" });
    try absent(r.err.written(), &.{ "credential-source", "credential%2Dsource" });
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
}

test "chat CLI: KR69 no-thinking guidance passes final discovery containment" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const models = "{\"data\": [{\"id\": \"safe\", \"capabilities\": {\"thinking_switch\": false}}]}";
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "unused")}));
    defer f.stop();
    const secret = "thinking_switch";
    try r.env.put("TENSORFOLD_API_KEY", secret);
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, ""), "--no-thinking" }));
    try absent(r.out.written(), &.{secret});
    try absent(r.err.written(), &.{secret});
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
}

fn unsafeModel(a: Allocator, secret: []const u8, model_json: []const u8) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{model_json});
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "unused")}));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", secret);
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
    try absent(r.out.written(), &.{secret});
    try absent(r.err.written(), &.{secret});
}

test "chat CLI: KR69 JSON model serialization cannot synthesize non-ASCII quote or backslash keys" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try unsafeModel(a, "\\u00e9", "\\u00e9");
    try unsafeModel(a, "\"", "\\\"");
    try unsafeModel(a, "\\\"", "\\\\");
}

test "chat CLI: KR69 a finished Unicode reply cannot synthesize a key in the following history POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try successful(a, "é"),
        try successful(a, "unused"),
    }));
    defer f.stop();
    const secret = "\\u00e9";
    try r.env.put("TENSORFOLD_API_KEY", secret);
    r.interactive = true;
    r.input = "first\nsecond\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqual(@as(usize, 1), postCount(f));
    try absent(r.out.written(), &.{secret});
    try absent(r.err.written(), &.{secret});
    for (0..f.count()) |i| try absent(f.request(i).body, &.{secret});
}

const Pipe = struct {
    fds: [2]std.posix.fd_t = .{ -1, -1 },
    buf: [256]u8 = undefined,
    file: std.Io.File.Reader = undefined,

    fn open(p: *Pipe) bool {
        if (platform.c.pipe(&p.fds) != 0) return false;
        p.file = (std.Io.File{ .handle = p.fds[0], .flags = .{ .nonblocking = false } }).readerStreaming(std.testing.io, &p.buf);
        return true;
    }

    fn close(p: *Pipe) void {
        for (p.fds) |fd| _ = platform.c.close(fd);
    }

    fn send(p: *const Pipe, text: []const u8) void {
        _ = platform.c.write(p.fds[1], text.ptr, text.len);
    }
};

fn hasPost(f: *fake.Fake) bool {
    return postCount(f) > 0;
}

fn handled(_: void) bool {
    return stop.requested() == null;
}

fn waitFor(comptime ready: anytype, arg: anytype) bool {
    var waited: u32 = 0;
    while (waited < 5000) : (waited += 5) {
        if (ready(arg)) return true;
        fake.nap(5);
    }
    return false;
}

test "chat CLI: KR69 a stopped Unicode reply cannot synthesize a key in the following history POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    stop.reset();
    defer stop.reset();
    var gate: std.Io.Event = .unset;
    const trickle = try a.dupe(fake.Write, &.{
        .{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), try delta(a, "é") }), .after = &gate },
        .{ .bytes = try delta(a, "unused") },
    });
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ trickle, try successful(a, "unused") }));
    defer {
        gate.set(io);
        f.stop();
    }
    var pipe: Pipe = .{};
    if (!pipe.open()) return error.SkipZigTest;
    defer pipe.close();
    r.interactive = true;
    r.reader = &pipe.file.interface;
    const Typist = struct {
        fn run(p: *const Pipe, fk: *fake.Fake, g: *std.Io.Event) void {
            p.send("first\n");
            if (waitFor(hasPost, fk)) fake.nap(300);
            stop.request(.INT);
            if (!waitFor(handled, {})) g.set(std.testing.io);
            p.send("second\n/bye\n");
        }
    };
    const t = try std.Thread.spawn(.{}, Typist.run, .{ &pipe, f, &gate });
    defer t.join();
    try r.env.put("TENSORFOLD_API_KEY", "\\u00e9");
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqual(@as(usize, 1), postCount(f));
    for (0..f.count()) |i| try absent(f.request(i).body, &.{"\\u00e9"});
}

test "chat CLI: KR71 same-delta and cross-delta prefixes stay out of output, history and every following POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const hostile = [_][]const u8{ "abcdef", "%61%62%63%64%65%66", "%2561%2562%2563%2564%2565%2566" };
    for ([_]bool{ false, true }) |same_delta| for (hostile) |attack| {
        var r = try fixtures.Run.init(a, &tmp);
        const first_delta = if (same_delta)
            try delta(a, try std.fmt.allocPrint(a, "{s}X", .{attack}))
        else
            try delta(a, attack);
        const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
            try stream(a, try a.dupe([]const u8, &.{
                first_delta,
                try delta(a, "X"),
                try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
                try fake.event(a, "[DONE]"),
            })),
            try successful(a, "done"),
        }));
        defer f.stop();
        try r.env.put("TENSORFOLD_API_KEY", "abcdefghi");
        r.interactive = true;
        r.input = "one\ntwo\n/bye\n";
        try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
        try absent(r.out.written(), &hostile);
        try absent(r.err.written(), &hostile);
        try std.testing.expectEqual(@as(usize, 2), postCount(f));
        for (0..f.count()) |i| try absent(f.request(i).body, &hostile);
    };
}

test "chat CLI: KR71 one hostile table covers discovery and HTTP or SSE errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const hostile = [_][]const u8{ "abcdef", "%61%62%63%64%65%66", "%2561%2562%2563%2564%2565%2566" };
    for (hostile) |attack| {
        var r = try fixtures.Run.init(a, &tmp);
        try r.env.put("TENSORFOLD_API_KEY", "abcdefghi");
        r.input = "hi";
        const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{attack});
        const discovered = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "unused")}));
        defer discovered.stop();
        try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try discovered.url(a, "") }));
        try absent(r.out.written(), &hostile);
        try absent(r.err.written(), &hostile);
        try std.testing.expectEqual(@as(usize, 0), postCount(discovered));
        const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ try httpError(a, attack), try sseError(a, attack) }));
        defer f.stop();
        for (0..2) |_| {
            try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, ""), "safe-model" }));
            try absent(r.out.written(), &hostile);
            try absent(r.err.written(), &hostile);
        }
        for (0..f.count()) |i| try absent(f.request(i).body, &hostile);
    }
}
