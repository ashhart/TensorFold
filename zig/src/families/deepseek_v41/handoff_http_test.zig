//! The HTTP hand-off over loopback: a fake producer serves a real export stream and every failure is refused by name.
const std = @import("std");
const fabric = @import("fabric");
const metal = @import("handoff/metal.zig");
const landing = @import("handoff/landing.zig");
const file = @import("handoff_file.zig");
const http = @import("handoff_http.zig");
const ft = @import("handoff_file_test.zig");
const producer = fabric.producer;

const gpa = std.testing.allocator;
const io = std.testing.io;
const model = "dsv41-test";
const key = "sk-gb10-test";

const Mode = enum {
    /// the whole stream, 200
    ok,
    /// the head promises the whole stream, the body stops at `cut` bytes and the line closes
    truncated,
    /// 404: nothing held under the id
    not_held,
    /// 409 with the producer's reason
    failed,
    /// accepts, reads the request, and never answers
    hang,
};

/// One producer on 127.0.0.1 answering `connections` requests.
const Fake = struct {
    server: std.Io.net.Server,
    mode: Mode,
    exp: *const producer.Export,
    needs_key: bool = false,
    connections: usize,
    cut: u64 = 0,
    gets: std.atomic.Value(u32) = .init(0),
    deletes: std.atomic.Value(u32) = .init(0),
    unauthorized: std.atomic.Value(u32) = .init(0),

    fn start(mode: Mode, exp: *const producer.Export, connections: usize) !*Fake {
        const f = try gpa.create(Fake);
        errdefer gpa.destroy(f);
        const loopback: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        f.* = .{ .server = try loopback.listen(io, .{}), .mode = mode, .exp = exp, .connections = connections };
        return f;
    }

    fn url(f: *const Fake, buf: []u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "http://127.0.0.1:{d}", .{f.server.socket.address.getPort()});
    }

    fn run(f: *Fake) void {
        for (0..f.connections) |_| f.one();
    }

    fn one(f: *Fake) void {
        const s = f.server.accept(io) catch return;
        defer s.close(io);
        var in: [512]u8 = undefined;
        var r = s.reader(io, &in);
        var head: [2048]u8 = undefined;
        var n: usize = 0;
        while (n < head.len and !std.mem.endsWith(u8, head[0..n], "\r\n\r\n")) : (n += 1) head[n] = r.interface.takeByte() catch return;
        const text = head[0..n];
        var out: [4096]u8 = undefined;
        var w = s.writer(io, &out);
        defer w.interface.flush() catch {};
        if (std.mem.startsWith(u8, text, "DELETE /v1/handoff/")) {
            _ = f.deletes.fetchAdd(1, .monotonic);
            w.interface.writeAll("HTTP/1.1 200 OK\r\nContent-Length: 18\r\nConnection: close\r\n\r\n{\"released\": true}") catch {};
            return;
        }
        _ = f.gets.fetchAdd(1, .monotonic);
        var line: [80]u8 = undefined;
        const want = std.fmt.bufPrint(&line, "GET /v1/handoff/{x} HTTP/1.1\r\n", .{&f.exp.handoff}) catch unreachable;
        if (!std.mem.startsWith(u8, text, want)) return refuse(&w.interface, "400 Bad Request");
        if (f.needs_key and std.ascii.findIgnoreCase(text, "authorization: Bearer " ++ key ++ "\r\n") == null) {
            _ = f.unauthorized.fetchAdd(1, .monotonic);
            return refuse(&w.interface, "401 Unauthorized");
        }
        switch (f.mode) {
            .not_held => return refuse(&w.interface, "404 Not Found"),
            .failed => {
                const body = "{\"error\": {\"message\": \"NotAHandoffPrompt\", \"type\": \"invalid_request_error\"}}";
                w.interface.print("HTTP/1.1 409 Conflict\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ body.len, body }) catch {};
            },
            .hang => while (true) { // hold the line until the client hangs up
                _ = r.interface.takeByte() catch return;
            },
            .ok, .truncated => {
                const replies = file.Replies.open(gpa, f.exp, model, 0, 1) catch return;
                defer replies.deinit();
                w.interface.print("HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{replies.size}) catch return;
                var sent: u64 = 0;
                while (replies.next() catch return) |message| {
                    if (f.mode == .truncated and sent + message.len > f.cut) {
                        w.interface.writeAll(message[0..@intCast(f.cut - sent)]) catch {};
                        return;
                    }
                    w.interface.writeAll(message) catch return;
                    sent += message.len;
                }
            },
        }
    }

    fn refuse(w: *std.Io.Writer, status: []const u8) void {
        w.print("HTTP/1.1 {s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{status}) catch {};
    }
};

/// What a fetch is given: the export's own tokens and checkpoint unless a test changes them.
const Case = struct {
    f: ft.Fixture,
    fake: *Fake,
    thread: std.Thread,
    base: [40]u8 = undefined,
    base_len: usize = 0,

    fn start(mode: Mode, connections: usize) !*Case {
        const c = try gpa.create(Case);
        errdefer gpa.destroy(c);
        c.f = try ft.Fixture.make(300, 128);
        errdefer c.f.deinit();
        c.fake = try Fake.start(mode, &c.f.built.exp, connections);
        errdefer {
            c.fake.server.deinit(io);
            gpa.destroy(c.fake);
        }
        c.base_len = (try c.fake.url(&c.base)).len;
        c.thread = try std.Thread.spawn(.{}, Fake.run, .{c.fake});
        return c;
    }

    fn finish(c: *Case) void {
        c.thread.join();
        c.fake.server.deinit(io);
        gpa.destroy(c.fake);
        c.f.deinit();
        gpa.destroy(c);
    }

    fn source(c: *const Case) http.Source {
        return .{ .gpa = gpa, .io = io, .base = c.base[0..c.base_len], .budget_ns = 30 * std.time.ns_per_s };
    }

    fn expect(c: *const Case) fabric.decoder.Expect {
        return .{ .model = model, .tokens = c.f.tokens, .protocol = 2 };
    }
};

fn target(arena: std.mem.Allocator) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = 1024 };
    const pk: metal.Packed = .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(1024)), .layers = 40, .capacity = 1024 };
    @memset(pk.storage, 0);
    return pk;
}

test "a stream's length is known before its first byte, and is the file's" {
    var f = try ft.Fixture.make(300, 128);
    defer f.deinit();
    const replies = try file.Replies.open(gpa, &f.built.exp, model, 0, 1);
    defer replies.deinit();
    try std.testing.expectEqual(@as(u64, f.bytes.len), replies.size);
    var all: std.Io.Writer.Allocating = .init(gpa);
    defer all.deinit();
    while (try replies.next()) |message| try all.writer.writeAll(message);
    try std.testing.expectEqualSlices(u8, f.bytes, all.written());
    try std.testing.expectEqual(@as(?[]const u8, null), try replies.next());
}

test "a hand-off fetched over HTTP lands in a Snapshot's buffers as the file replay lands it" {
    const c = try Case.start(.ok, 1);
    defer c.finish();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const via = try target(arena);
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = via });
    defer l.deinit();
    var reason: fabric.decoder.Reason = .{};
    const began = fabric.words.nowNs();
    const got = try c.source().pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason);
    const took = fabric.words.nowNs() - began;
    _ = try l.complete();
    try std.testing.expectEqual(@as(u32, 3), l.dropped); // the drafter's rings are read and dropped
    try std.testing.expectEqual(c.f.built.bytes, got.bytes);
    std.debug.print("\nhttp hand-off: {d} stream bytes ({d} part bytes) over loopback in {d:.1} ms\n", .{ c.f.bytes.len, got.bytes, @as(f64, @floatFromInt(took)) / 1e6 });

    const direct = try target(arena);
    var l2: landing.Landing = undefined;
    try l2.init(arena, 300, .{ .@"packed" = direct });
    defer l2.deinit();
    _ = try file.pull(arena, c.f.bytes, c.expect(), l2.decoderSink(), &reason);
    _ = try l2.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
    try std.testing.expectEqual(@as(u32, 1), c.fake.gets.load(.monotonic));
}

test "the key is sent as a Bearer token, and a producer that wants one and gets none answers 401" {
    const c = try Case.start(.ok, 2);
    defer c.finish();
    c.fake.needs_key = true;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = try target(arena) });
    defer l.deinit();
    var reason: fabric.decoder.Reason = .{};
    try std.testing.expectError(error.Unauthorized, c.source().pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason));
    var keyed = c.source();
    keyed.key = key;
    _ = try keyed.pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason);
    try std.testing.expectEqual(@as(u32, 1), c.fake.unauthorized.load(.monotonic));
}

test "an export that is not held is a 404, and one that failed carries the producer's reason" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = try target(arena) });
    defer l.deinit();
    var reason: fabric.decoder.Reason = .{};
    {
        const c = try Case.start(.not_held, 1);
        defer c.finish();
        try std.testing.expectError(error.NotHeld, c.source().pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason));
    }
    {
        const c = try Case.start(.failed, 1);
        defer c.finish();
        try std.testing.expectError(error.Refused, c.source().pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason));
        try std.testing.expectEqualStrings("NotAHandoffPrompt", reason.text());
    }
}

test "a stream cut short is refused as truncated, whole messages or not" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var reason: fabric.decoder.Reason = .{};
    for ([_]u64{ 40, 1000, 200_000 }) |cut| {
        const c = try Case.start(.truncated, 1);
        defer c.finish();
        c.fake.cut = @min(cut, c.f.bytes.len - 1);
        var l: landing.Landing = undefined;
        try l.init(arena, 300, .{ .@"packed" = try target(arena) });
        defer l.deinit();
        const err = c.source().pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason);
        if (err) |_| return error.TestExpectedError else |e| try std.testing.expect(e == error.Truncated or e == error.TransportFailed);
        try std.testing.expectError(error.Incomplete, l.complete()); // nothing it landed makes a state
    }
}

test "another checkpoint or another prompt is a mismatch, and the manifest's checks run before any row lands" {
    const c = try Case.start(.ok, 2);
    defer c.finish();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = try target(arena) });
    defer l.deinit();
    var reason: fabric.decoder.Reason = .{};
    var other_model = c.expect();
    other_model.model = "another-checkpoint";
    try std.testing.expectError(error.Mismatch, c.source().pull(arena, c.f.built.exp.handoff, other_model, l.decoderSink(), &reason));
    var other_tokens = c.expect();
    const changed = try arena.dupe(u32, c.f.tokens);
    changed[7] += 1;
    other_tokens.tokens = changed;
    try std.testing.expectError(error.Mismatch, c.source().pull(arena, c.f.built.exp.handoff, other_tokens, l.decoderSink(), &reason));
    try std.testing.expectError(error.Incomplete, l.complete());
}

test "a producer that never answers is given up on at the budget, and a stopping server cuts a fetch off" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var reason: fabric.decoder.Reason = .{};
    {
        const c = try Case.start(.hang, 1);
        defer c.finish();
        var l: landing.Landing = undefined;
        try l.init(arena, 300, .{ .@"packed" = try target(arena) });
        defer l.deinit();
        var s = c.source();
        s.budget_ns = 300 * std.time.ns_per_ms;
        const began = fabric.words.nowNs();
        try std.testing.expectError(error.TimedOut, s.pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason));
        try std.testing.expect(fabric.words.nowNs() - began < 10 * std.time.ns_per_s);
    }
    {
        var f = try ft.Fixture.make(300, 128); // a producer that is never reached: the server is stopping
        defer f.deinit();
        var l: landing.Landing = undefined;
        try l.init(arena, 300, .{ .@"packed" = try target(arena) });
        defer l.deinit();
        var stopping: std.atomic.Value(bool) = .init(true);
        const s: http.Source = .{ .gpa = gpa, .io = io, .base = "http://127.0.0.1:9", .closing = &stopping };
        try std.testing.expectError(error.Closed, s.pull(arena, f.built.exp.handoff, .{ .model = model, .tokens = f.tokens, .protocol = 2 }, l.decoderSink(), &reason));
    }
}

test "the producer's copy is released with DELETE, and a URL that is not plain http is refused" {
    const c = try Case.start(.ok, 1);
    defer c.finish();
    c.source().release(c.f.built.exp.handoff);
    try std.testing.expectEqual(@as(u32, 1), c.fake.deletes.load(.monotonic));
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var l: landing.Landing = undefined;
    try l.init(arena, 300, .{ .@"packed" = try target(arena) });
    defer l.deinit();
    var reason: fabric.decoder.Reason = .{};
    var s = c.source();
    s.base = "https://127.0.0.1:1";
    try std.testing.expectError(error.NotHttp, s.pull(arena, c.f.built.exp.handoff, c.expect(), l.decoderSink(), &reason));
}
