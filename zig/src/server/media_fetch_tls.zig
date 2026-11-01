//! Shared media TLS trust initialization and an absolute socket deadline, including the first handshake.
const std = @import("std");
const Io = std.Io;
const fetch = @import("media_fetch.zig");

pub const Trust = struct {
    lock: Io.RwLock = .init,
    bundle: std.crypto.Certificate.Bundle = .empty,
    loaded: bool = false,

    pub fn ensure(t: *Trust, io: Io, f: *fetch.Failure) fetch.Error!void {
        return t.ensureVia(io, f, System{});
    }

    fn ensureVia(t: *Trust, io: Io, f: *fetch.Failure, loader: anytype) fetch.Error!void {
        t.lock.lockUncancelable(io);
        defer t.lock.unlock(io);
        if (t.loaded) return;
        loader.rescan(&t.bundle, io) catch {
            f.text = "image download failed: no CA certificates on the server";
            return error.Media;
        };
        t.loaded = true;
    }

    const System = struct {
        fn rescan(_: System, b: *std.crypto.Certificate.Bundle, io: Io) !void {
            try b.rescan(std.heap.page_allocator, io, Io.Clock.real.now(io));
        }
    };
};

fn expire(io: Io, stream: Io.net.Stream, deadline: Io.Timestamp) void {
    const left = Io.Clock.awake.now(io).durationTo(deadline);
    if (left.nanoseconds > 0) io.sleep(left, .awake) catch return;
    if (comptime @import("builtin").os.tag != .windows) _ = std.c.shutdown(stream.socket.handle, 2);
}

pub fn request(t: *Trust, a: std.mem.Allocator, io: Io, stream: Io.net.Stream, u: fetch.Url, max_bytes: usize, deadline: Io.Timestamp, media_types: []const []const u8, f: *fetch.Failure) fetch.Error!fetch.Got {
    const failed = "image download failed or timed out";
    // Join the timer before the caller closes the socket, so an expired timer cannot shut down a reused descriptor.
    var timer = io.concurrent(expire, .{ io, stream, deadline }) catch {
        f.text = failed;
        return error.Media;
    };
    defer timer.cancel(io);
    const tls_len = std.crypto.tls.Client.min_buffer_len;
    const bufs = try a.alloc(u8, 4 * tls_len);
    defer a.free(bufs);
    var sr = stream.reader(io, bufs[0..tls_len]);
    var sw = stream.writer(io, bufs[tls_len .. 2 * tls_len]);
    var entropy: [std.crypto.tls.Client.Options.entropy_len]u8 = undefined;
    io.random(&entropy);
    var tls = std.crypto.tls.Client.init(&sr.interface, &sw.interface, .{
        .host = .{ .explicit = std.mem.trimEnd(u8, u.host, ".") },
        .ca = .{ .bundle = .{ .gpa = std.heap.page_allocator, .io = io, .lock = &t.lock, .bundle = &t.bundle } },
        .read_buffer = bufs[2 * tls_len .. 3 * tls_len],
        .write_buffer = bufs[3 * tls_len ..],
        .entropy = &entropy,
        .realtime_now = Io.Clock.real.now(io),
        .allow_truncation_attacks = true,
    }) catch {
        f.text = failed;
        return error.Media;
    };
    const accept = try std.mem.join(a, ", ", media_types);
    defer a.free(accept);
    fetch.ask(&tls.writer, u, accept) catch {
        f.text = failed;
        return error.Media;
    };
    sw.interface.flush() catch {
        f.text = failed;
        return error.Media;
    };
    return fetch.answer(a, io, &tls.reader, max_bytes, deadline, media_types, f);
}

test "media TLS: simultaneous first initialization is serialized; failed CA load remains retryable" {
    const Fake = struct {
        calls: std.atomic.Value(u32) = .init(0),
        broken: bool = false,
        fn rescan(s: *@This(), _: *std.crypto.Certificate.Bundle, _: Io) !void {
            _ = s.calls.fetchAdd(1, .monotonic);
            if (s.broken) return error.NoCertificates;
        }
        fn first(s: *@This(), trust: *Trust, ok: *bool) void {
            var f: fetch.Failure = .{};
            trust.ensureVia(std.testing.io, &f, s) catch return;
            ok.* = true;
        }
    };
    var fake: Fake = .{ .broken = true };
    var trust: Trust = .{};
    var f: fetch.Failure = .{};
    try std.testing.expectError(error.Media, trust.ensureVia(std.testing.io, &f, &fake));
    try std.testing.expect(!trust.loaded);
    try std.testing.expectEqualStrings("image download failed: no CA certificates on the server", f.text);
    fake.broken = false;
    var ok: [4]bool = @splat(false);
    var threads: [4]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (&threads, &ok) |*thread, *success| {
        thread.* = try std.Thread.spawn(.{}, Fake.first, .{ &fake, &trust, success });
        started += 1;
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    for (ok) |success| try std.testing.expect(success);
    try std.testing.expect(trust.loaded);
    try std.testing.expectEqual(@as(u32, 2), fake.calls.load(.monotonic));
}

test "media TLS: silent first handshake and invalid TLS return errors within the socket deadline" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const io = std.testing.io;
    for ([_]bool{ false, true }) |invalid| {
        var sockets: [2]std.c.fd_t = undefined;
        if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &sockets) != 0) return error.SocketPair;
        defer _ = std.c.close(sockets[0]);
        defer _ = std.c.close(sockets[1]);
        if (invalid) {
            const junk = "HTTP/1.1 400 not TLS\r\n\r\n";
            try std.testing.expectEqual(@as(isize, junk.len), std.c.write(sockets[1], junk, junk.len));
        }
        const stream: Io.net.Stream = .{ .socket = .{ .handle = sockets[0], .address = try Io.net.IpAddress.parse("127.0.0.1", 443) } };
        var trust: Trust = .{};
        defer trust.bundle.deinit(std.heap.page_allocator);
        var f: fetch.Failure = .{};
        const start = Io.Clock.awake.now(io);
        try std.testing.expectError(error.Media, request(&trust, std.testing.allocator, io, stream, .{ .host = "example.test", .target = "/image" }, 1024, start.addDuration(.fromMilliseconds(100)), &fetch.image_media, &f));
        const elapsed = Io.Clock.awake.now(io).nanoseconds - start.nanoseconds;
        try std.testing.expect(elapsed < 5 * std.time.ns_per_s);
        if (!invalid) try std.testing.expect(elapsed >= 100 * std.time.ns_per_ms);
        try std.testing.expectEqualStrings("image download failed or timed out", f.text);
    }
}

test "media TLS: four simultaneous first TLS handshakes refuse malformed peers without a panic" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const Worker = struct {
        stream: Io.net.Stream,
        trust: *Trust,
        refused: bool = false,
        fn run(w: *@This()) void {
            var f: fetch.Failure = .{};
            _ = request(w.trust, std.testing.allocator, std.testing.io, w.stream, .{ .host = "example.test", .target = "/first" }, 1024, Io.Clock.awake.now(std.testing.io).addDuration(.fromMilliseconds(100)), &fetch.image_media, &f) catch |err| {
                w.refused = err == error.Media and std.mem.eql(u8, f.text, "image download failed or timed out");
                return;
            };
        }
    };
    var trust: Trust = .{};
    defer trust.bundle.deinit(std.heap.page_allocator);
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    var pairs: [4][2]std.c.fd_t = undefined;
    var opened: usize = 0;
    var started: usize = 0;
    defer {
        for (threads[0..started]) |thread| thread.join();
        for (pairs[0..opened]) |pair| for (pair) |fd| {
            _ = std.c.close(fd);
        };
    }
    for (&workers, &threads, &pairs) |*worker, *thread, *pair| {
        if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
        opened += 1;
        const junk = "HTTP/1.1 400 not TLS\r\n\r\n";
        try std.testing.expectEqual(@as(isize, junk.len), std.c.write(pair[1], junk, junk.len));
        worker.* = .{ .trust = &trust, .stream = .{ .socket = .{
            .handle = pair[0],
            .address = try Io.net.IpAddress.parse("127.0.0.1", 443),
        } } };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
        started += 1;
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    for (workers) |worker| try std.testing.expect(worker.refused);
}
