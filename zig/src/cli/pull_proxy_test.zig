//! `pull` through a fake proxy and a fake hub on 127.0.0.1: tunnels per hop, NO_PROXY, refusals and stalled bodies.
const std = @import("std");
const pull = @import("pull.zig");
const TestClient = @import("pull_test_client.zig").Client;
const Io = std.Io;
const net = Io.net;

/// A loopback listener whose connections each run `handle` on their own thread.
fn Server(comptime Ctx: type) type {
    return struct {
        threaded: Io.Threaded,
        server: net.Server,
        stop: std.atomic.Value(bool) = .init(false),
        thread: ?std.Thread = null,
        workers: [64]std.Thread = undefined,
        n: usize = 0,
        ctx: *Ctx,

        const S = @This();

        fn start(ctx: *Ctx) !*S {
            const s = try std.heap.page_allocator.create(S);
            errdefer std.heap.page_allocator.destroy(s);
            s.* = .{ .threaded = .init(std.heap.page_allocator, .{}), .server = undefined, .ctx = ctx };
            const any: net.IpAddress = .{ .ip4 = .loopback(0) };
            s.server = try any.listen(s.threaded.io(), .{});
            s.thread = try std.Thread.spawn(.{}, loop, .{s});
            return s;
        }

        fn io(s: *S) Io {
            return s.threaded.io();
        }

        fn port(s: *S) u16 {
            return s.server.socket.address.getPort();
        }

        fn loop(s: *S) void {
            while (!s.stop.load(.acquire)) {
                const conn = s.server.accept(s.io()) catch return;
                if (s.stop.load(.acquire) or s.n == s.workers.len) {
                    conn.close(s.io());
                    continue;
                }
                s.workers[s.n] = std.Thread.spawn(.{}, work, .{ s, conn }) catch {
                    conn.close(s.io());
                    continue;
                };
                s.n += 1;
            }
        }

        fn work(s: *S, conn: net.Stream) void {
            defer conn.close(s.io());
            Ctx.handle(s.ctx, s.io(), conn);
        }

        fn finish(s: *S) void {
            s.stop.store(true, .release);
            if (s.server.socket.address.connect(s.io(), .{ .mode = .stream }) catch null) |wake| wake.close(s.io());
            if (s.thread) |t| t.join();
            for (s.workers[0..s.n]) |t| t.join();
            s.server.deinit(s.io());
            s.threaded.deinit();
            std.heap.page_allocator.destroy(s);
        }
    };
}

/// Reads one request head; null when the peer closes or sends more than `buf` holds first.
fn readHead(io: Io, conn: net.Stream, buf: []u8) ?[]const u8 {
    var end: usize = 0;
    var rbuf: [1]u8 = undefined; // one byte at a time: nothing past the head is taken from a tunnel
    var r = conn.reader(io, &rbuf);
    while (end < buf.len) {
        buf[end] = r.interface.takeByte() catch return null;
        end += 1;
        if (end >= 4 and std.mem.eql(u8, buf[end - 4 .. end], "\r\n\r\n")) return buf[0..end];
    }
    return null;
}

fn header(head: []const u8, name: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, head, "\r\n");
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(line[0..colon], name)) return std.mem.trim(u8, line[colon + 1 ..], " ");
    }
    return null;
}

fn send(io: Io, conn: net.Stream, bytes: []const u8) void {
    var w = conn.writer(io, &.{});
    w.interface.writeAll(bytes) catch {};
}

const sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1";
const config = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 262144}";
var weights_storage: [4000]u8 = @splat(0x3C);
const weights: []const u8 = &weights_storage;

/// The hub: revision, tree, config.json, and weights that may redirect to `cdn` or stall after 500 bytes.
const Hub = struct {
    cdn: ?[]const u8 = null,
    stalls: std.atomic.Value(u32) = .init(0),
    failures: std.atomic.Value(u32) = .init(0),
    resumed_at: std.atomic.Value(u64) = .init(0),
    requests: std.atomic.Value(u32) = .init(0),

    fn handle(h: *Hub, io: Io, conn: net.Stream) void {
        var buf: [4096]u8 = undefined;
        const head = readHead(io, conn, &buf) orelse return;
        _ = h.requests.fetchAdd(1, .monotonic);
        const line_end = std.mem.indexOf(u8, head, "\r\n").?;
        var parts = std.mem.tokenizeScalar(u8, head[0..line_end], ' ');
        _ = parts.next();
        const target = parts.next() orelse return;
        const path = if (std.mem.startsWith(u8, target, "http://")) target[std.mem.indexOfScalarPos(u8, target, 7, '/').?..] else target;
        var out: [1024]u8 = undefined;
        if (std.mem.eql(u8, path, "/api/models/Org/Tiny/revision/main")) return reply(io, conn, "{\"sha\": \"" ++ sha ++ "\"}");
        if (std.mem.startsWith(u8, path, "/api/models/Org/Tiny/tree/" ++ sha)) {
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(weights, &digest, .{});
            var git = std.crypto.hash.Sha1.init(.{});
            git.update(std.fmt.comptimePrint("blob {d}\x00", .{config.len}));
            git.update(config);
            const tree = std.fmt.bufPrint(&out, "[{{\"type\":\"file\",\"path\":\"config.json\",\"size\":{d},\"oid\":\"{s}\"}},{{\"type\":\"file\",\"path\":\"w.safetensors\",\"size\":{d},\"lfs\":{{\"oid\":\"{s}\",\"size\":{d}}}}}]", .{
                config.len, std.fmt.bytesToHex(git.finalResult(), .lower), weights.len, std.fmt.bytesToHex(digest, .lower), weights.len,
            }) catch return;
            return reply(io, conn, tree);
        }
        if (std.mem.eql(u8, path, "/Org/Tiny/resolve/" ++ sha ++ "/config.json")) return reply(io, conn, config);
        if (std.mem.eql(u8, path, "/Org/Tiny/resolve/" ++ sha ++ "/w.safetensors")) if (h.cdn) |cdn| {
            const host = header(head, "host").?;
            const port = host[std.mem.lastIndexOfScalar(u8, host, ':').? + 1 ..];
            const moved = std.fmt.bufPrint(&out, "HTTP/1.1 302 Found\r\nLocation: http://{s}:{s}/cdn/w.safetensors\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{ cdn, port }) catch return;
            return send(io, conn, moved);
        } else return h.body(io, conn, head);
        if (std.mem.eql(u8, path, "/cdn/w.safetensors")) return h.body(io, conn, head);
        send(io, conn, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    }

    fn body(h: *Hub, io: Io, conn: net.Stream, head: []const u8) void {
        var from: u64 = 0;
        if (header(head, "range")) |r| from = std.fmt.parseInt(u64, r["bytes=".len .. r.len - 1], 10) catch 0;
        if (from > 0) h.resumed_at.store(from, .monotonic);
        var out: [256]u8 = undefined;
        const status = if (from > 0) "206 Partial Content" else "200 OK";
        const head_out = std.fmt.bufPrint(&out, "HTTP/1.1 {s}\r\nContent-Length: {d}\r\nContent-Range: bytes {d}-{d}/{d}\r\nConnection: close\r\n\r\n", .{
            status, weights.len - from, from, weights.len - 1, weights.len,
        }) catch return;
        send(io, conn, head_out);
        const failure = h.failures.load(.monotonic) > 0 and h.failures.fetchSub(1, .monotonic) > 0;
        if (failure) return send(io, conn, weights[from..@min(weights.len - 1, from + 500)]);
        const stall = h.stalls.load(.monotonic) > 0 and h.stalls.fetchSub(1, .monotonic) > 0;
        if (!stall) return send(io, conn, weights[from..]);
        send(io, conn, weights[from..@min(weights.len - 1, from + 500)]);
        io.sleep(.fromMilliseconds(2500), .awake) catch {}; // longer than the 1 s stall timeout
    }

    fn reply(io: Io, conn: net.Stream, text: []const u8) void {
        var out: [128]u8 = undefined;
        send(io, conn, std.fmt.bufPrint(&out, "HTTP/1.1 200 OK\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{text.len}) catch return);
        send(io, conn, text);
    }
};

/// The proxy: records each CONNECT and its Proxy-Authorization, then refuses with `status` or pipes to 127.0.0.1.
const Proxy = struct {
    status: u16 = 200,
    lock: std.Io.Mutex = .init,
    seen: [16][96]u8 = undefined,
    seen_len: [16]usize = @splat(0),
    count: usize = 0,
    auth_ok: bool = true,
    want_auth: ?[]const u8 = null,

    fn handle(p: *Proxy, io: Io, conn: net.Stream) void {
        var buf: [2048]u8 = undefined;
        const head = readHead(io, conn, &buf) orelse return;
        const line = head[0..std.mem.indexOf(u8, head, "\r\n").?];
        p.lock.lockUncancelable(io);
        if (p.count < p.seen.len) {
            const n = @min(line.len, 96);
            @memcpy(p.seen[p.count][0..n], line[0..n]);
            p.seen_len[p.count] = n;
            p.count += 1;
        }
        const auth = header(head, "proxy-authorization");
        if (p.want_auth) |want| p.auth_ok = p.auth_ok and auth != null and std.mem.eql(u8, auth.?, want);
        p.lock.unlock(io);
        if (p.status != 200) {
            var out: [96]u8 = undefined;
            return send(io, conn, std.fmt.bufPrint(&out, "HTTP/1.1 {d} No\r\nContent-Length: 0\r\n\r\n", .{p.status}) catch return);
        }
        const target = line["CONNECT ".len..std.mem.lastIndexOfScalar(u8, line, ' ').?];
        const port = std.fmt.parseInt(u16, target[std.mem.lastIndexOfScalar(u8, target, ':').? + 1 ..], 10) catch return;
        const to: net.IpAddress = .{ .ip4 = .loopback(port) };
        const up = to.connect(io, .{ .mode = .stream }) catch return;
        defer up.close(io);
        send(io, conn, "HTTP/1.1 200 Connection established\r\n\r\n");
        const back = std.Thread.spawn(.{}, pump, .{ io, up, conn }) catch return;
        pump(io, conn, up);
        back.join();
    }

    fn pump(io: Io, from: net.Stream, to: net.Stream) void {
        var rbuf: [16 << 10]u8 = undefined;
        var r = from.reader(io, &rbuf);
        var w = to.writer(io, &.{});
        while (true) {
            var chunk: [16 << 10]u8 = undefined;
            var vecs = [_][]u8{&chunk};
            const n = r.interface.readVec(&vecs) catch break;
            if (n == 0) continue;
            w.interface.writeAll(chunk[0..n]) catch break;
        }
        to.shutdown(io, .both) catch {};
    }

    fn saw(p: *Proxy, line: []const u8) bool {
        for (0..p.count) |i| if (std.mem.eql(u8, p.seen[i][0..p.seen_len[i]], line)) return true;
        return false;
    }
};

const Run = struct { code: u8, out: []const u8, err: []const u8, root: []const u8 };

fn runPull(a: std.mem.Allocator, io: Io, pairs: []const [2][]const u8) !Run {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    for (pairs) |p| try env.put(p[0], p[1]);
    const root = try std.fmt.allocPrint(a, ".tf-pull-proxy-test-{d}", .{Io.Clock.awake.now(io).toNanoseconds()});
    var out: Io.Writer.Allocating = .init(a);
    var err: Io.Writer.Allocating = .init(a);
    const code = try pull.run(a, io, &out.writer, &err.writer, &env, root, "Org/Tiny");
    return .{ .code = code, .out = out.written(), .err = err.written(), .root = root };
}

fn weightsLanded(a: std.mem.Allocator, io: Io, root: []const u8) !void {
    const link = try std.fs.path.join(a, &.{ root, "models--Org--Tiny/snapshots", sha, "w.safetensors" });
    try std.testing.expectEqualSlices(u8, weights, try Io.Dir.cwd().readFileAlloc(io, link, a, .limited(1 << 20)));
}

fn closedPort(io: Io) !u16 {
    const any: net.IpAddress = .{ .ip4 = .loopback(0) };
    var s = try any.listen(io, .{});
    defer s.deinit(io);
    return s.socket.address.getPort();
}

fn noSecret(r: Run) !void {
    try std.testing.expect(std.mem.indexOf(u8, r.out, "s3cret") == null and std.mem.indexOf(u8, r.err, "s3cret") == null);
    try std.testing.expect(std.mem.indexOf(u8, r.out, "ana:") == null and std.mem.indexOf(u8, r.err, "ana:") == null);
}

test "pull proxy: http_proxy carries the hub request and the storage redirect hop, beating HTTP_PROXY" {
    var client: TestClient = undefined;
    client.init();
    defer client.deinit();
    var arena = std.heap.ArenaAllocator.init(client.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client.io();
    var hub: Hub = .{ .cdn = "cdn.test" };
    const hs = try Server(Hub).start(&hub);
    defer hs.finish();
    var proxy: Proxy = .{ .want_auth = "Basic YW5hOnMzY3JldA==" };
    const ps = try Server(Proxy).start(&proxy);
    defer ps.finish();
    const r = try runPull(a, io, &.{
        .{ "HF_ENDPOINT", try std.fmt.allocPrint(a, "http://hub.test:{d}", .{hs.port()}) },
        .{ "HTTP_PROXY", try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{try closedPort(io)}) },
        .{ "http_proxy", try std.fmt.allocPrint(a, "http://ana:s3cret@127.0.0.1:{d}", .{ps.port()}) },
    });
    defer Io.Dir.cwd().deleteTree(io, r.root) catch {};
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try weightsLanded(a, io, r.root);
    try std.testing.expect(proxy.saw(try std.fmt.allocPrint(a, "CONNECT hub.test:{d} HTTP/1.1", .{hs.port()})));
    try std.testing.expect(proxy.saw(try std.fmt.allocPrint(a, "CONNECT cdn.test:{d} HTTP/1.1", .{hs.port()})));
    try std.testing.expect(proxy.auth_ok);
    try noSecret(r);
}

test "pull proxy: HTTPS_PROXY carries an https hub request, and a refusal says which proxy and why" {
    var client: TestClient = undefined;
    client.init();
    defer client.deinit();
    var arena = std.heap.ArenaAllocator.init(client.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client.io();
    for ([_]struct { u16, []const u8 }{ .{ 403, "refused the tunnel (HTTP 403)" }, .{ 407, "wants credentials it did not accept (HTTP 407)" } }) |c| {
        var https_proxy: Proxy = .{ .status = c[0] };
        const ps = try Server(Proxy).start(&https_proxy);
        defer ps.finish();
        var http_proxy: Proxy = .{};
        const other = try Server(Proxy).start(&http_proxy);
        defer other.finish();
        const r = try runPull(a, io, &.{
            .{ "HF_ENDPOINT", "https://hub.test:8443" },
            .{ "HTTPS_PROXY", try std.fmt.allocPrint(a, "http://ana:s3cret@127.0.0.1:{d}", .{ps.port()}) },
            .{ "HTTP_PROXY", try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{other.port()}) },
        });
        defer Io.Dir.cwd().deleteTree(io, r.root) catch {};
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(https_proxy.saw("CONNECT hub.test:8443 HTTP/1.1"));
        try std.testing.expectEqual(@as(usize, 0), http_proxy.count);
        const name = if (@import("builtin").os.tag == .windows) "https_proxy" else "HTTPS_PROXY";
        const says = try std.fmt.allocPrint(a, "the proxy http://127.0.0.1:{d} (from {s}) {s}", .{ ps.port(), name, c[1] });
        if (std.mem.indexOf(u8, r.err, says) == null) {
            std.debug.print("pull said: {s}\n", .{r.err});
            return error.TestUnexpectedResult;
        }
        try noSecret(r);
    }
}

test "pull proxy: NO_PROXY sends the hub direct" {
    var client: TestClient = undefined;
    client.init();
    defer client.deinit();
    var arena = std.heap.ArenaAllocator.init(client.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client.io();
    var hub: Hub = .{};
    const hs = try Server(Hub).start(&hub);
    defer hs.finish();
    var proxy: Proxy = .{};
    const ps = try Server(Proxy).start(&proxy);
    defer ps.finish();
    const r = try runPull(a, io, &.{
        .{ "HF_ENDPOINT", try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{hs.port()}) },
        .{ "HTTP_PROXY", try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{ps.port()}) },
        .{ "no_proxy", "example.org, 127.0.0.1" },
    });
    defer Io.Dir.cwd().deleteTree(io, r.root) catch {};
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try weightsLanded(a, io, r.root);
    try std.testing.expectEqual(@as(usize, 0), proxy.count);
}

test "pull proxy: a stalled body times out and the .incomplete file resumes with a range" {
    var client: TestClient = undefined;
    client.init();
    defer client.deinit();
    var arena = std.heap.ArenaAllocator.init(client.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client.io();
    var hub: Hub = .{};
    hub.stalls.store(1, .monotonic);
    const hs = try Server(Hub).start(&hub);
    defer hs.finish();
    const r = try runPull(a, io, &.{
        .{ "HF_ENDPOINT", try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{hs.port()}) },
        .{ "TENSORFOLD_PULL_STALL_SECONDS", "1" },
    });
    defer Io.Dir.cwd().deleteTree(io, r.root) catch {};
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try weightsLanded(a, io, r.root);
    try std.testing.expectEqual(@as(u64, 500), hub.resumed_at.load(.monotonic));
}
test "pull download: a close-delimited read failure persists buffered bytes before range resume" {
    var client: TestClient = undefined;
    client.init();
    defer client.deinit();
    var arena = std.heap.ArenaAllocator.init(client.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client.io();
    var hub: Hub = .{};
    hub.failures.store(1, .monotonic);
    const hs = try Server(Hub).start(&hub);
    defer hs.finish();
    const r = try runPull(a, io, &.{
        .{ "HF_ENDPOINT", try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{hs.port()}) },
    });
    defer Io.Dir.cwd().deleteTree(io, r.root) catch {};
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try weightsLanded(a, io, r.root);
    try std.testing.expectEqual(@as(u64, 500), hub.resumed_at.load(.monotonic));
}

test "pull proxy: a body that keeps stalling fails after three resumes, naming the timeout and the proxy" {
    var client: TestClient = undefined;
    client.init();
    defer client.deinit();
    var arena = std.heap.ArenaAllocator.init(client.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client.io();
    var hub: Hub = .{};
    hub.stalls.store(100, .monotonic);
    const hs = try Server(Hub).start(&hub);
    defer hs.finish();
    var proxy: Proxy = .{};
    const ps = try Server(Proxy).start(&proxy);
    defer ps.finish();
    const r = try runPull(a, io, &.{
        .{ "HF_ENDPOINT", try std.fmt.allocPrint(a, "http://hub.test:{d}", .{hs.port()}) },
        .{ "http_proxy", try std.fmt.allocPrint(a, "http://ana:s3cret@127.0.0.1:{d}", .{ps.port()}) },
        .{ "TENSORFOLD_PULL_STALL_SECONDS", "1" },
    });
    defer Io.Dir.cwd().deleteTree(io, r.root) catch {};
    try std.testing.expectEqual(@as(u8, 1), r.code);
    const says = try std.fmt.allocPrint(a, "w.safetensors: no bytes arrived for 1 s (TENSORFOLD_PULL_STALL_SECONDS), and 3 range resumes stalled too; http through http://127.0.0.1:{d} (from http_proxy)", .{ps.port()});
    if (std.mem.indexOf(u8, r.err, says) == null) {
        std.debug.print("pull said: {s}\n", .{r.err});
        return error.TestUnexpectedResult;
    }
    try std.testing.expectEqual(@as(u64, 1500), hub.resumed_at.load(.monotonic));
    try noSecret(r);
}
