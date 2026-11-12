//! Scripted slow readers exercise the real parser, body reader and connection-admission path.
const std = @import("std");
const platform = @import("sys");
const http = @import("http_conn.zig");
const body = @import("http_body.zig");
const server = @import("server.zig");
const admission = @import("http_admission.zig");
const testing = std.testing;

const Slow = struct {
    fragments: []const []const u8,
    delays: []const i32,
    at: usize = 0,
    offset: usize = 0,
    ms: i64 = 0,
    waits: [32]i32 = @splat(0),
    calls: usize = 0,

    fn cast(ctx: *anyopaque) *Slow {
        return @ptrCast(@alignCast(ctx));
    }
    fn now(ctx: *anyopaque) i64 {
        return cast(ctx).ms;
    }
    fn read(ctx: *anyopaque, out: []u8, wait: i32) http.IoError!usize {
        const s = cast(ctx);
        if (s.at == s.fragments.len) return 0;
        s.waits[s.calls] = wait;
        s.calls += 1;
        const delay = if (s.offset == 0) s.delays[s.at] else 0;
        if (delay >= wait) {
            s.ms += wait;
            return error.Timeout;
        }
        s.ms += delay;
        const n = @min(out.len, s.fragments[s.at].len - s.offset);
        @memcpy(out[0..n], s.fragments[s.at][s.offset..][0..n]);
        s.offset += n;
        if (s.offset == s.fragments[s.at].len) {
            s.offset = 0;
            s.at += 1;
        }
        return n;
    }
    fn conn(s: *Slow) !http.Conn {
        var c = try http.Conn.init(testing.allocator, -1, "slow reader");
        c.quiet_log = true;
        c.read_hooks = .{ .ctx = s, .now = now, .read = read };
        c.timeouts = .{ .idle_ms = 900, .read_ms = 120, .request = .{ .head_ms = 100, .body_ms = 100, .body_bytes_per_s = 10 } };
        return c;
    }
};

test "HTTP limits: request-line drips cannot refresh the total head deadline" {
    var slow: Slow = .{ .fragments = &.{ "G", "E", "T", " /health HTTP/1.1\r\n\r\n" }, .delays = &.{ 30, 30, 30, 30 } };
    var c = try slow.conn();
    defer c.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(http.Outcome.closed, try c.readRequest(arena.allocator(), false));
    try testing.expectEqual(@as(i64, 100), slow.ms);
    try testing.expectEqual(@as(i32, 10), slow.waits[3]);
    try testing.expect(c.close);
}

test "HTTP limits: header lines share the request-line deadline and cannot become a ready request on timeout" {
    var slow: Slow = .{ .fragments = &.{ "GET /health HTTP/1.1\r\n", "A: 1\r\n", "B: 2\r\n", "\r\n" }, .delays = &.{ 40, 40, 40, 0 } };
    var c = try slow.conn();
    defer c.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(http.Outcome.handled, try c.readRequest(arena.allocator(), false));
    try testing.expect(c.close);
    try testing.expectEqual(@as(i64, 100), slow.ms);
}

test "HTTP limits: keep-alive idle wait is separate and first bytes start a fresh head budget" {
    var slow: Slow = .{ .fragments = &.{ "GET /health HTTP/1.1\r\n", "X: 1\r\n", "\r\n" }, .delays = &.{ 500, 60, 60 } };
    var c = try slow.conn();
    defer c.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(http.Outcome.handled, try c.readRequest(arena.allocator(), true));
    try testing.expectEqual(@as(i64, 600), slow.ms);
    try testing.expectEqual(@as(i32, 900), slow.waits[0]);
    try testing.expectEqual(@as(i32, 40), slow.waits[2]);
    try testing.expect(c.close);
}

fn fixed(slow: *Slow, length: []const u8) !http.Conn {
    var c = try slow.conn();
    c.headers = try testing.allocator.alloc(http.Header, 1);
    c.headers[0] = .{ .name = "Content-Length", .value = length };
    return c;
}

test "HTTP limits: trickling a length-scaled body expires even while every idle wait makes progress" {
    var slow: Slow = .{ .fragments = &.{ "12345", "67890", "12345", "67890" }, .delays = &.{ 600, 600, 600, 600 } };
    var c = try fixed(&slow, "20");
    defer c.deinit();
    defer testing.allocator.free(c.headers);
    c.timeouts.read_ms = 1000;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expect((try body.read(&c, arena.allocator(), body.limit)) == .refused);
    try testing.expectEqual(@as(i64, 2100), slow.ms);
    try testing.expectEqual(@as(i32, 300), slow.waits[3]);
    try testing.expect(c.close);
}

test "HTTP limits: an adequate declared upload completes and releases its timer before inference" {
    var slow: Slow = .{ .fragments = &.{ "1234567890", "1234567890" }, .delays = &.{ 900, 900 } };
    var c = try fixed(&slow, "20");
    defer c.deinit();
    defer testing.allocator.free(c.headers);
    c.timeouts.read_ms = 1000;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const result = try body.read(&c, arena.allocator(), body.limit);
    try testing.expectEqualStrings("12345678901234567890", result.ok);
    slow.ms += 1_000_000;
    try testing.expectEqual(@as(i32, 1000), try c.budget.wait(slow.ms, 1000));
}

test "HTTP limits: chunk metadata and trailers cannot reset the original body deadline" {
    var slow: Slow = .{ .fragments = &.{ "1\r\n", "x\r\n", "1\r\n", "y\r\n", "0\r\n", "\r\n" }, .delays = &.{ 60, 60, 60, 60, 60, 0 } };
    var c = try slow.conn();
    defer c.deinit();
    var headers = [_]http.Header{.{ .name = "Transfer-Encoding", .value = "chunked" }};
    c.headers = &headers;
    c.version = "HTTP/1.1";
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expect((try body.read(&c, arena.allocator(), body.limit)) == .refused);
    try testing.expectEqual(@as(i64, 300), slow.ms);
    try testing.expect(c.close);
}

fn pair() ![2]platform.posix.socket_t {
    var p: [2]platform.posix.socket_t = undefined;
    if (platform.windows) try platform.posix.socketpair(&p) else if (platform.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &p) != 0) return error.SocketPair;
    return p;
}

test "HTTP limits: production admission returns 503 and Retry-After without reserving an excess thread" {
    var srv: server.Server = undefined;
    srv.config = .{ .served_name = "test", .model_ids = &.{"test"}, .max_connections = 2 };
    srv.open_connections = .init(0);
    try testing.expect(srv.admitConnection(-1));
    try testing.expect(srv.admitConnection(-1));
    const p = try pair();
    defer _ = platform.posix.system.close(p[0]);
    const admitted = srv.admitConnection(p[1]);
    defer if (admitted) {
        _ = platform.posix.system.close(p[1]);
    };
    try testing.expect(!admitted);
    try testing.expectEqual(@as(u32, 2), srv.open_connections.load(.acquire));
    var buf: [256]u8 = undefined;
    var n: usize = 0;
    while (n < buf.len) {
        const got = try platform.posix.read(p[0], buf[n..]);
        if (got == 0) break;
        n += got;
    }
    try testing.expectEqualStrings(admission.response, buf[0..n]);
    _ = srv.open_connections.fetchSub(1, .acq_rel);
    try testing.expect(srv.admitConnection(-1));
    try testing.expectEqual(@as(u32, 2), srv.open_connections.load(.acquire));
}

const Race = struct {
    count: std.atomic.Value(u32) = .init(0),
    attempts: std.atomic.Value(u32) = .init(0),
    release: std.atomic.Value(bool) = .init(false),
    fn worker(r: *Race) void {
        const got = admission.acquire(&r.count, 3);
        _ = r.attempts.fetchAdd(1, .acq_rel);
        while (!r.release.load(.acquire)) std.atomic.spinLoopHint();
        if (got) _ = r.count.fetchSub(1, .acq_rel);
    }
};

test "HTTP limits: concurrent admission never exceeds the independent connection cap" {
    var race: Race = .{};
    var threads: [8]std.Thread = undefined;
    var spawned: usize = 0;
    defer {
        race.release.store(true, .release);
        for (threads[0..spawned]) |t| t.join();
    }
    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Race.worker, .{&race});
        spawned += 1;
    }
    while (race.attempts.load(.acquire) < 8) std.atomic.spinLoopHint();
    try testing.expectEqual(@as(u32, 3), race.count.load(.acquire));
    race.release.store(true, .release);
    for (threads[0..spawned]) |t| t.join();
    spawned = 0;
    try testing.expectEqual(@as(u32, 0), race.count.load(.acquire));
}

test "HTTP limits: successful keep-alive heads release and renew their budgets" {
    var slow: Slow = .{ .fragments = &.{ "GET /health HTTP/1.1\r\n\r\n", "GET /health HTTP/1.1\r\n\r\n" }, .delays = &.{ 90, 500 } };
    var c = try slow.conn();
    defer c.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(http.Outcome.ready, try c.readRequest(arena.allocator(), false));
    slow.ms += 1_000_000;
    try testing.expectEqual(@as(i32, 120), try c.budget.wait(slow.ms, 120));
    try testing.expectEqual(http.Outcome.ready, try c.readRequest(arena.allocator(), true));
    try testing.expect(!c.close);
}

test "HTTP limits: defaults and the maximum declared body have finite total budgets" {
    const Deadline = @import("http_deadline.zig");
    const policy: Deadline.Policy = .{};
    var budget: Deadline.Budget = .{};
    budget.head(0, false, policy);
    try testing.expectEqual(@as(i32, 30_000), try budget.wait(0, 120_000));
    budget.body(0, policy);
    budget.declared(body.limit, policy);
    try testing.expectEqual(@as(i64, 542_000), budget.end);
    try testing.expectError(error.Timeout, budget.wait(542_000, 120_000));
    try testing.expectEqual(@as(u32, 128), admission.default_limit);
}

test "security: one remote address holds at most its share of the connection slots" {
    const Ip = std.Io.net.IpAddress;
    var peers: admission.Peers = .{};
    const cap = admission.perPeer(8);
    try testing.expectEqual(@as(u32, 2), cap);
    const a: Ip = .{ .ip4 = .{ .bytes = .{ 192, 0, 2, 1 }, .port = 40001 } };
    const a_again: Ip = .{ .ip4 = .{ .bytes = .{ 192, 0, 2, 1 }, .port = 40002 } };
    const a_mapped: Ip = .{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 0, 2, 1 }, .port = 40003 } };
    const b: Ip = .{ .ip4 = .{ .bytes = .{ 192, 0, 2, 2 }, .port = 40001 } };
    const first = peers.take(testing.io, a, cap).?;
    const second = peers.take(testing.io, a_again, cap).?;
    try testing.expect(first != second);
    try testing.expectEqual(@as(?u16, null), peers.take(testing.io, a_again, cap));
    try testing.expectEqual(@as(?u16, null), peers.take(testing.io, a_mapped, cap));
    const other = peers.take(testing.io, b, cap).?;
    peers.give(testing.io, first);
    const freed = peers.take(testing.io, a_mapped, cap).?;
    try testing.expectEqual(@as(?u16, null), peers.take(testing.io, a, cap));
    for ([_]u16{ second, other, freed }) |slot| peers.give(testing.io, slot);
}

test "security: local clients share one address and are never limited by it" {
    var peers: admission.Peers = .{};
    const local: std.Io.net.IpAddress = .{ .ip4 = .loopback(40001) };
    for (0..admission.default_limit) |_| try testing.expectEqual(@as(?u16, admission.Peers.untracked), admission.admitPeer(&peers, testing.io, 128, 128, .{ .per_address = 1 }, local));
    peers.give(testing.io, admission.Peers.untracked);
}

fn doc(d: u8, port: u16) std.Io.net.IpAddress {
    return .{ .ip4 = .{ .bytes = .{ 192, 0, 2, d }, .port = port } };
}

test "security: below pressure a 33rd connection from one address is accepted; at pressure it is refused" {
    var peers: admission.Peers = .{};
    var slots: [34]u16 = undefined;
    // a reverse proxy, Docker's bridge gateway or an office NAT: 33 connections with 95 of 128 open, none refused
    for (slots[0..33], 0..) |*slot, i| slot.* = admission.admitPeer(&peers, testing.io, 95, 128, .{}, doc(1, @intCast(40000 + i))) orelse return error.Refused;
    // from 96 of 128 the address's share (32) applies to its next connection, and only to that address
    try testing.expectEqual(@as(?u16, null), admission.admitPeer(&peers, testing.io, 96, 128, .{}, doc(1, 41000)));
    slots[33] = admission.admitPeer(&peers, testing.io, 96, 128, .{}, doc(2, 41000)).?;
    // its 33 open connections stay open: nothing in flight is shed
    try testing.expectEqual(@as(?u16, null), admission.admitPeer(&peers, testing.io, 128, 128, .{}, doc(1, 41001)));
    // 0 turns the share off; another share is taken as set
    const unlimited = admission.admitPeer(&peers, testing.io, 128, 128, .{ .per_address = 0 }, doc(1, 41002)).?;
    peers.give(testing.io, unlimited);
    try testing.expectEqual(@as(?u16, null), admission.admitPeer(&peers, testing.io, 100, 128, .{ .per_address = 1 }, doc(2, 41001)));
    for (slots) |slot| peers.give(testing.io, slot);
    try testing.expectEqual(@as(?u32, null), (admission.Peering{}).cap(95, 128));
    try testing.expectEqual(@as(?u32, 32), (admission.Peering{}).cap(96, 128));
}

test "security: a trusted proxy is never limited by its address" {
    var peers: admission.Peers = .{};
    const proxy = doc(9, 0);
    const v6: std.Io.net.IpAddress = .{ .ip6 = .{ .bytes = .{ 0xfd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } };
    const rule: admission.Peering = .{ .per_address = 1, .trusted = &.{ proxy, v6 } };
    for (0..200) |i| {
        try testing.expectEqual(@as(?u16, admission.Peers.untracked), admission.admitPeer(&peers, testing.io, 128, 128, rule, doc(9, @intCast(50000 + i))));
        var at = v6;
        at.ip6.port = @intCast(50000 + i);
        try testing.expectEqual(@as(?u16, admission.Peers.untracked), admission.admitPeer(&peers, testing.io, 128, 128, rule, at));
    }
    // the list names exact addresses: a neighbour of the proxy is counted
    const next = admission.admitPeer(&peers, testing.io, 128, 128, rule, doc(10, 0)).?;
    try testing.expectEqual(@as(?u16, null), admission.admitPeer(&peers, testing.io, 128, 128, rule, doc(10, 1)));
    peers.give(testing.io, next);
}

test "security: IPv6 peers are counted by their /64" {
    var peers: admission.Peers = .{};
    const Ip = std.Io.net.IpAddress;
    const a: Ip = .{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 1 } };
    const same_64: Ip = .{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 1, 0, 0, 0xde, 0xad, 0xbe, 0xef, 0, 0, 0, 2 }, .port = 2 } };
    const next_64: Ip = .{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 3 } };
    try testing.expectEqualSlices(u8, &admission.key(a), &admission.key(same_64));
    const first = peers.take(testing.io, a, 1).?;
    try testing.expectEqual(@as(?u16, null), peers.take(testing.io, same_64, 1));
    const other = peers.take(testing.io, next_64, 1).?;
    // IPv4 keeps its whole address, in either form
    try testing.expect(!std.mem.eql(u8, &admission.key(doc(1, 0)), &admission.key(doc(2, 0))));
    for ([_]u16{ first, other }) |slot| peers.give(testing.io, slot);
}

test "security: an idle connection gives its slot up only with every slot taken" {
    var open: std.atomic.Value(u32) = .init(127);
    const shed: admission.Shed = .{ .open = &open, .limit = 128 };
    // a pooled keep-alive client (loopback or not) keeps its connection below the limit, however long it waits
    try testing.expect(!shed.due(60_000));
    open.store(96, .release);
    try testing.expect(!shed.due(60_000));
    open.store(128, .release);
    try testing.expect(!shed.due(admission.shed_after_ms - 1));
    try testing.expect(shed.due(admission.shed_after_ms));
    try testing.expectEqual(@as(u32, 96), admission.pressure(admission.default_limit));
    try testing.expectEqual(@as(u32, 32), admission.perPeer(admission.default_limit));
    try testing.expectEqual(@as(u32, 1), admission.perPeer(2));
}
