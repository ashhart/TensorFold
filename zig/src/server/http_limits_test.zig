//! Scripted slow readers exercise the real parser, body reader and connection-admission path.
const std = @import("std");
const platform = @import("sys");
const http = @import("http_conn.zig");
const body = @import("http_body.zig");
const server = @import("server.zig");
const admission = @import("http_admission.zig");
const json = @import("json");
const api = @import("engine_api");
const fields = @import("fields.zig");
const Value = json.Value;
const Cx = @import("errors.zig").Cx;
const MarkText = @import("mark_text.zig").MarkText;
const expand = @import("images.zig").expand;
const testing = std.testing;

test {
    _ = @import("png_limits_test.zig");
    _ = @import("completion_prompt_test.zig");
}

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

const strict_name = @import("core").resource_limits.switch_name;

const StrictEnv = struct {
    saved: ?[:0]u8,

    fn init(value: ?[:0]const u8) !StrictEnv {
        const saved = if (std.c.getenv(strict_name)) |v| try testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
        errdefer if (saved) |v| testing.allocator.free(v);
        try set(value);
        return .{ .saved = saved };
    }

    fn set(value: ?[:0]const u8) !void {
        const rc = if (value) |v| platform.env.setenv(strict_name, v, 1) else platform.env.unsetenv(strict_name);
        try testing.expectEqual(@as(c_int, 0), rc);
    }

    fn restore(s: StrictEnv) void {
        const rc = if (s.saved) |v| platform.env.setenv(strict_name, v, 1) else platform.env.unsetenv(strict_name);
        testing.expectEqual(@as(c_int, 0), rc) catch unreachable;
        if (s.saved) |v| testing.allocator.free(v);
    }
};

fn optionalSchema(a: std.mem.Allocator, count: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"type\":\"object\",\"properties\":{");
    for (0..count) |i| {
        if (i > 0) try out.append(a, ',');
        try out.print(a, "\"p{d}\":{{\"type\":\"string\"}}", .{i});
    }
    try out.appendSlice(a, "}}");
    return out.items;
}

test "HTTP limits: strict JSON rejects an expensive grammar schema before compilation" {
    const strict = try StrictEnv.init("1");
    defer strict.restore();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const parsed = try json.parse(arena.allocator(), try optionalSchema(arena.allocator(), 8000));
    try testing.expect(parsed == .err);
}

test "HTTP limits: strict JSON bounds recursion and individual string allocation" {
    const strict = try StrictEnv.init("1");
    defer strict.restore();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var nested: std.ArrayList(u8) = .empty;
    for (0..65) |_| try nested.append(a, '[');
    try nested.append(a, '0');
    for (0..65) |_| try nested.append(a, ']');
    try testing.expect((try json.parse(a, nested.items)) == .err);
    const text = try a.alloc(u8, (1 << 20) + 1);
    @memset(text, 'a');
    const quoted = try std.fmt.allocPrint(a, "\"{s}\"", .{text});
    try testing.expect((try json.parse(a, quoted)) == .err);
}

test "HTTP limits: strict stop and output bounds refuse excess but preserve their boundary" {
    const strict = try StrictEnv.init("1");
    defer strict.restore();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    var stops: std.ArrayList(u8) = .empty;
    try stops.appendSlice(a, "{\"stop\":[");
    for (0..17) |i| if (i > 0) try stops.appendSlice(a, ",\"x\"") else try stops.appendSlice(a, "\"x\"");
    try stops.appendSlice(a, "]}");
    try testing.expectError(error.Refused, fields.stopOptions(&cx, (try json.parse(a, stops.items)).ok));
    const stop_text = try a.alloc(u8, 257);
    @memset(stop_text, 'x');
    const long_stop = try std.fmt.allocPrint(a, "{{\"stop\":\"{s}\"}}", .{stop_text});
    try testing.expectError(error.Refused, fields.stopOptions(&cx, (try json.parse(a, long_stop)).ok));
    try testing.expectError(error.Refused, fields.parseNumbers(&cx, (try json.parse(a, "{\"max_tokens\":32769}")).ok));
    const accepted = try fields.parseNumbers(&cx, (try json.parse(a, "{\"max_tokens\":32768,\"stop\":\"ok\"}")).ok);
    try testing.expectEqualStrings("32768", accepted.get("max_tokens").?.int);
}

test "HTTP limits: strict switch is default-off and registered exactly once" {
    const strict = try StrictEnv.init(null);
    defer strict.restore();
    var env_count: usize = 0;
    for (@import("cli.zig").env) |name| env_count += @intFromBool(std.mem.eql(u8, name, strict_name));
    var registry_count: usize = 0;
    for (api.profile.entries) |entry| if (std.mem.eql(u8, entry.name(), strict_name)) {
        registry_count += 1;
        try testing.expectEqual(api.profile.Class.experimental, entry.class);
        try testing.expectEqual(api.profile.Syntax.one, entry.syntax);
        try testing.expect(entry.stable == null);
    };
    try testing.expectEqual(@as(usize, 1), env_count);
    try testing.expectEqual(@as(usize, 1), registry_count);
}

test "HTTP limits: unset strict switch keeps accepted stop and output values" {
    const strict = try StrictEnv.init(null);
    defer strict.restore();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const accepted = try fields.parseNumbers(&cx, (try json.parse(a, "{\"max_tokens\":32769,\"stop\":[\"a\",\"b\"]}")).ok);
    try testing.expectEqualStrings("32769", accepted.get("max_tokens").?.int);
    try testing.expectEqual(@as(usize, 2), (try fields.stopOptions(&cx, accepted)).strings.len);
}

fn discardQueueEvent(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

fn queueCase(strict: bool) !void {
    const gpa = testing.allocator;
    var cfg = try @import("lanes").Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: @import("lanes").fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: @import("lanes").fake.FixedClock = .{};
    var core = @import("lanes").Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, testing.io, &core, .{ .lanes = 1 });
    defer {
        for (host.queued.items) |job| gpa.destroy(job);
        host.queued.deinit(gpa);
    }
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 1 };
    var unused: u8 = 0;
    const sink: api.Sink = .{ .ctx = &unused, .event = discardQueueEvent };
    const engine = host.engine();
    for (0..128) |i| try engine.submit(@intCast(i + 1), &request, sink);
    if (strict) {
        try testing.expectError(error.Busy, engine.submit(129, &request, sink));
        try testing.expectEqual(@as(usize, 128), host.queued.items.len);
    } else {
        try engine.submit(129, &request, sink);
        try testing.expectEqual(@as(usize, 129), host.queued.items.len);
    }
}

test "HTTP limits: strict engine queue refuses excess while unset remains uncapped" {
    const plain = try StrictEnv.init(null);
    try queueCase(false);
    plain.restore();
    const strict = try StrictEnv.init("1");
    defer strict.restore();
    try queueCase(true);
}

fn longTextPair(a: std.mem.Allocator) ![2][]const u8 {
    const text = try a.alloc(u8, 600 << 10);
    @memset(text, 'x');
    return .{ text, text };
}

test "HTTP limits: strict chat preflight caps aggregate text but not image bytes" {
    const strict = try StrictEnv.init("1");
    defer strict.restore();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const texts = try longTextPair(a);
    const list = try a.alloc(Value, 2);
    for (texts, list) |text, *slot| {
        const message = try json.newObject(a);
        try message.put(a, "role", .{ .string = "user" });
        try message.put(a, "content", .{ .string = text });
        slot.* = .{ .object = message };
    }
    const raw: Value = .{ .array = list };
    try testing.expectError(error.Refused, @import("messages.zig").normalizeWith(&cx, raw, "system", false, .{}));
    const image_text = try a.alloc(u8, 2 << 20);
    @memset(image_text, 'a');
    const url = try std.fmt.allocPrint(a, "data:image/png,{s}", .{image_text});
    const image_url = try json.newObject(a);
    try image_url.put(a, "url", .{ .string = url });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "image_url" });
    try part.put(a, "image_url", .{ .object = image_url });
    const content = try a.alloc(Value, 1);
    content[0] = .{ .object = part };
    const message = try json.newObject(a);
    try message.put(a, "role", .{ .string = "user" });
    try message.put(a, "content", .{ .array = content });
    const valid = try a.alloc(Value, 1);
    valid[0] = .{ .object = message };
    _ = try @import("messages.zig").normalizeWith(&cx, .{ .array = valid }, "system", false, .{ .images = true, .max_images = 4 });
}

test "HTTP limits: strict completion preflight caps joined prompt source bytes" {
    const strict = try StrictEnv.init("1");
    defer strict.restore();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const texts = try longTextPair(a);
    const raw = try a.alloc(Value, texts.len);
    for (texts, raw) |text, *slot| slot.* = .{ .string = text };
    var srv: @import("server.zig").Server = undefined;
    try testing.expectError(error.Refused, @import("completion_prompt.zig").legacyPrompt(&srv, &cx, .{ .array = raw }));
}

fn countingPrepare(ctx: *anyopaque, a: std.mem.Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
    const calls: *u32 = @ptrCast(@alignCast(ctx));
    calls.* += 1;
    return .{ .pixels = try a.alloc(f32, 0), .gh = 1, .gw = 1, .tokens = 1, .hash = calls.* };
}

test "HTTP limits: strict decoded pixels admit exactly 32 MiPixels and refuse the third preparation" {
    const name = strict_name;
    const saved = if (std.c.getenv(name)) |v| try std.testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        _ = if (saved) |v| @import("sys").env.setenv(name, v, 1) else @import("sys").env.unsetenv(name);
        if (saved) |v| std.testing.allocator.free(v);
    }
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAIAAAAAgACAIAAAABmir1"; // signature and CRC-valid IHDR: 8192 x 2048
    const turn = "{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"" ++ url ++ "\"}}]}";
    const msgs = (try json.parse(a, "[" ++ turn ++ "," ++ turn ++ "," ++ turn ++ "]")).ok;
    const ids = [_]u32{ 9, 9, 9 };
    var calls: u32 = 0;
    var cx: Cx = .{ .a = a };
    const v: api.Vision = .{ .ctx = &calls, .prepare = countingPrepare, .image_token = 9, .image_tokens = 64, .max_images = 4 };
    for ([_]struct { strict: ?[:0]const u8, count: usize, refuse: bool }{
        .{ .strict = "1", .count = 2, .refuse = false },
        .{ .strict = "1", .count = 3, .refuse = true },
        .{ .strict = null, .count = 3, .refuse = false },
    }) |case| {
        _ = if (case.strict) |value| @import("sys").env.setenv(name, value, 1) else @import("sys").env.unsetenv(name);
        calls = 0;
        const result = expand(v, null, MarkText.text(), &cx, .{ .array = msgs.array[0..case.count] }, ids[0..case.count], 0);
        if (case.refuse) {
            try std.testing.expectError(error.Refused, result);
            try std.testing.expectEqual(@as(u32, 2), calls);
        } else {
            const x = (try result).?;
            try std.testing.expectEqual(case.count, x.images.len);
            try std.testing.expectEqual(@as(u32, @intCast(case.count)), calls);
        }
    }
}
