//! Media lookup and failover tests use a fake resolver and connector, with no external network.
const std = @import("std");
const Io = std.Io;
const net = Io.net;
const fetch = @import("media_fetch.zig");
const transport = @import("media_fetch_net.zig");
const Failure = fetch.Failure;
const Url = fetch.Url;

/// One host resolved through a lookup that gives `addresses`, on a thread its test can stop waiting for.
const Canned = struct {
    var addresses: []const net.IpAddress = &.{};
    /// The lookup fails once it has given its addresses.
    var fails = false;
    /// A delay after the addresses that ignores cancellation, as libc's getaddrinfo retries through the cancel signal.
    var delay_ms: i64 = 0;
    var active: std.atomic.Value(bool) = .init(false);
    var lookups: std.atomic.Value(u32) = .init(0);
    var table: Io.VTable = undefined;
    var failure: Failure = .{};
    var first: ?net.IpAddress = null;
    var done: std.atomic.Value(bool) = .init(false);

    /// Set when a test stops waiting, so a lookup left with no reader ends with its test.
    var abandoned: std.atomic.Value(bool) = .init(false);

    /// As std's lookup on a libc host: each address queued in turn, however many, waiting while the queue is full.
    fn lookup(_: ?*anyopaque, _: net.HostName, q: *Io.Queue(net.HostName.LookupResult), _: net.HostName.LookupOptions) net.HostName.LookupError!void {
        const io = std.testing.io;
        active.store(true, .release);
        defer active.store(false, .release);
        _ = lookups.fetchAdd(1, .monotonic);
        defer q.close(io);
        for (addresses) |addr| while ((q.put(io, &.{.{ .address = addr }}, 0) catch return error.NameServerFailure) == 0) {
            if (abandoned.load(.acquire)) return error.NameServerFailure;
            io.sleep(.fromMilliseconds(1), .awake) catch return error.NameServerFailure;
        };
        if (delay_ms > 0) {
            const end = Io.Clock.awake.now(io).addDuration(.fromMilliseconds(delay_ms));
            while (Io.Clock.awake.now(io).nanoseconds < end.nanoseconds) io.sleep(.fromMilliseconds(5), .awake) catch {};
        }
        if (fails) return error.NameServerFailure;
    }

    fn system() Io {
        const real = std.testing.io;
        table = real.vtable.*;
        table.netLookup = lookup;
        table.async = Io.noAsync; // a busy Io runs an async task in place, where the lookup would have no reader
        return .{ .userdata = real.userdata, .vtable = &table };
    }

    fn work(whole_request: bool) void {
        const io = system();
        const url: Url = .{ .host = "many.example", .target = "/" };
        failure = .{};
        first = null;
        const far = Io.Clock.awake.now(io).addDuration(.fromSeconds(10));
        if (whole_request) {
            var connector: Connector = .{};
            _ = transport.openVia(std.testing.allocator, io, url.host, far, &failure, &connector) catch {};
        } else {
            const resolved = transport.resolve(std.testing.allocator, io, url.host, far, &failure) catch {
                done.store(true, .release);
                return;
            };
            defer std.testing.allocator.free(resolved);
            first = resolved[0];
        }
        done.store(true, .release);
    }

    /// Whether the host resolves or is refused within five seconds: `resolve` alone, or a whole request for the URL.
    fn finishes(whole_request: bool) !bool {
        done.store(false, .release);
        abandoned.store(false, .release);
        const thread = try std.Thread.spawn(.{}, work, .{whole_request});
        defer thread.join();
        for (0..500) |_| {
            if (done.load(.acquire)) return true;
            std.testing.io.sleep(.fromMilliseconds(10), .awake) catch {};
        }
        abandoned.store(true, .release);
        return false;
    }
};

test "security: a host with more addresses than the lookup's queue holds resolves, each address still checked" {
    const inner = "image URLs must resolve only to public internet addresses";
    const unresolved = "image host could not be resolved";
    var many: [40]net.IpAddress = undefined;
    for (&many, 0..) |*addr, i| addr.* = .{ .ip4 = .{ .bytes = .{ 8, 8, 4, @intCast(i + 1) }, .port = 443 } };
    Canned.fails = false;
    // 32 addresses fill the queue; a 33rd used to wait forever, its reader being the thread that was waiting
    for ([_]usize{ 1, 32, 33, 40 }) |n| {
        Canned.addresses = many[0..n];
        try std.testing.expect(try Canned.finishes(false));
        try std.testing.expect(Canned.first.?.eql(&many[0]));
    }
    // one address the policy refuses, refusing the request: past the queue's end, then first with 39 still to come
    for ([_]usize{ 39, 0 }) |at| {
        const public = many[at];
        defer many[at] = public;
        many[at] = .{ .ip4 = .{ .bytes = .{ 10, 0, 0, 1 }, .port = 443 } };
        for ([_]bool{ true, false }) |whole_request| {
            try std.testing.expect(try Canned.finishes(whole_request));
            try std.testing.expect(Canned.first == null);
            try std.testing.expectEqualStrings(inner, Canned.failure.text);
        }
    }
    // a lookup that fails after giving public addresses, and one that gives none, resolve nothing
    Canned.addresses = many[0..3];
    Canned.fails = true;
    try std.testing.expect(try Canned.finishes(false));
    try std.testing.expect(Canned.first == null);
    try std.testing.expectEqualStrings(unresolved, Canned.failure.text);
    Canned.addresses = &.{};
    Canned.fails = false;
    try std.testing.expect(try Canned.finishes(false));
    try std.testing.expect(Canned.first == null);
    try std.testing.expectEqualStrings(unresolved, Canned.failure.text);
}

/// A connector that records attempts and returns a stand-in stream; it never opens a socket.
const Connector = struct {
    prepared: bool = false,
    calls: usize = 0,
    refuse: usize = 0,
    delay_ms: i64 = 0,
    attempted: [4]net.IpAddress = undefined,
    budgets: [4]i64 = undefined,

    pub fn prepare(c: *@This(), _: Io, _: *Failure) fetch.Error!void {
        c.prepared = true;
    }

    pub fn connect(c: *@This(), addr: net.IpAddress, ms: i64, f: *Failure) fetch.Error!net.Stream {
        std.debug.assert(c.calls < c.attempted.len);
        c.attempted[c.calls] = addr;
        c.budgets[c.calls] = ms;
        c.calls += 1;
        if (c.delay_ms > 0) std.testing.io.sleep(.fromMilliseconds(c.delay_ms), .awake) catch {};
        if (c.calls <= c.refuse) {
            f.text = "image download failed or timed out";
            return error.Media;
        }
        return .{ .socket = .{ .handle = undefined, .address = addr } };
    }
};

fn publicAddresses() [3]net.IpAddress {
    return .{
        .{ .ip4 = .{ .bytes = .{ 8, 8, 8, 8 }, .port = 443 } },
        net.IpAddress.parseIp6("2606:4700::1111", 443) catch unreachable,
        .{ .ip4 = .{ .bytes = .{ 1, 1, 1, 1 }, .port = 443 } },
    };
}

/// Waits for lookups a request stopped waiting for to end on their own threads and free their state.
fn lookupsEnd() !void {
    for (0..1_000) |_| {
        if (transport.lookups.load(.acquire) == 0) return;
        std.testing.io.sleep(.fromMilliseconds(10), .awake) catch {};
    }
    return error.LookupLeftRunning;
}

test "media lookup: a slow lookup that ignores cancellation, as getaddrinfo does, still ends at the deadline" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const addresses = publicAddresses();
    Canned.addresses = &addresses;
    Canned.fails = false;
    Canned.delay_ms = 4_000;
    defer Canned.delay_ms = 0;
    const io = Canned.system();
    for ([_]i64{ 100, 5_000 }) |budget_ms| {
        var connector: Connector = .{};
        var f: Failure = .{};
        const start = Io.Clock.awake.now(io);
        const deadline = start.addDuration(.fromMilliseconds(budget_ms));
        try std.testing.expectError(error.Media, transport.openVia(std.testing.allocator, io, "slow.example", deadline, &f, &connector));
        const elapsed = Io.Clock.awake.now(io).nanoseconds - start.nanoseconds;
        const limit_ms = @min(budget_ms, transport.lookup_timeout_s * 1000);
        try std.testing.expect(elapsed >= limit_ms * std.time.ns_per_ms);
        try std.testing.expect(elapsed < (limit_ms + 700) * std.time.ns_per_ms);
        try std.testing.expectEqualStrings("image host lookup timed out", f.text);
        try std.testing.expect(!connector.prepared);
        try std.testing.expectEqual(@as(usize, 0), connector.calls);
    }
    // the abandoned lookups end on their own and free what they hold
    try lookupsEnd();
    try std.testing.expect(!Canned.active.load(.acquire));
}

test "media lookup: past the running lookups' limit a request is refused without a lookup" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const addresses = publicAddresses();
    Canned.addresses = &addresses;
    Canned.fails = false;
    try lookupsEnd();
    Canned.lookups.store(0, .monotonic);
    _ = transport.lookups.fetchAdd(transport.max_lookups, .monotonic);
    defer _ = transport.lookups.fetchSub(transport.max_lookups, .monotonic);
    var connector: Connector = .{};
    var f: Failure = .{};
    const io = Canned.system();
    try std.testing.expectError(error.Media, transport.openVia(std.testing.allocator, io, "busy.example", Io.Clock.awake.now(io).addDuration(.fromSeconds(2)), &f, &connector));
    try std.testing.expectEqualStrings("too many image host lookups are running; try again later", f.text);
    try std.testing.expectEqual(@as(u32, 0), Canned.lookups.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), connector.calls);
}

test "media lookup: a refused first public address falls through once to the next checked address" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const addresses = publicAddresses();
    Canned.addresses = &addresses;
    Canned.fails = false;
    Canned.lookups.store(0, .monotonic);
    const io = Canned.system();
    var connector: Connector = .{ .refuse = 1, .delay_ms = 20 };
    var f: Failure = .{};
    const deadline = Io.Clock.awake.now(io).addDuration(.fromSeconds(2));
    const stream = try transport.openVia(std.testing.allocator, io, "failover.example", deadline, &f, &connector);
    try std.testing.expect(connector.prepared);
    try std.testing.expectEqual(@as(usize, 2), connector.calls);
    try std.testing.expectEqual(@as(u32, 1), Canned.lookups.load(.monotonic));
    try std.testing.expect(connector.attempted[0].eql(&addresses[0]));
    try std.testing.expect(connector.attempted[1].eql(&addresses[1]));
    try std.testing.expect(stream.socket.address.eql(&addresses[1]));
    try std.testing.expect(connector.budgets[1] < connector.budgets[0]);
    try std.testing.expectEqualStrings("", f.text);
}

test "media lookup: any private answer refuses before trust preparation or any connection attempt" {
    var addresses = publicAddresses();
    Canned.addresses = &addresses;
    Canned.fails = false;
    const io = Canned.system();
    for (0..addresses.len) |at| {
        const saved = addresses[at];
        defer addresses[at] = saved;
        addresses[at] = .{ .ip4 = .{ .bytes = .{ 10, 0, 0, 1 }, .port = 443 } };
        var connector: Connector = .{ .refuse = 1 };
        var f: Failure = .{};
        try std.testing.expectError(error.Media, transport.openVia(std.testing.allocator, io, "mixed.example", Io.Clock.awake.now(io).addDuration(.fromSeconds(2)), &f, &connector));
        try std.testing.expectEqualStrings("image URLs must resolve only to public internet addresses", f.text);
        try std.testing.expect(!connector.prepared);
        try std.testing.expectEqual(@as(usize, 0), connector.calls);
    }
}

test "media lookup: connection attempts share the request deadline and stop when it expires" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const addresses = publicAddresses();
    Canned.addresses = &addresses;
    Canned.fails = false;
    const io = Canned.system();
    var connector: Connector = .{ .refuse = addresses.len };
    var f: Failure = .{};
    const far = Io.Clock.awake.now(io).addDuration(.fromSeconds(2));
    try std.testing.expectError(error.Media, transport.openVia(std.testing.allocator, io, "refused.example", far, &f, &connector));
    try std.testing.expectEqual(addresses.len, connector.calls);
    try std.testing.expectEqualStrings("image download failed or timed out", f.text);
    connector = .{ .refuse = 1, .delay_ms = 150 };
    const near = Io.Clock.awake.now(io).addDuration(.fromMilliseconds(100));
    try std.testing.expectError(error.Media, transport.openVia(std.testing.allocator, io, "deadline.example", near, &f, &connector));
    try std.testing.expectEqual(@as(usize, 1), connector.calls);
    try std.testing.expectEqualStrings("image download timed out", f.text);
}

test "media lookup: failed or empty lookups never prepare trust or attempt an address" {
    const addresses = publicAddresses();
    const io = Canned.system();
    for ([_]bool{ false, true }) |failed| {
        Canned.addresses = if (failed) &addresses else &.{};
        Canned.fails = failed;
        defer Canned.fails = false;
        var connector: Connector = .{};
        var f: Failure = .{};
        const deadline = Io.Clock.awake.now(io).addDuration(.fromSeconds(2));
        try std.testing.expectError(error.Media, transport.openVia(std.testing.allocator, io, "unresolved.example", deadline, &f, &connector));
        try std.testing.expectEqualStrings("image host could not be resolved", f.text);
        try std.testing.expect(!connector.prepared);
        try std.testing.expectEqual(@as(usize, 0), connector.calls);
    }
}
