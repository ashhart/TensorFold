//! Media hostname lookup deadlines and TCP attempts restricted to one fully checked address list.
const std = @import("std");
const Io = std.Io;
const net = Io.net;
const fetch = @import("media_fetch.zig");
const Error = fetch.Error;
const Failure = fetch.Failure;

pub const lookup_timeout_s = 3;
/// Lookups running at once, those no request waits for any more included; past this a request is refused.
pub const max_lookups = 64;
/// Lookups running now; each ends with its own thread, after its request may have stopped waiting.
pub var lookups: std.atomic.Value(u32) = .init(0);

fn fail(f: *Failure, text: []const u8) Error {
    f.text = text;
    return error.Media;
}

const Outcome = struct {
    addresses: std.ArrayList(net.IpAddress) = .empty,
    failure: Failure = .{},
    err: ?Error = null,
};

/// One lookup on its own thread, freed by whichever of it and its request lets go last (see `resolve`).
const Pending = struct {
    io: Io,
    name: [net.HostName.max_len]u8 = undefined,
    len: usize,
    out: Outcome = .{},
    finished: Io.Event = .unset,
    holders: std.atomic.Value(u8) = .init(2),

    const gpa = std.heap.smp_allocator;

    fn run(p: *Pending) void {
        collectAddresses(gpa, p.io, .{ .bytes = p.name[0..p.len] }, &p.out) catch |err| {
            p.out.err = err;
        };
        p.finished.set(p.io);
        p.release();
    }

    fn release(p: *Pending) void {
        if (p.holders.fetchSub(1, .acq_rel) != 1) return;
        p.out.addresses.deinit(gpa);
        gpa.destroy(p);
        _ = lookups.fetchSub(1, .release);
    }
};

fn collectAddresses(a: std.mem.Allocator, io: Io, name: net.HostName, out: *Outcome) Error!void {
    const unresolved = "image host could not be resolved";
    var results: [32]net.HostName.LookupResult = undefined;
    var q: Io.Queue(net.HostName.LookupResult) = .init(&results);
    // the lookup runs beside its reader: alone on this thread it would wait forever to queue a 33rd address
    var looking = io.concurrent(net.HostName.lookup, .{ name, io, &q, .{ .port = 443 } }) catch return fail(&out.failure, unresolved);
    var joined = false;
    defer {
        if (!joined) _ = looking.cancel(io) catch {};
    }
    var public = true;
    while (q.getOneUncancelable(io)) |r| switch (r) {
        .address => |addr| {
            public = public and fetch.publicIp(addr);
            try out.addresses.append(a, addr);
        },
        .canonical_name => {},
    } else |_| {}
    joined = true;
    looking.await(io) catch return fail(&out.failure, unresolved);
    if (!public) return fail(&out.failure, "image URLs must resolve only to public internet addresses");
    if (out.addresses.items.len == 0) return fail(&out.failure, unresolved);
}

/// All addresses from one successful lookup, after every address has passed the public-address policy.
pub fn resolve(a: std.mem.Allocator, io: Io, host: []const u8, deadline: Io.Timestamp, f: *Failure) Error![]net.IpAddress {
    const unresolved = "image host could not be resolved";
    const inner = "image URLs must resolve only to public internet addresses";
    if (net.IpAddress.parseIp4(host, 443)) |literal| {
        if (!fetch.publicIp(literal)) return fail(f, inner);
        _ = try fetch.remaining(io, deadline, f);
        return a.dupe(net.IpAddress, &.{literal});
    } else |_| {}
    const name = net.HostName.init(std.mem.trimEnd(u8, host, ".")) catch return fail(f, unresolved);
    const limit = fetch.within(io, deadline, lookup_timeout_s);
    const timed_out = "image host lookup timed out";
    if (Io.Clock.awake.now(io).nanoseconds >= limit.nanoseconds) return fail(f, timed_out);
    if (lookups.fetchAdd(1, .acquire) >= max_lookups) {
        _ = lookups.fetchSub(1, .release);
        return fail(f, "too many image host lookups are running; try again later");
    }
    const p = Pending.gpa.create(Pending) catch {
        _ = lookups.fetchSub(1, .release);
        return error.OutOfMemory;
    };
    p.* = .{ .io = io, .len = name.bytes.len };
    @memcpy(p.name[0..p.len], name.bytes);
    // getaddrinfo can't be canceled (glibc and macOS retry through the signal): past the limit the request goes on
    const thread = std.Thread.spawn(.{}, Pending.run, .{p}) catch {
        p.holders.store(1, .monotonic);
        p.release();
        return fail(f, unresolved);
    };
    thread.detach();
    defer p.release();
    while (!p.finished.isSet()) {
        if (Io.Clock.awake.now(io).nanoseconds >= limit.nanoseconds) return fail(f, timed_out);
        p.finished.waitTimeout(io, .{ .deadline = .{ .raw = limit, .clock = .awake } }) catch |err| switch (err) {
            error.Timeout => {},
            error.Canceled => {
                io.recancel();
                return fail(f, unresolved);
            },
        };
    }
    if (p.out.err) |err| {
        f.* = p.out.failure;
        return err;
    }
    return a.dupe(net.IpAddress, p.out.addresses.items);
}

/// Resolve once, prepare trust, then try checked addresses in order within the same request deadline.
pub fn openVia(a: std.mem.Allocator, io: Io, host: []const u8, deadline: Io.Timestamp, f: *Failure, via: anytype) Error!net.Stream {
    const addresses = try resolve(a, io, host, deadline, f);
    defer a.free(addresses);
    try via.prepare(io, f);
    for (addresses) |addr| {
        const ms = try fetch.remaining(io, deadline, f);
        const stream = via.connect(addr, ms, f) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Media => continue,
        };
        f.text = "";
        return stream;
    }
    return error.Media;
}

/// A TCP connection to the checked address on port 443, its connect bounded by `ms` (std's connect has no timeout).
pub fn connect(addr: net.IpAddress, ms: i64, f: *Failure) Error!net.Stream {
    // the socket calls below are libc's on Linux and macOS; a Windows server refuses image URLs by name
    return if (comptime @import("builtin").os.tag == .windows) fail(f, "image URLs are not supported on Windows yet") else connectPosix(addr, ms, f);
}

fn connectPosix(addr: net.IpAddress, ms: i64, f: *Failure) Error!net.Stream {
    const c = std.c;
    const failed = "image download failed or timed out";
    var storage: c.sockaddr.storage = undefined;
    var len: c.socklen_t = undefined;
    switch (addr) {
        .ip4 => |v| {
            const sa: *c.sockaddr.in = @ptrCast(@alignCast(&storage));
            sa.* = std.mem.zeroes(c.sockaddr.in);
            sa.family = c.AF.INET;
            sa.port = std.mem.nativeToBig(u16, 443);
            sa.addr = @bitCast(v.bytes);
            len = @sizeOf(c.sockaddr.in);
        },
        .ip6 => |v| {
            const sa: *c.sockaddr.in6 = @ptrCast(@alignCast(&storage));
            sa.* = std.mem.zeroes(c.sockaddr.in6);
            sa.family = c.AF.INET6;
            sa.port = std.mem.nativeToBig(u16, 443);
            sa.addr = v.bytes;
            len = @sizeOf(c.sockaddr.in6);
        },
    }
    const fd = c.socket(@as(c_uint, storage.family), c.SOCK.STREAM, 0);
    if (fd < 0) return fail(f, failed);
    errdefer _ = c.close(fd);
    _ = c.fcntl(fd, c.F.SETFD, @as(c_int, c.FD_CLOEXEC));
    const flags = c.fcntl(fd, c.F.GETFL);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));
    if (flags < 0 or c.fcntl(fd, c.F.SETFL, flags | nonblock) < 0) return fail(f, failed);
    if (c.connect(fd, @ptrCast(&storage), len) != 0) {
        if (c.errno(@as(c_int, -1)) != .INPROGRESS) return fail(f, failed);
        var pfd = [_]c.pollfd{.{ .fd = fd, .events = c.POLL.OUT, .revents = 0 }};
        if (c.poll(&pfd, 1, @intCast(@min(ms, std.math.maxInt(c_int)))) != 1) return fail(f, failed);
        var err: c_int = 0;
        var err_len: c.socklen_t = @sizeOf(c_int);
        if (c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, &err, &err_len) != 0 or err != 0) return fail(f, failed);
    }
    if (c.fcntl(fd, c.F.SETFL, flags) < 0) return fail(f, failed);
    const tv: c.timeval = .{ .sec = @intCast(@divTrunc(ms, 1000)), .usec = @intCast(@mod(ms, 1000) * 1000) };
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVTIMEO, &tv, @sizeOf(c.timeval));
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDTIMEO, &tv, @sizeOf(c.timeval));
    return .{ .socket = .{ .handle = fd, .address = addr } };
}

test {
    _ = @import("media_fetch_net_test.zig");
}
