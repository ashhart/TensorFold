//! Two-rank ordered window writes over TCP, with a separate completion socket.
const std = @import("std");
const rma = @import("rdma.zig");
const socket = @import("tcp_socket.zig");
const words = @import("words.zig");
const c = std.c;

pub const Config = struct {
    rank: u32,
    host: []const u8,
    port: u16,
    /// Same nonempty session name on both ranks, to bind the two connections together.
    session: []const u8,
    timeout_ns: u64 = 30 * std.time.ns_per_s,
    /// Loading checkpoints can finish at different times; this never extends frame/flush waits.
    connect_timeout_ns: u64 = 30 * std.time.ns_per_s,
};

/// TCP writes, signals and flush; remote reads and atomics return AccessDenied.
pub const Endpoint = struct {
    gpa: std.mem.Allocator,
    rank_id: u32,
    memory: []align(rma.page) u8,
    data: c.fd_t,
    /// Separate completions let the receiver acknowledge writes without taking the application's transmit lock.
    acknowledgements: c.fd_t,
    receiver: std.Thread = undefined,
    failed: std.atomic.Value(bool) = .init(false),
    external_failed: ?*std.atomic.Value(bool),
    locked: std.atomic.Value(bool) = .init(false),
    timeout_ns: u64,
    deadline_ns: std.atomic.Value(u64) = .init(0),
    sequence: u64 = 0,
    peer_finished: std.atomic.Value(bool) = .init(false),

    /// Both applications have finished all exchanges before either tears down its receiver.
    pub fn finish(e: *Endpoint) !void {
        errdefer e.fail();
        const until = try e.end();
        try e.lock(until);
        defer e.locked.store(false, .release);
        const h = header(4, 0, 0, 0, 0);
        try socket.send(e.data, &h, until);
        while (!e.peer_finished.load(.acquire)) {
            if (e.failed.load(.acquire)) return error.PeerDown;
            if (try words.checkedNowNs() >= until) return error.Timeout;
            words.pause(10_000);
        }
    }

    pub fn create(gpa: std.mem.Allocator, config: Config, window_bytes: usize, failed: ?*std.atomic.Value(bool)) !*Endpoint {
        if (config.rank > 1 or config.port == 0 or config.port == 65535 or config.session.len == 0 or
            config.timeout_ns == 0 or config.connect_timeout_ns == 0 or window_bytes == 0) return error.BadTcpSettings;
        const until = try socket.end(config.connect_timeout_ns);
        const data = try socket.open(config.host, config.port, config.rank, until);
        errdefer socket.close(data);
        try hello(data, config, window_bytes, 0, until);
        const ack = try socket.open(config.host, config.port + 1, config.rank, until);
        errdefer socket.close(ack);
        try hello(ack, config, window_bytes, 1, until);
        return adopt(gpa, config.rank, window_bytes, data, ack, config.timeout_ns, failed);
    }

    /// Takes ownership of already-connected sockets only on success.
    pub fn adopt(gpa: std.mem.Allocator, rank_id: u32, window_bytes: usize, data: c.fd_t, ack: c.fd_t, timeout_ns: u64, failed: ?*std.atomic.Value(bool)) !*Endpoint {
        if (rank_id > 1 or window_bytes == 0 or timeout_ns == 0) return error.BadTcpSettings;
        try socket.prepare(data, false);
        try socket.prepare(ack, false);
        const e = try gpa.create(Endpoint);
        errdefer gpa.destroy(e);
        const memory = try gpa.alignedAlloc(u8, .fromByteUnits(rma.page), window_bytes);
        errdefer gpa.free(memory);
        @memset(memory, 0);
        e.* = .{ .gpa = gpa, .rank_id = rank_id, .memory = memory, .data = data, .acknowledgements = ack, .timeout_ns = timeout_ns, .external_failed = failed };
        e.receiver = try std.Thread.spawn(.{}, receive, .{e});
        return e;
    }

    pub fn deinit(e: *Endpoint) void {
        e.fail();
        e.receiver.join();
        _ = c.close(e.data);
        _ = c.close(e.acknowledgements);
        const gpa = e.gpa;
        gpa.free(e.memory);
        gpa.destroy(e);
    }

    fn fail(e: *Endpoint) void {
        e.failed.store(true, .release);
        if (e.external_failed) |f| f.store(true, .release);
        _ = c.shutdown(e.data, c.SHUT.RDWR);
        _ = c.shutdown(e.acknowledgements, c.SHUT.RDWR);
    }

    fn end(e: *Endpoint) rma.Error!u64 {
        const own = try socket.end(e.timeout_ns);
        const imposed = e.deadline_ns.load(.acquire);
        return if (imposed == 0) own else @min(own, imposed);
    }

    fn lock(e: *Endpoint, until: u64) rma.Error!void {
        while (e.locked.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            if (e.failed.load(.acquire)) return error.PeerDown;
            if (try words.checkedNowNs() >= until) return error.Timeout;
            words.pause(10_000);
        }
        if (e.failed.load(.acquire)) {
            e.locked.store(false, .release);
            return error.PeerDown;
        }
    }

    fn check(e: *Endpoint, peer: u32, offset: usize, bytes: usize) rma.Error!void {
        if (peer != 1 - e.rank_id) return error.NoSuchRank;
        if (offset > e.memory.len or bytes > e.memory.len - offset) return error.OutOfBounds;
    }

    /// Header: magic, kind, offset, payload bytes, signal offset, signal value.
    fn header(kind: u32, offset: usize, bytes: usize, flag: usize, value: u64) [40]u8 {
        var out: [40]u8 = undefined;
        @memcpy(out[0..4], "TFW1");
        std.mem.writeInt(u32, out[4..8], kind, .little);
        std.mem.writeInt(u64, out[8..16], offset, .little);
        std.mem.writeInt(u64, out[16..24], bytes, .little);
        std.mem.writeInt(u64, out[24..32], flag, .little);
        std.mem.writeInt(u64, out[32..40], value, .little);
        return out;
    }

    fn post(e: *Endpoint, peer: u32, offset: usize, head: []const u8, body: []const u8, flag: ?usize, value: u64) rma.Error!void {
        const count = std.math.add(usize, head.len, body.len) catch return error.OutOfBounds;
        try e.check(peer, offset, count);
        if (flag) |at| {
            try e.check(peer, at, 8);
            if (at % 8 != 0) return error.Unaligned;
        }
        errdefer e.fail();
        const until = try e.end();
        try e.lock(until);
        defer e.locked.store(false, .release);
        const h = header(if (flag == null) 1 else 2, offset, count, flag orelse 0, value);
        try socket.send(e.data, &h, until);
        try socket.send(e.data, head, until);
        try socket.send(e.data, body, until);
    }

    fn receive(e: *Endpoint) void {
        e.receiveLoop() catch {
            e.fail();
        };
    }

    fn receiveLoop(e: *Endpoint) !void {
        var fences: u64 = 0;
        while (true) {
            var h: [40]u8 = undefined;
            // An idle server may wait; an incomplete frame always has a deadline.
            try socket.receive(e.data, h[0..1], null);
            const until = try socket.end(e.timeout_ns);
            try socket.receive(e.data, h[1..], until);
            if (!std.mem.eql(u8, h[0..4], "TFW1")) return error.BadTcpFrame;
            const kind = std.mem.readInt(u32, h[4..8], .little);
            const offset = std.mem.readInt(u64, h[8..16], .little);
            const count = std.mem.readInt(u64, h[16..24], .little);
            const flag = std.mem.readInt(u64, h[24..32], .little);
            const value = std.mem.readInt(u64, h[32..40], .little);
            if (kind == 4) {
                if (offset != 0 or count != 0 or flag != 0 or value != 0) return error.BadTcpFrame;
                e.peer_finished.store(true, .release);
                return;
            }
            if (kind == 3) {
                if (offset != 0 or count != 0 or flag != 0 or value != fences + 1) return error.BadTcpFrame;
                fences = value;
                var ack: [8]u8 = undefined;
                std.mem.writeInt(u64, &ack, fences, .little);
                try socket.send(e.acknowledgements, &ack, until);
                continue;
            }
            if (kind != 1 and kind != 2) return error.BadTcpFrame;
            try e.check(1 - e.rank_id, offset, count);
            if (kind == 1 and (flag != 0 or value != 0)) return error.BadTcpFrame;
            if (kind == 2) {
                try e.check(1 - e.rank_id, flag, 8);
                if (flag % 8 != 0) return error.Unaligned;
            }
            try socket.receive(e.data, e.memory[offset..][0..count], until);
            if (kind == 2) {
                const word: *u64 = @ptrCast(@alignCast(e.memory.ptr + flag));
                @atomicStore(u64, word, value, .release);
            }
        }
    }

    pub fn rdma(e: *Endpoint) rma.Rdma {
        return .{ .ptr = e, .vtable = &vtable };
    }
    fn self(ptr: *anyopaque) *Endpoint {
        return @ptrCast(@alignCast(ptr));
    }
    fn rank(ptr: *anyopaque) u32 {
        return self(ptr).rank_id;
    }
    fn size(_: *anyopaque) u32 {
        return 2;
    }
    fn window(ptr: *anyopaque) []align(rma.page) u8 {
        return self(ptr).memory;
    }
    fn write(ptr: *anyopaque, peer: u32, offset: usize, bytes: []const u8) rma.Error!void {
        try self(ptr).post(peer, offset, bytes, &.{}, null, 0);
    }
    fn write2(ptr: *anyopaque, peer: u32, offset: usize, head: []const u8, body: []const u8) rma.Error!void {
        try self(ptr).post(peer, offset, head, body, null, 0);
    }
    fn signal(ptr: *anyopaque, peer: u32, offset: usize, value: u64) rma.Error!void {
        try self(ptr).post(peer, 0, &.{}, &.{}, offset, value);
    }
    fn write2Signal(ptr: *anyopaque, peer: u32, offset: usize, head: []const u8, body: []const u8, flag: usize, value: u64) rma.Error!void {
        try self(ptr).post(peer, offset, head, body, flag, value);
    }
    fn fetchAdd(_: *anyopaque, _: u32, _: usize, _: u64) rma.Error!u64 {
        return error.AccessDenied;
    }
    fn read(_: *anyopaque, _: u32, _: usize, _: []u8) rma.Error!void {
        return error.AccessDenied;
    }
    fn flush(ptr: *anyopaque) rma.Error!void {
        const e = self(ptr);
        errdefer e.fail();
        const until = try e.end();
        try e.lock(until);
        defer e.locked.store(false, .release);
        e.sequence += 1;
        const h = header(3, 0, 0, 0, e.sequence);
        try socket.send(e.data, &h, until);
        var ack: [8]u8 = undefined;
        try socket.receive(e.acknowledgements, &ack, until);
        if (std.mem.readInt(u64, &ack, .little) != e.sequence) return error.PeerDown;
    }
    fn link(_: *anyopaque, _: u32) rma.LinkKind {
        return .tcp;
    }
    fn deadline(ptr: *anyopaque, until: u64) void {
        self(ptr).deadline_ns.store(until, .release);
    }
    const vtable: rma.Rdma.VTable = .{ .rank = rank, .size = size, .window = window, .write = write, .write2 = write2, .signal = signal, .write2_signal = write2Signal, .fetch_add = fetchAdd, .read = read, .flush = flush, .link = link, .deadline = deadline };
};

fn hello(fd: c.fd_t, config: Config, bytes: usize, channel: u32, until: u64) !void {
    var mine: [64]u8 = @splat(0);
    @memcpy(mine[0..4], "TFT1");
    std.mem.writeInt(u32, mine[4..8], config.rank, .little);
    std.mem.writeInt(u32, mine[8..12], channel, .little);
    std.mem.writeInt(u64, mine[16..24], bytes, .little);
    std.crypto.hash.sha2.Sha256.hash(config.session, mine[32..64], .{});
    try socket.send(fd, &mine, until);
    var peer: [64]u8 = undefined;
    try socket.receive(fd, &peer, until);
    std.mem.writeInt(u32, mine[4..8], 1 - config.rank, .little);
    if (!std.mem.eql(u8, &mine, &peer)) return error.TcpIdentityMismatch;
}

test "TP TCP ordered writes complete at the peer, including simultaneous large payloads" {
    const Pair = struct {
        fn run(e: *Endpoint, failure: *?anyerror) void {
            exercise(e) catch |err| {
                failure.* = err;
            };
        }
        fn exercise(e: *Endpoint) !void {
            const r = e.rdma();
            const bytes = try std.testing.allocator.alloc(u8, 1 << 20);
            defer std.testing.allocator.free(bytes);
            @memset(bytes, @intCast(40 + r.rank()));
            var ch = @import("collective.zig").Channel.init(r, 0, bytes.len);
            for (0..16) |_| {
                var incoming: [2][]const u8 = undefined;
                try ch.exchange(&.{ bytes, bytes }, &incoming);
                try std.testing.expect(std.mem.allEqual(u8, incoming[1 - r.rank()], @intCast(41 - r.rank())));
            }
        }
    };
    var data: [2]c.fd_t = undefined;
    var ack: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &data));
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &ack));
    const a = try Endpoint.adopt(std.testing.allocator, 0, @import("collective.zig").Channel.bytes(2, 1 << 20), data[0], ack[0], std.time.ns_per_s, null);
    defer a.deinit();
    const b = try Endpoint.adopt(std.testing.allocator, 1, @import("collective.zig").Channel.bytes(2, 1 << 20), data[1], ack[1], std.time.ns_per_s, null);
    defer b.deinit();
    var errors: [2]?anyerror = .{ null, null };
    const thread = try std.Thread.spawn(.{}, Pair.run, .{ b, &errors[1] });
    Pair.run(a, &errors[0]);
    thread.join();
    for (errors) |err| if (err) |e| return e;
}

test "TP TCP flush requires remote completion and poisons a timed out endpoint" {
    var data: [2]c.fd_t = undefined;
    var ack: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &data));
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &ack));
    defer socket.close(data[1]);
    defer socket.close(ack[1]);
    var failed: std.atomic.Value(bool) = .init(false);
    const a = try Endpoint.adopt(std.testing.allocator, 0, 64, data[0], ack[0], 5 * std.time.ns_per_ms, &failed);
    defer a.deinit();
    try a.rdma().write(1, 8, "test");
    try std.testing.expectError(error.Timeout, a.rdma().flush());
    try std.testing.expect(failed.load(.acquire));
    try std.testing.expectError(error.PeerDown, a.rdma().signal(1, 0, 1));
}

test "TP TCP malformed or truncated frames fail the idle follower and bounded close joins its receiver" {
    for (0..3) |mode| {
        var data: [2]c.fd_t = undefined;
        var ack: [2]c.fd_t = undefined;
        try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &data));
        try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &ack));
        defer socket.close(data[1]);
        defer socket.close(ack[1]);
        var failed: std.atomic.Value(bool) = .init(false);
        const a = try Endpoint.adopt(std.testing.allocator, 0, 64, data[0], ack[0], 5 * std.time.ns_per_ms, &failed);
        defer a.deinit();
        try socket.prepare(data[1], false);
        const h = Endpoint.header(2, 64, 1, 0, 1); // outside the registered window
        if (mode == 0) {
            try socket.send(data[1], &h, try socket.end(std.time.ns_per_s));
        } else if (mode == 1) {
            try socket.send(data[1], h[0..1], try socket.end(std.time.ns_per_s));
        } else {
            _ = c.shutdown(data[1], c.SHUT.RDWR);
        }
        const until = try socket.end(std.time.ns_per_s);
        while (!failed.load(.acquire)) {
            if (try words.checkedNowNs() >= until) return error.Timeout;
            words.pause(100_000);
        }
        try std.testing.expectEqual(@as(u64, 0), a.rdma().local(0));
        try std.testing.expectError(error.PeerDown, a.rdma().flush());
    }
}

test "TP TCP handshake rejects a different rank, session or window layout" {
    const Worker = struct {
        fn run(fd: c.fd_t, config: Config, bytes: usize, result: *?anyerror) void {
            hello(fd, config, bytes, 0, socket.end(std.time.ns_per_s) catch unreachable) catch |err| {
                result.* = err;
            };
        }
    };
    for (0..4) |mode| {
        var pair: [2]c.fd_t = undefined;
        try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &pair));
        defer socket.close(pair[0]);
        defer socket.close(pair[1]);
        for (pair) |fd| try socket.prepare(fd, false);
        const a: Config = .{ .rank = 0, .host = "::1", .port = 19750, .session = "one" };
        var b = a;
        b.rank = if (mode == 1) 0 else 1;
        if (mode == 2) b.session = "two";
        var errors: [2]?anyerror = .{ null, null };
        const thread = try std.Thread.spawn(.{}, Worker.run, .{ pair[1], b, @as(usize, if (mode == 3) 65 else 64), &errors[1] });
        Worker.run(pair[0], a, 64, &errors[0]);
        thread.join();
        for (errors) |err| if (mode == 0) {
            try std.testing.expectEqual(@as(?anyerror, null), err);
        } else {
            try std.testing.expectEqual(@as(?anyerror, error.TcpIdentityMismatch), err);
        };
    }
}
