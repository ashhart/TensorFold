//! N ranks' registered windows in one process; queued mode keeps each pair's RC order and interleaves pairs at random.
const std = @import("std");
const rdma = @import("rdma.zig");
const Rdma = rdma.Rdma;
const Error = rdma.Error;

pub const Mode = enum { immediate, queued };

const Op = struct {
    kind: enum { write, signal, fetch_add, read },
    src: u32,
    dst: u32,
    offset: usize,
    bytes: []const u8 = &.{},
    out: []u8 = &.{},
    value: u64 = 0,
    done: ?*std.atomic.Value(u64) = null,
};

/// The fake fabric: windows of equal length, one per rank, and per-pair queues in queued mode.
pub const Cluster = struct {
    gpa: std.mem.Allocator,
    windows: [][]align(rdma.page) u8,
    access: []rdma.Access,
    endpoints: []Endpoint,
    links: []const rdma.LinkKind,
    mode: Mode,
    lock: std.atomic.Mutex = .unlocked,
    queues: []std.ArrayList(Op),
    posted: []std.atomic.Value(u64),
    delivered: []std.atomic.Value(u64),
    written: []std.atomic.Value(u64),
    prng: std.Random.DefaultPrng,

    /// `links[a * n + b]` is the pair's link kind; null means every pair is tb5.
    pub fn init(gpa: std.mem.Allocator, n: u32, window_bytes: usize, mode: Mode, links: ?[]const rdma.LinkKind) !*Cluster {
        const c = try gpa.create(Cluster);
        errdefer gpa.destroy(c);
        const tb5 = try gpa.alloc(rdma.LinkKind, n * n);
        @memset(tb5, .tb5);
        if (links) |l| @memcpy(tb5, l);
        c.* = .{
            .gpa = gpa,
            .windows = try gpa.alloc([]align(rdma.page) u8, n),
            .access = try gpa.alloc(rdma.Access, n),
            .endpoints = try gpa.alloc(Endpoint, n),
            .links = tb5,
            .mode = mode,
            .queues = try gpa.alloc(std.ArrayList(Op), n * n),
            .posted = try gpa.alloc(std.atomic.Value(u64), n),
            .delivered = try gpa.alloc(std.atomic.Value(u64), n),
            .written = try gpa.alloc(std.atomic.Value(u64), n * n),
            .prng = .init(0x7f4a),
        };
        for (c.windows, c.access, c.endpoints, 0..) |*w, *a, *e, i| {
            w.* = try std.posix.mmap(null, window_bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            a.* = .{};
            e.* = .{ .cluster = c, .me = @intCast(i) };
        }
        for (c.queues) |*q| q.* = .empty;
        for (c.posted, c.delivered) |*p, *d| {
            p.* = .init(0);
            d.* = .init(0);
        }
        for (c.written) |*w| w.* = .init(0);
        return c;
    }

    pub fn deinit(c: *Cluster) void {
        for (c.windows) |w| std.posix.munmap(w);
        for (c.queues) |*q| {
            for (q.items) |op| if (op.kind == .write) c.gpa.free(op.bytes);
            q.deinit(c.gpa);
        }
        const gpa = c.gpa;
        gpa.free(c.windows);
        gpa.free(c.access);
        gpa.free(c.endpoints);
        gpa.free(c.links);
        gpa.free(c.queues);
        gpa.free(c.posted);
        gpa.free(c.delivered);
        gpa.free(c.written);
        gpa.destroy(c);
    }

    pub fn endpoint(c: *Cluster, r: u32) Rdma {
        return .{ .ptr = &c.endpoints[r], .vtable = &Endpoint.vtable };
    }

    /// Bytes rank `a` has written into rank `b`'s window.
    pub fn bytesWritten(c: *Cluster, a: u32, b: u32) u64 {
        return c.written[a * c.size() + b].load(.monotonic);
    }

    fn size(c: *const Cluster) u32 {
        return @intCast(c.windows.len);
    }

    fn acquire(c: *Cluster) void {
        while (!c.lock.tryLock()) std.atomic.spinLoopHint();
    }

    fn check(c: *Cluster, op: Op, len: usize, word: bool) Error!void {
        if (op.dst >= c.size()) return error.NoSuchRank;
        if (op.offset + len > c.windows[op.dst].len) return error.OutOfBounds;
        if (word and op.offset % 8 != 0) return error.Unaligned;
        const a = c.access[op.dst];
        const ok = switch (op.kind) {
            .write, .signal => a.remote_write,
            .read => a.remote_read,
            .fetch_add => a.remote_atomic,
        };
        if (!ok) return error.AccessDenied;
    }

    fn post(c: *Cluster, op: Op) Error!void {
        const len = switch (op.kind) {
            .write => op.bytes.len,
            .read => op.out.len,
            .signal, .fetch_add => 8,
        };
        try c.check(op, len, op.kind == .signal or op.kind == .fetch_add);
        _ = c.posted[op.src].fetchAdd(1, .monotonic);
        if (c.mode == .immediate) return c.execute(op);
        var queued = op;
        // a queued write owns its bytes, as a real link's staging copy does: callers may reuse theirs before a flush
        if (op.kind == .write) queued.bytes = c.gpa.dupe(u8, op.bytes) catch @panic("fake queue");
        c.acquire();
        defer c.lock.unlock();
        c.queues[op.src * c.size() + op.dst].append(c.gpa, queued) catch @panic("fake queue");
    }

    fn execute(c: *Cluster, op: Op) void {
        const w = c.windows[op.dst];
        switch (op.kind) {
            .write => {
                @memcpy(w[op.offset..][0..op.bytes.len], op.bytes);
                _ = c.written[op.src * c.size() + op.dst].fetchAdd(op.bytes.len, .monotonic);
                if (c.mode == .queued) c.gpa.free(op.bytes);
            },
            .read => @memcpy(op.out, w[op.offset..][0..op.out.len]),
            .signal => @atomicStore(u64, @as(*u64, @ptrCast(@alignCast(w.ptr + op.offset))), op.value, .release),
            .fetch_add => {
                const old = @atomicRmw(u64, @as(*u64, @ptrCast(@alignCast(w.ptr + op.offset))), .Add, op.value, .acq_rel);
                op.done.?.store(old | 1 << 63, .release);
            },
        }
        _ = c.delivered[op.src].fetchAdd(1, .release);
    }

    /// Deliver the oldest operation of one random non-empty pair; false when every queue is empty.
    pub fn deliverOne(c: *Cluster) bool {
        c.acquire();
        const n = c.queues.len;
        const start = c.prng.random().uintLessThan(usize, n);
        var op: ?Op = null;
        for (0..n) |k| {
            const q = &c.queues[(start + k) % n];
            if (q.items.len > 0) {
                op = q.orderedRemove(0);
                break;
            }
        }
        c.lock.unlock();
        c.execute(op orelse return false);
        return true;
    }

    /// Pump deliveries until `stop` is set (run it on its own thread in queued mode).
    pub fn pump(c: *Cluster, stop: *const std.atomic.Value(bool)) void {
        while (!stop.load(.acquire)) {
            if (!c.deliverOne()) std.atomic.spinLoopHint();
        }
        while (c.deliverOne()) {}
    }
};

const Endpoint = struct {
    cluster: *Cluster,
    me: u32,

    const vtable: Rdma.VTable = .{ .rank = rank, .size = size, .window = window, .write = write, .signal = signal, .fetch_add = fetchAdd, .read = read, .flush = flush, .link = link };

    fn self(ptr: *anyopaque) *Endpoint {
        return @ptrCast(@alignCast(ptr));
    }
    fn rank(ptr: *anyopaque) u32 {
        return self(ptr).me;
    }
    fn size(ptr: *anyopaque) u32 {
        return self(ptr).cluster.size();
    }
    fn window(ptr: *anyopaque) []align(rdma.page) u8 {
        const e = self(ptr);
        return e.cluster.windows[e.me];
    }
    fn write(ptr: *anyopaque, peer: u32, offset: usize, bytes: []const u8) Error!void {
        const e = self(ptr);
        return e.cluster.post(.{ .kind = .write, .src = e.me, .dst = peer, .offset = offset, .bytes = bytes });
    }
    fn signal(ptr: *anyopaque, peer: u32, offset: usize, value: u64) Error!void {
        const e = self(ptr);
        return e.cluster.post(.{ .kind = .signal, .src = e.me, .dst = peer, .offset = offset, .value = value });
    }
    fn fetchAdd(ptr: *anyopaque, peer: u32, offset: usize, add: u64) Error!u64 {
        const e = self(ptr);
        var done: std.atomic.Value(u64) = .init(0);
        try e.cluster.post(.{ .kind = .fetch_add, .src = e.me, .dst = peer, .offset = offset, .value = add, .done = &done });
        while (true) {
            const v = done.load(.acquire);
            if (v >> 63 == 1) return v & ~(@as(u64, 1) << 63);
            std.atomic.spinLoopHint();
        }
    }
    fn read(ptr: *anyopaque, peer: u32, offset: usize, out: []u8) Error!void {
        const e = self(ptr);
        try e.cluster.post(.{ .kind = .read, .src = e.me, .dst = peer, .offset = offset, .out = out });
        return flush(ptr);
    }
    fn flush(ptr: *anyopaque) Error!void {
        const e = self(ptr);
        const c = e.cluster;
        while (c.delivered[e.me].load(.acquire) < c.posted[e.me].load(.monotonic)) std.atomic.spinLoopHint();
    }
    fn link(ptr: *anyopaque, peer: u32) rdma.LinkKind {
        const e = self(ptr);
        return if (peer == e.me) .local else e.cluster.links[e.me * e.cluster.size() + peer];
    }
};

test "writes, signals, reads and atomics land in the peer's window, checked like an MR" {
    const c = try Cluster.init(std.testing.allocator, 2, 1 << 16, .immediate, null);
    defer c.deinit();
    const a = c.endpoint(0);
    const b = c.endpoint(1);
    try a.write(1, 4096, "hello");
    try a.signal(1, 64, 42);
    try std.testing.expectEqual(@as(u64, 42), b.local(64));
    try std.testing.expectEqualStrings("hello", b.window()[4096..4101]);
    try std.testing.expectEqual(@as(u64, 0), try a.fetchAdd(1, 128, 5));
    try std.testing.expectEqual(@as(u64, 5), try a.fetchAdd(1, 128, 1));
    var out: [5]u8 = undefined;
    try b.read(1, 4096, &out);
    try std.testing.expectEqualStrings("hello", &out);
    try std.testing.expectError(error.OutOfBounds, a.write(1, (1 << 16) - 2, "abc"));
    try std.testing.expectError(error.Unaligned, a.signal(1, 4, 1));
    c.access[1].remote_atomic = false;
    try std.testing.expectError(error.AccessDenied, a.fetchAdd(1, 128, 1));
    try std.testing.expectEqual(@as(u64, 5), c.bytesWritten(0, 1));
}

test "queued delivery keeps each pair's order and interleaves pairs" {
    const c = try Cluster.init(std.testing.allocator, 3, 1 << 16, .queued, null);
    defer c.deinit();
    const a = c.endpoint(0);
    try a.write(1, 4096, "first");
    try a.signal(1, 0, 1);
    try a.write(2, 4096, "other");
    try a.write(1, 4096, "later");
    try std.testing.expectEqual(@as(u64, 0), c.endpoint(1).local(0));
    while (c.deliverOne()) {}
    try std.testing.expectEqualStrings("later", c.windows[1][4096..4101]);
    try std.testing.expectEqualStrings("other", c.windows[2][4096..4101]);
    try a.flush();
}
