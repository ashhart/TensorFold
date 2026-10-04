//! N ranks' two-sided links in one process: ordered messages per pair, pairs delivered in random interleavings.
const std = @import("std");
const sr = @import("sendrecv.zig");

const Box = struct {
    msgs: std.ArrayList([]u8) = .empty,
    returned: usize = 0,
};

pub const Cluster = struct {
    gpa: std.mem.Allocator,
    n: u32,
    endpoints: []Endpoint,
    inflight: []std.ArrayList([]u8),
    boxes: []Box,
    queued: bool,
    lock: std.atomic.Mutex = .unlocked,
    prng: std.Random.DefaultPrng,
    sent: []std.atomic.Value(u64),
    delivered: []std.atomic.Value(u64),
    peak: usize = 0,

    pub fn init(gpa: std.mem.Allocator, n: u32, queued: bool) !*Cluster {
        const c = try gpa.create(Cluster);
        c.* = .{ .gpa = gpa, .n = n, .endpoints = try gpa.alloc(Endpoint, n), .inflight = try gpa.alloc(std.ArrayList([]u8), n * n), .boxes = try gpa.alloc(Box, n * n), .queued = queued, .prng = .init(0x5eed), .sent = try gpa.alloc(std.atomic.Value(u64), n), .delivered = try gpa.alloc(std.atomic.Value(u64), n) };
        for (c.endpoints, c.sent, c.delivered, 0..) |*e, *s, *d, i| {
            e.* = .{ .cluster = c, .me = @intCast(i) };
            s.* = .init(0);
            d.* = .init(0);
        }
        for (c.inflight, c.boxes) |*q, *b| {
            q.* = .empty;
            b.* = .{};
        }
        return c;
    }

    pub fn deinit(c: *Cluster) void {
        const gpa = c.gpa;
        for (c.inflight) |*q| {
            for (q.items) |m| gpa.free(m);
            q.deinit(gpa);
        }
        for (c.boxes) |*b| {
            for (b.msgs.items) |m| gpa.free(m);
            b.msgs.deinit(gpa);
        }
        inline for (.{ "endpoints", "inflight", "boxes", "sent", "delivered" }) |f| gpa.free(@field(c, f));
        gpa.destroy(c);
    }

    pub fn link(c: *Cluster, r: u32) sr.Link {
        return .{ .ptr = &c.endpoints[r], .vtable = &Endpoint.vtable };
    }

    fn acquire(c: *Cluster) void {
        while (!c.lock.tryLock()) std.atomic.spinLoopHint();
    }

    /// Append a message to `dst`'s box for `src`; the caller holds the lock.
    fn land(c: *Cluster, src: u32, dst: u32, m: []u8) void {
        const b = &c.boxes[dst * c.n + src];
        b.msgs.append(c.gpa, m) catch @panic("fake box");
        c.peak = @max(c.peak, b.msgs.items.len);
        _ = c.delivered[src].fetchAdd(1, .release);
    }

    /// Deliver the oldest message of one random non-empty pair; false when none is in flight.
    pub fn deliverOne(c: *Cluster) bool {
        c.acquire();
        defer c.lock.unlock();
        const k = c.inflight.len;
        const start = c.prng.random().uintLessThan(usize, k);
        for (0..k) |j| {
            const at = (start + j) % k;
            if (c.inflight[at].items.len == 0) continue;
            c.land(@intCast(at / c.n), @intCast(at % c.n), c.inflight[at].orderedRemove(0));
            return true;
        }
        return false;
    }

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

    const vtable: sr.Link.VTable = .{ .rank = rank, .size = size, .send = send, .next = next, .release = release, .flush = flush };

    fn self(ptr: *anyopaque) *Endpoint {
        return @ptrCast(@alignCast(ptr));
    }
    fn rank(ptr: *anyopaque) u32 {
        return self(ptr).me;
    }
    fn size(ptr: *anyopaque) u32 {
        return self(ptr).cluster.n;
    }
    fn send(ptr: *anyopaque, peer: u32, bytes: []const u8) sr.Error!void {
        const e = self(ptr);
        const c = e.cluster;
        if (peer >= c.n) return error.NoSuchRank;
        const copy = c.gpa.dupe(u8, bytes) catch return error.PeerDown;
        _ = c.sent[e.me].fetchAdd(1, .monotonic);
        c.acquire();
        defer c.lock.unlock();
        if (c.queued) c.inflight[e.me * c.n + peer].append(c.gpa, copy) catch return error.PeerDown else c.land(e.me, peer, copy);
    }
    fn next(ptr: *anyopaque, peer: u32) sr.Error!?[]const u8 {
        const e = self(ptr);
        const c = e.cluster;
        c.acquire();
        defer c.lock.unlock();
        const b = &c.boxes[e.me * c.n + peer];
        if (b.returned == b.msgs.items.len) return null;
        b.returned += 1;
        return b.msgs.items[b.returned - 1];
    }
    fn release(ptr: *anyopaque, peer: u32) sr.Error!void {
        const e = self(ptr);
        const c = e.cluster;
        c.acquire();
        defer c.lock.unlock();
        const b = &c.boxes[e.me * c.n + peer];
        if (b.returned == 0) return error.Protocol;
        c.gpa.free(b.msgs.orderedRemove(0));
        b.returned -= 1;
    }
    fn flush(ptr: *anyopaque) sr.Error!void {
        const e = self(ptr);
        const c = e.cluster;
        while (c.delivered[e.me].load(.acquire) < c.sent[e.me].load(.monotonic)) std.atomic.spinLoopHint();
    }
};

test "messages arrive in order per pair, empty ones included, and stay until released" {
    const c = try Cluster.init(std.testing.allocator, 2, false);
    defer c.deinit();
    const a = c.link(0);
    const b = c.link(1);
    try b.send(0, "");
    try b.send(0, "two");
    try std.testing.expectEqualStrings("", (try a.next(1)).?);
    try std.testing.expectEqualStrings("two", (try a.next(1)).?);
    try std.testing.expectEqual(@as(?[]const u8, null), try a.next(1));
    try a.release(1);
    try a.release(1);
    try std.testing.expectError(error.Protocol, a.release(1));
}
