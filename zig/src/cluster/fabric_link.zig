//! The cluster on lead/zig-fabric's links: two-sided Thunderbolt (`TwoSided`), or one-sided windows where writes exist (`Link`).
const std = @import("std");
const fabric = @import("fabric");
const exchange = @import("exchange.zig");
const barrier = @import("barrier.zig");
const tr = @import("transport.zig");

const Rdma = fabric.Rdma;
const Channel = fabric.collective.Channel;
const Pipe = fabric.collective.Pipe;

/// Where each part lives in every rank's registered window (the same layout on every rank).
pub const Layout = struct {
    ranks: u32,
    step_bytes: usize,
    message_bytes: usize,
    channel: usize = 0,
    words: usize = 0,
    pipes: usize = 0,
    total: usize = 0,

    pub fn init(ranks: u32, step_bytes: usize, message_bytes: usize) Layout {
        var l: Layout = .{ .ranks = ranks, .step_bytes = step_bytes, .message_bytes = message_bytes };
        l.words = Channel.bytes(ranks, step_bytes);
        l.pipes = l.words + std.mem.alignForward(usize, ranks * 8, fabric.rdma.page);
        l.total = l.pipes + ranks * ranks * Pipe.bytes(message_bytes);
        return l;
    }

    /// The pipe from `src` to `dst`: both ends use this base (flags and slots at dst, the credit word at src).
    fn pipe(l: Layout, src: u32, dst: u32) usize {
        return l.pipes + (src * l.ranks + dst) * Pipe.bytes(l.message_bytes);
    }
};

/// One rank's cluster plumbing over one fabric endpoint.
pub const Link = struct {
    gpa: std.mem.Allocator,
    ep: Rdma,
    layout: Layout,
    ch: Channel,
    out: []Pipe,
    in: []Pipe,
    /// Messages waiting for a credit, so a send never blocks the membership loop.
    backlog: []std.ArrayList([]u8),

    pub fn init(gpa: std.mem.Allocator, ep: Rdma, layout: Layout) !Link {
        const n = ep.size();
        const me = ep.rank();
        const l: Link = .{ .gpa = gpa, .ep = ep, .layout = layout, .ch = Channel.init(ep, layout.channel, layout.step_bytes), .out = try gpa.alloc(Pipe, n), .in = try gpa.alloc(Pipe, n), .backlog = try gpa.alloc(std.ArrayList([]u8), n) };
        for (0..n) |i| {
            const p: u32 = @intCast(i);
            l.out[i] = Pipe.init(ep, layout.pipe(me, p), layout.message_bytes, p);
            l.in[i] = Pipe.init(ep, layout.pipe(p, me), layout.message_bytes, p);
            l.backlog[i] = .empty;
        }
        return l;
    }

    pub fn deinit(l: *Link) void {
        for (l.backlog) |*b| {
            for (b.items) |m| l.gpa.free(m);
            b.deinit(l.gpa);
        }
        l.gpa.free(l.out);
        l.gpa.free(l.in);
        l.gpa.free(l.backlog);
    }

    pub fn steps(l: *Link) exchange.Exchange {
        return .{ .ptr = l, .vtable = &.{ .rank = rankFn, .size = sizeFn, .exchange = exchangeFn } };
    }

    pub fn roundWords(l: *Link) barrier.Words {
        return .{ .ptr = l, .vtable = &.{ .rank = rankFn, .size = sizeFn, .post = post, .load = load } };
    }

    /// Links are addressed by peer rank; the membership learns which node is at each.
    pub fn transport(l: *Link) tr.Transport {
        return .{ .ptr = l, .vtable = &.{ .send = send, .recv = recv, .links = links, .limit = limit } };
    }

    fn self(ptr: *anyopaque) *Link {
        return @ptrCast(@alignCast(ptr));
    }

    fn rankFn(ptr: *anyopaque) u32 {
        return self(ptr).ep.rank();
    }

    fn sizeFn(ptr: *anyopaque) u32 {
        return self(ptr).ep.size();
    }

    fn exchangeFn(ptr: *anyopaque, chunks: []const []const u8, out: [][]const u8) exchange.Error!void {
        self(ptr).ch.exchange(chunks, out) catch |err| return switch (err) {
            error.TooLarge => error.TooLarge,
            error.PeerTimeout, error.PeerDown => error.PeerTimeout,
            else => error.Corrupt,
        };
    }

    fn post(ptr: *anyopaque, peer: u32, slot: u32, value: u64) barrier.Error!void {
        const l = self(ptr);
        const at = l.layout.words + slot * 8;
        if (peer == l.ep.rank()) {
            @atomicStore(u64, @as(*u64, @ptrCast(@alignCast(l.ep.window().ptr + at))), value, .release);
            return;
        }
        l.ep.signal(peer, at, value) catch return error.LinkDown;
    }

    fn load(ptr: *anyopaque, slot: u32) u64 {
        const l = self(ptr);
        return l.ep.local(l.layout.words + slot * 8);
    }

    /// A pipe has room when the receiver released all but one of the sender's messages (two slots).
    fn room(p: *const Pipe) bool {
        const next = p.sent + 1;
        return next <= 2 or p.ep.local(p.base + 16) >= next - 2;
    }

    fn send(ptr: *anyopaque, link: tr.LinkId, bytes: []const u8) tr.Error!void {
        const l = self(ptr);
        if (link >= l.out.len or link == l.ep.rank()) return error.LinkDown;
        if (bytes.len > l.layout.message_bytes) return error.TooLarge;
        if (l.backlog[link].items.len == 0 and room(&l.out[link])) {
            l.out[link].send(bytes) catch return error.LinkDown;
            return;
        }
        const copy = l.gpa.dupe(u8, bytes) catch return error.LinkDown;
        l.backlog[link].append(l.gpa, copy) catch return error.LinkDown;
    }

    fn flush(l: *Link) void {
        for (l.backlog, l.out) |*b, *p| {
            while (b.items.len > 0 and room(p)) {
                p.send(b.items[0]) catch return;
                l.gpa.free(b.orderedRemove(0));
            }
        }
    }

    fn recv(ptr: *anyopaque, buf: []u8) ?tr.Received {
        const l = self(ptr);
        l.flush();
        for (l.in, 0..) |*p, i| {
            if (i == l.ep.rank()) continue;
            const next = p.received + 1;
            if (p.ep.local(p.base + (next & 1) * 8) >> 32 != next & 0xFFFF_FFFF) continue;
            const msg = p.recv() catch continue;
            if (msg.len > buf.len) continue;
            @memcpy(buf[0..msg.len], msg);
            p.release() catch {};
            return .{ .link = @intCast(i), .len = msg.len };
        }
        return null;
    }

    fn links(ptr: *anyopaque, out: []tr.Link) usize {
        const l = self(ptr);
        var k: usize = 0;
        for (0..l.ep.size()) |i| {
            if (i == l.ep.rank() or k == out.len) continue;
            const kind = l.ep.link(@intCast(i));
            out[k] = .{ .id = @intCast(i), .up = true, .kind = if (kind == .cx5) .cx5 else if (kind == .nccl) .cx7 else .tb5, .gbps = 80 };
            k += 1;
        }
        return k;
    }

    fn limit(ptr: *anyopaque) usize {
        return self(ptr).layout.message_bytes;
    }
};

/// Thunderbolt between Macs: SENDs only, so steps run on one two-sided link and membership on another (its own QP).
pub const TwoSided = struct {
    x: fabric.sendrecv.Exchange,
    control: fabric.sendrecv.Link,
    /// The largest control message (the membership's receive buffer).
    limit_bytes: usize = 1 << 20,

    pub fn init(data: fabric.sendrecv.Link, control: fabric.sendrecv.Link) TwoSided {
        return .{ .x = fabric.sendrecv.Exchange.init(data), .control = control };
    }

    /// Exchange steps; a step's lockstep is the round barrier, since a peer can be at most one step ahead.
    pub fn steps(t: *TwoSided) exchange.Exchange {
        return .{ .ptr = t, .vtable = &.{ .rank = rank2, .size = size2, .exchange = exchange2 } };
    }

    pub fn transport(t: *TwoSided) tr.Transport {
        return .{ .ptr = t, .vtable = &.{ .send = send2, .recv = recv2, .links = links2, .limit = limit2 } };
    }

    fn of(ptr: *anyopaque) *TwoSided {
        return @ptrCast(@alignCast(ptr));
    }

    fn rank2(ptr: *anyopaque) u32 {
        return of(ptr).x.myRank();
    }

    fn size2(ptr: *anyopaque) u32 {
        return of(ptr).x.ranks();
    }

    fn exchange2(ptr: *anyopaque, chunks: []const []const u8, out: [][]const u8) exchange.Error!void {
        of(ptr).x.exchange(chunks, out) catch |err| return switch (err) {
            error.TooLarge => error.TooLarge,
            error.PeerTimeout, error.PeerDown => error.PeerTimeout,
            else => error.Corrupt,
        };
    }

    fn send2(ptr: *anyopaque, link: tr.LinkId, bytes: []const u8) tr.Error!void {
        const t = of(ptr);
        if (link >= t.control.size() or link == t.control.rank()) return error.LinkDown;
        if (bytes.len > t.limit_bytes) return error.TooLarge;
        t.control.send(link, bytes) catch return error.LinkDown;
    }

    fn recv2(ptr: *anyopaque, buf: []u8) ?tr.Received {
        const t = of(ptr);
        for (0..t.control.size()) |i| {
            const p: u32 = @intCast(i);
            if (p == t.control.rank()) continue;
            const msg = (t.control.next(p) catch continue) orelse continue;
            const n = @min(msg.len, buf.len);
            @memcpy(buf[0..n], msg[0..n]);
            t.control.release(p) catch {};
            if (n < msg.len) continue;
            return .{ .link = @intCast(p), .len = n };
        }
        return null;
    }

    fn links2(ptr: *anyopaque, out: []tr.Link) usize {
        const t = of(ptr);
        var k: usize = 0;
        for (0..t.control.size()) |i| {
            if (i == t.control.rank() or k == out.len) continue;
            out[k] = .{ .id = @intCast(i), .up = true, .kind = .tb5, .gbps = 80 };
            k += 1;
        }
        return k;
    }

    fn limit2(ptr: *anyopaque) usize {
        return of(ptr).limit_bytes;
    }
};

test {
    _ = @import("fabric_link_test.zig");
}
