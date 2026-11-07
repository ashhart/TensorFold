//! TCP between ranks with no RDMA (#2066): checked frame headers, deadlines and no resend, rank-order sums.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const sock = @import("sock.zig");
const c = std.c;
const posix = std.posix;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

pub const max_ranks = 8;
const magic: u32 = 0x31545054; // "TPT1"
/// How long `abort` waits for room to queue its abort frame on each link.
const abort_send_ns = 50 * std.time.ns_per_ms;
/// Linux marks a peer's plain hang-up only as POLLRDHUP (0x2000, not in std's POLL), and only when asked for it.
const rdhup: i16 = if (@hasDecl(posix.POLL, "RDHUP")) posix.POLL.RDHUP else if (@import("builtin").os.tag == .linux) 0x2000 else 0;

const Call = enum(u8) { all_reduce = 1, all_gather, all_gather_v, exchange, send, barrier, abort };

const Header = extern struct {
    magic: u32 = magic,
    call: u8,
    t: u8 = 0,
    op: u8 = 0,
    pad: u8 = 0,
    /// This link's frame number, from 0, as the sender counts.
    seq: u64,
    count: u64 = 0,
    /// Payload bytes after the header.
    bytes: u64 = 0,
    sender: u32,
    pad2: u32 = 0,
};

/// The first failure, kept for `check` (later calls are `Aborted`: the streams are out of step).
const Fail = enum(u8) { none, aborted, invalid, timeout, peer_failed, backend };

pub const Options = struct {
    /// A frame that has not come this long after its call started is `Timeout`.
    timeout_ns: u64 = 10 * std.time.ns_per_s,
};

pub const Rank = struct {
    gpa: std.mem.Allocator,
    me: u32,
    world: u32,
    /// The socket to each peer (-1 at `me`).
    fds: [max_ranks]sock.Fd = @splat(-1),
    opts: Options,
    sent: [max_ranks]u64 = @splat(0),
    got: [max_ranks]u64 = @splat(0),
    /// A frame to this peer is partly sent: no other frame may start on that link.
    mid: [max_ranks]bool = @splat(false),
    /// Held for each send and for `abort`'s frames, so an abort frame never splits a frame (as `Fate.send_lock`).
    send_lock: std.atomic.Value(bool) = .init(false),
    fail: std.atomic.Value(Fail) = .init(.none),
    scratch: std.ArrayList(u8) = .empty,
    var_agreed: ?bool = null,

    /// A rank over connected sockets (`fds[peer]`, -1 at `me`); it owns them from here and makes them non-blocking.
    pub fn init(gpa: std.mem.Allocator, me: u32, world: u32, fds: []const sock.Fd, opts: Options) Rank {
        std.debug.assert(world >= 1 and world <= max_ranks and me < world and fds.len == world);
        var r: Rank = .{ .gpa = gpa, .me = me, .world = world, .opts = opts };
        for (fds, 0..) |fd, i| {
            r.fds[i] = if (i == me) -1 else fd;
            if (i != me and fd >= 0) nonBlocking(fd);
        }
        return r;
    }

    /// A macOS send can block despite MSG_DONTWAIT and raise SIGPIPE without SO_NOSIGPIPE, so: non-blocking sockets.
    fn nonBlocking(fd: sock.Fd) void {
        const flags = c.fcntl(fd, posix.F.GETFL);
        if (flags >= 0) _ = c.fcntl(fd, posix.F.SETFL, flags | @as(c_int, @bitCast(posix.O{ .NONBLOCK = true })));
        if (@hasDecl(posix.SO, "NOSIGPIPE")) {
            const one: c_int = 1;
            _ = c.setsockopt(fd, posix.SOL.SOCKET, posix.SO.NOSIGPIPE, &one, @sizeOf(c_int));
        }
    }

    pub fn deinit(r: *Rank) void {
        for (r.fds[0..r.world]) |fd| if (fd >= 0) sock.close(fd);
        r.fds = @splat(-1);
        r.scratch.deinit(r.gpa);
    }

    pub fn iface(r: *Rank) collective.Collective {
        return .{ .ptr = r, .vtable = &vtable };
    }

    const vtable: collective.Collective.VTable = .{
        .kind = .tcp,
        .rank = rankOf,
        .world = worldOf,
        .all_reduce = allReduce,
        .all_gather = allGather,
        .exchange = exchange,
        .send = send,
        .recv = recv,
        .barrier = barrier,
        .check = check,
        .abort = abort,
        .all_gather_v = allGatherV,
        .var_fits = varFits,
        .var_agreed = varAgreed,
    };

    fn self_(ptr: *anyopaque) *Rank {
        return @ptrCast(@alignCast(ptr));
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return self_(ptr).me;
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return self_(ptr).world;
    }

    fn errorOf(f: Fail) Error {
        return switch (f) {
            .none, .aborted => error.Aborted,
            .invalid => error.Invalid,
            .timeout => error.Timeout,
            .peer_failed => error.PeerFailed,
            .backend => error.BackendFailed,
        };
    }

    fn lock(r: *Rank) void {
        while (r.send_lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn unlock(r: *Rank) void {
        r.send_lock.store(false, .release);
    }

    /// `shut`, then the failure's error.
    fn close(r: *Rank, f: Fail) Error {
        r.shut(f);
        return errorOf(f);
    }

    /// Keeps the first failure, queues an abort frame on every link with no frame half sent, and hangs up every socket.
    fn shut(r: *Rank, f: Fail) void {
        _ = r.fail.cmpxchgStrong(.none, f, .acq_rel, .acquire);
        r.lock();
        defer r.unlock();
        const deadline = sock.nowNs() +| abort_send_ns;
        for (r.fds[0..r.world], 0..) |fd, i| {
            if (fd < 0) continue;
            if (!r.mid[i]) {
                const h: Header = .{ .call = @backingInt(Call.abort), .seq = r.sent[i], .sender = r.me };
                sock.sendBefore(fd, std.mem.asBytes(&h), deadline) catch {};
            }
            sock.hangup(fd);
        }
    }

    fn live(r: *Rank) Error!void {
        const f = r.fail.load(.acquire);
        if (f != .none) return error.Aborted;
    }

    fn check(ptr: *anyopaque) Error!void {
        const f = self_(ptr).fail.load(.acquire);
        if (f != .none) return errorOf(f);
    }

    fn abort(ptr: *anyopaque) void {
        self_(ptr).shut(.aborted);
    }

    /// One frame out to `peer` and one in: the header must equal `want`, the prefix `want_prefix`, the rest to `into`.
    const Leg = struct {
        peer: u32,
        out_hdr: Header,
        out: []const u8 = &.{},
        out_prefix: []const u8 = &.{},
        sent: usize = 0,
        sending: bool = true,
        /// Our frame could not be sent: the call fails whatever the peer's frame says.
        broken: bool = false,
        want: Header,
        want_prefix: []const u8 = &.{},
        into: []u8 = &.{},
        receiving: bool = true,
        in_hdr: [@sizeOf(Header)]u8 = undefined,
        got: usize = 0,
        prefix_buf: [max_ranks * 4]u8 = undefined,
    };

    fn outLen(l: *const Leg) usize {
        return @sizeOf(Header) + l.out_prefix.len + l.out.len;
    }

    /// The unsent rest of a leg's frame as up to three pieces, so header, prefix and payload go in one `sendmsg`.
    fn outVecs(l: *const Leg, iov: *[3]posix.iovec_const) usize {
        const parts = [3][]const u8{ std.mem.asBytes(&l.out_hdr), l.out_prefix, l.out };
        var skip = l.sent;
        var k: usize = 0;
        for (parts) |p| {
            if (skip >= p.len) {
                skip -= p.len;
                continue;
            }
            iov[k] = .{ .base = p.ptr + skip, .len = p.len - skip };
            skip = 0;
            k += 1;
        }
        return k;
    }

    /// Every leg's frames both ways at once, so no rank blocks on a full buffer, until done, a refusal or the deadline.
    fn move(r: *Rank, legs: []Leg) Error!void {
        const deadline = sock.nowNs() +| r.opts.timeout_ns;
        // the peers this call doesn't use are watched for an abort frame, so an abort anywhere ends this call
        var watch: [max_ranks]bool = @splat(false);
        for (0..r.world) |p| watch[p] = p != r.me and r.fds[p] >= 0;
        for (legs) |l| watch[l.peer] = false;
        var pfd: [2 * max_ranks]c.pollfd = undefined;
        while (true) {
            try r.live();
            var n: usize = 0;
            for (legs) |*l| {
                if (!l.sending and !l.receiving) continue;
                var ev: i16 = 0;
                if (l.sending) ev |= posix.POLL.OUT;
                if (l.receiving) ev |= posix.POLL.IN;
                pfd[n] = .{ .fd = r.fds[l.peer], .events = ev, .revents = 0 };
                n += 1;
            }
            if (n == 0) {
                for (legs) |l| if (l.broken) return r.close(.peer_failed);
                return;
            }
            const active = n;
            for (0..r.world) |p| if (watch[p]) {
                pfd[n] = .{ .fd = r.fds[p], .events = posix.POLL.IN | rdhup, .revents = 0 };
                n += 1;
            };
            const now = sock.nowNs();
            if (now >= deadline) return r.close(.timeout);
            const ms: c_int = @intCast(@min((deadline - now) / std.time.ns_per_ms + 1, 100));
            if (c.poll(&pfd, @intCast(n), ms) < 0) continue;
            try r.live();
            var k: usize = 0;
            for (legs) |*l| {
                if (!l.sending and !l.receiving) continue;
                const re = pfd[k].revents;
                k += 1;
                if (l.sending and re & (posix.POLL.OUT | posix.POLL.ERR | posix.POLL.HUP) != 0) try r.push(l);
                if (l.receiving and re & (posix.POLL.IN | posix.POLL.ERR | posix.POLL.HUP) != 0) try r.pull(l);
            }
            k = active;
            for (0..r.world) |p| if (watch[p]) {
                if (pfd[k].revents != 0) try r.peek(@intCast(p), &watch[p], pfd[k].revents);
                k += 1;
            };
        }
    }

    /// A peer outside this call: its abort frame ends the call; its next frame, a hang-up or an error stops the watch.
    fn peek(r: *Rank, p: u32, watching: *bool, revents: i16) Error!void {
        var h: Header = undefined;
        const got = c.recv(r.fds[p], std.mem.asBytes(&h), @sizeOf(Header), posix.MSG.PEEK | posix.MSG.DONTWAIT);
        if (got < 0) {
            const e = posix.errno(got);
            watching.* = e == .AGAIN or e == .INTR; // a reset link stays readable and would spin this loop
            return;
        }
        if (got < @sizeOf(Header)) {
            watching.* = got != 0 and revents & (posix.POLL.HUP | rdhup) == 0; // a part header the peer will never finish
            return;
        }
        if (h.magic == magic and h.call == @backingInt(Call.abort)) return r.close(.aborted);
        watching.* = false;
    }

    fn push(r: *Rank, l: *Leg) Error!void {
        var bad = false;
        {
            r.lock();
            defer r.unlock();
            if (r.fail.load(.acquire) != .none) return error.Aborted;
            // a part-sent frame from an earlier call: a new frame would land inside it
            if (l.sent == 0 and r.mid[l.peer]) bad = true;
            while (!bad and l.sent < outLen(l)) {
                var iov: [3]posix.iovec_const = undefined;
                const k = outVecs(l, &iov);
                const msg: c.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = @intCast(k), .control = null, .controllen = 0, .flags = 0 };
                const sent = c.sendmsg(r.fds[l.peer], &msg, posix.MSG.NOSIGNAL | posix.MSG.DONTWAIT);
                if (sent < 0) switch (posix.errno(sent)) {
                    .INTR => continue,
                    .AGAIN, .NOBUFS, .NOMEM => {
                        r.mid[l.peer] = l.sent > 0;
                        return;
                    },
                    else => bad = true,
                } else if (sent == 0) bad = true else l.sent += @intCast(sent);
            }
            if (bad) {
                r.mid[l.peer] = r.mid[l.peer] or l.sent > 0;
            } else {
                r.mid[l.peer] = false;
                r.sent[l.peer] += 1;
            }
        }
        l.sending = false;
        if (bad) {
            // the call fails; a leg still reading reads first, so a refusal or an abort from the peer names it
            l.broken = true;
            if (!l.receiving) return r.close(.peer_failed);
        }
    }

    fn pull(r: *Rank, l: *Leg) Error!void {
        const fd = r.fds[l.peer];
        while (true) {
            const hlen = @sizeOf(Header);
            const w: Header = l.want;
            const plen = l.want_prefix.len;
            var dst: []u8 = undefined;
            if (l.got < hlen) {
                dst = l.in_hdr[l.got..];
            } else if (l.got < hlen + plen) {
                dst = l.prefix_buf[l.got - hlen .. plen];
            } else if (l.got < hlen + plen + l.into.len) {
                dst = l.into[l.got - hlen - plen ..];
            } else break;
            const got = c.recv(fd, dst.ptr, dst.len, posix.MSG.DONTWAIT);
            if (got < 0) {
                const e = posix.errno(got);
                if (e == .INTR) continue;
                if (e == .AGAIN) return;
                return r.close(if (r.fail.load(.acquire) != .none) .aborted else .peer_failed);
            }
            if (got == 0) return r.close(if (r.fail.load(.acquire) != .none) .aborted else .peer_failed);
            const before = l.got;
            l.got += @intCast(got);
            if (before < hlen and l.got >= hlen) {
                const h = std.mem.bytesToValue(Header, &l.in_hdr);
                if (h.magic != magic) return r.close(.invalid);
                if (h.call == @backingInt(Call.abort)) return r.close(.aborted);
                if (h.sender != l.peer or h.seq != r.got[l.peer] or h.call != w.call or h.t != w.t or h.op != w.op or
                    h.count != w.count or h.bytes != plen + l.into.len) return r.close(.invalid);
            }
            if (plen > 0 and before < hlen + plen and l.got >= hlen + plen) {
                if (!std.mem.eql(u8, l.prefix_buf[0..plen], l.want_prefix)) return r.close(.invalid);
            }
        }
        r.got[l.peer] += 1;
        l.receiving = false;
    }

    fn leg(r: *Rank, peer: u32, call: Call, t: DType, op: Op, count: usize, out: []const u8, into: []u8) Leg {
        const h: Header = .{ .call = @backingInt(call), .t = @backingInt(t), .op = @backingInt(op), .seq = r.sent[peer], .count = count, .bytes = out.len, .sender = r.me };
        return .{ .peer = peer, .out_hdr = h, .out = out, .want = .{ .call = @backingInt(call), .t = @backingInt(t), .op = @backingInt(op), .seq = 0, .count = count, .sender = peer }, .into = into };
    }

    fn bytesAt(p: DevicePtr, n: usize) []u8 {
        if (n == 0) return &.{};
        return @as([*]u8, @ptrFromInt(p))[0..n];
    }

    fn allReduce(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, count: usize, t: DType, op: Op, _: collective.Stream) Error!void {
        const r = self_(ptr);
        try r.live();
        const n = count * t.size();
        // every rank's input, then the sum in rank order into a copy (the output may alias the input)
        r.scratch.resize(r.gpa, n * (r.world + 1)) catch return r.close(.backend);
        const all = r.scratch.items;
        const mine = bytesAt(s, n);
        var legs: [max_ranks]Leg = undefined;
        var k: usize = 0;
        for (0..r.world) |p| {
            if (p == r.me) continue;
            legs[k] = r.leg(@intCast(p), .all_reduce, t, op, count, mine, all[p * n ..][0..n]);
            k += 1;
        }
        try r.move(legs[0..k]);
        @memcpy(all[r.me * n ..][0..n], mine);
        var views: [max_ranks][]const u8 = undefined;
        for (0..r.world) |p| views[p] = all[p * n ..][0..n];
        const sum = all[r.world * n ..][0..n];
        if (n > 0) reference.reduce(sum, views[0..r.world], t, op);
        @memcpy(bytesAt(d, n), sum);
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, count: usize, t: DType, _: collective.Stream) Error!void {
        const r = self_(ptr);
        try r.live();
        const n = count * t.size();
        const out = bytesAt(d, n * r.world);
        const mine = bytesAt(s, n);
        var legs: [max_ranks]Leg = undefined;
        var k: usize = 0;
        for (0..r.world) |p| {
            if (p == r.me) continue;
            legs[k] = r.leg(@intCast(p), .all_gather, t, .sum, count, mine, out[p * n ..][0..n]);
            k += 1;
        }
        try r.move(legs[0..k]);
        if (n > 0) std.mem.copyForwards(u8, out[r.me * n ..][0..n], mine);
    }

    /// Each rank's lens[r] bytes; the lengths travel with the data, and a rank whose lengths differ is refused.
    fn allGatherV(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, stride: usize, lens: DevicePtr, t: DType, _: collective.Stream) Error!void {
        const r = self_(ptr);
        try r.live();
        const n = stride * t.size();
        var ls: [max_ranks]i32 = undefined; // a copy: the lengths may sit in a buffer this call writes
        @memcpy(ls[0..r.world], @as([*]const i32, @ptrFromInt(lens))[0..r.world]);
        for (ls[0..r.world]) |x| if (x < 0 or @as(usize, @intCast(x)) > n) return r.close(.invalid);
        const mine_len: usize = @intCast(ls[r.me]);
        const out = bytesAt(d, n * r.world);
        const mine = bytesAt(s, mine_len);
        const lbytes = std.mem.sliceAsBytes(ls[0..r.world]);
        var legs: [max_ranks]Leg = undefined;
        var k: usize = 0;
        for (0..r.world) |p| {
            if (p == r.me) continue;
            var l = r.leg(@intCast(p), .all_gather_v, t, .sum, stride, mine, out[p * n ..][0..@as(usize, @intCast(ls[p]))]);
            l.out_prefix = lbytes;
            l.out_hdr.bytes = lbytes.len + mine.len;
            l.want_prefix = lbytes;
            legs[k] = l;
            k += 1;
        }
        try r.move(legs[0..k]);
        if (mine_len > 0) std.mem.copyForwards(u8, out[r.me * n ..][0..mine_len], mine);
    }

    fn varFits(_: *anyopaque, bytes: usize) bool {
        return bytes > 0;
    }

    /// One int all-gather of `varFits(16)` the first time, then the kept answer.
    fn varAgreed(ptr: *anyopaque) Error!bool {
        const r = self_(ptr);
        if (r.var_agreed) |v| return v;
        const mine: i32 = @intFromBool(varFits(ptr, 16));
        var all: [max_ranks]i32 = undefined;
        try allGather(ptr, @intFromPtr(&mine), @intFromPtr(&all), 1, .i32, null);
        r.var_agreed = std.mem.min(i32, all[0..r.world]) == 1;
        return r.var_agreed.?;
    }

    /// Two ranks trade; as with NCCL, `s` and `d` must not overlap.
    fn exchange(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, count: usize, t: DType, peer: u32, _: collective.Stream) Error!void {
        const r = self_(ptr);
        try collective.checkPeer(r.me, r.world, peer);
        try r.live();
        const n = count * t.size();
        var legs = [_]Leg{r.leg(peer, .exchange, t, .sum, count, bytesAt(s, n), bytesAt(d, n))};
        try r.move(&legs);
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, _: collective.Stream) Error!void {
        const r = self_(ptr);
        try collective.checkPeer(r.me, r.world, peer);
        try r.live();
        var l = r.leg(peer, .send, t, .sum, count, bytesAt(buf, count * t.size()), &.{});
        l.receiving = false;
        var legs = [_]Leg{l};
        try r.move(&legs);
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, _: collective.Stream) Error!void {
        const r = self_(ptr);
        try collective.checkPeer(r.me, r.world, peer);
        try r.live();
        var l = r.leg(peer, .send, t, .sum, count, &.{}, bytesAt(buf, count * t.size()));
        l.sending = false;
        var legs = [_]Leg{l};
        try r.move(&legs);
    }

    fn barrier(ptr: *anyopaque) Error!void {
        const r = self_(ptr);
        try r.live();
        var legs: [max_ranks]Leg = undefined;
        var k: usize = 0;
        for (0..r.world) |p| {
            if (p == r.me) continue;
            legs[k] = r.leg(@intCast(p), .barrier, .u8, .sum, 0, &.{}, &.{});
            k += 1;
        }
        try r.move(legs[0..k]);
    }
};

/// Each pair of ranks' socket in one process over loopback (tests): `fds[i][j]` is rank i's end to rank j.
pub fn loopback(world: u32) ![max_ranks][max_ranks]sock.Fd {
    var fds: [max_ranks][max_ranks]sock.Fd = @splat(@splat(-1));
    errdefer for (0..world) |i| for (0..world) |j| if (fds[i][j] >= 0) sock.close(fds[i][j]);
    for (0..world) |i| for (i + 1..world) |j| {
        const l = try sock.listen(0);
        defer sock.close(l);
        var addr: posix.sockaddr.in = undefined;
        var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
        if (c.getsockname(l, @ptrCast(&addr), &len) != 0) return error.SocketFailed;
        const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
        fds[i][j] = try sock.connect("127.0.0.1", std.mem.bigToNative(u16, addr.port), deadline);
        fds[j][i] = try sock.accept(l, deadline);
    };
    return fds;
}

/// Runs `body` on `world` threads over loopback sockets; a rank that fails aborts, and the first error returns.
pub fn run(gpa: std.mem.Allocator, world: u32, opts: Options, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    const fds = try loopback(world);
    var ranks: [max_ranks]Rank = undefined;
    var row: [max_ranks]sock.Fd = undefined;
    for (0..world) |i| {
        for (0..world) |j| row[j] = fds[i][j];
        ranks[i] = Rank.init(gpa, @intCast(i), world, row[0..world], opts);
    }
    defer for (ranks[0..world]) |*r| r.deinit();
    var errs: [max_ranks]?anyerror = @splat(null);
    var threads: [max_ranks]std.Thread = undefined;
    const Wrap = struct {
        fn go(r: *Rank, cx: @TypeOf(ctx), e: *?anyerror) void {
            body(r.iface(), r.me, cx) catch |err| {
                e.* = err;
                r.iface().abort();
            };
        }
    };
    var started: usize = 0;
    defer for (threads[0..started]) |t| t.join();
    errdefer for (ranks[0..world]) |*r| r.iface().abort();
    while (started < world) : (started += 1) threads[started] = try std.Thread.spawn(.{}, Wrap.go, .{ &ranks[started], ctx, &errs[started] });
    for (threads[0..started]) |t| t.join();
    started = 0;
    for (errs[0..world]) |e| if (e) |err| return err;
}
