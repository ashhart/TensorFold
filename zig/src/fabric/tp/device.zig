//! In-process ranks on one GPU, for tests (NCCL refuses two ranks on a device): host.zig's rendezvous, device copies.
const std = @import("std");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const host = @import("host.zig");
const reference = @import("reference.zig");
const DevicePtr = collective.DevicePtr;
const Stream = collective.Stream;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

pub const max_ranks = host.max_ranks;

/// One-way messages from one rank to another. Each count has one writer, so a receiver never reads an old post.
const Slot = struct {
    buf: DevicePtr = 0,
    bytes: usize = 0,
    posted: std.atomic.Value(u64) = .init(0),
    taken: std.atomic.Value(u64) = .init(0),
};

pub const Group = struct {
    base: host.Group,
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    /// Each rank's `all_gather_v` lengths, read back from the device before it posts.
    lens: [max_ranks][max_ranks]i32 = @splat(@splat(0)),
    /// Each rank's `varFits(16)`, for `varAgreed`.
    fits: [max_ranks]bool = @splat(false),
    slots: [max_ranks][max_ranks]Slot = @splat(@splat(.{})),

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator, world: u32) Group {
        return .{ .base = host.Group.init(world), .d = d, .gpa = gpa };
    }

    pub fn rank(g: *Group, r: u32) Rank {
        std.debug.assert(r < g.base.world);
        return .{ .group = g, .me = r };
    }
};

/// Each call syncs its caller's stream and returns with its copies done: not stream-ordered, and never captured.
pub const Rank = struct {
    group: *Group,
    me: u32,
    /// `varAgreed`'s answer, once this communicator agreed.
    var_agreed: ?bool = null,

    pub fn iface(self: *Rank) collective.Collective {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: collective.Collective.VTable = .{
        .kind = .device,
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
        return self_(ptr).group.base.world;
    }

    /// A driver call; a failure aborts every rank, so none waits for this one.
    fn call(self: *Rank, res: cuda.abi.Result, what: []const u8) Error!void {
        self.group.d.check(res, what) catch {
            abort(self);
            return error.BackendFailed;
        };
    }

    /// The caller's earlier work on `stream` is done, so its buffers hold what it wrote. A capturing stream is refused.
    fn ready(self: *Rank, stream: Stream) Error!void {
        const d = self.group.d;
        var status: cuda.abi.CaptureStatus = .none;
        try self.call(d.api.cuStreamIsCapturing(stream, &status), "cuStreamIsCapturing");
        if (status != .none) return error.Unsupported;
        try self.finish(stream);
    }

    fn finish(self: *Rank, stream: Stream) Error!void {
        try self.call(self.group.d.api.cuStreamSynchronize(stream), "cuStreamSynchronize");
    }

    fn copy(self: *Rank, dst: DevicePtr, src: DevicePtr, n: usize, stream: Stream) Error!void {
        if (n == 0 or dst == src) return;
        try self.call(self.group.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream), "cuMemcpyDtoDAsync");
    }

    fn download(self: *Rank, dst: []u8, src: DevicePtr, stream: Stream) Error!void {
        if (dst.len == 0) return;
        try self.call(self.group.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(dst.ptr), src, dst.len, stream), "cuMemcpyDtoHAsync");
    }

    fn upload(self: *Rank, dst: DevicePtr, src: []const u8, stream: Stream) Error!void {
        if (src.len == 0) return;
        try self.call(self.group.d.api.cuMemcpyHtoDAsync_v2(dst, @ptrCast(src.ptr), src.len, stream), "cuMemcpyHtoDAsync");
    }

    /// Every rank's input read back and reduced on the host in rank order (`reference.reduce`), then written out.
    fn allReduce(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, op: Op, stream: Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        const w = g.base.world;
        try self.ready(stream);
        try g.base.enter(self.me, .{ .call = .all_reduce, .send = s, .recv = r, .count = count, .t = t, .op = op });
        const n = count * t.size();
        const buf = g.gpa.alloc(u8, n * (w + 1)) catch {
            abort(self);
            return error.BackendFailed;
        };
        defer g.gpa.free(buf);
        var ins: [max_ranks][]const u8 = undefined;
        for (0..w) |i| {
            ins[i] = buf[i * n ..][0..n];
            try self.download(buf[i * n ..][0..n], g.base.posts[i].send, stream);
        }
        try self.finish(stream);
        const out = buf[w * n ..][0..n];
        if (n > 0) reference.reduce(out, ins[0..w], t, op);
        // the output may alias an input: every rank has read them all before any rank writes
        try g.base.meet();
        try self.upload(r, out, stream);
        try self.finish(stream);
        try g.base.meet();
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try self.ready(stream);
        try g.base.enter(self.me, .{ .call = .all_gather, .send = s, .recv = r, .count = count, .t = t });
        const n = count * t.size();
        for (0..g.base.world) |i| try self.copy(r + i * n, g.base.posts[i].send, n, stream);
        try self.finish(stream);
        try g.base.meet();
    }

    /// Copies only each rank's lens[r] bytes (`var_refuse` ranks take whole strides); unequal lengths are refused.
    fn allGatherV(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        const w = g.base.world;
        try self.ready(stream);
        try self.download(std.mem.sliceAsBytes(g.lens[self.me][0..w]), lens, stream);
        try self.finish(stream);
        try g.base.enter(self.me, .{ .call = .all_gather_v, .send = s, .recv = r, .count = stride, .t = t, .lens = lens });
        const n = stride * t.size();
        const mine = g.lens[self.me][0..w];
        for (0..w) |i| if (!std.mem.eql(i32, mine, g.lens[i][0..w])) {
            abort(self);
            return error.Invalid;
        };
        const whole = g.base.var_refuse & (@as(u32, 1) << @intCast(self.me)) != 0;
        for (0..w) |i| {
            if (mine[i] < 0 or mine[i] > n) {
                abort(self);
                return error.Invalid;
            }
            const len: usize = if (whole) n else @intCast(mine[i]);
            try self.copy(r + i * n, g.base.posts[i].send, len, stream);
        }
        try self.finish(stream);
        try g.base.meet();
    }

    fn varFits(ptr: *anyopaque, bytes: usize) bool {
        const self = self_(ptr);
        return bytes > 0 and self.group.base.var_refuse & (@as(u32, 1) << @intCast(self.me)) == 0;
    }

    /// Every rank's `varFits(16)` met once, then the kept answer.
    fn varAgreed(ptr: *anyopaque) Error!bool {
        const self = self_(ptr);
        if (self.var_agreed) |v| return v;
        const g = self.group;
        g.fits[self.me] = varFits(ptr, 16);
        try g.base.enter(self.me, .{ .call = .barrier });
        var all = true;
        for (g.fits[0..g.base.world]) |f| all = all and f;
        try g.base.meet();
        self.var_agreed = all;
        return all;
    }

    /// Posts `buf` for `peer` without waiting; `delivered` waits until the peer has copied it.
    fn post(self: *Rank, buf: DevicePtr, n: usize, peer: u32) u64 {
        const slot = &self.group.slots[self.me][peer];
        slot.buf = buf;
        slot.bytes = n;
        const k = slot.posted.load(.monotonic) + 1;
        slot.posted.store(k, .release);
        return k;
    }

    fn delivered(self: *Rank, peer: u32, k: u64) Error!void {
        const slot = &self.group.slots[self.me][peer];
        try waitFor(self.group, &slot.taken, k);
    }

    /// Waits for `peer`'s next post to this rank and copies it into `buf`, finished on return.
    fn take(self: *Rank, buf: DevicePtr, n: usize, peer: u32, stream: Stream) Error!void {
        const slot = &self.group.slots[peer][self.me];
        const k = slot.taken.load(.monotonic) + 1;
        try waitFor(self.group, &slot.posted, k);
        if (slot.bytes != n) {
            abort(self);
            return error.Invalid;
        }
        try self.copy(buf, slot.buf, n, stream);
        try self.finish(stream);
        slot.taken.store(k, .release);
    }

    fn waitFor(g: *Group, count: *std.atomic.Value(u64), want: u64) Error!void {
        const t0 = host.now();
        while (count.load(.acquire) < want) {
            if (g.base.aborted.load(.acquire)) return error.Aborted;
            if (host.now() - t0 > g.base.timeout_ns) return error.Timeout;
            std.Thread.yield() catch {};
        }
    }

    /// Two ranks trade without the others, as NCCL's pairs do: the last rank of an odd world can sit out.
    fn exchange(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        const self = self_(ptr);
        try collective.checkPeer(self.me, self.group.base.world, peer);
        if (self.group.base.aborted.load(.acquire)) return error.Aborted;
        try self.ready(stream);
        const n = count * t.size();
        const k = self.post(s, n, peer);
        try self.take(r, n, peer, stream);
        try self.delivered(peer, k);
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        const self = self_(ptr);
        try collective.checkPeer(self.me, self.group.base.world, peer);
        if (self.group.base.aborted.load(.acquire)) return error.Aborted;
        try self.ready(stream);
        try self.delivered(peer, self.post(buf, count * t.size(), peer));
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        const self = self_(ptr);
        try collective.checkPeer(self.me, self.group.base.world, peer);
        if (self.group.base.aborted.load(.acquire)) return error.Aborted;
        try self.ready(stream);
        try self.take(buf, count * t.size(), peer, stream);
    }

    fn barrier(ptr: *anyopaque) Error!void {
        const self = self_(ptr);
        try self.group.base.enter(self.me, .{ .call = .barrier });
        try self.group.base.meet();
    }

    fn check(ptr: *anyopaque) Error!void {
        if (self_(ptr).group.base.aborted.load(.acquire)) return error.Aborted;
    }

    fn abort(ptr: *anyopaque) void {
        self_(ptr).group.base.aborted.store(true, .release);
    }
};

/// Runs `body(rank_collective, rank, ctx)` on `world` threads with `context` current; returns any rank's first error.
pub fn run(context: *const cuda.Context, gpa: std.mem.Allocator, world: u32, opts: host.Options, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    var group = Group.init(context.d, gpa, world);
    group.base.var_refuse = opts.var_refuse;
    var ranks: [max_ranks]Rank = undefined;
    var errs: [max_ranks]?anyerror = @splat(null);
    var threads: [max_ranks]std.Thread = undefined;
    const Wrap = struct {
        fn go(r: *Rank, cx: *const cuda.Context, c: @TypeOf(ctx), e: *?anyerror) void {
            cx.makeCurrent() catch |err| {
                e.* = err;
                r.group.base.aborted.store(true, .release);
                return;
            };
            body(r.iface(), r.me, c) catch |err| {
                e.* = err;
                r.group.base.aborted.store(true, .release);
            };
        }
    };
    for (0..world) |i| {
        ranks[i] = group.rank(@intCast(i));
        threads[i] = std.Thread.spawn(.{}, Wrap.go, .{ &ranks[i], context, ctx, &errs[i] }) catch |err| {
            // the ranks already running hold this frame's group: stop them and wait before returning
            group.base.aborted.store(true, .release);
            for (threads[0..i]) |t| t.join();
            return err;
        };
    }
    for (threads[0..world]) |t| t.join();
    for (errs[0..world]) |e| if (e) |err| return err;
}
