//! In-process ranks over host memory: the `Collective` contract without a GPU (desyncs refused, abort wakes waiters).
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

pub const max_ranks = 8;

const Call = enum(u8) { none, all_reduce, all_gather, all_gather_v, exchange, barrier };

const Post = struct { call: Call = .none, send: DevicePtr = 0, recv: DevicePtr = 0, count: usize = 0, t: DType = .u8, op: Op = .sum, peer: u32 = 0, lens: DevicePtr = 0 };

/// One-way message slot from rank `src` to rank `dst` (send / recv).
const Slot = struct { buf: std.atomic.Value(DevicePtr) = .init(0), bytes: usize = 0, taken: std.atomic.Value(bool) = .init(false) };

pub const Group = struct {
    world: u32,
    /// A wait longer than this is `error.Timeout` (a peer that never calls).
    timeout_ns: u64 = 10 * std.time.ns_per_s,
    posts: [max_ranks]Post = @splat(.{}),
    arrived: std.atomic.Value(u32) = .init(0),
    generation: std.atomic.Value(u32) = .init(0),
    aborted: std.atomic.Value(bool) = .init(false),
    /// Ranks (a bit each) whose transport refuses device lengths: their `all_gather_v` moves whole strides (tests).
    var_refuse: u32 = 0,
    slots: [max_ranks][max_ranks]Slot = @splat(@splat(.{})),

    pub fn init(world: u32) Group {
        std.debug.assert(world >= 1 and world <= max_ranks);
        return .{ .world = world };
    }

    pub fn rank(g: *Group, r: u32) Rank {
        std.debug.assert(r < g.world);
        return .{ .group = g, .me = r };
    }

    /// Sense-reversing barrier over every rank; `error.Aborted` once any rank aborts.
    fn meet(g: *Group) Error!void {
        const gen = g.generation.load(.acquire);
        if (g.arrived.fetchAdd(1, .acq_rel) + 1 == g.world) {
            g.arrived.store(0, .release);
            _ = g.generation.fetchAdd(1, .acq_rel);
            return if (g.aborted.load(.acquire)) error.Aborted else {};
        }
        const t0 = now();
        while (g.generation.load(.acquire) == gen) {
            if (g.aborted.load(.acquire)) return error.Aborted;
            if (now() - t0 > g.timeout_ns) return error.Timeout;
            std.Thread.yield() catch {};
        }
        if (g.aborted.load(.acquire)) return error.Aborted;
    }
};

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn bytesAt(p: DevicePtr, n: usize) []u8 {
    if (n == 0) return &.{};
    return @as([*]u8, @ptrFromInt(p))[0..n];
}

pub const Rank = struct {
    group: *Group,
    me: u32,
    /// `varAgreed`'s answer, once this communicator agreed.
    var_agreed: ?bool = null,

    pub fn iface(self: *Rank) collective.Collective {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: collective.Collective.VTable = .{
        .kind = .host,
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
        return self_(ptr).group.world;
    }

    /// Posts this rank's call, waits for all, and refuses calls that do not match rank 0's.
    fn enter(self: *Rank, p: Post) Error!void {
        const g = self.group;
        if (g.aborted.load(.acquire)) return error.Aborted;
        g.posts[self.me] = p;
        try g.meet();
        const r0 = g.posts[0];
        if (r0.call != p.call or r0.count != p.count or r0.t != p.t or r0.op != p.op) {
            abort(self);
            return error.Invalid;
        }
    }

    fn allReduce(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, op: Op, _: collective.Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try self.enter(.{ .call = .all_reduce, .send = s, .recv = r, .count = count, .t = t, .op = op });
        const n = count * t.size();
        var ins: [max_ranks][]const u8 = undefined;
        for (0..g.world) |i| ins[i] = bytesAt(g.posts[i].send, n);
        // the output may alias this rank's input: reduce into scratch first, then meet before writing
        var scratch: [1 << 16]u8 = undefined;
        if (n > scratch.len) return error.Unsupported;
        if (n > 0) reference.reduce(scratch[0..n], ins[0..g.world], t, op);
        try g.meet();
        @memcpy(bytesAt(r, n), scratch[0..n]);
        try g.meet();
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, _: collective.Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try self.enter(.{ .call = .all_gather, .send = s, .recv = r, .count = count, .t = t });
        const n = count * t.size();
        const out = bytesAt(r, n * g.world);
        for (0..g.world) |i| @memcpy(out[i * n ..][0..n], bytesAt(g.posts[i].send, n));
        try g.meet();
    }

    fn lensAt(p: DevicePtr, world: u32) []const i32 {
        return @as([*]const i32, @ptrFromInt(p))[0..world];
    }

    /// Copies only each rank's lens[r] bytes (`var_refuse` ranks take whole strides); unequal lengths are refused.
    fn allGatherV(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, stride: usize, lens: DevicePtr, t: DType, _: collective.Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try self.enter(.{ .call = .all_gather_v, .send = s, .recv = r, .count = stride, .t = t, .lens = lens });
        const n = stride * t.size();
        const mine = lensAt(lens, g.world);
        for (0..g.world) |i| if (!std.mem.eql(i32, mine, lensAt(g.posts[i].lens, g.world))) {
            abort(self);
            return error.Invalid;
        };
        const whole = g.var_refuse & (@as(u32, 1) << @intCast(self.me)) != 0;
        const out = bytesAt(r, n * g.world);
        for (0..g.world) |i| {
            if (mine[i] < 0 or mine[i] > n) {
                abort(self);
                return error.Invalid;
            }
            const len: usize = if (whole) n else @intCast(mine[i]);
            @memcpy(out[i * n ..][0..len], bytesAt(g.posts[i].send, len));
        }
        try g.meet();
    }

    fn varFits(ptr: *anyopaque, bytes: usize) bool {
        const self = self_(ptr);
        return bytes > 0 and self.group.var_refuse & (@as(u32, 1) << @intCast(self.me)) == 0;
    }

    /// One int all-gather of `varFits(16)` the first time, then the kept answer.
    fn varAgreed(ptr: *anyopaque) Error!bool {
        const self = self_(ptr);
        if (self.var_agreed) |v| return v;
        const mine: i32 = @intFromBool(varFits(ptr, 16));
        var all: [max_ranks]i32 = undefined;
        try allGather(ptr, @intFromPtr(&mine), @intFromPtr(&all), 1, .i32, null);
        self.var_agreed = std.mem.min(i32, all[0..self.group.world]) == 1;
        return self.var_agreed.?;
    }

    fn exchange(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, peer: u32, _: collective.Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try collective.checkPeer(self.me, g.world, peer);
        try self.enter(.{ .call = .exchange, .send = s, .recv = r, .count = count, .t = t, .peer = peer });
        if (g.posts[peer].peer != self.me) {
            abort(self);
            return error.Invalid;
        }
        const n = count * t.size();
        @memcpy(bytesAt(r, n), bytesAt(g.posts[peer].send, n));
        try g.meet();
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, _: collective.Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try collective.checkPeer(self.me, g.world, peer);
        const slot = &g.slots[self.me][peer];
        slot.bytes = count * t.size();
        slot.taken.store(false, .release);
        slot.buf.store(if (buf == 0) 1 else buf, .release);
        try waitFor(g, &slot.taken, true);
        slot.buf.store(0, .release);
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, _: collective.Stream) Error!void {
        const self = self_(ptr);
        const g = self.group;
        try collective.checkPeer(self.me, g.world, peer);
        const slot = &g.slots[peer][self.me];
        const t0 = now();
        var src = slot.buf.load(.acquire);
        while (src == 0) : (src = slot.buf.load(.acquire)) {
            if (g.aborted.load(.acquire)) return error.Aborted;
            if (now() - t0 > g.timeout_ns) return error.Timeout;
            std.Thread.yield() catch {};
        }
        const n = count * t.size();
        if (slot.bytes != n) {
            abort(self);
            return error.Invalid;
        }
        @memcpy(bytesAt(buf, n), bytesAt(src, n));
        slot.taken.store(true, .release);
    }

    fn waitFor(g: *Group, flag: *std.atomic.Value(bool), want: bool) Error!void {
        const t0 = now();
        while (flag.load(.acquire) != want) {
            if (g.aborted.load(.acquire)) return error.Aborted;
            if (now() - t0 > g.timeout_ns) return error.Timeout;
            std.Thread.yield() catch {};
        }
    }

    fn barrier(ptr: *anyopaque) Error!void {
        const self = self_(ptr);
        try self.enter(.{ .call = .barrier });
        try self.group.meet();
    }

    fn check(ptr: *anyopaque) Error!void {
        if (self_(ptr).group.aborted.load(.acquire)) return error.Aborted;
    }

    fn abort(ptr: *anyopaque) void {
        self_(ptr).group.aborted.store(true, .release);
    }
};

/// Runs `body(rank_collective, rank, ctx)` on `world` threads at once and returns the first error any rank hit.
pub fn run(world: u32, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    return runWith(world, .{}, ctx, body);
}

pub const Options = struct {
    /// `Group.var_refuse`: ranks whose transport takes no device lengths.
    var_refuse: u32 = 0,
};

pub fn runWith(world: u32, opts: Options, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    var group = Group.init(world);
    group.var_refuse = opts.var_refuse;
    var ranks: [max_ranks]Rank = undefined;
    var errs: [max_ranks]?anyerror = @splat(null);
    var threads: [max_ranks]std.Thread = undefined;
    const Wrap = struct {
        fn go(r: *Rank, c: @TypeOf(ctx), e: *?anyerror) void {
            body(r.iface(), r.me, c) catch |err| {
                e.* = err;
                r.group.aborted.store(true, .release);
            };
        }
    };
    for (0..world) |i| {
        ranks[i] = group.rank(@intCast(i));
        threads[i] = try std.Thread.spawn(.{}, Wrap.go, .{ &ranks[i], ctx, &errs[i] });
    }
    for (threads[0..world]) |t| t.join();
    for (errs[0..world]) |e| if (e) |err| return err;
}
