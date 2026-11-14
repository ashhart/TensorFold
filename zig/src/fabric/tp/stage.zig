//! The TCP Kind for device memory: a call's stream waits on a word while a proxy thread copies and moves its bytes.
const std = @import("std");
const collective = @import("collective.zig");
const log = @import("log.zig");
const sock = @import("sock.zig");
const tcp = @import("tcp.zig");
const DevicePtr = collective.DevicePtr;
const Stream = collective.Stream;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

/// What staging asks of a GPU backend: the proxy's own copies, and 64-bit words written and awaited on a stream.
pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// host[0..n] <- dev[0..n], finished on return (the proxy thread's copy, on the backend's own stream).
        download: *const fn (ptr: *anyopaque, host: [*]u8, dev: DevicePtr, n: usize) Error!void,
        /// dev[0..n] <- host[0..n], finished on return.
        upload: *const fn (ptr: *anyopaque, dev: DevicePtr, host: [*]const u8, n: usize) Error!void,
        /// The stream sets a slab word to `value` once its earlier work is done.
        write: *const fn (ptr: *anyopaque, word: *u64, value: u64, stream: Stream) Error!void,
        /// The stream's later work waits until a slab word is at least `value`.
        wait: *const fn (ptr: *anyopaque, word: *u64, value: u64, stream: Stream) Error!void,
        /// Whether the stream is capturing a graph (refused: a replay would repeat the words' values).
        capturing: *const fn (ptr: *anyopaque, stream: Stream) Error!bool,
        /// Pinned host memory the device copies to and from and whose words it reads and writes.
        alloc: *const fn (ptr: *anyopaque, n: usize) Error![]align(64) u8,
        free: *const fn (ptr: *anyopaque, slab: []align(64) u8) void,
    };
};

pub const Options = struct {
    /// Bytes the proxy stages each way at a time; a larger call moves in parts, split alike on every rank.
    part_bytes: usize = 4 << 20,
};

/// Calls the host may queue ahead of the proxy, each with its own words.
const ring = 256;
/// A done word past every call: a failed or stopped proxy frees every stream's wait with it.
const released: u64 = 1 << 62;
/// How long an idle proxy or host wait spins before it naps between looks.
const spin_ns = 20 * std.time.ns_per_ms;
const nap_ns = 50 * std.time.ns_per_us;

/// A call's words, one cache line: the stream writes ready and passed, the proxy writes done.
const Words = extern struct { ready: u64, done: u64, passed: u64, pad: [5]u64 };

const Call = enum(u8) { all_reduce, all_gather, exchange, send, recv, barrier };

/// A queued call as the proxy runs it: device addresses in, parts over `tcp`, device addresses out.
const Job = struct {
    call: Call,
    t: DType = .u8,
    op: Op = .sum,
    count: usize = 0,
    peer: u32 = 0,
    s: DevicePtr = 0,
    d: DevicePtr = 0,
};

/// Own failures, beside the ones `tcp` keeps.
const Fail = enum(u8) { none, backend };

/// Waits that spin while work is recent, then nap, so an idle rank costs no core.
const Idle = struct {
    since: u64 = 0,

    fn pause(i: *Idle) void {
        const now = sock.nowNs();
        if (i.since == 0) i.since = now;
        if (now - i.since < spin_ns) std.atomic.spinLoopHint() else sock.sleepNs(nap_ns);
    }
};

/// Each call returns once queued on its stream; a failure shows at a later call or `check`, and frees the streams.
pub const Rank = struct {
    net: *tcp.Rank,
    dev: Device,
    opts: Options,
    /// `ring` calls' words, then the bytes the proxy sends and receives.
    slab: []align(64) u8 = &.{},
    jobs: [ring]Job = undefined,
    /// Calls the host queued and the proxy finished, counted from 1; call q uses words and jobs [q % ring].
    queued: std.atomic.Value(u64) = .init(0),
    finished: std.atomic.Value(u64) = .init(0),
    fail: std.atomic.Value(Fail) = .init(.none),
    stop: std.atomic.Value(bool) = .init(false),
    proxy: ?std.Thread = null,

    /// Staging over `net`, used only by the proxy from here on; every rank opens alike and checks the part size.
    pub fn open(r: *Rank, net: *tcp.Rank, dev: Device, opts: Options) Error!void {
        r.* = .{ .net = net, .dev = dev, .opts = opts };
        // sizes first, then the checks: a bad size is refused on every rank, not left to a peer's deadline
        const mine = [1]u64{opts.part_bytes};
        var all: [tcp.max_ranks]u64 = undefined;
        try net.iface().allGather(@intFromPtr(&mine), @intFromPtr(&all), 1, .i64, null);
        for (all[0..net.world]) |p| if (p != opts.part_bytes or p % 64 != 0 or p < 8 * @as(u64, net.world)) return error.Invalid;
        r.slab = try dev.vtable.alloc(dev.ptr, ring * @sizeOf(Words) + 2 * opts.part_bytes);
        @memset(r.words()[0..ring], std.mem.zeroes(Words));
        r.proxy = std.Thread.spawn(.{}, loop, .{r}) catch {
            dev.vtable.free(dev.ptr, r.slab);
            return error.BackendFailed;
        };
    }

    /// Stops the proxy and frees the slab once the streams are past their last calls (the caller synchronised them).
    pub fn deinit(r: *Rank) void {
        r.stop.store(true, .release);
        if (r.proxy) |t| t.join();
        r.proxy = null;
        // a stream still in a call has until the deadline to pass it; past that the slab leaks, never freed early
        const deadline = sock.nowNs() +| r.net.opts.timeout_ns;
        const last = r.queued.raw;
        var q = @max(1, last -| (ring - 1));
        while (q <= last) {
            if (r.jobs[q % ring].call == .barrier or @atomicLoad(u64, &r.words()[q % ring].passed, .acquire) >= q) {
                q += 1;
            } else if (sock.nowNs() > deadline) {
                log.warn("[tensorfold] tp: a stream never passed its staged call {d}; the slab of {d} bytes is kept", .{ q, r.slab.len });
                return;
            } else std.Thread.yield() catch {};
        }
        r.dev.vtable.free(r.dev.ptr, r.slab);
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
    };

    fn self_(ptr: *anyopaque) *Rank {
        return @ptrCast(@alignCast(ptr));
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return self_(ptr).net.me;
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return self_(ptr).net.world;
    }

    fn words(r: *Rank) [*]Words {
        return @ptrCast(r.slab.ptr);
    }

    /// The bytes the proxy sends; the bytes it receives follow them.
    fn tx(r: *Rank) [*]u8 {
        return r.slab.ptr + ring * @sizeOf(Words);
    }

    fn rx(r: *Rank) [*]u8 {
        return r.tx() + r.opts.part_bytes;
    }

    fn failed(r: *Rank) bool {
        if (r.fail.load(.acquire) != .none) return true;
        r.net.iface().check() catch return true;
        return false;
    }

    fn check(ptr: *anyopaque) Error!void {
        const r = self_(ptr);
        if (r.fail.load(.acquire) == .backend) return error.BackendFailed;
        return r.net.iface().check();
    }

    fn abort(ptr: *anyopaque) void {
        self_(ptr).net.iface().abort();
    }

    /// A device call that failed: the communicator fails here and at the peers, and the streams are freed.
    fn device(r: *Rank, res: Error!void) Error!void {
        res catch {
            _ = r.fail.cmpxchgStrong(.none, .backend, .acq_rel, .acquire);
            r.net.iface().abort();
            r.release();
            return error.BackendFailed;
        };
    }

    /// Every done word past any call, so no stream waits on a proxy that will not answer.
    fn release(r: *Rank) void {
        for (r.words()[0..ring]) |*w| @atomicStore(u64, &w.done, released, .release);
    }

    fn loop(r: *Rank) void {
        var q: u64 = 1;
        var idle: Idle = .{};
        while (true) {
            if (r.stop.load(.acquire)) return r.release();
            if (r.queued.load(.acquire) < q) {
                idle.pause();
                continue;
            }
            idle = .{};
            const j = r.jobs[q % ring];
            const moved = !r.failed() and (j.call == .barrier or r.ready(q));
            if (moved) r.move(j) catch {}; // the failure stays in `tcp` or `fail`, for `check`
            if (!moved or r.failed()) r.release() else if (j.call != .barrier) @atomicStore(u64, &r.words()[q % ring].done, q, .release);
            r.finished.store(q, .release);
            q += 1;
        }
    }

    /// Waits for the stream to reach call `q`; false on a failure or a stop.
    fn ready(r: *Rank, q: u64) bool {
        var idle: Idle = .{};
        while (@atomicLoad(u64, &r.words()[q % ring].ready, .acquire) < q) {
            if (r.stop.load(.acquire) or r.failed()) return false;
            idle.pause();
        }
        return true;
    }

    /// One call in parts of at most `part_bytes`: each part copied out, moved over `tcp`, and copied back.
    fn move(r: *Rank, j: Job) Error!void {
        const c = r.net.iface();
        r.net.part = 0;
        if (j.call == .barrier) return c.barrier();
        const d = r.dev;
        const w = r.net.world;
        const size = j.t.size();
        const n = j.count * size;
        const per = (if (j.call == .all_gather) r.opts.part_bytes / w else r.opts.part_bytes) / size;
        const s = @intFromPtr(r.tx());
        const o = @intFromPtr(r.rx());
        var a: usize = 0;
        var k: u32 = 0;
        while (true) : (k += 1) {
            const m = @min(per, j.count - a);
            const off = a * size;
            const b = m * size;
            r.net.part = k;
            if (j.call != .recv and b > 0) try r.device(d.vtable.download(d.ptr, r.tx(), j.s + off, b));
            switch (j.call) {
                .all_reduce => try c.allReduce(s, o, m, j.t, j.op, null),
                .all_gather => try c.allGather(s, o, m, j.t, null),
                .exchange => try c.exchange(s, o, m, j.t, j.peer, null),
                .send => try c.send(s, m, j.t, j.peer, null),
                .recv => try c.recv(o, m, j.t, j.peer, null),
                .barrier => unreachable,
            }
            if (b > 0) switch (j.call) {
                .all_gather => for (0..w) |i| try r.device(d.vtable.upload(d.ptr, j.d + i * n + off, r.rx() + i * b, b)),
                .send => {},
                else => try r.device(d.vtable.upload(d.ptr, j.d + off, r.rx(), b)),
            };
            a += m;
            if (a >= j.count) return;
        }
    }

    /// Queues `j` for the proxy; with a stream, the stream marks it ready, waits for its done word and passes it.
    fn hand(r: *Rank, j: Job, stream: ?Stream) Error!u64 {
        const q = r.queued.raw + 1;
        // call q takes the words of call q - ring, which the proxy must have finished
        var idle: Idle = .{};
        while (r.finished.load(.acquire) + ring < q) {
            if (r.failed()) return error.Aborted;
            idle.pause();
        }
        const d = r.dev;
        const w = &r.words()[q % ring];
        if (stream) |s| try r.device(d.vtable.write(d.ptr, &w.ready, q, s));
        r.jobs[q % ring] = j;
        r.queued.store(q, .release);
        if (stream) |s| {
            try r.device(d.vtable.wait(d.ptr, &w.done, q, s));
            try r.device(d.vtable.write(d.ptr, &w.passed, q, s));
        }
        return q;
    }

    fn staged(r: *Rank, j: Job, stream: Stream) Error!void {
        if (r.failed()) return error.Aborted;
        if (try r.dev.vtable.capturing(r.dev.ptr, stream)) return error.Unsupported;
        _ = try r.hand(j, stream);
        if (r.failed()) return check(r);
    }

    fn allReduce(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, count: usize, t: DType, op: Op, stream: Stream) Error!void {
        return self_(ptr).staged(.{ .call = .all_reduce, .t = t, .op = op, .count = count, .s = s, .d = d }, stream);
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, count: usize, t: DType, stream: Stream) Error!void {
        return self_(ptr).staged(.{ .call = .all_gather, .t = t, .count = count, .s = s, .d = d }, stream);
    }

    /// Two ranks trade; as with NCCL, `s` and `d` must not overlap.
    fn exchange(ptr: *anyopaque, s: DevicePtr, d: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        const r = self_(ptr);
        try collective.checkPeer(r.net.me, r.net.world, peer);
        return r.staged(.{ .call = .exchange, .t = t, .count = count, .peer = peer, .s = s, .d = d }, stream);
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        const r = self_(ptr);
        try collective.checkPeer(r.net.me, r.net.world, peer);
        return r.staged(.{ .call = .send, .t = t, .count = count, .peer = peer, .s = buf }, stream);
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        const r = self_(ptr);
        try collective.checkPeer(r.net.me, r.net.world, peer);
        return r.staged(.{ .call = .recv, .t = t, .count = count, .peer = peer, .d = buf }, stream);
    }

    /// Returns once every rank has called it and the proxy has moved this rank's earlier calls.
    fn barrier(ptr: *anyopaque) Error!void {
        const r = self_(ptr);
        if (r.failed()) return error.Aborted;
        const q = try r.hand(.{ .call = .barrier }, null);
        var idle: Idle = .{};
        while (r.finished.load(.acquire) < q) idle.pause();
        if (r.failed()) return check(r);
    }
};

/// Runs `body` on `world` threads, rank i a staged rank on `devs[i]` over loopback; the first error returns.
pub fn run(gpa: std.mem.Allocator, net: tcp.Options, devs: []const Device, opts: Options, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    const world: u32 = @intCast(devs.len);
    const fds = try tcp.loopback(world);
    var nets: [tcp.max_ranks]tcp.Rank = undefined;
    var row: [tcp.max_ranks]sock.Fd = undefined;
    for (0..world) |i| {
        for (0..world) |j| row[j] = fds[i][j];
        nets[i] = tcp.Rank.init(gpa, @intCast(i), world, row[0..world], net);
    }
    defer for (nets[0..world]) |*n| n.deinit();
    var errs: [tcp.max_ranks]?anyerror = @splat(null);
    var threads: [tcp.max_ranks]std.Thread = undefined;
    const Wrap = struct {
        fn go(n: *tcp.Rank, d: Device, o: Options, cx: @TypeOf(ctx), e: *?anyerror) void {
            var r: Rank = undefined;
            r.open(n, d, o) catch |err| {
                e.* = err;
                n.iface().abort();
                return;
            };
            defer r.deinit();
            body(r.iface(), n.me, cx) catch |err| {
                e.* = err;
                r.iface().abort();
            };
        }
    };
    var started: usize = 0;
    defer for (threads[0..started]) |t| t.join();
    errdefer for (nets[0..world]) |*n| n.iface().abort();
    while (started < world) : (started += 1) threads[started] = try std.Thread.spawn(.{}, Wrap.go, .{ &nets[started], devs[started], opts, ctx, &errs[started] });
    for (threads[0..started]) |t| t.join();
    started = 0;
    for (errs[0..world]) |e| if (e) |err| return err;
}
