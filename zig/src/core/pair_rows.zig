//! Two Macs' exchanges: GPU part, posted(x), host MCDMA relay, landed(x); the event wait is a real scheduler hand-off.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const fabric = @import("fabric");
const sources = @import("kernel_sources");
const settings = @import("../families/flashnext/tp_settings.zig");
const Fence = @import("fence.zig").Fence;
const Owner = @import("../families/flashnext/owned.zig").Owner;
const pair_layout = @import("pair_rows_layout.zig");
const pair_loopback = @import("pair_rows_loopback.zig");

pub const Settings = settings.Settings;
pub const Ref = @import("exlr8_projection.zig").Ref;

pub const command_words = pair_layout.command_words;
pub const Slot = pair_layout.Slot;
pub const seam_failed = pair_layout.seam_failed;
pub const Seam = pair_layout.Seam;
pub const elems_per_word = pair_layout.elems_per_word;
pub const stageBytes = pair_layout.stageBytes;
pub const gatherBytes = pair_layout.gatherBytes;
pub const pickBytes = pair_layout.pickBytes;
pub const frameBytes = pair_layout.frameBytes;
pub const toFrame = pair_layout.toFrame;
pub const seedValue = pair_layout.seedValue;
pub const Loopback = pair_loopback.Loopback;

const PAGE = pair_layout.PAGE;
const pages = pair_layout.pages;
const CMD_BYTES = pair_layout.CMD_BYTES;
const JOBS = pair_layout.JOBS;
const slotOf = pair_layout.slotOf;
const Layout = pair_layout.Layout;
const FLAG = pair_layout.FLAG;
const HELLO = pair_layout.HELLO;
const CMD_FLAG = pair_layout.CMD_FLAG;
const CMD_ACK = pair_layout.CMD_ACK;
const READY = pair_layout.READY;
const BEAT = pair_layout.BEAT;
const POSTED = pair_layout.POSTED;
const FAILED = pair_layout.FAILED;
const OBSERVED = pair_layout.OBSERVED;
const PairRows = pair_layout.PairRows;
const seam = pair_layout.seam;

/// Mach's clock: 24 MHz, so a millisecond is 24,000 ticks -- the clock the probes' seam numbers are in.
const ticks_per_ms: u64 = 24_000;

/// How long the host waits for the GPU to reach a published seam: a hung GPU must fail the request, not wedge it.
const post_timeout_ms: u64 = 5_000;
/// The peer's part wait: 250 ms, 300x the measured tail (p99 ~700 us), so a dead peer fails fast.
const peer_timeout_ms: u64 = 250;
/// The cold-link peer wait: the first request pays one-time Metal pipelines (1.07 s measured, 0.14 s warm).
const cold_peer_timeout_ms: u64 = 5_000;
/// How long after the link's first relay the cold bound applies: 8 s covers a longer first prompt (prefill chunks).
const cold_window_ms: u64 = 8_000;

/// How long rank 1 waits for rank 0's acknowledgement of a replay: a stuck peer fails the request, it is not wedged.
const cmd_ack_timeout_ms: u64 = 60_000;

/// How long either rank waits for the other's ready word: a cold 27B load on the far Mac, five minutes at most.
const ready_timeout_ms: u64 = 300_000;

/// The close's last flush: microseconds when the peer is there, and never the library's whole timeout when it is gone.
const last_flush_ns: u64 = 2 * std.time.ns_per_s;

threadlocal var raised = false;

/// This thread at the user-interactive QoS once: busy CPUs must not delay the pair's sends and commands.
pub fn raise() void {
    if (raised) return;
    _ = platform.c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
    raised = true;
}

/// Whether 32-bit counter a has reached b across the wrap.
fn reached(a: u32, b: u32) bool {
    return @as(i32, @bitCast(a -% b)) >= 0;
}

fn ticksNow() u64 {
    return platform.c.mach_absolute_time();
}

/// A short sleep, for the host threads' waits that are not on the seam's critical path.
fn sleepUs(us: u64) void {
    const ts: platform.c.timespec = .{ .sec = @intCast(us / 1_000_000), .nsec = @intCast(us % 1_000_000 * 1000) };
    _ = platform.c.nanosleep(&ts, null);
}

pub const Pair = struct {
    rank: u32,
    peer: u32,
    l: Layout,
    ep: *fabric.mcdma.Endpoint,
    rd: fabric.rdma.Rdma,
    win: []u8,
    wbuf: mtl.Buffer,
    stage_pipe: mtl.Pipeline,
    gather_pipe: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    /// Orders the window's untracked writes across the encoders a seam splits.
    fence: Fence,
    /// GPU -> host: seam x's part is written, relay it. Host -> GPU: the peer's part for x is in place, add it.
    posted: mtl.SharedEvent,
    landed: mtl.SharedEvent,
    /// Exchanges encoded so far: the same count on both Macs.
    x: u64 = 0,
    bytes: []u32,
    queued: std.atomic.Value(u32) = .init(0),
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    broken: std.atomic.Value(bool) = .init(false),
    /// The exchange the service thread is on: the round number in a failure line.
    seam_x: std.atomic.Value(u32) = .init(0),
    /// The request this pair is on, for failure lines; `beginRound` bumps it -- a label only, failures are permanent.
    round: std.atomic.Value(u32) = .init(0),
    /// Commands sent (rank 0) or taken (rank 1) so far.
    cmd: u64 = 0,
    /// Beats this Mac's watchdog has signalled to the peer: its counter for the peer's silence check.
    beats: u64 = 0,
    /// The GPU objects this pair holds: released in `deinit`, the window's buffer before the memory behind it.
    owned: Owner(mtl) = .{},
    /// The allocator `init` was given: `bytes`, the settings arena and this struct go back to it.
    gpa: std.mem.Allocator,
    /// The parsed settings file: the endpoint's config points into it, so it outlives `ep`.
    arena: *std.heap.ArenaAllocator,

    /// Connect to the peer named in `settings_path` (speed-up mode's file), exchanges up to `bytes`, and start sending.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, settings_path: []const u8, bytes: usize) !*Pair {
        const l: Layout = .{ .slot = pages(bytes) };
        const text = try std.Io.Dir.cwd().readFileAlloc(io, settings_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        const arena = try gpa.create(std.heap.ArenaAllocator);
        errdefer gpa.destroy(arena);
        arena.* = .init(gpa);
        errdefer arena.deinit();
        // the endpoint's config keeps `s.links`, so the parse outlives `text` in the arena, not on the stack
        const s = try settings.read(arena.allocator(), text);
        if (s.rank > 1 or s.links.len != 1) return error.PairTwoRanksOneLink;
        const lib = try gpa.dupeSentinel(u8, s.library, 0);
        defer gpa.free(lib);
        const ep = try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = l.window(), .staging_bytes = 4 << 20, .links = s.links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        errdefer ep.deinit();
        var owned: Owner(mtl) = .{};
        errdefer owned.deinit();
        const rd = ep.rdma();
        const win = rd.window();
        const lib_m = try mtl.Library.fromSource(device, sources.core_pair_rows, mtl.CompileOptions.mlx());
        defer lib_m.deinit();
        const fence = try Fence.init(device);
        errdefer fence.deinit();
        const posted = try device.sharedEvent();
        errdefer posted.deinit();
        const landed = try device.sharedEvent();
        errdefer landed.deinit();
        const p = try gpa.create(Pair);
        errdefer gpa.destroy(p);
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        p.* = .{
            .rank = s.rank,
            .peer = 1 - s.rank,
            .l = l,
            .ep = ep,
            .rd = rd,
            .win = win,
            .wbuf = try owned.keep(gpa, try device.bufferNoCopy(win.ptr, win.len, opts)),
            .stage_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "pair_stage", false)),
            .gather_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "pair_gather", false)),
            .argmax_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "pair_argmax", false)),
            .pick_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "pair_pick", false)),
            .fence = fence,
            .posted = posted,
            .landed = landed,
            .bytes = try gpa.alloc(u32, JOBS),
            .gpa = gpa,
            .arena = arena,
            .owned = owned,
        };
        owned = .{}; // the pair owns them now: the error path must not release them twice
        errdefer gpa.free(p.bytes);
        // both Macs up before the first exchange: a word each way
        try rd.signal(p.peer, l.flags() + HELLO, 1);
        while (@atomicLoad(u64, p.word64(l.flags() + HELLO), .acquire) < 1) std.atomic.spinLoopHint();
        p.thread = try std.Thread.spawn(.{}, service, .{p});
        return p;
    }

    /// Rank 0 tells rank 1 it is done (its `next` returns null); then the sending thread and the link close.
    pub fn deinit(p: *Pair) void {
        if (p.rank == 0 and !p.failed()) p.send(0, &.{}) catch {};
        p.stop.store(true, .release);
        if (p.thread) |t| t.join();
        if (!p.failed()) flushLast(p);
        p.posted.deinit();
        p.landed.deinit();
        p.fence.deinit();
        p.owned.deinit(); // the window's buffer before the endpoint frees the memory it aliases
        p.ep.deinit();
        const gpa = p.gpa;
        gpa.free(p.bytes);
        p.arena.deinit();
        gpa.destroy(p.arena);
        gpa.destroy(p);
    }

    /// Hand the last seam over without the library's whole timeout: a peer that is gone leaves nothing to deliver.
    fn flushLast(p: *Pair) void {
        if (fabric.words.checkedNowNs()) |now| p.rd.setDeadline(now + last_flush_ns) else |_| return;
        p.rd.flush() catch {};
    }

    fn word64(p: *const Pair, off: usize) *u64 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }
    fn word32(p: *const Pair, off: usize) *u32 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }

    /// Seams the host gave up on: nonzero means the run is invalid.
    pub fn gaveUp(p: *const Pair) u32 {
        return @atomicLoad(u32, p.word32(p.l.sync() + FAILED * 4), .acquire);
    }

    /// Seams the GPU itself refused after seeing FAILED: this side's own record of a failed relay.
    pub fn observed(p: *const Pair) u32 {
        return @atomicLoad(u32, p.word32(p.l.sync() + OBSERVED * 4), .acquire);
    }

    /// The exchange the service thread is on (or the one it gave up on, if it has).
    pub fn failedSeam(p: *const Pair) u32 {
        return p.seam_x.load(.acquire);
    }

    /// Labels a request; never a reset: after a seam failure the pair is dead (a half-written ring corrupts).
    pub fn beginRound(p: *Pair) u32 {
        return p.round.fetchAdd(1, .monotonic) + 1;
    }

    /// Where this rank's part for exchange x goes, and where the peer's lands: the relay's two addresses.
    pub fn sendOffset(p: *const Pair, x: u32) usize {
        return p.l.send(x);
    }
    pub fn recvOffset(p: *const Pair, x: u32) usize {
        return p.l.recv(x);
    }

    /// The link is down: failed relay, dead peer, GPU never at the seam; check after every seam and command buffer.
    pub fn failed(p: *const Pair) bool {
        return p.broken.load(.acquire) or p.gaveUp() != 0;
    }

    /// The peer's watchdog is alive: its last beat, a word its watchdog wrote here; silence means the peer is gone.
    pub fn heartbeats(p: *const Pair) u64 {
        return @atomicLoad(u64, p.word64(p.l.flags() + pair_layout.BEAT), .acquire);
    }

    /// Tell the peer this rank's watchdog is alive; a failed signal is the peer gone, not nothing to do.
    pub fn beat(p: *Pair) void {
        p.beats +%= 1;
        p.rd.signal(p.peer, p.l.flags() + pair_layout.BEAT, p.beats) catch p.broken.store(true, .release);
    }

    /// The next exchange: take it before the producer writes, then pass the same x to stage/gather/argmax/pick.
    pub fn nextX(p: *const Pair) u32 {
        return @truncate(p.x + 1);
    }

    /// The exchange slot for this seam's part: where this rank's rows go, and where the peer's land.
    pub fn part(p: *const Pair) Seam {
        const x: u32 = @truncate(p.x + 1);
        return .{ .x = x, .ref = .{ .buf = p.wbuf, .off = p.l.send(x) } };
    }

    /// Publish x and its bytes to the service thread; x must be `part`'s, so an out-of-order one panics in release too.
    fn queue(p: *Pair, x: u32, bytes: usize) void {
        if (x != @as(u32, @truncate(p.x + 1))) @panic("pair: an exchange was encoded out of order (part() not called first)");
        if (bytes > p.l.slot) @panic("pair: this exchange's part does not fit the window's slot (speed-up's rows too large)");
        p.x += 1;
        p.bytes[x % JOBS] = @intCast(bytes);
        p.queued.store(x, .release);
    }

    /// Before `seam`: this rank's `width` columns of `rows` rows from `src` into the slot, bf16 to bf16, no arithmetic.
    pub fn stage(p: *Pair, e: *mtl.ComputeEncoder, x: u32, src: Ref, rows: u32, width: u32, stride: u32) void {
        if (rows == 0 or width == 0 or width > stride) @panic("pair: stage() wants rows > 0 and a width inside the source row");
        e.setPipeline(p.stage_pipe);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setValue(PairRows{ .width = width, .stride = stride, .rank = p.rank, .x = x }, 1);
        e.setBuffer(src.buf, src.off, 2);
        e.setBuffer(p.wbuf, p.l.send(x), 3);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// After `stage` (or the producer): relay, then pack this rank's `width` at `rank * width` and the peer's after.
    pub fn gather(p: *Pair, e: *mtl.ComputeEncoder, x: u32, mine: Ref, out: Ref, rows: u32, width: u32, out_stride: u32) void {
        if (rows == 0 or width == 0 or width % elems_per_word != 0) @panic("pair: gather() wants rows > 0 and a width that fills whole window words");
        if (out_stride < 2 * width) @panic("pair: gather()'s frame row is narrower than both halves");
        if (mine.buf.id != p.wbuf.id or mine.off != p.l.send(x)) @panic("pair: gather()'s part is not the region part() returned");
        p.queue(x, stageBytes(rows, width));
        seam(e, p.fence, p.posted, p.landed, x);
        e.setPipeline(p.gather_pipe);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setValue(PairRows{ .width = width, .stride = out_stride, .rank = p.rank, .x = x }, 1);
        e.setBuffer(mine.buf, mine.off, 2);
        e.setBuffer(p.wbuf, p.l.recv(x), 3);
        e.setBuffer(out.buf, out.off, 4);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// Each row's argmax over this rank's n fp32 logits from id lo, row stride `stride`, then the pick over both Macs'.
    pub fn argmax(p: *Pair, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        p.argmaxWidth(e, x, logits, stride, lo, n, rows, picks, 0);
    }

    /// The same seam over bf16 logits (Qwen3.8-27B's `frame.logits`): one kernel, `dims.w` selects the load width.
    pub fn argmaxBf16(p: *Pair, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        p.argmaxWidth(e, x, logits, stride, lo, n, rows, picks, 1);
    }

    /// `dtype` 0 is fp32 logits, 1 is bf16; the exchange is (fp32 value, id) either way, so pair_pick is one path.
    fn argmaxWidth(p: *Pair, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref, dtype: u32) void {
        if (rows == 0 or n == 0) @panic("pair: argmax() wants rows > 0 and n > 0");
        p.queue(x, pickBytes(rows));
        e.setPipeline(p.argmax_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setValue([4]u32{ stride, lo, n, dtype }, 1);
        e.setBuffer(p.wbuf, p.l.send(x), 2);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
        seam(e, p.fence, p.posted, p.landed, x); // the encoder boundary orders the argmax before the part is relayed
        e.setPipeline(p.pick_pipe);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setValue([2]u32{ rows, x }, 1);
        e.setBuffer(p.wbuf, p.l.send(x), 2);
        e.setBuffer(p.wbuf, p.l.recv(x), 3);
        e.setBuffer(picks.buf, picks.off, 4);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    }

    /// Rank 0: command `kind` (never 0, which ends rank 1's loop) and its words, once rank 1 took the one before.
    pub fn send(p: *Pair, kind: u32, words: []const u32) !void {
        if (p.rank != 0) return error.FollowsPeer;
        if (words.len > command_words) return error.CommandTooLong;
        raise();
        try p.reach(p.l.flags() + CMD_ACK, p.cmd, cmd_ack_timeout_ms);
        p.cmd += 1;
        const head: [2]u32 = .{ kind, @intCast(words.len) };
        const out = p.l.cmdOut();
        @memcpy(p.win[out..][0..8], std.mem.asBytes(&head));
        @memcpy(p.win[out + 8 ..][0 .. 4 * words.len], std.mem.sliceAsBytes(words));
        try p.ep.writeSignalFrom(p.peer, out, p.l.cmdIn(), 8 + 4 * words.len, p.l.flags() + CMD_FLAG, p.cmd);
    }

    /// Rank 1: rank 0's next command's words into `out` and its kind; null once rank 0 is done or this Mac stops.
    pub fn next(p: *Pair, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !?u32 {
        raise();
        p.reach(p.l.flags() + CMD_FLAG, p.cmd + 1, 0) catch |err| return if (err == error.PairStopped) null else err;
        p.cmd += 1;
        const in = p.l.cmdIn();
        const head: *const [2]u32 = @ptrCast(@alignCast(p.win.ptr + in));
        if (head[1] > command_words) return error.CommandOutOfStep;
        out.clearRetainingCapacity();
        try out.appendSlice(gpa, @as([*]const u32, @ptrCast(@alignCast(p.win.ptr + in + 8)))[0..head[1]]);
        const kind = head[0];
        try p.rd.signal(p.peer, p.l.flags() + CMD_ACK, p.cmd);
        return if (kind == 0) null else kind;
    }

    /// Both Macs loaded before the first command: a word each way, each waited for with the load's bound.
    pub fn ready(p: *Pair) !void {
        try p.rd.signal(p.peer, p.l.flags() + READY, 1);
        try p.reach(p.l.flags() + READY, 1, ready_timeout_ms);
    }

    /// This Mac's command wait ends (rank 1 at close).
    pub fn stopFollowing(p: *Pair) void {
        p.stop.store(true, .release);
    }

    /// Wait until this Mac's word at `off` is `value`: spins, then short sleeps, or fails after `limit_ms` (0: none).
    fn reach(p: *Pair, off: usize, value: u64, limit_ms: u64) !void {
        const t0 = ticksNow();
        var spins: usize = 0;
        while (@atomicLoad(u64, p.word64(off), .acquire) < value) {
            if (p.stop.load(.acquire)) return error.PairStopped;
            if (p.failed()) return error.PairLinkFailed;
            if (limit_ms != 0 and ticksNow() -% t0 > limit_ms * ticks_per_ms) return error.PairPeerSilent;
            spins += 1;
            if (spins > 200_000) sleepUs(20) else std.atomic.spinLoopHint();
        }
    }

    /// Give up on x: release every event wait with the sentinel, so the blocked GPU resumes and the request aborts.
    fn fail(p: *Pair, x: u32, comptime what: []const u8, peer_ms: u64) void {
        p.broken.store(true, .release);
        _ = @atomicRmw(u32, p.word32(p.l.sync() + FAILED * 4), .Add, 1, .monotonic);
        std.log.err("pair rank {d}: request {d} seam {d} failed: {s}; the link is down and stays down (posted {d}, flag {d}, queued {d}, GPU-observed {d}; deadlines {d} ms post, {d} ms peer): restart both processes. The blocked GPU is released with the failure sentinel", .{
            p.rank, p.round.load(.acquire), x, what, p.posted.value(), @atomicLoad(u64, p.word64(p.l.flags() + FLAG), .acquire), p.queued.load(.acquire), p.observed(), post_timeout_ms, peer_ms,
        });
        p.landed.set(seam_failed);
    }

    /// One exchange at a time: part written, relay it, peer's lands, release the GPU; every wait has a deadline.
    fn service(p: *Pair) void {
        raise();
        var x: u32 = 1;
        // Ticks after which this link is no longer cold: set at the first relay, when both ranks have just loaded.
        var cold_until: u64 = 0;
        while (!p.stop.load(.acquire)) {
            p.seam_x.store(x, .release);
            // 1. The GPU must reach the seam: sleep while idle, spin once published, fail if it never arrives.
            var armed: u64 = 0;
            while (p.posted.value() < @as(u64, x)) {
                if (p.stop.load(.acquire)) return;
                if (!reached(p.queued.load(.acquire), x)) {
                    sleepUs(20);
                    continue;
                }
                if (armed == 0) armed = ticksNow();
                if (ticksNow() -% armed > post_timeout_ms * ticks_per_ms) return p.fail(x, "this Mac's GPU never reached the seam", peer_timeout_ms);
                std.atomic.spinLoopHint();
            }
            // 2. Relay: part to the peer's recv slot, arrival flag in the same MCDMA write, data and count together.
            const at = x % JOBS;
            p.ep.writeSignalFrom(p.peer, p.l.send(x), p.l.recv(x), p.bytes[at], p.l.flags() + FLAG, x) catch |err| {
                std.log.err("pair rank {d}: seam {d} relay write failed: {s}", .{ p.rank, x, @errorName(err) });
                return p.fail(x, "the relay write failed", peer_timeout_ms);
            };
            // 3. The peer's relay puts its part in our slot; its flag is proof; the cold bound, then the steady one.
            if (x == 1) cold_until = ticksNow() +% cold_window_ms * ticks_per_ms;
            const cold = cold_until != 0 and ticksNow() < cold_until;
            const budget_ms = if (cold) cold_peer_timeout_ms else peer_timeout_ms;
            const t1 = ticksNow();
            while (@atomicLoad(u64, p.word64(p.l.flags() + FLAG), .acquire) < x) {
                if (p.stop.load(.acquire)) return;
                if (ticksNow() -% t1 > budget_ms * ticks_per_ms) return p.fail(x, "the peer's part never landed", budget_ms);
                std.atomic.spinLoopHint();
            }
            // 4. Release the GPU: both parts are in; the event's wait is the acquire that makes them visible.
            @atomicStore(u32, p.word32(p.l.sync() + POSTED * 4), x, .release);
            p.landed.set(x);
            x +%= 1;
        }
    }
};

test "the window's regions do not overlap and its words sit on their pages" {
    const l: Layout = .{ .slot = pages(stageBytes(256, 6144)) };
    try std.testing.expect(l.send(1) >= l.recv(1) + l.slot and l.send(0) >= 2 * l.slot + PAGE and l.send(0) + l.slot <= l.send(1));
    try std.testing.expect(l.cmdIn() >= l.send(1) + l.slot and l.cmdOut() >= l.cmdIn() + CMD_BYTES + 64);
    try std.testing.expect(l.flags() >= l.cmdOut() + 8 + 4 * command_words and BEAT + 8 <= PAGE and l.window() % PAGE == 0);
    try std.testing.expect(reached(1, 0xFFFFFFFF) and !reached(5, 6));
}

test "the parity ring alternates and one sentinel releases every exchange" {
    try std.testing.expect(slotOf(1) == .b and slotOf(2) == .a and slotOf(3) == .b);
    const l: Layout = .{ .slot = pages(stageBytes(16, 5120)) };
    try std.testing.expect(l.recv(1) == l.slot and l.recv(2) == 0 and l.send(2) == 2 * l.slot + PAGE);
    try std.testing.expectEqual(pages(stageBytes(16, 5120)), l.slot); // the 27B's 16-row prefill part, page-rounded
    try std.testing.expect(seam_failed > std.math.maxInt(u32)); // every waitFor(x) in a buffer is satisfied by it
}
