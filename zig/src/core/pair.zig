//! Two Macs' tensor-parallel exchanges across retired command buffers, and rank 0's commands to rank 1.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const fabric = @import("fabric");
const sources = @import("kernel_sources");
const Handoff = @import("pair_handoff.zig").Handoff;
const deadlines = @import("pair_deadline.zig");
const shutdown = @import("pair_shutdown.zig");
const heartbeat = @import("pair_heartbeat.zig");
const settings = @import("../families/flashnext/tp_settings.zig");

test {
    _ = shutdown;
    _ = heartbeat;
}

pub const Settings = settings.Settings;
pub const Ref = @import("exlr8_projection.zig").Ref;

const PAGE = 16384;
fn pages(n: usize) usize {
    return (n + PAGE - 1) / PAGE * PAGE;
}
/// A command's words at most: a 262,144-token prompt and its head.
pub const command_words = 262144 + 64;
const CMD_BYTES = pages(8 + 4 * command_words);
const JOBS = 4096;

/// The window on both Macs: parts in and out by parity (a page before each out), commands in and out, flags, GPU words.
const Layout = struct {
    slot: usize,

    fn recv(l: Layout, x: u64) usize {
        return @as(usize, @intCast(x % 2)) * l.slot;
    }
    fn send(l: Layout, x: u64) usize {
        return 2 * l.slot + @as(usize, @intCast(x % 2)) * (PAGE + l.slot) + PAGE;
    }
    fn cmdIn(l: Layout) usize {
        return 4 * l.slot + 2 * PAGE;
    }
    fn cmdOut(l: Layout) usize {
        return l.cmdIn() + CMD_BYTES + PAGE;
    }
    fn flags(l: Layout) usize {
        return l.cmdOut() + CMD_BYTES;
    }
    fn sync(l: Layout) usize {
        return l.flags() + PAGE;
    }
    fn window(l: Layout) usize {
        return l.sync() + PAGE;
    }
};
const FLAG = 0; // u64 in the flag page: the last exchange whose peer part has landed
const HELLO = 64;
const CMD_FLAG = 128; // u64, at rank 1: the last command rank 0 has written
const CMD_ACK = 192; // u64, at rank 0: the last command rank 1 has copied out
const READY = 256; // u64: the peer has loaded its model

threadlocal var raised = false;

/// This thread at the user-interactive QoS once: busy CPUs must not delay the pair's sends and commands.
pub fn raise() void {
    if (raised) return;
    _ = platform.c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
    raised = true;
}

const PairRows = extern struct { width: u32, stride: u32, rank: u32, x: u32, eps: f32, pad0: f32 = 0, pad1: f32 = 0, pad2: f32 = 0 };
const Control = struct { local: usize, remote: usize, bytes: usize, flag: usize, value: u64, until: u64 };

pub const Pair = struct {
    benchmark_group: ?[]const u8 = null,
    rank: u32,
    peer: u32,
    l: Layout,
    ep: *fabric.mcdma.Endpoint,
    rd: fabric.rdma.Rdma,
    win: []u8,
    wbuf: mtl.Buffer,
    partial: mtl.Buffer,
    handoff: Handoff,
    stage_pipe: mtl.Pipeline,
    add_rms: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    /// Exchanges encoded so far: the same count on both Macs.
    x: u64 = 0,
    bytes: []u32,
    queued: std.atomic.Value(u64) = .init(0),
    submitted: std.atomic.Value(u64) = .init(0),
    thread: ?std.Thread = null,
    monitor: ?std.Thread = null,
    relay_done: std.atomic.Value(bool) = .init(false),
    endpoint_done: std.atomic.Value(bool) = .init(false),
    goodbye: std.atomic.Value(bool) = .init(false),
    heartbeat_ready: std.atomic.Value(bool) = .init(false),
    control_job: Control = undefined,
    control_posted: std.atomic.Value(u64) = .init(0),
    control_done: std.atomic.Value(u64) = .init(0),
    arena: std.heap.ArenaAllocator,
    gpa: std.mem.Allocator,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    /// Commands sent (rank 0) or taken (rank 1) so far.
    cmd: u64 = 0,

    /// Connect to the peer named in `settings_path` (speed-up mode's file), exchanges up to `bytes`, and start sending.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, settings_path: []const u8, bytes: usize) !*Pair {
        var relay_owns = false;
        const l: Layout = .{ .slot = pages(bytes) };
        const text = try std.Io.Dir.cwd().readFileAlloc(io, settings_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer if (!relay_owns) arena.deinit();
        const s = try settings.read(arena.allocator(), text);
        if (s.rank > 1 or s.links.len != 1) return error.PairTwoRanksOneLink;
        const lib = try gpa.dupeSentinel(u8, s.library, 0);
        defer gpa.free(lib);
        const ep = try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = l.window(), .staging_bytes = 4 << 20, .links = s.links, .timeout_ns = std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        errdefer if (!relay_owns) ep.deinit();
        const rd = ep.rdma();
        const win = rd.window();
        const lib_m = try mtl.Library.fromSource(device, sources.core_pair, mtl.CompileOptions.mlx());
        defer lib_m.deinit();
        const p = try gpa.create(Pair);
        errdefer if (!relay_owns) gpa.destroy(p);
        const wbuf = try device.bufferNoCopy(win.ptr, win.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer if (!relay_owns) wbuf.deinit();
        const partial = try device.buffer(l.slot, mtl.ResourceOptions.shared);
        errdefer if (!relay_owns) partial.deinit();
        var pipes: [4]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer if (!relay_owns) for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{ "pair_stage", "pair_add_rms", "pair_argmax", "pair_pick" }) |name| {
            pipes[made] = try mtl.Pipeline.init(device, lib_m, name, false);
            made += 1;
        }
        const jobs = try gpa.alloc(u32, JOBS);
        errdefer if (!relay_owns) gpa.free(jobs);
        p.* = .{
            .rank = s.rank,
            .peer = 1 - s.rank,
            .l = l,
            .ep = ep,
            .rd = rd,
            .win = win,
            .wbuf = wbuf,
            .partial = partial,
            .handoff = undefined,
            .stage_pipe = pipes[0],
            .add_rms = pipes[1],
            .argmax_pipe = pipes[2],
            .pick_pipe = pipes[3],
            .bytes = jobs,
            .gpa = gpa,
            .arena = arena,
        };
        p.handoff = try Handoff.init(device, &p.failed, &p.stop, p.rank);
        errdefer if (!relay_owns) p.handoff.deinit();
        p.monitor = try std.Thread.spawn(.{}, Handoff.watch, .{&p.handoff});
        errdefer if (!relay_owns) {
            p.stop.store(true, .release);
            p.monitor.?.join();
        };
        p.thread = try std.Thread.spawn(.{}, service, .{p});
        // Once a native call can start, only bounded relay cleanup may retire its resources.
        relay_owns = true;
        errdefer p.deinit();
        var inventory: Inventory = .{ .pair = p };
        p.benchmark_group = try @import("benchmark_group.zig").metalFor(gpa, io, device, &inventory, l.flags() + 4096, l.flags() + HELLO, 2, 60 * std.time.ns_per_s);
        return p;
    }

    /// The inventory uses the same bounded relay as later commands, including its native write and flush.
    const Inventory = struct {
        pair: *Pair,
        pub fn size(_: *@This()) u32 {
            return 2;
        }
        pub fn rank(i: *@This()) u32 {
            return i.pair.rank;
        }
        pub fn window(i: *@This()) []u8 {
            return i.pair.win;
        }
        pub fn write2Signal(i: *@This(), peer: u32, offset: usize, head: []const u8, body: []const u8, flag: usize, value: u64) !void {
            const p = i.pair;
            std.debug.assert(peer == p.peer and body.len == 0 and head.len < PAGE - 4096);
            const local = p.l.sync() + 4096;
            @memcpy(p.win[local..][0..head.len], head);
            try p.control(local, offset, head.len, flag, value);
        }
        pub fn flush(_: *@This()) !void {} // control's completion includes flush
    };

    /// Rank 0 tells rank 1 it is done (its `next` returns null); then the sending thread and the link close.
    pub fn deinit(p: *Pair) void {
        p.goodbye.store(p.rank == 0 and !p.failed.load(.acquire) and
            @atomicLoad(u64, p.word64(p.l.flags() + CMD_ACK), .acquire) >= p.cmd, .release);
        p.stopFollowing();
        if (p.monitor) |t| t.join();
        if (p.thread) |t| if (!shutdown.join(t, &p.relay_done, 2 * std.time.ns_per_s)) {
            std.log.warn("pair rank {d}: native relay did not retire; retaining its window and library until process exit", .{p.rank});
            return;
        };
        for ([_]mtl.Pipeline{ p.stage_pipe, p.add_rms, p.argmax_pipe, p.pick_pipe }) |pipe| pipe.deinit();
        p.handoff.deinit();
        p.partial.deinit();
        p.wbuf.deinit();
        const closer = std.Thread.spawn(.{}, closeEndpoint, .{p}) catch {
            std.log.warn("pair rank {d}: retaining native endpoint after close thread could not start", .{p.rank});
            return;
        };
        if (!shutdown.join(closer, &p.endpoint_done, 2 * std.time.ns_per_s)) {
            std.log.warn("pair rank {d}: native endpoint close did not retire; retaining its owner until process exit", .{p.rank});
            return;
        }
        if (p.benchmark_group) |b| p.gpa.free(b);
        p.gpa.free(p.bytes);
        p.arena.deinit();
        p.gpa.destroy(p);
    }
    fn closeEndpoint(p: *Pair) void {
        p.ep.deinit();
        p.endpoint_done.store(true, .release);
    }

    fn word64(p: *const Pair, off: usize) *u64 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }
    /// The producer uses GPU-owned scratch; only the guarded stage writes a transport send slot.
    pub fn part(p: *const Pair) Ref {
        return .{ .buf = p.partial };
    }

    pub fn submit(p: *Pair) void {
        p.submitted.store(p.x, .release);
    }

    fn exchange(p: *Pair, e: *mtl.ComputeEncoder, bytes: usize) !void {
        std.debug.assert(bytes <= p.l.slot);
        p.x += 1;
        const x = p.x;
        p.bytes[x % JOBS] = @intCast(bytes);
        p.queued.store(x, .release);
        e.setPipeline(p.stage_pipe);
        p.handoff.bind(e.*, x - 1, 0);
        e.setBuffer(p.partial, 0, 2);
        e.setBuffer(p.wbuf, p.l.send(x), 3);
        e.setValue(@as(u32, @intCast(bytes / 4)), 4);
        e.dispatchThreads(.{ .width = bytes / 4 }, .{ .width = 256 });
        p.submit();
        try p.handoff.seam(e, x);
    }

    /// After `part()` was written: send it, wait for the peer's, h += rank 0's + rank 1's, y = rms(h) w.
    pub fn addRms(p: *Pair, e: *mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32) !void {
        try p.exchange(e, @as(usize, rows) * width * 4);
        const x = p.x;
        e.setPipeline(p.add_rms);
        p.handoff.bind(e.*, x, 0);
        e.setValue(PairRows{ .width = width, .stride = width, .rank = p.rank, .x = 0, .eps = eps }, 2);
        e.setBuffer(p.wbuf, p.l.send(x), 3);
        e.setBuffer(p.wbuf, p.l.recv(x), 4);
        e.setBuffer(h.buf, h.off, 5);
        e.setBuffer(w.buf, w.off, 6);
        e.setBuffer(y.buf, y.off, 7);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// Each row's pick over both Macs' logits (this one's n columns from id lo, row stride `stride`) into picks.
    pub fn argmax(p: *Pair, e: *mtl.ComputeEncoder, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) !void {
        e.setPipeline(p.argmax_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setValue([4]u32{ stride, lo, n, 0 }, 1);
        e.setBuffer(p.partial, 0, 2);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
        e.barrier();
        try p.exchange(e, @as(usize, rows) * 8);
        const x = p.x;
        e.setPipeline(p.pick_pipe);
        p.handoff.bind(e.*, x, 0);
        e.setValue(rows, 2);
        e.setBuffer(p.wbuf, p.l.send(x), 3);
        e.setBuffer(p.wbuf, p.l.recv(x), 4);
        e.setBuffer(picks.buf, picks.off, 5);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    }

    /// Rank 0: command `kind` (never 0, which ends rank 1's loop) and its words, once rank 1 took the one before.
    pub fn send(p: *Pair, kind: u32, words: []const u32) !void {
        if (p.failed.load(.acquire)) return error.PairLinkFailed;
        if (p.stop.load(.acquire)) return error.PairStopped;
        if (p.rank != 0) return error.FollowsPeer;
        if (words.len > command_words) return error.CommandTooLong;
        raise();
        try p.reach(p.l.flags() + CMD_ACK, p.cmd, 60);
        p.cmd += 1;
        const head: [2]u32 = .{ kind, @intCast(words.len) };
        const out = p.l.cmdOut();
        @memcpy(p.win[out..][0..8], std.mem.asBytes(&head));
        @memcpy(p.win[out + 8 ..][0 .. 4 * words.len], std.mem.sliceAsBytes(words));
        try p.control(out, p.l.cmdIn(), 8 + 4 * words.len, p.l.flags() + CMD_FLAG, p.cmd);
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
        try p.control(0, 0, 0, p.l.flags() + CMD_ACK, p.cmd);
        return if (kind == 0) null else kind;
    }

    /// Both Macs loaded before the first command: a word each way, with ten minutes for a slower peer's load.
    pub fn ready(p: *Pair) !void {
        try p.control(0, 0, 0, p.l.flags() + READY, 1);
        try p.reach(p.l.flags() + READY, 1, 600);
        p.heartbeat_ready.store(true, .release);
    }

    /// This Mac's command wait ends (rank 1 at close).
    pub fn stopFollowing(p: *Pair) void {
        p.stop.store(true, .release);
        p.handoff.cancel();
    }

    /// The single command caller publishes work to the relay; no native call can trap that caller in close.
    fn control(p: *Pair, local: usize, remote: usize, bytes: usize, flag: usize, value: u64) !void {
        if (p.stop.load(.acquire)) return error.PairStopped;
        if (p.failed.load(.acquire)) return error.PairLinkFailed;
        const until = (try fabric.words.checkedNowNs()) +| std.time.ns_per_s;
        const seq = p.control_posted.load(.acquire) + 1;
        p.control_job = .{ .local = local, .remote = remote, .bytes = bytes, .flag = flag, .value = value, .until = until };
        p.control_posted.store(seq, .release);
        shutdown.await(&p.control_done, seq, &p.stop, &p.failed, until) catch |err| {
            p.handoff.fail(p.x, err);
            return err;
        };
    }

    fn pumpControl(p: *Pair) void {
        const seq = p.control_posted.load(.acquire);
        if (seq == p.control_done.load(.acquire)) return;
        const job = p.control_job;
        p.ep.writeSignalFromUntil(p.peer, job.local, job.remote, job.bytes, job.flag, job.value, job.until) catch |err| return p.handoff.fail(p.x, err);
        p.ep.flushPeerUntil(p.peer, job.until) catch |err| return p.handoff.fail(p.x, err);
        p.control_done.store(seq, .release);
    }

    /// Idle commands use peer heartbeats; model readiness retains its separate long deadline.
    fn reach(p: *Pair, off: usize, value: u64, seconds: u64) !void {
        return heartbeat.reach(p, p.word64(off), p.word64(p.l.flags() + heartbeat.BEAT), value, seconds);
    }

    /// A retired GPU producer posts its send slot; the host proves arrival before opening the consumer encoder.
    fn service(p: *Pair) void {
        defer p.relay_done.store(true, .release);
        defer p.sayGoodbye();
        raise();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var x: u64 = 1;
        var alive: heartbeat.State = .{};
        while (!p.stop.load(.acquire) and !p.failed.load(.acquire)) : (x += 1) {
            var began: ?u64 = null;
            while (p.handoff.posted.value() < x or p.queued.load(.acquire) < x) {
                if (p.stop.load(.acquire) or p.failed.load(.acquire)) return;
                p.pumpControl();
                heartbeat.pulse(p, &alive, p.word64(p.l.flags() + heartbeat.BEAT), p.l.flags() + heartbeat.BEAT);
                if (p.submitted.load(.acquire) < x) {
                    fabric.words.pause(20_000);
                } else {
                    const now = platform.c.mach_absolute_time();
                    if (began == null) began = now;
                    if (now -% began.? > 30 * 24_000_000) return p.handoff.fail(x, error.GpuPostTimeout);
                    std.atomic.spinLoopHint();
                }
            }
            const until = p.handoff.arm(x, deadlines.send_budget_ns, error.PairTransferTimeout) orelse return;
            p.ep.writeSignalFromUntil(p.peer, p.l.send(x), p.l.recv(x), p.bytes[x % JOBS], p.l.flags() + FLAG, x, until) catch |err| return p.handoff.fail(x, err);
            p.ep.flushPeerUntil(p.peer, until) catch |err| return p.handoff.fail(x, err);
            _ = p.handoff.arm(x, deadlines.arrival_budget_ns, error.PairPeerSilent) orelse return;
            while (true) {
                if (p.stop.load(.acquire) or p.failed.load(.acquire)) return;
                p.pumpControl();
                heartbeat.pulse(p, &alive, p.word64(p.l.flags() + heartbeat.BEAT), p.l.flags() + heartbeat.BEAT);
                const arrived = @atomicLoad(u64, p.word64(p.l.flags() + FLAG), .acquire);
                if (arrived > x + 1) return p.handoff.fail(x, error.PairOutOfStep);
                if (arrived >= x) break;
                std.atomic.spinLoopHint();
            }
            if (!p.handoff.publish(x)) return;
        }
    }

    fn sayGoodbye(p: *Pair) void {
        if (!p.goodbye.load(.acquire)) return;
        const now = fabric.words.checkedNowNs() catch return;
        const until = now + std.time.ns_per_s;
        const head: [2]u32 = .{ 0, 0 };
        @memcpy(p.win[p.l.cmdOut()..][0..8], std.mem.asBytes(&head));
        p.ep.writeSignalFromUntil(p.peer, p.l.cmdOut(), p.l.cmdIn(), 8, p.l.flags() + CMD_FLAG, p.cmd + 1, until) catch return;
        p.ep.flushPeerUntil(p.peer, until) catch {};
    }
};

test "the window's regions do not overlap and its words sit on their pages" {
    const l: Layout = .{ .slot = pages(256 * 6144 * 4) };
    try std.testing.expect(l.send(1) >= l.recv(1) + l.slot and l.send(0) >= 2 * l.slot + PAGE and l.send(0) + l.slot <= l.send(1));
    try std.testing.expect(l.cmdIn() >= l.send(1) + l.slot and l.cmdOut() >= l.cmdIn() + CMD_BYTES + 64);
    try std.testing.expect(l.flags() >= l.cmdOut() + 8 + 4 * command_words and heartbeat.BEAT + 8 <= PAGE and l.window() % PAGE == 0);
}

test "pair shutdown: terminal commands refuse already satisfied flags without entering the native transport" {
    var p: Pair = undefined;
    p.rank = 0;
    p.cmd = 0;
    p.l = .{ .slot = PAGE };
    p.win = try std.testing.allocator.alloc(u8, p.l.window());
    defer std.testing.allocator.free(p.win);
    @memset(p.win, 0);
    p.word64(p.l.flags() + CMD_FLAG).* = 1;
    p.stop = .init(false);
    p.failed = .init(true);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    try std.testing.expectError(error.PairLinkFailed, p.send(1, &.{}));
    try std.testing.expectError(error.PairLinkFailed, p.next(std.testing.allocator, &out));
    p.failed.store(false, .release);
    p.stop.store(true, .release);
    try std.testing.expectError(error.PairStopped, p.send(1, &.{}));
    try std.testing.expectEqual(@as(?u32, null), try p.next(std.testing.allocator, &out));
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
}

/// A real Pair's GPU path with host loopback in place of MCDMA; tests never call its transport or deinit.
fn testPair(device: mtl.Device, rank: u32) !*Pair {
    const a = std.testing.allocator;
    const l: Layout = .{ .slot = pages(16 * 6144 * 4) };
    const win = try std.heap.page_allocator.alloc(u8, l.window());
    errdefer std.heap.page_allocator.free(win);
    @memset(win, 0);
    const wbuf = try device.bufferNoCopy(win.ptr, win.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer wbuf.deinit();
    const partial = try device.buffer(l.slot, mtl.ResourceOptions.shared);
    errdefer partial.deinit();
    const lib = try mtl.Library.fromSource(device, sources.core_pair, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: [4]mtl.Pipeline = undefined;
    var made: usize = 0;
    errdefer for (pipes[0..made]) |pipe| pipe.deinit();
    for ([_][]const u8{ "pair_stage", "pair_add_rms", "pair_argmax", "pair_pick" }) |name| {
        pipes[made] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    const bytes = try a.alloc(u32, JOBS);
    errdefer a.free(bytes);
    const p = try a.create(Pair);
    errdefer a.destroy(p);
    p.* = .{ .rank = rank, .peer = 1 - rank, .l = l, .ep = undefined, .rd = undefined, .win = win, .wbuf = wbuf, .partial = partial, .handoff = undefined, .stage_pipe = pipes[0], .add_rms = pipes[1], .argmax_pipe = pipes[2], .pick_pipe = pipes[3], .bytes = bytes, .arena = .init(a), .gpa = a };
    p.handoff = try Handoff.init(device, &p.failed, &p.stop, rank);
    return p;
}

fn dropTestPair(p: *Pair) void {
    for ([_]mtl.Pipeline{ p.stage_pipe, p.add_rms, p.argmax_pipe, p.pick_pipe }) |pipe| pipe.deinit();
    p.handoff.deinit();
    p.partial.deinit();
    p.wbuf.deinit();
    std.heap.page_allocator.free(p.win);
    p.gpa.free(p.bytes);
    p.gpa.destroy(p);
}

const Loopback = struct {
    pub const create = testPair;
    pub const drop = dropTestPair;
    pub fn receive(p: *Pair, x: u64) []u8 {
        return p.win[p.l.recv(x)..][0..p.bytes[x % JOBS]];
    }
    pub fn send(p: *Pair, x: u64) []const u8 {
        return p.win[p.l.send(x)..][0..p.bytes[x % JOBS]];
    }
};

test "GLM DSA pair loopback: three exchanges order GPU posts, host copies and payload consumers on both ranks" {
    try @import("pair_loopback.zig").ordered(Pair, Loopback);
}

test "GLM DSA pair loopback: terminal wake refuses stale receive epochs and preserves transport send slots" {
    try @import("pair_loopback.zig").stale(Pair, Loopback);
}
