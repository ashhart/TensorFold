//! Two Macs' tensor-parallel exchanges in a command buffer over MCDMA, and rank 0's commands to rank 1.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const sources = @import("kernel_sources");
const settings = @import("../families/flashnext/tp_settings.zig");

pub const Settings = settings.Settings;
pub const Ref = @import("exlr8_projection.zig").Ref;

const PAGE = 16384;
fn pages(n: usize) usize {
    return (n + PAGE - 1) / PAGE * PAGE;
}
/// One exchange's bytes at most: 16 rows of 6,144 fp32 values.
pub const slot_bytes = 16 * 6144 * 4;
/// A command's words at most: a 262,144-token prompt and its head.
pub const command_words = 262144 + 64;
const CMD_BYTES = pages(8 + 4 * command_words);
// The window on both Macs: parts in and out by parity (a page before each out), commands in and out, flags, GPU words.
const RECV = 0;
const SEND = RECV + 2 * slot_bytes;
const CMD_IN = SEND + 2 * (PAGE + slot_bytes);
const CMD_OUT = CMD_IN + CMD_BYTES + PAGE;
const FLAGS = CMD_OUT + CMD_BYTES;
const FLAG = FLAGS; // u64: the last exchange whose peer part has landed
const HELLO = FLAGS + 64;
const CMD_FLAG = FLAGS + 128; // u64, at rank 1: the last command rank 0 has written
const CMD_ACK = FLAGS + 192; // u64, at rank 0: the last command rank 1 has copied out
const READY = FLAGS + 256; // u64: the peer has loaded its model
const SYNC = FLAGS + PAGE; // u32 words the GPU writes: [0] the last exchange posted, [16] waits that gave up
pub const window = SYNC + PAGE;
const JOBS = 4096;

fn recvAt(x: u64) usize {
    return RECV + @as(usize, @intCast(x % 2)) * slot_bytes;
}
fn sendAt(x: u64) usize {
    return SEND + @as(usize, @intCast(x % 2)) * (PAGE + slot_bytes) + PAGE;
}
/// Whether 32-bit counter a has reached b across the wrap.
fn reached(a: u32, b: u32) bool {
    return @as(i32, @bitCast(a -% b)) >= 0;
}

const PairRows = extern struct { width: u32, stride: u32, rank: u32, x: u32, eps: f32, pad0: f32 = 0, pad1: f32 = 0, pad2: f32 = 0 };

pub const Pair = struct {
    rank: u32,
    peer: u32,
    ep: *fabric.mcdma.Endpoint,
    rd: fabric.rdma.Rdma,
    win: []u8,
    wbuf: mtl.Buffer,
    add_rms: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    /// Exchanges encoded so far: the same count on both Macs.
    x: u64 = 0,
    bytes: []u32,
    queued: std.atomic.Value(u32) = .init(0),
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    /// Commands sent (rank 0) or taken (rank 1) so far.
    cmd: u64 = 0,

    /// Connect to the peer named in `settings_path` (speed-up mode's file) and start the sending thread.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, settings_path: []const u8) !*Pair {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, settings_path, gpa, .limited(1 << 20));
        const s = try settings.read(gpa, text);
        if (s.rank > 1 or s.links.len != 1) return error.PairTwoRanksOneLink;
        const lib = try gpa.dupeSentinel(u8, s.library, 0);
        const ep = try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = window, .staging_bytes = 4 << 20, .links = s.links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        errdefer ep.deinit();
        const rd = ep.rdma();
        const win = rd.window();
        const lib_m = try mtl.Library.fromSource(device, sources.core_pair, mtl.CompileOptions.mlx());
        defer lib_m.deinit();
        const p = try gpa.create(Pair);
        p.* = .{
            .rank = s.rank,
            .peer = 1 - s.rank,
            .ep = ep,
            .rd = rd,
            .win = win,
            .wbuf = try device.bufferNoCopy(win.ptr, win.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked),
            .add_rms = try mtl.Pipeline.init(device, lib_m, "pair_add_rms", false),
            .argmax_pipe = try mtl.Pipeline.init(device, lib_m, "pair_argmax", false),
            .pick_pipe = try mtl.Pipeline.init(device, lib_m, "pair_pick", false),
            .bytes = try gpa.alloc(u32, JOBS),
        };
        // both Macs up before the first exchange: a word each way
        try rd.signal(p.peer, HELLO, 1);
        while (@atomicLoad(u64, p.word64(HELLO), .acquire) < 1) std.atomic.spinLoopHint();
        p.thread = try std.Thread.spawn(.{}, service, .{p});
        return p;
    }

    /// Rank 0 tells rank 1 it is done (its `next` returns null); then the sending thread and the link close.
    pub fn deinit(p: *Pair) void {
        if (p.rank == 0 and !p.failed.load(.acquire)) p.send(0, &.{}) catch {};
        p.rd.flush() catch {};
        p.stop.store(true, .release);
        if (p.thread) |t| t.join();
        p.ep.deinit();
    }

    fn word64(p: *const Pair, off: usize) *u64 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }
    fn word32(p: *const Pair, off: usize) *u32 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }

    /// GPU waits that gave up (a peer that never answered): nonzero means the run is invalid.
    pub fn gaveUp(p: *const Pair) u32 {
        return @atomicLoad(u32, p.word32(SYNC + 16 * 4), .acquire);
    }

    /// Where the next exchange's part goes (rows x width fp32, rows packed): the producer writes it here.
    pub fn part(p: *const Pair) Ref {
        return .{ .buf = p.wbuf, .off = sendAt(p.x + 1) };
    }

    fn queue(p: *Pair, bytes: usize) u32 {
        p.x += 1;
        const x: u32 = @truncate(p.x);
        p.bytes[x % JOBS] = @intCast(bytes);
        p.queued.store(x, .release);
        return x;
    }

    /// After `part()` was written: send it, wait for the peer's, h += rank 0's + rank 1's, y = rms(h) w.
    pub fn addRms(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32) void {
        const x = p.queue(@as(usize, rows) * width * 4);
        e.setPipeline(p.add_rms);
        e.setBuffer(p.wbuf, SYNC, 0);
        e.setBuffer(p.wbuf, FLAG, 1);
        e.setValue(PairRows{ .width = width, .stride = width, .rank = p.rank, .x = x, .eps = eps }, 2);
        e.setBuffer(p.wbuf, sendAt(x), 3);
        e.setBuffer(p.wbuf, recvAt(x), 4);
        e.setBuffer(h.buf, h.off, 5);
        e.setBuffer(w.buf, w.off, 6);
        e.setBuffer(y.buf, y.off, 7);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// Each row's pick over both Macs' logits (this one's n columns from id lo, row stride `stride`) into picks.
    pub fn argmax(p: *Pair, e: mtl.ComputeEncoder, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        const x = p.queue(@as(usize, rows) * 8);
        e.setPipeline(p.argmax_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setValue([4]u32{ stride, lo, n, 0 }, 1);
        e.setBuffer(p.wbuf, sendAt(x), 2);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
        e.barrier();
        e.setPipeline(p.pick_pipe);
        e.setBuffer(p.wbuf, SYNC, 0);
        e.setBuffer(p.wbuf, FLAG, 1);
        e.setValue([2]u32{ rows, x }, 2);
        e.setBuffer(p.wbuf, sendAt(x), 3);
        e.setBuffer(p.wbuf, recvAt(x), 4);
        e.setBuffer(picks.buf, picks.off, 5);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    }

    /// Rank 0: command `kind` (never 0, which ends rank 1's loop) and its words, once rank 1 took the one before.
    pub fn send(p: *Pair, kind: u32, words: []const u32) !void {
        if (p.rank != 0) return error.FollowsPeer;
        if (words.len > command_words) return error.CommandTooLong;
        try p.reach(CMD_ACK, p.cmd, true);
        p.cmd += 1;
        const head: [2]u32 = .{ kind, @intCast(words.len) };
        @memcpy(p.win[CMD_OUT..][0..8], std.mem.asBytes(&head));
        @memcpy(p.win[CMD_OUT + 8 ..][0 .. 4 * words.len], std.mem.sliceAsBytes(words));
        try p.ep.writeSignalFrom(p.peer, CMD_OUT, CMD_IN, 8 + 4 * words.len, CMD_FLAG, p.cmd);
    }

    /// Rank 1: rank 0's next command's words into `out` and its kind; null once rank 0 is done or this Mac stops.
    pub fn next(p: *Pair, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !?u32 {
        p.reach(CMD_FLAG, p.cmd + 1, false) catch |err| return if (err == error.PairStopped) null else err;
        p.cmd += 1;
        const head: *const [2]u32 = @ptrCast(@alignCast(p.win.ptr + CMD_IN));
        if (head[1] > command_words) return error.CommandOutOfStep;
        out.clearRetainingCapacity();
        try out.appendSlice(gpa, @as([*]const u32, @ptrCast(@alignCast(p.win.ptr + CMD_IN + 8)))[0..head[1]]);
        const kind = head[0];
        try p.rd.signal(p.peer, CMD_ACK, p.cmd);
        return if (kind == 0) null else kind;
    }

    /// Both Macs loaded before the first command: a word each way, waited for without a bound.
    pub fn ready(p: *Pair) !void {
        try p.rd.signal(p.peer, READY, 1);
        try p.reach(READY, 1, false);
    }

    /// This Mac's command wait ends (rank 1 at close).
    pub fn stopFollowing(p: *Pair) void {
        p.stop.store(true, .release);
    }

    /// Wait until this Mac's word at `off` reaches `value`: spins, then short sleeps (`bounded`: 60 s at most).
    fn reach(p: *Pair, off: usize, value: u64, bounded: bool) !void {
        const t0 = std.c.mach_absolute_time();
        var spins: usize = 0;
        while (@atomicLoad(u64, p.word64(off), .acquire) < value) {
            if (p.stop.load(.acquire)) return error.PairStopped;
            if (p.failed.load(.acquire)) return error.PairLinkFailed;
            if (bounded and std.c.mach_absolute_time() - t0 > 60 * 24_000_000) return error.PairPeerSilent;
            spins += 1;
            if (spins > 200_000) {
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                _ = std.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
    }

    /// Each exchange the GPU posts goes to the peer, then its flag; a failed link lands its own flags (no GPU hang).
    fn service(p: *Pair) void {
        var next_x: u32 = 1;
        var idle: usize = 0;
        while (!p.stop.load(.acquire)) {
            const posted = @atomicLoad(u32, p.word32(SYNC), .acquire);
            if (!reached(posted, next_x) or !reached(p.queued.load(.acquire), next_x)) {
                idle += 1;
                if (idle > 1_000_000) {
                    const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                    _ = std.c.nanosleep(&ts, null);
                } else std.atomic.spinLoopHint();
                continue;
            }
            idle = 0;
            if (!p.failed.load(.acquire)) p.ep.writeSignalFrom(p.peer, sendAt(next_x), recvAt(next_x), p.bytes[next_x % JOBS], FLAG, next_x) catch |err| {
                std.log.err("pair rank {d}: exchange {d} failed: {s}", .{ p.rank, next_x, @errorName(err) });
                p.failed.store(true, .release);
            };
            if (p.failed.load(.acquire)) @atomicStore(u64, p.word64(FLAG), next_x, .release);
            next_x +%= 1;
        }
    }
};

test "the window's regions do not overlap and its words sit on their pages" {
    try std.testing.expect(sendAt(1) >= recvAt(1) + slot_bytes and sendAt(0) >= SEND + PAGE);
    try std.testing.expect(CMD_IN >= sendAt(1) + slot_bytes and CMD_OUT >= CMD_IN + CMD_BYTES + 64);
    try std.testing.expect(FLAGS >= CMD_OUT + 8 + 4 * command_words and CMD_ACK + 8 <= SYNC and window % PAGE == 0);
    try std.testing.expect(reached(1, 0xFFFFFFFF) and !reached(5, 6));
}
