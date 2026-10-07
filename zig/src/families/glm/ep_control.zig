//! The expert-parallel pair's control protocol over a link's window: identities, requests and stop decisions, each acked.
const std = @import("std");
const fabric = @import("fabric");
const Rdma = fabric.rdma.Rdma;

const PAGE = 16384;
// Words in the control region (this Mac's window; each written by the peer).
const HELLO = 0; // u64: the peer's identity has landed
const CTRL = 8; // u64 by step parity: rank 0's decision at step s, (s << 1) | quit
const CTRL_ACK = 24; // u64, at rank 0: the last step rank 1 has read
const REQ_FLAG = 32; // u64, at rank 1: rank 0's last request
const REQ_ACK = 40; // u64, at rank 0: the last request rank 1 has copied out
const BYE = 48; // u64, at rank 1: rank 0 has closed
const REPLY = 56; // u64, at rank 0: rank 1's result of the last command, (command << 1) | ok
const IDENT = 64; // the peer's Identity
const REQ = PAGE; // the request: a 64-byte head, then the prompt's tokens
pub const REQ_TOKENS = 262144;
pub const MAX_EOS = 11;
/// The control region's bytes (page-aligned).
pub const BYTES = REQ + std.mem.alignForward(usize, 64 + 4 * REQ_TOKENS, PAGE);

/// What a rank runs; the pair works only when both agree on all but their expert ranges, which must tile [0, experts).
pub const Identity = extern struct {
    magic: u32 = 0x474c4d45,
    version: u32 = 1,
    layers: u32,
    run: u32,
    mtp: u32,
    experts: u32,
    own_lo: u32,
    own_hi: u32,
    cap: u32,
    pad: u32 = 0,
    model: u64, // the checkpoint's config and weight index, hashed
    rest: [16]u8 = @splat(0),

    pub fn compatible(a: Identity, b: Identity) bool {
        const same = a.magic == b.magic and a.version == b.version and a.layers == b.layers and a.run == b.run and
            a.mtp == b.mtp and a.experts == b.experts and a.cap == b.cap and a.model == b.model and std.mem.eql(u8, &a.rest, &b.rest);
        const tile = (a.own_lo == 0 and a.own_hi == b.own_lo and b.own_hi == a.experts) or
            (b.own_lo == 0 and b.own_hi == a.own_lo and a.own_hi == a.experts);
        return same and tile and a.own_lo < a.own_hi and b.own_lo < b.own_hi;
    }
};

comptime {
    std.debug.assert(@sizeOf(Identity) == 64);
}

pub const Request = struct { max_tokens: usize, depth: usize, eos: []const u32, prompt: []const u32 };

/// What rank 0 hands over: a request rank 1 runs in step, or a lane command (its kind, never 0, and its words).
pub const Command = union(enum) { request: Request, lanes: struct { kind: u32, words: []const u32 } };

fn ticks() u64 {
    return std.c.mach_absolute_time(); // 24 MHz on Apple silicon
}

pub const Control = struct {
    rank: u32,
    peer: u32,
    rd: Rdma,
    base: usize, // the control region's offset in the window
    step: u64 = 0, // steps agreed so far (the same count on both Macs)
    req: u64 = 0, // requests handed over so far
    eos: [MAX_EOS]u32 = undefined, // rank 1: the request's end tokens, copied out of the window
    prompt: []u32, // rank 1: the request's prompt, copied out of the window
    stop: std.atomic.Value(bool) = .init(false), // a local close: every wait ends
    failed: *const std.atomic.Value(bool), // the link's failure bit: every wait ends
    limit: u64 = 30 * 24_000_000, // ticks a wait for the peer may take (30 s) before the peer counts as silent

    pub fn init(gpa: std.mem.Allocator, rd: Rdma, base: usize, failed: *const std.atomic.Value(bool)) !Control {
        const r = rd.rank();
        return .{ .rank = r, .peer = 1 - r, .rd = rd, .base = base, .prompt = try gpa.alloc(u32, REQ_TOKENS), .failed = failed };
    }

    pub fn deinit(c: *Control, gpa: std.mem.Allocator) void {
        gpa.free(c.prompt);
    }

    fn word(c: *const Control, off: usize) *u64 {
        return @ptrCast(@alignCast(c.rd.window().ptr + c.base + off));
    }

    /// Wait until this Mac's word at `off` reaches `value`; null when closed (a local stop, the peer's goodbye).
    fn reach(c: *Control, off: usize, value: u64, bounded: bool) !?u64 {
        const t0 = ticks();
        var spins: usize = 0;
        while (true) {
            const v = @atomicLoad(u64, c.word(off), .acquire);
            if (v >= value) return v;
            if (c.stop.load(.acquire)) return null;
            if (c.rank == 1 and @atomicLoad(u64, c.word(BYE), .acquire) != 0) return null;
            if (c.failed.load(.acquire)) return error.EpLinkFailed;
            if (bounded and ticks() - t0 > c.limit) return error.EpPeerSilent;
            spins += 1;
            if (spins > 100_000) { // a long wait: back off
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000 };
                _ = std.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
    }

    /// Send this Mac's identity, wait for the peer's, and refuse a peer that runs something else (both Macs refuse).
    pub fn hello(c: *Control, me: Identity) !Identity {
        try c.rd.write2Signal(c.peer, c.base + IDENT, std.mem.asBytes(&me), &.{}, c.base + HELLO, 1);
        _ = try c.reach(HELLO, 1, true) orelse return error.EpClosed;
        const peer: *const Identity = @ptrCast(@alignCast(c.rd.window().ptr + c.base + IDENT));
        if (!me.compatible(peer.*)) {
            std.log.warn("expert parallel: this Mac runs {any}, the peer {any}", .{ me, peer.* });
            return error.EpIdentityMismatch;
        }
        return peer.*;
    }

    /// A step both Macs take in order: rank 0's stop decision, read and acknowledged by rank 1 before its parity word is reused.
    pub fn agree(c: *Control, quit: bool) !bool {
        c.step += 1;
        const at = CTRL + 8 * (c.step % 2);
        if (c.rank == 0) {
            if (c.step > 2) _ = try c.reach(CTRL_ACK, c.step - 2, true) orelse return error.EpClosed;
            try c.rd.signal(c.peer, c.base + at, (c.step << 1) | @intFromBool(quit));
            return quit;
        }
        const v = try c.reach(at, c.step << 1, true) orelse return error.EpClosed;
        if (v >> 1 != c.step) return error.EpOutOfStep;
        try c.rd.signal(c.peer, c.base + CTRL_ACK, c.step);
        return v & 1 != 0;
    }

    /// Rank 0: the next request, once rank 1 has copied out the one before.
    pub fn sendRequest(c: *Control, r: Request) !void {
        if (r.prompt.len > REQ_TOKENS or r.eos.len > MAX_EOS) return error.EpRequestTooLarge;
        if (c.req > 0) _ = try c.reach(REQ_ACK, c.req, true) orelse return error.EpClosed;
        var head: [16]u32 = @splat(0);
        head[0] = @intCast(@min(r.max_tokens, std.math.maxInt(u32)));
        head[1] = @intCast(r.depth);
        head[2] = @intCast(r.prompt.len);
        head[3] = @intCast(r.eos.len);
        @memcpy(head[4..][0..r.eos.len], r.eos);
        c.req += 1;
        try c.rd.write2Signal(c.peer, c.base + REQ, std.mem.sliceAsBytes(&head), std.mem.sliceAsBytes(r.prompt), c.base + REQ_FLAG, c.req);
    }

    /// Rank 0: a lane command (kind > 0) in the request region, once rank 1 has copied out the one before.
    pub fn sendCommand(c: *Control, kind: u32, words: []const u32) !void {
        if (kind == 0 or words.len > REQ_TOKENS) return error.EpRequestTooLarge;
        if (c.req > 0) _ = try c.reach(REQ_ACK, c.req, true) orelse return error.EpClosed;
        var head: [16]u32 = @splat(0);
        head[2] = @intCast(words.len);
        head[15] = kind;
        c.req += 1;
        try c.rd.write2Signal(c.peer, c.base + REQ, std.mem.sliceAsBytes(&head), std.mem.sliceAsBytes(words), c.base + REQ_FLAG, c.req);
    }

    /// Rank 1: rank 0's next request or command, copied out and acked; null once closed; a skipped one is refused.
    pub fn waitCommand(c: *Control) !?Command {
        const v = try c.reach(REQ_FLAG, c.req + 1, false) orelse return null;
        if (v != c.req + 1) return error.EpOutOfStep;
        c.req = v;
        const head: [*]const u32 = @ptrCast(@alignCast(c.rd.window().ptr + c.base + REQ));
        const tokens: [*]const u32 = @ptrCast(@alignCast(c.rd.window().ptr + c.base + REQ + 64));
        const n = @min(head[2], REQ_TOKENS);
        @memcpy(c.prompt[0..n], tokens[0..n]);
        const out: Command = if (head[15] != 0) .{ .lanes = .{ .kind = head[15], .words = c.prompt[0..n] } } else blk: {
            const n_eos = @min(head[3], MAX_EOS);
            @memcpy(c.eos[0..n_eos], head[4..][0..n_eos]);
            break :blk .{ .request = .{ .max_tokens = head[0], .depth = head[1], .eos = c.eos[0..n_eos], .prompt = c.prompt[0..n] } };
        };
        try c.rd.signal(c.peer, c.base + REQ_ACK, c.req);
        return out;
    }

    /// Rank 1: the result of the command just taken (a learned state written or read), for rank 0's `waitReply`.
    pub fn reply(c: *Control, ok: bool) !void {
        try c.rd.signal(c.peer, c.base + REPLY, (c.req << 1) | @intFromBool(ok));
    }

    /// Rank 0: whether rank 1 carried out the last command sent.
    pub fn waitReply(c: *Control) !bool {
        const v = try c.reach(REPLY, c.req << 1, true) orelse return error.EpClosed;
        if (v >> 1 != c.req) return error.EpOutOfStep;
        return v & 1 != 0;
    }

    /// Rank 1: rank 0's next request; null once closed; a command in its place is refused.
    pub fn waitRequest(c: *Control) !?Request {
        return switch (try c.waitCommand() orelse return null) {
            .request => |r| r,
            .lanes => error.EpOutOfStep,
        };
    }

    /// Rank 0 closing: rank 1's waits end.
    pub fn bye(c: *Control) void {
        if (c.rank == 0) c.rd.signal(c.peer, c.base + BYE, 1) catch {};
    }
};

// Fake-transport tests: both ranks' Controls on one fabric.fake cluster, rank 1 held back where the protocol must wait.

const Pair = struct {
    cluster: *fabric.fake.Cluster,
    failed: std.atomic.Value(bool) = .init(false),
    c: [2]Control = undefined,

    fn init(gpa: std.mem.Allocator) !*Pair {
        const p = try gpa.create(Pair);
        p.* = .{ .cluster = try fabric.fake.Cluster.init(gpa, 2, BYTES, .immediate, null) };
        for (0..2) |r| p.c[r] = try Control.init(gpa, p.cluster.endpoint(@intCast(r)), 0, &p.failed);
        for (&p.c) |*c| c.limit = 24_000_000; // 1 s in tests
        return p;
    }

    fn deinit(p: *Pair, gpa: std.mem.Allocator) void {
        for (&p.c) |*c| c.deinit(gpa);
        p.cluster.deinit();
        gpa.destroy(p);
    }
};

fn ident(lo: u32, hi: u32) Identity {
    return .{ .layers = 45, .run = 45, .mtp = 1, .experts = 288, .own_lo = lo, .own_hi = hi, .cap = 8256, .model = 0x1234 };
}

test "identities: the halves tile the experts and everything else matches" {
    try std.testing.expect(ident(0, 144).compatible(ident(144, 288)));
    try std.testing.expect(ident(144, 288).compatible(ident(0, 144)));
    try std.testing.expect(!ident(0, 144).compatible(ident(0, 144)));
    try std.testing.expect(!ident(0, 144).compatible(ident(100, 288)));
    var other = ident(144, 288);
    other.run = 8;
    try std.testing.expect(!ident(0, 144).compatible(other));
    other = ident(144, 288);
    other.cap = 4160;
    try std.testing.expect(!ident(0, 144).compatible(other));
}

test "hello refuses a peer that runs another layer subset, on both ranks" {
    const gpa = std.testing.allocator;
    const p = try Pair.init(gpa);
    defer p.deinit(gpa);
    var one = ident(144, 288);
    one.run = 8;
    const T = struct {
        fn run(c: *Control, me: Identity, out: *?anyerror) void {
            _ = c.hello(me) catch |e| {
                out.* = e;
                return;
            };
            out.* = null;
        }
    };
    var e1: ?anyerror = undefined;
    const th = try std.Thread.spawn(.{}, T.run, .{ &p.c[1], one, &e1 });
    var e0: ?anyerror = undefined;
    T.run(&p.c[0], ident(0, 144), &e0);
    th.join();
    try std.testing.expectEqual(@as(?anyerror, error.EpIdentityMismatch), e0);
    try std.testing.expectEqual(@as(?anyerror, error.EpIdentityMismatch), e1);
}

test "early cancels: rank 0 waits for rank 1's acks, nothing is overwritten or skipped" {
    const gpa = std.testing.allocator;
    const p = try Pair.init(gpa);
    defer p.deinit(gpa);
    const prompts = [_][3]u32{ .{ 1, 2, 3 }, .{ 4, 5, 6 }, .{ 7, 8, 9 }, .{ 10, 11, 12 } };
    const Rank0 = struct { // three requests cancelled at their first step, then one that runs three steps
        fn run(c: *Control, out: *?anyerror) void {
            out.* = null;
            for (prompts, 0..) |pr, i| {
                c.sendRequest(.{ .max_tokens = 8, .depth = 3, .eos = &.{ 7, 9 }, .prompt = &pr }) catch |e| {
                    out.* = e;
                    return;
                };
                const steps: usize = if (i < 3) 1 else 3;
                for (0..steps) |s| _ = c.agree(i < 3 or s == 2) catch |e| {
                    out.* = e;
                    return;
                };
            }
        }
    };
    var e0: ?anyerror = undefined;
    const th = try std.Thread.spawn(.{}, Rank0.run, .{ &p.c[0], &e0 });
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000_000 }; // rank 1 held back while rank 0 runs ahead
    _ = std.c.nanosleep(&ts, null);
    for (prompts, 0..) |pr, i| {
        const r = (try p.c[1].waitRequest()).?;
        try std.testing.expectEqualSlices(u32, &pr, r.prompt);
        try std.testing.expectEqualSlices(u32, &.{ 7, 9 }, r.eos);
        const steps: usize = if (i < 3) 1 else 3;
        for (0..steps) |s| try std.testing.expectEqual(i < 3 or s == 2, try p.c[1].agree(false));
    }
    th.join();
    try std.testing.expectEqual(@as(?anyerror, null), e0);
    p.c[0].bye();
    try std.testing.expectEqual(@as(?Request, null), try p.c[1].waitRequest()); // the goodbye ends rank 1's wait
}

test "lane commands and requests arrive in order, each word intact" {
    const gpa = std.testing.allocator;
    const p = try Pair.init(gpa);
    defer p.deinit(gpa);
    const Rank0 = struct {
        fn run(c: *Control, out: *?anyerror) void {
            out.* = null;
            c.sendCommand(3, &.{ 7, 8, 9 }) catch |e| return set(out, e);
            c.sendRequest(.{ .max_tokens = 4, .depth = 2, .eos = &.{5}, .prompt = &.{ 1, 2 } }) catch |e| return set(out, e);
            c.sendCommand(6, &.{}) catch |e| return set(out, e);
        }
        fn set(out: *?anyerror, e: anyerror) void {
            out.* = e;
        }
    };
    var e0: ?anyerror = undefined;
    const th = try std.Thread.spawn(.{}, Rank0.run, .{ &p.c[0], &e0 });
    const a = (try p.c[1].waitCommand()).?;
    try std.testing.expectEqual(@as(u32, 3), a.lanes.kind);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, a.lanes.words);
    const r = (try p.c[1].waitCommand()).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, r.request.prompt);
    try std.testing.expectEqualSlices(u32, &.{5}, r.request.eos);
    try std.testing.expectError(error.EpOutOfStep, p.c[1].waitRequest()); // a command where a request was due
    th.join();
    try std.testing.expectEqual(@as(?anyerror, null), e0);
    try std.testing.expectError(error.EpRequestTooLarge, p.c[0].sendCommand(0, &.{}));
}

test "a skipped request or step is refused, and a dead link ends a wait" {
    const gpa = std.testing.allocator;
    const p = try Pair.init(gpa);
    defer p.deinit(gpa);
    const at = p.cluster.endpoint(1).window();
    @as(*u64, @ptrCast(@alignCast(at.ptr + REQ_FLAG))).* = 2; // request 2 landed where 1 was expected
    try std.testing.expectError(error.EpOutOfStep, p.c[1].waitRequest());
    @as(*u64, @ptrCast(@alignCast(at.ptr + CTRL + 8))).* = (3 << 1) | 1; // step 3 in step 1's word
    try std.testing.expectError(error.EpOutOfStep, p.c[1].agree(false));
}

test "a dead link or a local close ends a wait at once" {
    const gpa = std.testing.allocator;
    const p = try Pair.init(gpa);
    defer p.deinit(gpa);
    p.failed.store(true, .release);
    try std.testing.expectError(error.EpLinkFailed, p.c[1].agree(false)); // step 1 never comes
    p.failed.store(false, .release);
    p.c[1].stop.store(true, .release);
    try std.testing.expectEqual(@as(?Request, null), try p.c[1].waitRequest());
}
