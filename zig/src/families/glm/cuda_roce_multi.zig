//! TF_GLM_ROCE_MULTI=1 at three ranks or more: the main stream's small all-gathers on fabric/tp's multirank RoCE.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const tp = @import("tp");

const Group = cuda.tp_group.Group;

/// Every rank opens the wire and lends it to `g` once every probe matched NCCL; one rank failing leaves all on NCCL.
pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group, on: bool) !void {
    if (comptime builtin.os.tag != .linux) return; // fabric/tp's sockets and verbs
    if (!on) return;
    if (g.world < 3) {
        if (g.rank == 0) std.log.info("TF_GLM_ROCE_MULTI=1 takes three ranks or more; {d} keep their own route", .{g.world});
        return;
    }
    var cfg = try tp.Config.fromEnv(); // TF_TP_MASTER, TF_TP_PORT and the wire's settings: a bad value refuses
    cfg.rank = @intCast(g.rank);
    cfg.world = @intCast(g.world);
    var entered = false; // this rank's probe enqueued NCCL's gather
    const w = borrow(gpa, io, d, g, cfg, &entered) catch |err| {
        // TF_TP_ROCE_FALLBACK=error refuses rather than run on NCCL, as does a failure NCCL's order may not survive
        if (cfg.roce_fallback == .@"error" or !fallsBack(err, entered)) return err;
        std.log.warn("rank {d}: TF_GLM_ROCE_MULTI=1, but the wire did not open ({s}); NCCL keeps the small gathers", .{ g.rank, @errorName(err) });
        return;
    };
    g.multi = .{ .ctx = w, .fits = fits, .gather = gather, .check = check, .abort = abort, .deinit = deinit };
}

/// Whether NCCL keeps the small gathers after `err`: no gather of the probe's entered yet, or every rank's verdict.
fn fallsBack(err: anyerror, entered: bool) bool {
    return !entered or err == error.ProbeMismatch;
}

/// fabric/tp's bootstrap, ports, wire and kernel, and a probe against the group's NCCL: each step every rank's or none.
fn borrow(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *const Group, cfg: tp.Config, entered: *bool) !*Wire {
    const path = std.mem.span(std.c.getenv("TF_TP_KERNEL_IMAGE") orelse return error.NoKernelImage);
    const image = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(1 << 26), .@"16", null);
    defer gpa.free(image); // the module copies it at load
    var b = try tp.Bootstrap.open(cfg);
    defer b.close(); // the wire is connected by the end of init
    try b.agree(&cfg.digest(), "fabric/tp settings (TF_TP_MAILBOX_MAX_KB, TF_TP_ROCE_WIRE, TF_TP_ROCE_FALLBACK)");
    var specs: [tp.roce_verbs.max_hcas]tp.roce_verbs.HcaSpec = undefined;
    var picked = true;
    const n = if (cfg.roce_wire == .verbs) tp.roce_pick.resolve(cfg, &b, &specs) catch blk: {
        picked = false;
        break :blk 0;
    } else 0;
    if (!try everyRank(&b, picked)) return error.RoceHcaSelectionFailed;
    const r = try tp.roce_multi.Roce.init(d, &b, image, .{ .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = specs[0..n], .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = cfg.mailbox_max_bytes, .timeout_ns = cfg.mailbox_timeout_ns, .fast = cfg.roce_fast, .blocks = cfg.roce_blocks });
    errdefer r.deinit();
    const w = try gpa.create(Wire); // before the last vote: nothing after it can fail
    errdefer gpa.destroy(w);
    w.* = .{ .gpa = gpa, .r = r, .rank = g.rank };
    var p: Probe = .{ .gpa = gpa, .d = d, .g = g, .r = r };
    defer p.deinit();
    try vote(&b, p.steps(), entered);
    var line: [384]u8 = undefined;
    std.log.info("rank {d}: TF_GLM_ROCE_MULTI: gathers up to {d} KiB a rank on the wire, its probe equal to NCCL's: {s}", .{ g.rank, r.slot_bytes / 1024, r.describeLink(&line) });
    return w;
}

/// A rank's probe in steps (made ready before any gather, both gathers, the wait, the verdict); a test fails any.
const Steps = struct {
    ctx: *anyopaque,
    prepare: *const fn (*anyopaque) anyerror!void,
    wire: *const fn (*anyopaque) anyerror!void,
    nccl: *const fn (*anyopaque) anyerror!void,
    wait: *const fn (*anyopaque) anyerror!void,
    equal: *const fn (*anyopaque) bool,
};

/// Every rank is ready before any enters a gather, then all enter both; a failed NCCL gather or wait refuses at once.
fn vote(b: *tp.Bootstrap, s: Steps, entered: *bool) !void {
    const ready = if (s.prepare(s.ctx)) |_| true else |_| false;
    if (!try everyRank(b, ready)) return error.ProbeUnprepared;
    const wired = if (s.wire(s.ctx)) |_| true else |_| false; // NCCL's gather follows it either way
    entered.* = true;
    try s.nccl(s.ctx);
    try s.wait(s.ctx);
    try lendIfAll(b, wired and s.equal(s.ctx));
}

/// One 4 KiB gather in place, on the wire and on the group's NCCL (no wire lent yet): every byte must be equal.
const Probe = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: *const Group,
    r: *tp.roce_multi.Roce,
    a: ?cuda.DeviceBuffer = null, // the wire's rows, then NCCL's
    s: ?cuda.Stream = null,

    const n = 4096;

    fn deinit(p: *Probe) void {
        if (p.a) |*a| a.free();
        if (p.s) |*s| s.deinit();
    }

    fn steps(p: *Probe) Steps {
        return .{ .ctx = p, .prepare = prepare, .wire = wire, .nccl = nccl, .wait = wait, .equal = equal };
    }

    fn from(ctx: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ctx));
    }

    fn prepare(ctx: *anyopaque) anyerror!void {
        const p = from(ctx);
        const world = p.g.world;
        p.a = try cuda.DeviceBuffer.alloc(p.d, 2 * world * n);
        p.s = try cuda.Stream.init(p.d, true);
        var mine: [n]u8 = undefined;
        tp.reference.fill(&mine, .bf16, @intCast(p.g.rank), 0x7072_6f62);
        try p.a.?.uploadAsync(p.g.rank * n, &mine, p.s.?.handle);
        try p.a.?.uploadAsync((world + p.g.rank) * n, &mine, p.s.?.handle);
        try p.s.?.synchronize(); // the rows are up before this rank says it is ready
    }

    fn wire(ctx: *anyopaque) anyerror!void {
        const p = from(ctx);
        try p.r.allGatherBytes(p.a.?.ptr + p.g.rank * n, p.a.?.ptr, n, p.s.?.handle);
    }

    fn nccl(ctx: *anyopaque) anyerror!void {
        const p = from(ctx);
        try p.g.allGather(p.s.?, p.a.?.ptr + (p.g.world + p.g.rank) * n, p.a.?.ptr + p.g.world * n, n);
    }

    fn wait(ctx: *anyopaque) anyerror!void {
        try from(ctx).s.?.synchronize();
    }

    fn equal(ctx: *anyopaque) bool {
        const p = from(ctx);
        const world = p.g.world;
        p.r.check() catch return false;
        const got = p.gpa.alloc(u8, 2 * world * n) catch return false;
        defer p.gpa.free(got);
        p.a.?.download(0, got) catch return false;
        const same = std.mem.eql(u8, got[0 .. world * n], got[world * n ..]);
        if (!same) std.log.warn("rank {d}: TF_GLM_ROCE_MULTI's probe: the wire's gather DIFFERS from NCCL's", .{p.g.rank});
        return same;
    }
};

/// The wire is lent only when every rank's probe matched NCCL's bytes: one rank's mismatch keeps every rank on NCCL.
fn lendIfAll(b: *tp.Bootstrap, good: bool) !void {
    if (!try everyRank(b, good)) return error.ProbeMismatch;
}

/// Every rank's vote over the bootstrap, the same answer on every rank.
fn everyRank(b: *tp.Bootstrap, yes: bool) !bool {
    var blobs: [tp.bootstrap.max_ranks][]const u8 = undefined;
    var buf: [tp.bootstrap.max_ranks]u8 = undefined;
    return allSaid(try b.allGather(&.{@intFromBool(yes)}, &buf, &blobs));
}

/// Whether every rank's one-byte vote is yes: every rank reads the same votes, so every rank decides alike.
fn allSaid(votes: []const []const u8) bool {
    for (votes) |v| if (v.len != 1 or v[0] != 1) return false;
    return true;
}

/// The lent wire and what went down it, for its account at the end.
const Wire = struct {
    gpa: std.mem.Allocator,
    r: *tp.roce_multi.Roce,
    rank: usize,
    gathers: std.atomic.Value(u64) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
};

fn of(ctx: *anyopaque) *Wire {
    return @ptrCast(@alignCast(ctx));
}

fn fits(ctx: *anyopaque, bytes: usize) bool {
    return of(ctx).r.fits(bytes);
}

fn gather(ctx: *anyopaque, s: cuda.abi.Stream, mine: cuda.abi.DevicePtr, all: cuda.abi.DevicePtr, bytes: usize) anyerror!void {
    const w = of(ctx);
    try w.r.allGatherBytes(mine, all, bytes, s);
    // once: the engine's own gathers take the wire (a server's stop exits before any close prints the account)
    if (w.gathers.fetchAdd(1, .monotonic) == 0) std.log.info("rank {d}: TF_GLM_ROCE_MULTI: the first gather is on the wire ({d} bytes a rank)", .{ w.rank, bytes });
    _ = w.bytes.fetchAdd(bytes, .monotonic);
}

fn check(ctx: *anyopaque) anyerror!void {
    return of(ctx).r.check(); // a failed wait's line names the rank and seq it waited on
}

fn abort(ctx: *anyopaque) void {
    const w = of(ctx);
    w.r.abort();
    report(w, "aborted");
}

fn deinit(ctx: *anyopaque) void {
    const w = of(ctx);
    report(w, "closing");
    w.r.deinit();
    w.gpa.destroy(w);
}

/// Gathers, bytes, the proxy's last post and failure, each HCA's QPs: a rank whose posts stop short is the stuck one.
fn report(w: *const Wire, when: []const u8) void {
    var line: [384]u8 = undefined;
    const r = w.r;
    const fmt = "rank {d}: TF_GLM_ROCE_MULTI {s}: {d} gathers, {d} bytes, posted to seq {d}, failed {s}{s}; {s}";
    const args = .{ w.rank, when, w.gathers.load(.monotonic), w.bytes.load(.monotonic), r.last_seq, if (r.failed()) "yes " else "no", r.proxyError(), r.describeLink(&line) };
    if (r.failed()) std.log.warn(fmt, args) else std.log.info(fmt, args);
}

test "TF_GLM_ROCE_MULTI off, or under three ranks: the group keeps its own route and nothing opens" {
    var g: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 3 };
    try open(std.testing.allocator, undefined, undefined, &g, false);
    try std.testing.expect(g.multi == null);
    g.world = 2;
    try open(std.testing.allocator, undefined, undefined, &g, true);
    try std.testing.expect(g.multi == null);
}

test "the wire is lent only when every rank's vote is one byte saying yes" {
    try std.testing.expect(allSaid(&.{ &.{1}, &.{1}, &.{1} }));
    try std.testing.expect(!allSaid(&.{ &.{1}, &.{0}, &.{1} }));
    try std.testing.expect(!allSaid(&.{ &.{1}, &.{}, &.{1} }));
    try std.testing.expect(!allSaid(&.{ &.{1}, &.{ 1, 1 }, &.{1} }));
}

test "a probe step failing on any one rank: every rank refuses, or every rank keeps NCCL, and none enters a gather unready" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    const Point = enum { none, prepare, wire, nccl, wait, equal };
    const Run = struct {
        port: u16,
        lfd: tp.sock.Fd,
        at: Point,
        on: u32,
        took: [6]std.atomic.Value(u32) = @splat(.init(0)),
        errs: [3]?anyerror = .{ null, null, null },
        entered: [3]bool = .{ false, false, false },

        const R = @This();
        const Rank = struct {
            run: *R,
            me: u32,

            fn step(k: *Rank, p: Point) anyerror!void {
                _ = k.run.took[@backingInt(p)].fetchAdd(1, .monotonic);
                if (k.run.at == p and k.run.on == k.me) return error.Injected;
            }
            fn from(ctx: *anyopaque) *Rank {
                return @ptrCast(@alignCast(ctx));
            }
            fn prepare(ctx: *anyopaque) anyerror!void {
                return from(ctx).step(.prepare);
            }
            fn wire(ctx: *anyopaque) anyerror!void {
                return from(ctx).step(.wire);
            }
            fn nccl(ctx: *anyopaque) anyerror!void {
                return from(ctx).step(.nccl);
            }
            fn wait(ctx: *anyopaque) anyerror!void {
                return from(ctx).step(.wait);
            }
            fn equal(ctx: *anyopaque) bool {
                from(ctx).step(.equal) catch return false;
                return true;
            }
        };

        fn rank(run: *@This(), me: u32) void {
            var b = (if (me == 0) tp.Bootstrap.acceptOn(run.lfd, 3, 5 * std.time.ns_per_s) else tp.Bootstrap.open(.{ .world = 3, .rank = me, .link_port = run.port, .connect_timeout_ns = 5 * std.time.ns_per_s })) catch |e| {
                run.errs[me] = e;
                return;
            };
            defer b.close();
            var k: Rank = .{ .run = run, .me = me };
            vote(&b, .{ .ctx = &k, .prepare = Rank.prepare, .wire = Rank.wire, .nccl = Rank.nccl, .wait = Rank.wait, .equal = Rank.equal }, &run.entered[me]) catch |e| {
                run.errs[me] = e;
            };
        }
    };
    const Verdict = enum { lend, nccl, refuse };
    for (std.enums.values(Point)) |at| for (0..3) |on| {
        if (at == .none and on > 0) continue;
        const l = try tp.bootstrap.testListener();
        defer tp.sock.close(l.fd);
        var run: Run = .{ .port = l.port, .lfd = l.fd, .at = at, .on = @intCast(on) };
        var t: [2]std.Thread = undefined;
        for (&t, 1..) |*x, me| x.* = try std.Thread.spawn(.{}, Run.rank, .{ &run, @as(u32, @intCast(me)) });
        run.rank(0);
        for (t) |x| x.join();
        const want: Verdict = switch (at) {
            .none => .lend,
            .prepare, .wire, .equal => .nccl,
            .nccl, .wait => .refuse,
        };
        for (run.errs, run.entered) |e, entered| {
            const got: Verdict = if (e) |err| (if (fallsBack(err, entered)) .nccl else .refuse) else .lend;
            try std.testing.expectEqual(want, got);
        }
        // unready anywhere: no rank entered a gather; past that, every rank entered NCCL's, the wire failing or not
        const gathers: u32 = if (at == .prepare) 0 else 3;
        try std.testing.expectEqual(gathers, run.took[@backingInt(Point.wire)].load(.monotonic));
        try std.testing.expectEqual(gathers, run.took[@backingInt(Point.nccl)].load(.monotonic));
    };
}
