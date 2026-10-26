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
    const w = borrow(gpa, io, d, g, cfg) catch |err| {
        if (cfg.roce_fallback == .@"error") return err; // TF_TP_ROCE_FALLBACK=error refuses rather than run on NCCL
        std.log.warn("rank {d}: TF_GLM_ROCE_MULTI=1, but the wire did not open ({s}); NCCL keeps the small gathers", .{ g.rank, @errorName(err) });
        return;
    };
    g.multi = .{ .ctx = w, .fits = fits, .gather = gather, .check = check, .abort = abort, .deinit = deinit };
}

/// fabric/tp's bootstrap, ports, wire and kernel, and a probe against the group's NCCL: each step every rank's or none.
fn borrow(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *const Group, cfg: tp.Config) !*Wire {
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
    if (!try everyRank(&b, probe(gpa, d, g, r) catch false)) return error.ProbeMismatch;
    var line: [384]u8 = undefined;
    std.log.info("rank {d}: TF_GLM_ROCE_MULTI: gathers up to {d} KiB a rank on the wire, its probe equal to NCCL's: {s}", .{ g.rank, r.slot_bytes / 1024, r.describeLink(&line) });
    return w;
}

/// One 4 KiB gather in place, on the wire and on the group's NCCL (no wire lent yet): every byte must be equal.
fn probe(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *const Group, r: *tp.roce_multi.Roce) !bool {
    const n = 4096;
    const world = g.world;
    var a = try cuda.DeviceBuffer.alloc(d, 2 * world * n); // the wire's rows, then NCCL's
    defer a.free();
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var mine: [n]u8 = undefined;
    tp.reference.fill(&mine, .bf16, @intCast(g.rank), 0x7072_6f62);
    try a.uploadAsync(g.rank * n, &mine, s.handle);
    try a.uploadAsync((world + g.rank) * n, &mine, s.handle);
    try r.allGatherBytes(a.ptr + g.rank * n, a.ptr, n, s.handle);
    try g.allGather(s, a.ptr + (world + g.rank) * n, a.ptr + world * n, n);
    try s.synchronize();
    try r.check();
    const got = try gpa.alloc(u8, 2 * world * n);
    defer gpa.free(got);
    try a.download(0, got);
    const equal = std.mem.eql(u8, got[0 .. world * n], got[world * n ..]);
    if (!equal) std.log.warn("rank {d}: TF_GLM_ROCE_MULTI's probe: the wire's gather DIFFERS from NCCL's", .{g.rank});
    return equal;
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
