//! Multirank gather setup and an exact NCCL probe; the two-rank session setup stays separate.
const std = @import("std");
const cuda = @import("cuda");
const roce = @import("roce_multi.zig");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const bootstrap = @import("bootstrap.zig");
const pick = @import("roce_pick.zig");
const rv = @import("roce_verbs.zig");
const reference = @import("reference.zig");
const log = @import("log.zig");

pub fn open(s: anytype, d: *const cuda.Driver, b: *Bootstrap, image: []const u8) !void {
    const cfg = s.cfg;
    if (cfg.world < 3 or cfg.world > roce.max_world) return fallback(s, error.UnsupportedRoceWorld);
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var picked = true;
    const n = if (cfg.roce_wire == .verbs) pick.resolve(cfg, b, &specs) catch blk: {
        picked = false;
        break :blk 0;
    } else 0;
    var pick_blobs: [bootstrap.max_ranks][]const u8 = undefined;
    var pick_buf: [bootstrap.max_ranks]u8 = undefined;
    for (try b.allGather(&.{@intFromBool(picked)}, &pick_buf, &pick_blobs)) |x|
        if (x.len != 1 or x[0] != 1) return fallback(s, error.RoceHcaSelectionFailed);
    const r = roce.Roce.init(d, b, image, .{ .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = specs[0..n], .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = cfg.mailbox_max_bytes, .timeout_ns = cfg.mailbox_timeout_ns, .fast = cfg.roce_fast, .blocks = cfg.roce_blocks }) catch |e| return fallback(s, e);
    s.roce_multi = r;
    if (cfg.roce_chunked_bytes > 0 and !r.chunk_abi) {
        r.deinit();
        s.roce_multi = null;
        return error.ChunkAbiMissing;
    }
    s.route = .{ .small = .{ .roce_multi = r }, .nccl = &s.nccl, .max_bytes = cfg.mailbox_max_bytes, .chunked_bytes = cfg.roce_chunked_bytes };
    try keepIfAll(s, b, r, probe(s, d) catch false);
}

/// The wire stays only when every rank's probe matched NCCL's bytes: one rank's mismatch drops it on every rank.
fn keepIfAll(s: anytype, b: *Bootstrap, r: *roce.Roce, good: bool) !void {
    var blobs: [bootstrap.max_ranks][]const u8 = undefined;
    var buf: [bootstrap.max_ranks]u8 = undefined;
    for (try b.allGather(&.{@intFromBool(good)}, &buf, &blobs)) |x| if (x.len != 1 or x[0] != 1) {
        r.deinit();
        s.roce_multi = null;
        return fallback(s, error.ProbeMismatch);
    };
}

fn fallback(s: anytype, e: anyerror) !void {
    if (s.cfg.roce_fallback == .@"error") return e;
    if (s.cfg.rank == 0) log.warn("tp: multirank RoCE unavailable ({t}); every rank uses NCCL", .{e});
}

/// No rank-order surrogate or tolerance: both actual results must equal NCCL's bytes.
fn probe(s: anytype, d: *const cuda.Driver) !bool {
    const n = 4096;
    const w: usize = s.cfg.world;
    var a = try cuda.DeviceBuffer.alloc(d, (3 + 2 * w) * n);
    defer a.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var input: [n]u8 = undefined;
    reference.fill(&input, .bf16, s.cfg.rank, 0x7072_6f62);
    try a.uploadAsync(0, &input, stream.handle);
    const fast = s.route.iface();
    const slow = s.nccl.iface();
    try fast.allGather(a.ptr, a.ptr + n, n, .u8, stream.handle);
    try slow.allGather(a.ptr, a.ptr + (1 + w) * n, n, .u8, stream.handle);
    try fast.allReduce(a.ptr, a.ptr + (1 + 2 * w) * n, n / 2, .bf16, .sum, stream.handle);
    try slow.allReduce(a.ptr, a.ptr + (2 + 2 * w) * n, n / 2, .bf16, .sum, stream.handle);
    try stream.synchronize();
    try fast.check();
    const got = try std.heap.page_allocator.alloc(u8, (2 + 2 * w) * n);
    defer std.heap.page_allocator.free(got);
    try a.download(n, got);
    return std.mem.eql(u8, got[0 .. w * n], got[w * n .. 2 * w * n]) and std.mem.eql(u8, got[2 * w * n ..][0..n], got[(2 * w + 1) * n ..][0..n]);
}

test "multirank probe: one rank's mismatch leaves every rank without the wire; matches on every rank keep it on every rank" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest; // the shm wire: /dev/shm, a pinned proxy
    const sock = @import("sock.zig");
    const Vote = struct {
        port: u16,
        lfd: sock.Fd,
        odd: ?u32,
        kept: [3]bool = .{ false, false, false },
        errs: [3]?anyerror = .{ null, null, null },

        fn rank(v: *@This(), me: u32) void {
            v.errs[me] = v.body(me);
        }

        fn body(v: *@This(), me: u32) ?anyerror {
            var b = (if (me == 0) Bootstrap.acceptOn(v.lfd, 3, 5 * std.time.ns_per_s) else Bootstrap.open(.{ .world = 3, .rank = me, .link_port = v.port, .connect_timeout_ns = 5 * std.time.ns_per_s })) catch |e| return e;
            defer b.close();
            const r = roce.Roce.init(null, &b, "", .{ .wire = .shm, .max_bytes = 4096 }) catch |e| return e;
            var s: struct { cfg: struct { roce_fallback: enum { nccl, @"error" } = .nccl, rank: u32 }, roce_multi: ?*roce.Roce } = .{ .cfg = .{ .rank = me }, .roce_multi = r };
            keepIfAll(&s, &b, r, v.odd != me) catch |e| return e;
            v.kept[me] = s.roce_multi != null;
            if (s.roce_multi) |x| {
                b.agree("kept", "probe test end") catch |e| return e; // no rank unmaps its region while a peer may read it
                x.deinit();
            }
            return null;
        }
    };
    for ([_]?u32{ 1, null }) |odd| {
        const l = try bootstrap.testListener();
        defer sock.close(l.fd);
        var v: Vote = .{ .port = l.port, .lfd = l.fd, .odd = odd };
        var t: [2]std.Thread = undefined;
        for (&t, 1..) |*x, me| x.* = try std.Thread.spawn(.{}, Vote.rank, .{ &v, @as(u32, @intCast(me)) });
        v.rank(0);
        for (t) |x| x.join();
        for (v.errs) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (v.kept) |k| try std.testing.expectEqual(odd == null, k);
    }
}
