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
    const good = probe(s, d) catch false;
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
