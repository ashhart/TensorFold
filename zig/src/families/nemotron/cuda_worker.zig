//! TP over hosts for a latent MoE: rank 0 runs the model and asks rank 1, which holds only its expert shares, per layer.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const Kind = @import("config.zig").Kind;
const kern = @import("cuda_kernels.zig");
const W = @import("cuda_weights.zig");
const latent = @import("cuda_latent.zig");
const fwd = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const Source = @import("cuda_source.zig").Source;
const Engine = @import("cuda_engine.zig").Engine;

/// Rows a job's first message carries after its header; a longer job sends its rows in a second message.
pub const small_rows = 16;
const magic: u32 = 0x4a4f4253;
const header_bytes = 64;
const Job = enum(u32) { rows = 1, end = 2 };

/// Byte offsets of a job's sections, `per` rows each, 256 apart: picks (int32), latent rows (bf16), hidden rows (bf16).
fn sections(c: Config, per: usize) [3]usize {
    const picks = std.mem.alignForward(usize, per * c.top_k * 4, 256); // rows stay 16-byte aligned for odd counts
    return .{ 0, picks, picks + std.mem.alignForward(usize, per * c.latent * 2, 256) };
}

/// Bytes of a job's sections for `rows` rows.
pub fn payloadBytes(c: Config, rows: usize) usize {
    return sections(c, rows)[2] + rows * c.hidden * 2;
}

/// Bytes of a job's first message: the header, then a small job's sections.
pub fn firstBytes(c: Config) usize {
    return header_bytes + payloadBytes(c, small_rows);
}

/// Rank 0's job buffer: the first message, then a long job's sections.
pub fn jobBytes(c: Config) usize {
    return firstBytes(c) + payloadBytes(c, state.pass_rows);
}

/// Rank 0: layer `id`'s job of `rows` rows to rank 1, its picks, latent rows and hidden rows.
pub fn send(f: *const fwd.Forward, id: u32, rows: usize) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    const tp = f.tp.?;
    if (tp.world != 2) return error.TensorParallelNotBuilt;
    for ([4]u32{ magic, @intFromEnum(Job.rows), id, @intCast(rows) }, 0..) |v, i| try o.fill32(b.job + i * 4, v, 1);
    const small = rows <= small_rows;
    const base: u64 = if (small) b.job + header_bytes else b.job + firstBytes(c);
    const at = sections(c, if (small) small_rows else rows);
    try o.copy(base + at[0], b.pick, rows * c.top_k * 4);
    try o.copy(base + at[1], b.lat, rows * c.latent * 2);
    try o.copy(base + at[2], b.y, rows * c.hidden * 2);
    try tp.send(o.s, b.job, firstBytes(c), 1);
    if (!small) try tp.send(o.s, base, payloadBytes(c, rows), 1);
}

/// Rank 0: rank 1's expert rows for its job, into rank 1's region of ymoe.
pub fn receive(f: *const fwd.Forward, rows: usize) !void {
    const region: u64 = latent.partFloats(f.c, state.pass_rows) * 4;
    try f.tp.?.recv(f.ops.s, f.b.ymoe + region, latent.partFloats(f.c, rows) * 4, 1);
}

/// Rank 0 as its engine closes: the end header, so rank 1 leaves its loop.
pub fn end(e: *Engine) !void {
    cuda.tp_watch.ended(); // the other ranks' lines close on purpose from here
    const o = e.ops();
    for ([4]u32{ magic, @intFromEnum(Job.end), 0, 0 }, 0..) |v, i| try o.fill32(e.b.job + i * 4, v, 1);
    try e.tp.?.send(o.s, e.b.job, firstBytes(e.c), 1);
    try e.stream.synchronize();
}

/// Rank 1's tables: each MoE block's (by block index) and the MTP head's.
const Layers = struct { blocks: []?latent.Tables, mtp: ?latent.Tables = null };

fn loadLayers(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, c: Config, w: *W.Weights) !Layers {
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    var L: W.Loader = .{ .gpa = gpa, .ops = ops, .w = w, .src = &src, .scratch = try cuda.DeviceBuffer.alloc(ops.k.d, 1 << 20) };
    defer {
        src.flush() catch {};
        ops.s.synchronize() catch {};
        L.scratch.free();
        L.host.deinit(gpa);
        for (L.held.items) |h| gpa.free(h);
        L.held.deinit(gpa);
    }
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try src.add(&ck);
    var out: Layers = .{ .blocks = try gpa.alloc(?latent.Tables, c.layers) };
    errdefer gpa.free(out.blocks);
    var nm: [128]u8 = undefined;
    var pb: [128]u8 = undefined;
    for (c.kinds[0..c.layers], out.blocks, 0..) |kind, *t, i| {
        t.* = null;
        if (kind != Kind.moe) continue;
        const pre = try std.fmt.bufPrint(&pb, "{s}backbone.layers.{d}.mixer.", .{ c.prefix(), i });
        t.* = try latent.tables(&L, &ck, try std.fmt.bufPrint(&nm, "blocks.{d}.moe", .{i}), pre, c);
    }
    const mtp_pre = try std.fmt.bufPrint(&pb, "{s}mtp.layers.1.mixer.", .{c.prefix()});
    var gate: [160]u8 = undefined;
    if (ck.has(try std.fmt.bufPrint(&gate, "{s}gate.weight", .{mtp_pre}))) out.mtp = try latent.tables(&L, &ck, "mtp.moe", mtp_pre, c);
    try src.flush();
    try ops.s.synchronize();
    return out;
}

/// Rank r > 0: its expert shares of every MoE layer, computed for rank 0's jobs until rank 0 ends.
pub fn serve(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, precision: ?@import("config.zig").Precision) !void {
    const t0 = std.Io.Clock.awake.now(io);
    var c = try Config.read(gpa, io, dir);
    try latent.place(&c);
    if (c.rank == 0 or c.latent == 0) return error.NotAnExpertRank;
    try latent.checkHeld(c);
    c.ck = try @import("cuda_latent_ck.zig").wanted(ctx, c, precision);
    try cuda.tp_watch.start(io, c.rank, c.world); // rank 0's line before NCCL: either rank's failure stops both
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try kern.Kernels.load(gpa, io, ctx, null);
    defer k.deinit();
    try k.loadModelopt(ctx);
    try k.loadLatent();
    if (c.ck) try k.loadCk(ctx);
    var g = try cuda.tp_group.Group.open(c.rank, c.world);
    defer g.close();
    try latent.agree(gpa, io, c, &g, ctx.d, stream, dir);
    const ops: kern.Ops = .{ .k = &k, .s = stream };
    var w: W.Weights = .{ .gpa = gpa, .config = c };
    defer w.deinit();
    const layers = try loadLayers(gpa, io, ops, dir, c, &w);
    defer gpa.free(layers.blocks);
    const R = state.pass_rows;
    const own = c.ownShares();
    const pairs = R * c.top_k * own;
    const sw = c.expert_width / c.shares;
    const ssw = c.shared_width / c.shares;
    const items = kern.maxItems(pairs, c.experts * own, 16);
    const ck4 = if (c.ck) @import("cuda_latent_ck.zig").rowsBytes(c, R) else 0; // NVFP4 rows (checkpoint math)
    const sizes = [_]usize{ firstBytes(c), payloadBytes(c, R), pairs * sw * 2, R * own * ssw * 2, R * own * 4, latent.partFloats(c, R) * 4, pairs * 4, items * 12, 8, pairs * 4, (pairs + 1023) / 1024 * c.experts * own * 4, ck4 };
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    var arena = try cuda.DeviceBuffer.alloc(ctx.d, total);
    defer arena.free();
    var at: [sizes.len]u64 = undefined;
    var used: usize = 0;
    for (sizes, &at) |n, *p| {
        p.* = arena.ptr + used;
        used += std.mem.alignForward(usize, n, 256);
    }
    const first = at[0];
    const long = at[1];
    const out = at[5];
    const scratch: latent.Scratch = .{ .plan = .{ .members = at[6], .items = at[7], .counts = at[8], .rank = at[9], .hist = at[10] }, .act = at[2], .sact = at[3], .sids = at[4], .ck4 = if (c.ck) at[11] else 0 };
    try latent.initShareIds(k.latent.?, stream, scratch.sids, R, own);
    try stream.synchronize();
    var blocks: usize = 0;
    for (layers.blocks) |t| blocks += @intFromBool(t != null);
    std.log.info("rank {d} of {d}: {d} MoE layers{s} of share {d} ({s}) in {d:.1} s, {d:.2} GiB on the device; serving rank 0's jobs", .{ c.rank, c.world, blocks, if (layers.mtp != null) " and the MTP head's" else "", c.rank, if (c.ck) "the checkpoint's math" else "full precision", @import("cuda_engine.zig").seconds(io, t0), @as(f64, @floatFromInt(cuda.usage(false).device)) / (1 << 30) });
    try cuda.tp_watch.ready(io); // every rank loaded before rank 0's first job
    var jobs: usize = 0;
    while (true) {
        try g.recv(stream, first, firstBytes(c), 0);
        try stream.synchronize();
        var h: [4]u32 = undefined;
        try ops.download(std.mem.sliceAsBytes(&h), first);
        try stream.synchronize();
        if (h[0] != magic) return error.BadJob;
        if (h[1] == @intFromEnum(Job.end)) {
            cuda.tp_watch.ended();
            break;
        }
        const rows: usize = h[3];
        if (rows < 1 or rows > R) return error.BadJob;
        const t = (if (h[2] == latent.mtp_id) layers.mtp else if (h[2] < layers.blocks.len) layers.blocks[h[2]] else null) orelse return error.BadJob;
        const small = rows <= small_rows;
        const base = if (small) first + header_bytes else long;
        if (!small) try g.recv(stream, long, payloadBytes(c, rows), 0);
        const s = sections(c, if (small) small_rows else rows);
        try latent.partials(ops, t, scratch, base + s[0], base + s[1], base + s[2], rows, own, c, out);
        try g.send(stream, out, latent.partFloats(c, rows) * 4, 0);
        jobs += 1;
    }
    try stream.synchronize();
    std.log.info("rank {d}: rank 0 ended after {d} jobs; leaving", .{ c.rank, jobs });
}

test "a job's sections and messages" {
    var c: Config = undefined;
    c.top_k = 22;
    c.latent = 1024;
    c.hidden = 4096;
    try std.testing.expectEqual([3]usize{ 0, 1536, 1536 + 16 * 2048 }, sections(c, 16));
    try std.testing.expectEqual(@as(usize, 64 + 1536 + 16 * 2048 + 16 * 8192), firstBytes(c));
    try std.testing.expectEqual(@as(usize, 0), sections(c, 1821)[1] % 256); // an odd job's rows start aligned
    try std.testing.expectEqual(@as(usize, 0), sections(c, 1821)[2] % 256);
}
