//! flashnext-exl3-tp: EXL3 expert parallel (cuda_exl3_tp.zig) on two ranks, one process a GPU: layer, head, memory.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const T = @import("../model/flashnext_exl3_tests.zig");

const X = flash.exl3;
const tp = X.tp;
const wt = X.prompt_wt;
const pl = X.prompt_load;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const Gpu = check.Gpu;

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;

// -- the two ranks' control link -----------------------------------------------------------------------------------

const Ranks = struct {
    rank: usize,
    link: wire.Link,
    collective: Collective,
    seq: u64 = 10,

    fn open(gpu: Gpu, rank: usize, port: u16) !Ranks {
        const link = try wire.Link.open(gpu.io, "127.0.0.1", port, rank, 120_000);
        errdefer link.close();
        // rank 0 does float64 work between collectives: a generous deadline (the waits are bounded by the stream)
        return .{ .rank = rank, .link = link, .collective = try Collective.init(link, rank, 3_600_000) };
    }

    fn close(r: *Ranks) void {
        r.collective.deinit();
        r.link.close();
    }

    /// Both ranks here; rank 1's `value` reaches rank 0 (returned there), rank 0's reaches rank 1.
    fn swap(r: *Ranks, value: u64) !u64 {
        r.seq += 1;
        const until = wire.deadline(3_600_000);
        if (r.rank == 0) {
            const got = try r.link.receiveHeader(until);
            if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
            try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
            return got.arg;
        }
        try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
        const got = try r.link.receiveHeader(until);
        if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
        return got.arg;
    }
};

// -- one decoder layer, stepped -------------------------------------------------------------------------------------

/// A layer's output streams, routing and canonical fp32 branch (slot scratch is rank-local).
const Out = struct { h: []u16, picks: []i32, wts: []f32, branch: []f32 };

fn grab(comptime E: type, a: std.mem.Allocator, gpu: Gpu, p: *T.Pass, ptr: u64, n: usize) ![]E {
    const got = try T.download(E, gpu, p.stream, ptr, n);
    defer gpu.gpa.free(got);
    return a.dupe(E, got);
}

/// cuda_batch.zig Pass.chunk's loop body for one layer (T.deviceLayer's sequence), the MoE's outputs read back.
fn step(gpu: Gpu, a: std.mem.Allocator, p: *T.Pass, lw: *const pl.Layer, l: usize, ids: []const u32, pos0: usize, h_in: []const u16) !Out {
    const s = p.s;
    const n = ids.len;
    p.rows = n;
    p.pos0 = pos0;
    p.pending = null;
    try p.stage(ids);
    try T.uploadTo(gpu, p.stream, s.h, std.mem.sliceAsBytes(h_in));
    try p.writeback(0, n);
    try p.readout(lw.attn_hc, 0, n);
    if (lw.linear) try p.gdnBlock(lw, l, wt.linearIndex(l), .{}) else try p.attnBlock(lw, wt.attnIndex(l));
    try p.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", n * dims, n, dims, streams, 4, 1, 1, wt.out_sk);
    try p.readout(lw.mlp_hc, 0, n);
    try p.moeBlock(lw);
    var o: Out = undefined;
    o.picks = try grab(i32, a, gpu, p, s.picks, n * slots);
    o.wts = try grab(f32, a, gpu, p, s.wts, n * slots);
    o.branch = try grab(f32, a, gpu, p, s.moe_f32, n * dims);
    p.pending = l;
    try p.writeback(0, n);
    o.h = try grab(u16, a, gpu, p, s.h, n * wide);
    return o;
}

/// prefill 8 rows then 4 single decode rows (T.layerCheck's calls)
const calls = [_][2]usize{ .{ 0, 8 }, .{ 8, 1 }, .{ 9, 1 }, .{ 10, 1 }, .{ 11, 1 } };

fn embedRows(a: std.mem.Allocator, raw: *const T.RawPack, toks: []const u32) ![]u16 {
    const table = try raw.get("model.language_model.embed_tokens.weight", .{});
    const hb = try a.alloc(u16, toks.len * wide);
    for (toks, 0..) |tk, t| for (0..dims) |i| {
        const bits = std.mem.readInt(u16, table.bytes[(tk * dims + i) * 2 ..][0..2], .little);
        for (0..streams) |sidx| hb[t * wide + sidx * dims + i] = bits;
    };
    return hb;
}

/// The 5 calls on `lw` from a reset rig.
fn sequence(gpu: Gpu, a: std.mem.Allocator, rig: *T.Rig, raw: *const T.RawPack, lw: *const pl.Layer, l: usize) ![calls.len]Out {
    try rig.reset();
    var p = rig.pass(lw[0..1]);
    var outs: [calls.len]Out = undefined;
    for (calls, &outs) |c, *o| {
        const toks = T.default_ids[c[0] .. c[0] + c[1]];
        o.* = try step(gpu, a, &p, lw, l, toks, c[0], try embedRows(a, raw, toks));
    }
    return outs;
}

fn hashOf(o: Out) u64 {
    var h = std.hash.Wyhash.init(0x7e_2);
    h.update(std.mem.sliceAsBytes(o.h));
    h.update(std.mem.sliceAsBytes(o.picks));
    h.update(std.mem.sliceAsBytes(o.wts));
    h.update(std.mem.sliceAsBytes(o.branch));
    return h.final();
}

fn layerMode(gpu: Gpu, rk: *Ranks, dir: []const u8, which: []const usize) !void {
    const rank = rk.rank;
    var rig: T.Rig = undefined;
    try rig.init(gpu, dir, 8, 64);
    defer rig.deinit();
    var raw = try T.RawPack.open(gpu, dir);
    defer raw.close(gpu);
    var failed: usize = 0;
    for (which) |l| {
        var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var t0 = check.now(gpu.io);
        var lw2 = try pl.loadLayerRank(gpu.gpa, @constCast(gpu.d), &rig.stream, &rig.store, rig.pack_fn, l, 2, rank, null);
        defer lw2.free(gpu.gpa);
        const x2 = lw2.exl3 orelse return error.UnexpectedTensor;
        try check.expect(x2.tp != null and x2.tp.?.rank == rank, "layer {d}: rank {d} experts carry no rank", .{ l, rank });
        var held: usize = 0;
        for (x2.trellis[0]) |ptr| held += @intFromBool(ptr != 0);
        std.debug.print("== rank {d} layer {d} ({s}): tp=2 layer loaded in {d:.0} ms, {d} of 513 experts held (ids {d}..{d})\n", .{ rank, l, if (wt.linearLayer(l)) "GDN" else "attention", @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6, held, x2.tp.?.lo, x2.tp.?.hi - 1 });
        try check.expect(held == x2.tp.?.hi - x2.tp.?.lo, "layer {d}: rank {d} holds {d} experts", .{ l, rank, held });

        // tp=1 on rank 0 alone (rank 1 waits at the swap)
        var ref: [calls.len]Out = undefined;
        if (rank == 0) {
            t0 = check.now(gpu.io);
            var lw1 = try pl.loadLayerRank(gpu.gpa, @constCast(gpu.d), &rig.stream, &rig.store, rig.pack_fn, l, 1, 0, null);
            defer lw1.free(gpu.gpa);
            rig.collective = null;
            ref = try sequence(gpu, a, &rig, &raw, &lw1, l);
            std.debug.print("  tp=1 reference (all 513 experts on this card) in {d:.0} ms\n", .{@as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6});
        }
        _ = try rk.swap(0);
        rig.collective = &rk.collective;
        t0 = check.now(gpu.io);
        const got = try sequence(gpu, a, &rig, &raw, &lw2, l);
        const tp_ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
        // both ranks must hold the same bits (replicated streams): rank 1's hashes to rank 0
        var same_ranks = true;
        for (got) |o| {
            const theirs = try rk.swap(hashOf(o));
            if (rank == 0) same_ranks = same_ranks and theirs == hashOf(o);
        }
        if (rank == 1) {
            check.pass("rank 1 layer {d}: tp=2 mirrored the 5 calls in {d:.0} ms", .{ l, tp_ms });
        } else {
            try check.expect(same_ranks, "layer {d}: the ranks' outputs or routing differ in bits", .{l});
            var picks_equal = true;
            var wts_equal = true;
            var branch_equal = true;
            var h_equal: usize = 0;
            var h_total: usize = 0;
            for (ref, got) |want, have| {
                picks_equal = picks_equal and std.mem.eql(i32, want.picks, have.picks);
                wts_equal = wts_equal and std.mem.eql(u8, std.mem.sliceAsBytes(want.wts), std.mem.sliceAsBytes(have.wts));
                branch_equal = branch_equal and std.mem.eql(u8, std.mem.sliceAsBytes(want.branch), std.mem.sliceAsBytes(have.branch));
                for (want.h, have.h) |u, v| {
                    h_equal += @intFromBool(u == v);
                    h_total += 1;
                }
            }
            std.debug.print("  tp=2 ({d:.0} ms for the 5 calls): picks {s}, weights {s}, canonical fp32 branch {s}; layer output bf16 equal {d}/{d}\n", .{ tp_ms, if (picks_equal) "equal" else "DIFFER", if (wts_equal) "equal" else "DIFFER", if (branch_equal) "equal" else "DIFFER", h_equal, h_total });
            var ok = true;
            check.expect(picks_equal and wts_equal, "layer {d}: tp=2 router picks or weights differ from tp=1", .{l}) catch {
                ok = false;
            };
            check.expect(branch_equal, "layer {d}: tp=2's canonical fp32 branch differs from tp=1", .{l}) catch {
                ok = false;
            };
            check.expect(h_equal == h_total, "layer {d}: tp=2's layer output is not tp=1's bits ({d}/{d})", .{ l, h_equal, h_total }) catch {
                ok = false;
            };
            if (ok) check.pass("layer {d} tp=2 vs tp=1: picks, weights, canonical branch and layer output bit-identical ({d}/{d} outputs); ranks bit-identical", .{ l, h_equal, h_total }) else failed += 1;
        }
        // tp=2 against float64 (T.layerCheck on rank 0; rank 1 mirrors its device calls)
        _ = try rk.swap(1);
        if (rank == 0) {
            T.layerCheck(gpu, &rig, &raw, &lw2, l, &T.default_ids) catch |err| if (err == error.TestFailed) {
                failed += 1;
            } else return err;
        } else {
            _ = try sequence(gpu, a, &rig, &raw, &lw2, l);
        }
        _ = try rk.swap(2);
    }
    const all = try rk.swap(failed);
    try check.expect(failed == 0 and all == 0, "flashnext-exl3-tp layer: {d} checks failed on rank {d}", .{ failed, rank });
}

// -- the head's vocabulary split (one card) -------------------------------------------------------------------------

fn headMode(gpu: Gpu, dir: []const u8) !void {
    var rig: T.Rig = undefined;
    try rig.init(gpu, dir, 8, 64);
    defer rig.deinit();
    const d: *cuda.Driver = @constCast(gpu.d);
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var full = try X.projection_load.load(gpu.gpa, d, &rig.store, "lm_head", wt.vocab, dims);
    defer full.free();
    var halves: [2]flash.projection.Projection = undefined;
    for (&halves, 0..) |*h, r| h.* = try tp.headProjection(gpu.gpa, d, &rig.store, "lm_head", 2, r);
    defer for (&halves) |*h| h.free();
    var p = rig.pass(&.{});
    var prng = std.Random.DefaultPrng.init(0x4ead);
    var all_equal = true;
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    for ([_]usize{ 1, 5 }) |m| {
        const xb = try a.alloc(u16, m * dims);
        for (xb) |*v| {
            const f: f32 = prng.random().floatNorm(f32);
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xb));
        defer x.free();
        var yf = try cuda.DeviceBuffer.alloc(gpu.d, m * wt.vocab * 2);
        defer yf.free();
        var yh = try cuda.DeviceBuffer.alloc(gpu.d, m * wt.vocab * 2);
        defer yh.free();
        _ = try p.lane(x.ptr, p.s.dummy, full, yf.ptr, p.s.dummy, m, 1);
        try e0.record(p.stream);
        _ = try p.lane(x.ptr, p.s.dummy, full, yf.ptr, p.s.dummy, m, 1);
        try e1.record(p.stream);
        try e1.synchronize();
        const t_full = try e0.elapsedMs(e1);
        var t_half: f32 = 0;
        for (halves, 0..) |h, r| {
            // the gathered layout: rank r's half at columns [r * N/2, (r + 1) * N/2) of every row
            const tmp = try a.alloc(u16, m * h.n);
            var yr = try cuda.DeviceBuffer.alloc(gpu.d, m * h.n * 2);
            defer yr.free();
            _ = try p.lane(x.ptr, p.s.dummy, h, yr.ptr, p.s.dummy, m, 1);
            try e0.record(p.stream);
            _ = try p.lane(x.ptr, p.s.dummy, h, yr.ptr, p.s.dummy, m, 1);
            try e1.record(p.stream);
            try e1.synchronize();
            t_half = @max(t_half, try e0.elapsedMs(e1));
            try yr.download(0, std.mem.sliceAsBytes(tmp));
            for (0..m) |row| {
                const dst = (row * wt.vocab + r * h.n) * 2;
                try yh.upload(dst, std.mem.sliceAsBytes(tmp[row * h.n ..][0..h.n]));
            }
        }
        const got = try a.alloc(u16, m * wt.vocab);
        try yh.download(0, std.mem.sliceAsBytes(got));
        const want = try a.alloc(u16, m * wt.vocab);
        try yf.download(0, std.mem.sliceAsBytes(want));
        var equal: usize = 0;
        var worst: f64 = 0;
        for (got, want) |g, w| {
            equal += @intFromBool(g == w);
            worst = @max(worst, @abs(T.bfValue(g) - T.bfValue(w)));
        }
        all_equal = all_equal and equal == got.len;
        std.debug.print("RESULT head split rows {d}: halves (124,160 columns a rank) joined in vocabulary order equal the whole head on {d}/{d} bf16 logits (max diff {e:.3}); whole {d:.3} ms, one half {d:.3} ms\n", .{ m, equal, got.len, worst, t_full, t_half });
    }
    try check.expect(all_equal, "head: the halves' logits differ from the whole head's", .{});
    check.pass("head: the EXL3 lm_head split at 128-column blocks (2 x 124,160) gives the whole head's bf16 logits bit for bit", .{});
}

// -- memory ---------------------------------------------------------------------------------------------------------

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

fn memoryMode(gpu: Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    const io = gpu.io;
    const o = try X.memory.options();
    const head = try X.memory.headBytes(a, io, dir, o);
    const one = try tp.deviceBytes(a, io, dir, o, try tp.Rank.init(1, 0));
    const ranks = [2]u64{ try tp.deviceBytes(a, io, dir, o, try tp.Rank.init(2, 0)), try tp.deviceBytes(a, io, dir, o, try tp.Rank.init(2, 1)) };
    const all_dev = try tp.deviceBytes(a, io, dir, .{ .routed_on_host = false }, try tp.Rank.init(2, 0));
    std.debug.print("RESULT memory (routed trellis {s}): tp=1 main {d:.3} GiB; tp=2 rank 0 {d:.3} GiB + MTP head {d:.3} GiB, rank 1 {d:.3} GiB; (rank 0 with its 256 routed experts resident: {d:.3} GiB)\n", .{ if (o.routed_on_host) "on the host (offload)" else "resident", gib(one), gib(ranks[0]), gib(head), gib(ranks[1]), gib(all_dev) });
    // the lm_head's split would save half its bytes on each rank
    var store = try wt.Store.open(a, io, dir);
    defer store.deinit();
    const lm = try X.load.mat(&store, "lm_head");
    const head_bytes: u64 = lm.trellis.len + lm.suh.len + lm.svh.len;
    std.debug.print("RESULT memory lm_head {d:.3} GiB a rank replicated; split: {d:.3} GiB a rank\n", .{ gib(head_bytes), gib(head_bytes / 2) });
    const prompt = flash.prompt;
    const window: usize = 200_000;
    const seq = prompt.seqBytes(window, .full);
    const kv = wt.attn_n * window * 2 * flash.Heads.full.kvRow();
    const kv8 = kv / 2 + wt.attn_n * window * 2 * wt.kv_heads * 4; // int8 rows plus an fp32 scale a row and head
    const draft = flash.mtp.Draft.bytes(window + flash.mtp.max_rows, 1, flash.prompt.batch.default_chunk, .full);
    const scratch = flash.prompt.batch.Scratch.bytes(flash.prompt.batch.default_chunk, .{ .format = .exl3 });
    std.debug.print("RESULT memory a sequence at {d} tokens: {d:.3} GiB (bf16 K/V {d:.3} GiB of it; int8 K/V would be {d:.3}); MTP draft cache {d:.3} GiB; pass scratch at chunk {d}: {d:.3} GiB\n", .{ window, gib(seq), gib(kv), gib(kv8), gib(draft), flash.prompt.batch.default_chunk, gib(scratch) });
    check.pass("memory: per-rank placement read from the pack's headers", .{});
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 4) return error.MissingArgument;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const port = try std.fmt.parseInt(u16, rest[1], 10);
    const dir = rest[2];
    const mode = rest[3];
    if (std.mem.eql(u8, mode, "head")) return headMode(gpu, dir);
    if (std.mem.eql(u8, mode, "memory")) return memoryMode(gpu, dir);
    var rk = try Ranks.open(gpu, rank, port);
    defer rk.close();
    if (std.mem.eql(u8, mode, "layer")) {
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(gpu.gpa);
        var it = std.mem.splitScalar(u8, if (rest.len > 4) rest[4] else "0,3,17", ',');
        while (it.next()) |tok| if (tok.len > 0) try list.append(gpu.gpa, try std.fmt.parseInt(usize, tok, 10));
        return layerMode(gpu, &rk, dir, list.items);
    }
    return error.UnknownMode;
}
