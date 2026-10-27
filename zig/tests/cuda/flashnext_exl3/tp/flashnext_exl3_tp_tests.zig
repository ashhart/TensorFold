//! flashnext-exl3-tp: EXL3 expert parallel (cuda_exl3_tp.zig) on two ranks, one process a GPU, through the Flash Next
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

/// What one call of a layer leaves: the output streams (bf16), and the MoE's picks, weights and per-slot y.
const Out = struct { h: []u16, picks: []i32, wts: []f32, y: []f32 };

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
    o.y = try grab(f32, a, gpu, p, s.y, n * slots * dims);
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
    // slot 0 of every row: the ranks' sum after placement
    const n = o.picks.len / slots;
    for (0..n) |r| h.update(std.mem.sliceAsBytes(o.y[r * slots * dims ..][0..dims]));
    return h.final();
}

fn layerMode(gpu: Gpu, rk: *Ranks, dir: []const u8, which: []const usize) !void {
    const rank = rk.rank;
    var rig: T.Rig = undefined;
    try rig.init(gpu, dir, 8, 64);
    defer rig.deinit();
    var raw = try T.RawPack.open(gpu, dir);
    defer raw.close(gpu);
    const r0 = try tp.Rank.init(2, 0);
    const r1 = try tp.Rank.init(2, 1);
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
            try check.expect(same_ranks, "layer {d}: the ranks' outputs or sums differ in bits", .{l});
            var picks_equal = true;
            var sum_exact: usize = 0;
            var sum_total: usize = 0;
            var branch_abs: f64 = 0;
            var branch_peak: f64 = 0;
            var h_equal: usize = 0;
            var h_total: usize = 0;
            var h_abs: f64 = 0;
            var h_peak: f64 = 0;
            for (ref, got, calls) |want, have, c| {
                const n = c[1];
                picks_equal = picks_equal and std.mem.eql(i32, want.picks, have.picks);
                for (0..n) |r| for (0..dims) |d| {
                    // tp=1: the write-back's chain over the 11 slots; tp=2: each rank's chain over its own, then a + b
                    var one: f32 = 0;
                    var acc: [2]f32 = .{ 0, 0 };
                    for (0..slots) |k| {
                        const at = r * slots + k;
                        const w = want.wts[at];
                        const v = want.y[at * dims + d];
                        one = @mulAdd(f32, v, w, one);
                        const id: usize = @intCast(want.picks[at]);
                        if (r0.owns(id)) acc[0] = @mulAdd(f32, w, v, acc[0]);
                        if (r1.owns(id)) acc[1] = @mulAdd(f32, w, v, acc[1]);
                    }
                    const predicted = acc[0] + acc[1];
                    const placed = have.y[r * slots * dims + d];
                    sum_exact += @intFromBool(placed == predicted);
                    sum_total += 1;
                    branch_abs = @max(branch_abs, @abs(@as(f64, placed) - one));
                    branch_peak = @max(branch_peak, @abs(@as(f64, one)));
                };
                for (want.h, have.h) |u, v| {
                    h_equal += @intFromBool(u == v);
                    h_total += 1;
                    h_abs = @max(h_abs, @abs(T.bfValue(u) - T.bfValue(v)));
                    h_peak = @max(h_peak, @abs(T.bfValue(u)));
                }
            }
            std.debug.print("  tp=2 ({d:.0} ms for the 5 calls): picks {s}; summed MoE branch equals (rank 0's chain + rank 1's chain) of the tp=1 per-slot outputs on {d}/{d} values; vs tp=1's one chain max_abs {e:.3} (rel {e:.3}, peak {d:.3}); layer output bf16 equal {d}/{d}, max_abs {e:.3} (rel {e:.3})\n", .{ tp_ms, if (picks_equal) "equal" else "DIFFER", sum_exact, sum_total, branch_abs, branch_abs / branch_peak, branch_peak, h_equal, h_total, h_abs, h_abs / h_peak });
            var ok = true;
            check.expect(picks_equal, "layer {d}: tp=2 router picks differ from tp=1", .{l}) catch {
                ok = false;
            };
            check.expect(sum_exact == sum_total, "layer {d}: tp=2's MoE sum is not the fp32 regrouping of tp=1's slots ({d}/{d})", .{ l, sum_exact, sum_total }) catch {
                ok = false;
            };
            // an fp32 regrouping of 11 terms: a few ulps of the branch's peak; bf16 outputs.
            check.expect(branch_abs <= 1e-5 * branch_peak, "layer {d}: tp=2 MoE branch off tp=1 by {e:.3} of peak", .{ l, branch_abs / branch_peak }) catch {
                ok = false;
            };
            check.expect(h_abs <= 8e-3 * h_peak, "layer {d}: tp=2 layer output off tp=1 by {e:.3} of peak", .{ l, h_abs / h_peak }) catch {
                ok = false;
            };
            if (ok) check.pass("layer {d} tp=2 vs tp=1: picks equal, MoE sum = fp32 regrouping of tp=1's slots on every value (bit-exact), output bf16 equal {d}/{d}, max diff {e:.3} of peak; ranks bit-identical", .{ l, h_equal, h_total, h_abs / h_peak }) else failed += 1;
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

// -- the MoE sum's cost ---------------------------------------------------------------------------------------------

fn fill(a: std.mem.Allocator, rows: usize, seed: u64) ![]f32 {
    const out = try a.alloc(f32, rows * dims);
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
    return out;
}

fn waitComm(c: *Collective, s: cuda.Stream) !void {
    try ready(c);
    try s.synchronize();
}

/// A nonblocking communicator takes its next call only once the last one's enqueue finished (state 0, not 7).
fn ready(c: *Collective) !void {
    while (true) {
        var state: cuda.nccl.Result = 0;
        try c.lib.check(c.api.ncclCommGetAsyncError(c.comm, &state), "ncclCommGetAsyncError");
        if (state == 0) break;
        if (state != 7) {
            std.debug.print("waitComm: NCCL async state {d}\n", .{state});
            return c.lib.check(state, "NCCL async");
        }
    }
}

/// Rank order: the peer's partial into `scratch` (send/recv), then out = rank 0's term + rank 1's (tf_rank_sum).
fn rankOrder(c: *Collective, k: *const cuda.rank_scatter.RankScatter, s: cuda.Stream, rank: usize, partial: u64, scratch: u64, out: u64, n: usize) !void {
    const peer: c_int = @intCast(1 - rank);
    const api = c.lib.api;
    const ok = struct {
        fn f(r: cuda.nccl.Result) !void {
            if (r != 0 and r != 7) {
                std.debug.print("rank-order: NCCL result {d}\n", .{r});
                return error.NcclFailed;
            }
        }
    }.f;
    try ok(api.ncclGroupStart());
    try ok(api.ncclSend(partial, n, .f32, peer, c.comm, s.handle));
    try ok(api.ncclRecv(scratch + (1 - rank) * n * 4, n, .f32, peer, c.comm, s.handle));
    try ok(api.ncclGroupEnd());
    try ready(c);
    try k.rankSum(s, scratch, partial, 2, rank, n, out);
}

fn timeIt(s: cuda.Stream, e0: *cuda.Event, e1: *cuda.Event, reps: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
    for (0..3) |_| try f(ctx);
    try s.synchronize();
    var best: f64 = std.math.inf(f64);
    for (0..5) |_| {
        try e0.record(s);
        for (0..reps) |_| try f(ctx);
        try e1.record(s);
        try e1.synchronize();
        best = @min(best, @as(f64, try e0.elapsedMs(e1.*)) / @as(f64, @floatFromInt(reps)));
    }
    return best;
}

fn sumMode(gpu: Gpu, rk: *Ranks) !void {
    const rank = rk.rank;
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    var rs = try cuda.rank_scatter.RankScatter.load(gpu.d);
    defer rs.unload();
    var ops = try X.load.Ops.create(gpu.gpa, gpu.d);
    defer ops.release();
    const t = try tp.scratchOf(ops);
    const c = &rk.collective;
    const sizes = [_]usize{ 1, 128, 512, 2048 };
    var all_equal = true;
    for (sizes) |rows| {
        const n = rows * dims;
        const mine = try fill(a, rows, 0x5eed + rank);
        const theirs = try fill(a, rows, 0x5eed + (1 - rank));
        var part = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mine));
        defer part.free();
        var buf = try cuda.DeviceBuffer.alloc(gpu.d, n * 4);
        defer buf.free();
        var scratch = try cuda.DeviceBuffer.alloc(gpu.d, 2 * n * 4);
        defer scratch.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, n * 4);
        defer out.free();
        // bits: the all-reduce, the rank-order sum and the host's rank0 + rank1 in fp32
        try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(buf.ptr, part.ptr, n * 4, s.handle), "cuMemcpyDtoDAsync");
        try c.sum(s, buf.ptr, n);
        try c.finish(s);
        try rankOrder(c, &rs, s, rank, part.ptr, scratch.ptr, out.ptr, n);
        try waitComm(c, s);
        const ar = try a.alloc(f32, n);
        try buf.download(0, std.mem.sliceAsBytes(ar));
        const ro = try a.alloc(f32, n);
        try out.download(0, std.mem.sliceAsBytes(ro));
        var eq_host: usize = 0;
        var eq_ro: usize = 0;
        for (0..n) |i| {
            const want = if (rank == 0) mine[i] + theirs[i] else theirs[i] + mine[i];
            eq_host += @intFromBool(@as(u32, @bitCast(ar[i])) == @as(u32, @bitCast(want)));
            eq_ro += @intFromBool(@as(u32, @bitCast(ar[i])) == @as(u32, @bitCast(ro[i])));
        }
        all_equal = all_equal and eq_host == n and eq_ro == n;
        // times: the pass's Collective.sum over [rows, 2560] fp32; rank order; upstream's per-slot [rows, 11, 2560]
        const reps: usize = if (rows <= 128) 100 else if (rows <= 512) 20 else 5;
        const Ctx = struct { c: *Collective, rs: *const cuda.rank_scatter.RankScatter, s: cuda.Stream, rank: usize, p: u64, sc: u64, o: u64, n: usize, t: *const tp.Scratch, picks: u64, wts: u64, y: u64, rows: usize };
        var per_slot = try cuda.DeviceBuffer.alloc(gpu.d, n * slots * 4);
        defer per_slot.free();
        try per_slot.fill8(0, s.handle);
        var picks = try cuda.DeviceBuffer.alloc(gpu.d, rows * slots * 4);
        defer picks.free();
        try picks.fill8(0, s.handle);
        var wts = try cuda.DeviceBuffer.alloc(gpu.d, rows * slots * 4 * 2);
        defer wts.free();
        try wts.fill8(0, s.handle);
        try s.synchronize();
        const ctx: Ctx = .{ .c = c, .rs = &rs, .s = s, .rank = rank, .p = part.ptr, .sc = scratch.ptr, .o = out.ptr, .n = n, .t = t, .picks = picks.ptr, .wts = wts.ptr, .y = per_slot.ptr, .rows = rows };
        const t_ar = try timeIt(s, &e0, &e1, reps, ctx, struct {
            fn f(x: Ctx) anyerror!void {
                try x.c.sum(x.s, x.o, x.n);
            }
        }.f);
        const t_raw = try timeIt(s, &e0, &e1, reps, ctx, struct {
            fn f(x: Ctx) anyerror!void {
                const r = x.c.lib.api.ncclAllReduce(x.o, x.o, x.n, .f32, .sum, x.c.comm, x.s.handle);
                if (r != 0 and r != 7) return error.NcclFailed;
                if (r == 7) try ready(x.c);
            }
        }.f);
        const t_ro = try timeIt(s, &e0, &e1, reps, ctx, struct {
            fn f(x: Ctx) anyerror!void {
                try rankOrder(x.c, x.rs, x.s, x.rank, x.p, x.sc, x.o, x.n);
            }
        }.f);
        try waitComm(c, s);
        const t_slot = try timeIt(s, &e0, &e1, @max(1, reps / 4), ctx, struct {
            fn f(x: Ctx) anyerror!void {
                try x.c.sum(x.s, x.y, x.n * slots);
            }
        }.f);
        const t_mask = try timeIt(s, &e0, &e1, reps, ctx, struct {
            fn f(x: Ctx) anyerror!void {
                try x.t.mask(x.s, x.picks, x.wts, x.picks, x.wts + x.rows * slots * 4, x.rows * slots, try tp.Rank.init(2, x.rank));
            }
        }.f);
        const t_place = try timeIt(s, &e0, &e1, reps, ctx, struct {
            fn f(x: Ctx) anyerror!void {
                try x.t.place(x.s, x.o, x.y, x.wts, x.rows);
            }
        }.f);
        if (rank == 0) std.debug.print("RESULT tp2 MoE sum rows {d:>4} ({d:>9} B): all-reduce {d:.4} ms (raw ncclAllReduce, no peer/async checks: {d:.4}) | rank-order send/recv+sum {d:.4} ms | upstream per-slot all-reduce ({d:>9} B) {d:.4} ms | mask {d:.4} ms | place {d:.4} ms | bits: all-reduce == host fp32 a+b {d}/{d}, == rank-order {d}/{d}\n", .{ rows, n * 4, t_ar, t_raw, t_ro, n * slots * 4, t_slot, t_mask, t_place, eq_host, n, eq_ro, n });
        _ = try rk.swap(rows);
    }
    try check.expect(all_equal, "rank {d}: an all-reduce sum differs from the fp32 a + b or the rank-order sum", .{rank});
    check.pass("rank {d}: two-rank fp32 all-reduce of the combined MoE branch is bit-identical to rank-order a + b at 1..2048 rows", .{rank});
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
    const o = X.memory.options();
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
    const seq = prompt.seqBytes(window);
    const kv = wt.attn_n * window * 2 * prompt.kv_row_bytes;
    const kv8 = kv / 2 + wt.attn_n * window * 2 * wt.kv_heads * 4; // int8 rows plus an fp32 scale a row and head
    const draft = flash.mtp.Draft.bytes(window + flash.mtp.max_rows, 1, flash.prompt.batch.default_chunk);
    const scratch = flash.prompt.batch.Scratch.bytes(flash.prompt.batch.default_chunk);
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
    if (std.mem.eql(u8, mode, "sum")) return sumMode(gpu, &rk);
    if (std.mem.eql(u8, mode, "layer")) {
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(gpu.gpa);
        var it = std.mem.splitScalar(u8, if (rest.len > 4) rest[4] else "0,3,17", ',');
        while (it.next()) |tok| if (tok.len > 0) try list.append(gpu.gpa, try std.fmt.parseInt(usize, tok, 10));
        return layerMode(gpu, &rk, dir, list.items);
    }
    return error.UnknownMode;
}
