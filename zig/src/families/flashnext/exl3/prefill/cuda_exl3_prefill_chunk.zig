//! Deep-prefill chunks: the sub-block views, writeback, the window stages and one chunk's layers and head.

const std = @import("std");
const cuda = @import("cuda");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const ring_ = @import("cuda_exl3_prefill_ring.zig");
const history_ = @import("cuda_exl3_prefill_history.zig");
const window_ = @import("exl3_prefill_window.zig");
const state = @import("cuda_exl3_prefill_state.zig");
const experts = @import("cuda_exl3_prefill_moe.zig");
const ringmoe = @import("cuda_exl3_prefill_ringmoe.zig");
const tiles = @import("cuda_exl3_prefill_tiles.zig");

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const eps = wt.eps;
const vocab = wt.vocab;
const out_sk = wt.out_sk;
const ple_layer = wt.ple_layer;
const ple_tail_n = wt.ple_tail_n;
const routed_n_ = wt.experts_n;
const att_rows = batch.att_rows;
const pss_n = batch.pss_n;

const Deep = state.Deep;
const Options = state.Options;
const max_sub = state.max_sub;
const now = state.now;
const profile = state.profile;
const stats = &state.stats;
const upload = state.upload;
const moe = experts.moe;
const rankSum = experts.rankSum;
const keepFree = ringmoe.keepFree;
const ringMoe = ringmoe.ringMoe;

/// A sub-block's view of the pass's scratch: the per-row buffers a layer keeps across its sub-blocks point into the
pub fn view(base: *const batch.Scratch, dp: *const Deep, r0: usize) batch.Scratch {
    var v = base.*;
    const l = dp.lay;
    v.h = dp.dptr(l.h) + r0 * wide * 2;
    v.pss = dp.dptr(l.pss) + r0 * pss_n * 4;
    v.inj = dp.dptr(l.inj) + r0 * streams * 2;
    v.mixed = dp.dptr(l.mixed) + r0 * dims * 2;
    v.xsm = dp.dptr(l.xsm) + r0 * (dims / 32) * 4;
    v.y = dp.dptr(l.y) + r0 * slots * dims * 4;
    v.wts = dp.dptr(l.wts) + r0 * slots * 4;
    v.picks = dp.dptr(l.picks) + r0 * slots * 4;
    if (dp.world > 1) v.moe_f32 = dp.dptr(l.sum) + r0 * dims * 4;
    v.ple_emb = dp.dptr(l.ple_emb) + r0 * dims * 2;
    v.ple_xs = dp.dptr(l.ple_xs) + r0 * (dims / 32) * 4;
    v.ids = dp.dptr(l.ids) + r0 * 4;
    v.pos = dp.dptr(l.pos) + (r0 / att_rows) * 16;
    return v;
}

pub fn selectWriteback(dp: *Deep, reduced_branch: bool) !void {
    if (reduced_branch and dp.unit_weights == null) {
        const weights: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dp.hbytes(dp.hl.ids, dp.rows * 4)));
        @memset(weights, 1);
        dp.unit_weights = try cuda.DeviceBuffer.fromHost(dp.dev.d, std.mem.sliceAsBytes(weights));
    }
    if (dp.reduced_branch != reduced_branch) @import("../graph/cuda_exl3_graph.zig").clearPrefill();
    dp.reduced_branch = reduced_branch;
}

pub fn pendingWriteback(p: anytype, dp: *const Deep, layer: ?usize, r0: usize) void {
    p.pending = layer;
    p.pending_f32 = layer != null and dp.reduced_branch;
    p.pending_f32_weights = if (p.pending_f32) dp.unit_weights.?.ptr + r0 * 4 else null;
}

pub fn compactLast(p: anytype, reduced_branch: bool, row: usize) !void {
    if (row == 0) return;
    const s = p.s;
    try p.copy(s.h, s.h + row * wide * 2, wide * 2);
    try p.copy(s.pss, s.pss + row * pss_n * 4, pss_n * 4);
    try p.copy(s.inj, s.inj + row * streams * 2, streams * 2);
    if (reduced_branch) {
        try p.copy(s.moe_f32, s.moe_f32 + row * dims * 4, dims * 4);
    } else {
        try p.copy(s.y, s.y + row * slots * dims * 4, slots * dims * 4);
        try p.copy(s.wts, s.wts + row * slots * 4, slots * 4);
    }
}

const Gather = struct {
    ngram: *const @import("../../cuda_ple.zig").Ngram,
    history: []i64,
    ids: []const u32,
    emb: []u16,
    xs: []f32,
    failed: ?anyerror = null,
    ns: i96 = 0,
    io: std.Io,

    fn main(g: *Gather) void {
        const t0 = now(g.io);
        defer g.ns = now(g.io) - t0;
        g.ngram.rows(g.history, g.ids, g.emb, g.xs) catch |err| {
            g.failed = err;
        };
    }
};

fn windowStage(dp: *Deep, p: anytype, ids: []const u32, pos0: usize) !void {
    const n = ids.len;
    try p.stream.synchronize();
    const id32 = std.mem.bytesAsSlice(i32, dp.hbytes(dp.hl.ids, n * 4));
    for (id32, ids) |*destination, id| destination.* = @intCast(id);
    const blocks = (n + att_rows - 1) / att_rows;
    const positions = std.mem.bytesAsSlice(i32, dp.hbytes(dp.hl.pos, blocks * 16));
    @memset(positions, 0);
    for (0..blocks) |block| positions[4 * block] = @intCast(pos0 + block * att_rows);
    try upload(p.stream, dp.dptr(dp.lay.ids), dp.hbytes(dp.hl.ids, n * 4));
    try upload(p.stream, dp.dptr(dp.lay.pos), dp.hbytes(dp.hl.pos, blocks * 16));
    if (p.m.table_dense) {
        try p.tri.embedDense(dp.dptr(dp.lay.ids), p.m.table_w, dp.dptr(dp.lay.h), dims, streams, n);
    } else {
        try p.tri.embed(dp.dptr(dp.lay.ids), p.m.table_w, p.m.table_s, p.m.table_b, dp.dptr(dp.lay.h), dims, streams, n);
    }
}

pub fn windowInputs(dp: *Deep, p: anytype, ids: []const u32, pos0: usize, layer: usize) !void {
    const blocks = (ids.len + att_rows - 1) / att_rows;
    const positions_offset = dp.hl.pos + (layer + 1) * dp.hl.pos_stride;
    const positions: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(positions_offset, blocks * 16)));
    @memset(positions, 0);
    for (0..blocks) |block| positions[4 * block] = @intCast(pos0 + block * att_rows);
    try upload(p.stream, dp.dptr(dp.lay.pos), dp.hbytes(positions_offset, blocks * 16));
    if (layer == ple_layer) {
        const embedding: []u16 = @alignCast(std.mem.bytesAsSlice(u16, dp.hbytes(dp.hl.emb, ids.len * dims * 2)));
        const sums: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dp.hbytes(dp.hl.xs, ids.len * (dims / 32) * 4)));
        try p.m.ngram.rows(p.seq.history[0..], ids, embedding, sums);
        try upload(p.stream, dp.dptr(dp.lay.ple_emb), std.mem.sliceAsBytes(embedding));
        try upload(p.stream, dp.dptr(dp.lay.ple_xs), std.mem.sliceAsBytes(sums));
    }
}

fn windowDense(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, layer: usize, n: usize, pos0: usize, sub: usize, waits: []const cuda.Event) !void {
    const graph = @import("../graph/cuda_exl3_graph.zig");
    const lw = &p.m.layers[layer];
    const n_sub = (n + sub - 1) / sub;
    var views: [max_sub]batch.Scratch = undefined;
    for (0..n_sub) |block| views[block] = view(base, dp, block * sub);
    defer p.s = base;
    p.s = base;
    p.rows = n;
    p.pos0 = pos0;
    const capture = try graph.prefillBegin(p, lw.linear, layer, n, sub, pos0, dp.dev.ptr, base, tiles.tiles_on, waits, dp.dptr(dp.lay.pos));
    errdefer graph.prefillAbort(p, capture);
    if (capture != .replayed) for (0..n_sub) |block| {
        const at = block * sub;
        const count = @min(sub, n - at);
        p.s = &views[block];
        p.rows = count;
        p.pos0 = pos0 + at;
        pendingWriteback(p, dp, if (layer == 0) null else layer - 1, at);
        if (block < waits.len) switch (capture) {
            .eager => try p.stream.wait(waits[block]),
            .capturing => try p.stream.waitExternal(waits[block]),
            .replayed => unreachable,
        };
        try layerBlock(Tri, p, lw, layer, count);
    };
    try graph.prefillEnd(p, capture);
    dp.red_n = 0;
}

fn windowFinish(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, n: usize, sub: usize, logits: batch.Logits, streams_out: u64, logits_host: ?[]u8) !void {
    const n_sub = (n + sub - 1) / sub;
    var views: [max_sub]batch.Scratch = undefined;
    for (0..n_sub) |block| views[block] = view(base, dp, block * sub);
    defer p.s = base;
    stats.chunks += 1;
    stats.rows += n;
    const last_block = n_sub - 1;
    const last_rows = n - last_block * sub;
    if (streams_out != 0) {
        for (0..n_sub) |block| {
            const count = @min(sub, n - block * sub);
            p.s = &views[block];
            p.rows = count;
            pendingWriteback(p, dp, wt.layers_n - 1, block * sub);
            try p.writeback(0, count);
            try p.copy(streams_out + block * sub * wide * 2, views[block].h, count * wide * 2);
        }
    }
    if (logits == .none) return;
    p.s = &views[last_block];
    p.rows = last_rows;
    const scratch = p.s;
    const row = last_rows - 1;
    if (streams_out != 0) {
        if (row > 0) {
            try p.copy(scratch.h, scratch.h + row * wide * 2, wide * 2);
            try p.copy(scratch.pss, scratch.pss + row * pss_n * 4, pss_n * 4);
        }
    } else {
        try compactLast(p, dp.reduced_branch, row);
        pendingWriteback(p, dp, wt.layers_n - 1, last_block * sub);
        try p.writeback(0, 1);
    }
    try p.readout(p.m.mixer, 0, 1);
    if (p.m.head.n == vocab) {
        _ = try p.lane(scratch.mixed, scratch.xsm, p.m.head, scratch.logits, scratch.dummy, 1, 1);
    } else {
        const collective = p.m.collective orelse return error.TensorParallelUnavailable;
        _ = try p.lane(scratch.mixed, scratch.xsm, p.m.head, scratch.head_local, scratch.dummy, 1, 1);
        try collective.gather(p.stream, scratch.head_local, scratch.logits, vocab / 2);
    }
    if (logits_host) |destination| {
        if (p.m.collective) |collective| try collective.finish(p.stream);
        try p.stream.synchronize();
        try p.stream.d.check(p.stream.d.api.cuMemcpyDtoH_v2(destination.ptr, scratch.logits, vocab * 2), "cuMemcpyDtoH");
    }
}

pub fn twoChunk(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, ids: []const u32, pos0: usize, sub: usize, hooks: batch.Hooks, output_at: usize, last: bool, logits_host: []u8, o: Options, ring: *ring_.Ring, plan_: window_.Plan) !void {
    const buffers = [_]*Deep{ dp, dp.second.? };
    for (plan_.segments[0..plan_.count]) |segment| {
        if ((segment.rows + sub - 1) / sub > max_sub) return error.UnexpectedTensor;
    }
    try p.stream.synchronize();
    try ring.newChunk();
    history_.prepare(&dp.history, &dp.history_shape, .{ .sub = sub, .lo = ring.own_lo, .hi = ring.own_hi });
    for (0..@min(o.ring_ahead + 1, wt.layers_n)) |layer| _ = try ring.prefetch(layer, @min(routed_n_, dp.used[layer]), keepFree(ring, o), .initial_prefetch);
    for (plan_.segments[0..plan_.count], 0..) |segment, index| {
        try windowStage(buffers[index], p, ids[segment.offset..][0..segment.rows], pos0 + segment.offset);
    }
    var schedule = plan_.iterator(p.m.layers.len);
    while (schedule.next()) |step| {
        const active = buffers[step.chunk];
        const segment = plan_.segments[step.chunk];
        try windowInputs(active, p, ids[segment.offset..][0..segment.rows], pos0 + segment.offset, step.layer);
        try windowDense(Tri, active, p, base, step.layer, segment.rows, pos0 + segment.offset, sub, active.red_ev[0..active.red_n]);
        if (active != dp) {
            active.used = dp.used;
            active.history = dp.history;
        }
        const layer = &p.m.layers[step.layer];
        try ringMoe(Tri, active, ring, p, layer, step.layer, segment.rows, o);
        if (active != dp) {
            dp.used = active.used;
            dp.history = active.history;
        }
        try rankSum(active, p, layer.exl3.?, segment.rows, sub);
        if (step.layer + 1 == p.m.layers.len) {
            for (active.red_ev[0..active.red_n]) |event| try p.stream.wait(event);
            active.red_n = 0;
        }
    }
    for (plan_.segments[0..plan_.count], 0..) |segment, index| {
        const final = last and index + 1 == plan_.count;
        const output = if (hooks.streams == 0) 0 else hooks.streams + (output_at + segment.offset) * wide * 2;
        try windowFinish(Tri, buffers[index], p, base, segment.rows, sub, if (final) hooks.logits else .none, output, if (final and hooks.logits != .none) logits_host else null);
    }
}

pub fn chunk(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, ids: []const u32, pos0: usize, sub: usize, logits: batch.Logits, streams_out: u64, logits_host: ?[]u8, o: Options, rg: ?*ring_.Ring) !void {
    const n = ids.len;
    const st = p.stream;
    const hl = dp.hl;
    // the previous chunk's uploads read the pinned rows: they landed before its head (a synchronize), or below
    try st.synchronize();
    if (rg) |r| {
        try r.newChunk();
        history_.prepare(&dp.history, &dp.history_shape, .{ .sub = sub, .lo = r.own_lo, .hi = r.own_hi });
        // the first layers' experts start streaming now, under the embedding and layer 0's blocks
        for (0..@min(o.ring_ahead + 1, wt.layers_n)) |l| _ = try r.prefetch(l, @min(routed_n_, dp.used[l]), keepFree(r, o), .initial_prefetch);
    }
    const id32 = std.mem.bytesAsSlice(i32, dp.hbytes(hl.ids, n * 4));
    for (id32, ids) |*x, v| x.* = @intCast(v);
    const blocks = (n + att_rows - 1) / att_rows;
    const pos = std.mem.bytesAsSlice(i32, dp.hbytes(hl.pos, blocks * 16));
    @memset(pos, 0);
    for (0..blocks) |b| pos[4 * b] = @intCast(pos0 + b * att_rows);
    try upload(st, dp.dptr(dp.lay.ids), dp.hbytes(hl.ids, n * 4));
    try upload(st, dp.dptr(dp.lay.pos), dp.hbytes(hl.pos, blocks * 16));
    const emb: []u16 = @alignCast(std.mem.bytesAsSlice(u16, dp.hbytes(hl.emb, n * dims * 2)));
    const xs: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dp.hbytes(hl.xs, n * (dims / 32) * 4)));
    var gather: Gather = .{ .ngram = p.m.ngram, .history = p.seq.history[0..], .ids = ids, .emb = emb, .xs = xs, .io = p.io };
    var thread: ?std.Thread = try std.Thread.spawn(.{}, Gather.main, .{&gather});
    defer if (thread) |th| th.join();
    const t0 = now(p.io);
    defer if (profile()) {
        st.synchronize() catch {};
        std.debug.print("exl3 prefill: {d} rows at {d} in {d:.1} ms (n-gram gather {d:.1} ms on the host)\n", .{ n, pos0, @as(f64, @floatFromInt(now(p.io) - t0)) / 1e6, @as(f64, @floatFromInt(gather.ns)) / 1e6 });
    };
    p.rows = n;
    p.pos0 = pos0;
    pendingWriteback(p, dp, null, 0);
    if (p.m.table_dense) {
        try p.tri.embedDense(dp.dptr(dp.lay.ids), p.m.table_w, dp.dptr(dp.lay.h), dims, streams, n);
    } else {
        try p.tri.embed(dp.dptr(dp.lay.ids), p.m.table_w, p.m.table_s, p.m.table_b, dp.dptr(dp.lay.h), dims, streams, n);
    }
    const n_sub = (n + sub - 1) / sub;
    var views: [max_sub]batch.Scratch = undefined;
    if (n_sub > views.len) return error.UnexpectedTensor;
    for (0..n_sub) |b| views[b] = view(base, dp, b * sub);
    defer p.s = base;
    for (p.m.layers, 0..) |*lw, layer| {
        if (layer == ple_layer) {
            if (thread) |th| th.join();
            thread = null;
            if (gather.failed) |err| return err;
            try upload(st, dp.dptr(dp.lay.ple_emb), std.mem.sliceAsBytes(emb));
            try upload(st, dp.dptr(dp.lay.ple_xs), std.mem.sliceAsBytes(xs));
        }
        const pg = try @import("../graph/cuda_exl3_graph.zig").prefillBegin(p, lw.linear, layer, n, sub, pos0, dp.dev.ptr, base, tiles.tiles_on, dp.red_ev[0..dp.red_n], dp.dptr(dp.lay.pos));
        errdefer @import("../graph/cuda_exl3_graph.zig").prefillAbort(p, pg);
        if (pg != .replayed) for (0..n_sub) |b| {
            const m = @min(sub, n - b * sub);
            p.s = &views[b];
            p.rows = m;
            p.pos0 = pos0 + b * sub;
            pendingWriteback(p, dp, if (layer > 0) layer - 1 else null, b * sub);
            if (b < dp.red_n) switch (pg) {
                .eager => try st.wait(dp.red_ev[b]),
                .capturing => try st.waitExternal(dp.red_ev[b]),
                .replayed => unreachable,
            };
            try layerBlock(Tri, p, lw, layer, m);
        };
        try @import("../graph/cuda_exl3_graph.zig").prefillEnd(p, pg);
        dp.red_n = 0;
        if (rg) |r| try ringMoe(Tri, dp, r, p, lw, layer, n, o) else try moe(Tri, dp, p, lw, layer, n, o);
        try rankSum(dp, p, lw.exl3.?, n, sub);
    }
    for (dp.red_ev[0..dp.red_n]) |ev| try st.wait(ev);
    dp.red_n = 0;
    stats.chunks += 1;
    stats.rows += n;
    // the last layer's experts written back where the caller keeps the streams, then the head on the last row
    const last_b = n_sub - 1;
    const m_last = n - last_b * sub;
    if (streams_out != 0) {
        for (0..n_sub) |b| {
            const m = @min(sub, n - b * sub);
            p.s = &views[b];
            p.rows = m;
            pendingWriteback(p, dp, wt.layers_n - 1, b * sub);
            try p.writeback(0, m);
            try p.copy(streams_out + b * sub * wide * 2, views[b].h, m * wide * 2);
        }
    }
    if (logits == .none) return;
    p.s = &views[last_b];
    p.rows = m_last;
    const s = p.s;
    const r = m_last - 1;
    if (streams_out != 0) {
        if (r > 0) {
            try p.copy(s.h, s.h + r * wide * 2, wide * 2);
            try p.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
        }
    } else {
        try compactLast(p, dp.reduced_branch, r);
        pendingWriteback(p, dp, wt.layers_n - 1, last_b * sub);
        try p.writeback(0, 1);
    }
    try p.readout(p.m.mixer, 0, 1);
    if (p.m.head.n == vocab) {
        _ = try p.lane(s.mixed, s.xsm, p.m.head, s.logits, s.dummy, 1, 1);
    } else {
        // the head split by vocabulary (expert parallel): each rank's half, gathered in vocabulary order (projectHead)
        const c = p.m.collective orelse return error.TensorParallelUnavailable;
        _ = try p.lane(s.mixed, s.xsm, p.m.head, s.head_local, s.dummy, 1, 1);
        try c.gather(st, s.head_local, s.logits, vocab / 2);
    }
    if (logits_host) |lh| {
        if (p.m.collective) |c| try c.finish(st);
        try st.synchronize();
        try st.d.check(st.d.api.cuMemcpyDtoH_v2(lh.ptr, s.logits, vocab * 2), "cuMemcpyDtoH");
    }
}

/// One sub-block of one layer, as Pass.chunk runs it, up to the router's picks (the experts come after every
fn layerBlock(comptime Tri: type, p: *batch.Pass(Tri), lw: *const @import("../../cuda_prompt_load.zig").Layer, layer: usize, m: usize) !void {
    const s = p.s;
    if (layer == ple_layer) {
        if (p.pending != null) try p.writeback(0, m);
        // Pass.pleBlock for a prompt's rows (no lanes)
        _ = try p.lane(s.ple_emb, s.ple_xs, lw.key, s.ple_keys, s.dummy, m, 1);
        _ = try p.lane(s.ple_emb, s.ple_xs, lw.val, s.dummy, s.ple_part, m, 4);
        try p.sumSlices(s.ple_part, s.ple_vals, true, m * dims, 4);
        try p.tri.pleGate(s.ple_keys, s.ple_vals, s.h, lw.nk.ptr, lw.nq.ptr, s.ple_gated, s.ple_pss, eps, m, dims, streams);
        try p.tri.pleConv(s.ple_gated, s.ple_pss, lw.nc.ptr, p.seq.tail.ptr, lw.ple_cw.ptr, s.h, s.h, s.ple_nrow, eps, m, dims, streams, 4, 3);
        try p.shiftWindow(p.seq.tail.ptr, s.ple_nrow, wide, m, ple_tail_n, wide);
    }
    try p.writeback(0, m);
    try p.readout(lw.attn_hc, 0, m);
    if (lw.linear) {
        try p.gdnBlock(lw, layer, wt.linearIndex(layer), .{});
    } else {
        try p.attnBlock(lw, wt.attnIndex(layer));
    }
    try p.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, out_sk);
    try p.readout(lw.mlp_hc, 0, m);
    try p.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, m);
    try p.tri.topkMany(s.router_l, s.picks, s.wts, m);
}
