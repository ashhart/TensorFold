//! Deep-prefill experts, pass by pass: the pass plan, members and items, prefetch and the rank sum.

const std = @import("std");
const cuda = @import("cuda");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const load = @import("../pack/cuda_exl3_load.zig");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const pk = @import("cuda_exl3_prefill_k.zig");
const tp_ = @import("../tp/cuda_exl3_tp.zig");
const state = @import("cuda_exl3_prefill_state.zig");
const leaves = @import("cuda_exl3_prefill_leaves.zig");

const dims = wt.dims;
const slots = wt.slots;
const table_n = wt.experts_n + 1;
const shared_id = wt.experts_n;

const Deep = state.Deep;
const Options = state.Options;
const Pass = state.Pass;
const now = state.now;
const profile = state.profile;
const stats = &state.stats;
const upload = state.upload;

/// Lines a pass may fill: half the layer's lines (the other half holds the previous pass, still computing).
fn passLines(pool: *const pool_.Pool, o: Options) usize {
    const lines = pool.opt.lines;
    return @max(1, if (o.pass_lines_half) lines / 2 else lines);
}

/// The layer's experts over all `n` rows of the deep chunk, expert-major.
pub fn moe(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), lw: *const @import("../../cuda_prompt_load.zig").Layer, layer: usize, n: usize, o: Options) !void {
    const x = lw.exl3.?;
    const st = p.stream;
    const pairs = n * slots;
    const hl = dp.hl;
    const t0 = now(p.io);
    // the picks of every row on the host: the layer's only wait
    const host_picks: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.picks, pairs * 4)));
    try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(host_picks.ptr, dp.dptr(dp.lay.picks), pairs * 4, st.handle), "cuMemcpyDtoHAsync");
    try st.synchronize();
    maskOwned(host_picks, x.tp);
    const t1 = now(p.io);
    try plan(dp, x, host_picks, o);
    // members and items of every pass in one upload (the pinned bytes stay put until the next layer's wait)
    const members: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.members, pairs * 4)));
    const items: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.items, 3 * pk.maxItems(pairs, table_n) * 4)));
    const counts = build(dp, host_picks, members, items);
    try upload(st, dp.dptr(dp.lay.members), std.mem.sliceAsBytes(members[0..counts.members]));
    try upload(st, dp.dptr(dp.lay.items), std.mem.sliceAsBytes(items[0 .. 3 * counts.items]));
    const t2 = now(p.io);
    stats.host_wait_ns += @intCast(t1 - t0);
    stats.plan_ns += @intCast(t2 - t1);
    stats.members += counts.members;
    const pool = x.pool;
    var fake_rows: usize = 0;
    for (dp.passes[0..dp.n_passes]) |ps| {
        var r: ?*const pool_.Round = null;
        if (pool) |pl| {
            const ids_ = dp.order[ps.id_lo..ps.id_hi];
            fake_rows = fakePicks(dp, ids_);
            if (fake_rows > 0) {
                r = try pl.planUnobserved(dp.fake[0 .. fake_rows * slots], fake_rows);
                if (r.?.n_passes != 1) return error.Exl3PrefillPass;
                try pl.stage(r.?, 0, st);
                stats.copies += r.?.misses_();
            }
        }
        const l = if (pool) |pl| pl.tables() else x.layer;
        if (o.slots_kernel) {
            try slotsPass(dp, x, st, l, host_picks, ps, n);
            if (r) |rr| try pool.?.release(rr, 0, st);
            stats.passes += 1;
            continue;
        }
        const dv: pk.Device = .{ .members = dp.dptr(dp.lay.members) + ps.m_lo * 4, .items = dp.dptr(dp.lay.items) + ps.i_lo * 12, .scratch = dp.dptr(dp.lay.kscratch) };
        try pk.run(&dp.kern, st, l, dp.dptr(dp.lay.mixed), dims, dp.dptr(dp.lay.picks), slots, ps.m_n, ps.i_n, dv, dp.dptr(dp.lay.y));
        if (r) |rr| try pool.?.release(rr, 0, st);
        stats.passes += 1;
    }
    // the next layer's first pass, staged on the copy engine while this layer's experts and the next blocks run
    if (o.prefetch and layer + 1 < p.m.layers.len) {
        if (p.m.layers[layer + 1].exl3) |nx| if (nx.pool) |npl| try prefetch(dp, npl, x.tp, o);
    }
    p.pending = layer;
}

/// Expert parallel: another rank's picks become -1 on the host (this rank plans, copies and computes its own pairs).
pub fn maskOwned(picks: []i32, r: ?tp_.Rank) void {
    const rk = r orelse return;
    for (picks) |*pk_| if (pk_.* >= 0 and !rk.owns(@intCast(pk_.*))) {
        pk_.* = -1;
    };
}

/// The canonical reduction at any degree: block partials, an fp32 all-gather, the tree finished on every rank, placed.
pub fn rankSum(dp: *Deep, p: anytype, x: *const load.Experts, n: usize, sub: usize) !void {
    const st = p.stream;
    const sum = dp.dptr(dp.lay.sum);
    const y = dp.dptr(dp.lay.y);
    const wts = dp.dptr(dp.lay.wts);
    const world: usize = if (x.tp) |r| r.world else 1;
    const rank: usize = if (x.tp) |r| r.rank else 0;
    const m = leaves.partials(world);
    const gath = sum + dp.rows * dims * 4;
    const t0 = if (profile()) now(p.io) else 0;
    const use_overlap = world > 1 and overlap() and dp.red != null;
    const per = if (use_overlap) sub else n;
    try pk.tpCombine(&dp.kern, st, y, wts, dp.dptr(dp.lay.picks), if (world == 1) sum else gath, n, slots, dims, per, world, rank, m);
    if (world == 1) {
        if (!dp.reduced_branch) try pk.tpPlace(&dp.kern, st, sum, y, wts, n, slots, dims);
        return;
    }
    const c = p.m.collective orelse return error.TensorParallelUnavailable;
    if (!use_overlap) {
        try c.gatherF32(st, gath + rank * m * n * dims * 4, gath, m * n * dims);
        try pk.tpFinish(&dp.kern, st, gath, sum, n, dims, world, m);
        if (!dp.reduced_branch) try pk.tpPlace(&dp.kern, st, sum, y, wts, n, slots, dims);
    } else {
        const rs = dp.red.?;
        try dp.red_go.record(st);
        try rs.wait(dp.red_go);
        const n_sub = (n + sub - 1) / sub;
        for (0..n_sub) |b| {
            const r0 = b * sub;
            const rows = @min(sub, n - r0);
            const region = gath + r0 * world * m * dims * 4;
            try c.gatherF32(rs, region + rank * m * rows * dims * 4, region, m * rows * dims);
            try pk.tpFinish(&dp.kern, rs, region, sum + r0 * dims * 4, rows, dims, world, m);
            if (!dp.reduced_branch) try pk.tpPlace(&dp.kern, rs, sum + r0 * dims * 4, y + r0 * slots * dims * 4, wts + r0 * slots * 4, rows, slots, dims);
            try dp.red_ev[b].record(rs);
        }
        dp.red_n = n_sub;
    }
    if (profile()) {
        if (dp.red) |rs| try c.finish(rs);
        try c.finish(st);
        stats.reduce_ns += @intCast(now(p.io) - t0);
    }
    stats.reduces += 1;
}

fn overlap() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_OVERLAP") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Picks for a pool round that holds `ids` (routed, < 512): rows of 11 slots, -1 after the last.
fn fakePicks(dp: *Deep, ids: []const u16) usize {
    var routed: usize = 0;
    for (ids) |id| routed += @intFromBool(id != shared_id);
    if (routed == 0) return 0;
    const rows = (routed + slots - 1) / slots;
    std.debug.assert(rows * slots <= dp.fake.len);
    @memset(dp.fake[0 .. rows * slots], -1);
    var k: usize = 0;
    for (ids) |id| {
        if (id == shared_id) continue;
        dp.fake[k] = id;
        k += 1;
    }
    return rows;
}

/// Stages up to a pass's worth of the next layer's experts that its pool does not hold (lowest ids first): a deep
pub fn prefetch(dp: *Deep, pool: *pool_.Pool, own: ?tp_.Rank, o: Options) !void {
    const want = passLines(pool, o);
    var ids: [table_n]u16 = undefined;
    var k: usize = 0;
    for (0..wt.experts_n) |id| {
        if (k == want) break;
        if (own) |rk| if (!rk.owns(id)) continue;
        if (pool_.offload.lineOf(pool.v, pool.opt.layer, @intCast(id)) != null) continue;
        ids[k] = @intCast(id);
        k += 1;
    }
    const rows = fakePicks(dp, ids[0..k]);
    if (rows == 0) return;
    const r = try pool.plan(dp.fake[0 .. rows * slots], rows);
    if (r.n_passes != 1) return error.Exl3PrefillPass;
    // the copies wait for nothing on the compute stream: the aux stream takes the gates and the release
    try pool.stage(r, 0, dp.aux);
    try pool.release(r, 0, dp.aux);
    stats.prefetched += r.misses_();
    stats.copies += r.misses_();
}

/// The layer's passes: pass 0 the shared expert and every expert the pool holds (no copy), then the others in id
fn plan(dp: *Deep, x: *const load.Experts, picks: []const i32, o: Options) !void {
    @memset(&dp.cnt, 0);
    for (picks) |pk_| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        dp.cnt[@intCast(pk_)] += 1;
    }
    const cap = dp.lay.member_cap;
    var k: usize = 0;
    dp.n_passes = 0;
    var held: [table_n]bool = @splat(false);
    held[shared_id] = true;
    const per: usize = if (x.pool) |pl| passLines(pl, o) else table_n;
    if (x.pool) |pl| {
        for (0..wt.experts_n) |id| held[id] = pool_.offload.lineOf(pl.v, pl.opt.layer, @intCast(id)) != null;
    } else {
        @memset(&held, true);
    }
    // pass 0 (and more if the held experts' members exceed the cap): held experts, the shared one first
    var lo: usize = 0;
    var members: usize = 0;
    const order_ids = struct {
        fn close(d: *Deep, lo_: *usize, k_: usize, members_: *usize) void {
            if (k_ == lo_.*) return;
            d.passes[d.n_passes] = .{ .id_lo = @intCast(lo_.*), .id_hi = @intCast(k_), .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
            d.n_passes += 1;
            lo_.* = k_;
            members_.* = 0;
        }
    };
    if (dp.cnt[shared_id] > 0) {
        dp.order[k] = shared_id;
        k += 1;
        members += dp.cnt[shared_id];
    }
    for (0..wt.experts_n) |id| {
        if (!held[id] or dp.cnt[id] == 0) continue;
        if (members + dp.cnt[id] > cap) order_ids.close(dp, &lo, k, &members);
        dp.order[k] = @intCast(id);
        k += 1;
        members += dp.cnt[id];
    }
    order_ids.close(dp, &lo, k, &members);
    // the misses, `per` new experts a pass
    var in_pass: usize = 0;
    for (0..wt.experts_n) |id| {
        if (held[id] or dp.cnt[id] == 0) continue;
        if (in_pass == per or members + dp.cnt[id] > cap) {
            order_ids.close(dp, &lo, k, &members);
            in_pass = 0;
        }
        dp.order[k] = @intCast(id);
        k += 1;
        members += dp.cnt[id];
        in_pass += 1;
    }
    order_ids.close(dp, &lo, k, &members);
}

/// Members (pairs grouped by expert in pass order, pairs ascending) and items (first relative to the pass's first
pub fn build(dp: *Deep, picks: []const i32, members: []i32, items: []i32) struct { members: usize, items: usize } {
    var m_at: u32 = 0;
    var i_at: u32 = 0;
    @memset(&dp.start, 0);
    for (dp.passes[0..dp.n_passes]) |*ps| {
        ps.m_lo = m_at;
        ps.i_lo = i_at;
        for (dp.order[ps.id_lo..ps.id_hi]) |id| {
            const c = dp.cnt[id];
            dp.start[id] = m_at;
            var off: u32 = 0;
            while (off < c) : (off += pk.tile) {
                items[3 * i_at] = id;
                items[3 * i_at + 1] = @intCast(m_at - ps.m_lo + off);
                items[3 * i_at + 2] = @intCast(@min(@as(u32, pk.tile), c - off));
                i_at += 1;
            }
            m_at += c;
        }
        ps.m_n = m_at - ps.m_lo;
        ps.i_n = i_at - ps.i_lo;
    }
    for (picks, 0..) |pk_, i| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        const id: usize = @intCast(pk_);
        members[dp.start[id]] = @intCast(i);
        dp.start[id] += 1;
    }
    return .{ .members = m_at, .items = i_at };
}

/// One pass through Experts.runSlots: the deep chunk's picks with every expert outside the pass masked to -1, in the
pub fn slotsPass(dp: *Deep, x: *load.Experts, st: cuda.Stream, l: cuda.exl3_experts.Layer, picks: []const i32, ps: Pass, n: usize) !void {
    var in_pass: [table_n]bool = @splat(false);
    for (dp.order[ps.id_lo..ps.id_hi]) |id| in_pass[id] = true;
    const pairs = n * slots;
    // the members' pinned region holds the masked picks (members are not used in this mode); the previous pass's
    try st.synchronize();
    const masked: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(dp.hl.members, pairs * 4)));
    for (picks[0..pairs], masked) |pk_, *o| o.* = if (pk_ >= 0 and pk_ < table_n and in_pass[@intCast(pk_)]) pk_ else -1;
    try upload(st, dp.dptr(dp.lay.members), std.mem.sliceAsBytes(masked));
    const ex = try x.ops.grouped();
    const block: usize = @import("../model/cuda_exl3_pass.zig").moe_rows;
    const bytes = try cuda.exl3_experts.scratchBytes(l, block, slots);
    const base = try x.ops.scratch(st, &x.ops.moe, bytes);
    var r0: usize = 0;
    while (r0 < n) : (r0 += block) {
        const rows = @min(block, n - r0);
        var sc = try cuda.exl3_experts.scratch(base, l, block, slots);
        sc.y = dp.dptr(dp.lay.y) + r0 * slots * dims * 4;
        try ex.runSlots(st, l, dp.dptr(dp.lay.mixed) + r0 * dims * 2, dims, .bf16, dp.dptr(dp.lay.members) + r0 * slots * 4, rows, sc);
    }
}
