//! Deep-prefill experts through the line ring: its pass order, copies and cross-layer prefetch.

const std = @import("std");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const pk = @import("cuda_exl3_prefill_k.zig");
const prompt = @import("../../cuda_prompt.zig");
const ring_ = @import("cuda_exl3_prefill_ring.zig");
const history_ = @import("cuda_exl3_prefill_history.zig");
const state = @import("cuda_exl3_prefill_state.zig");
const experts = @import("cuda_exl3_prefill_moe.zig");

const dims = wt.dims;
const slots = wt.slots;
const table_n = wt.experts_n + 1;
const routed_n_ = wt.experts_n;
const shared_id = wt.experts_n;

const Deep = state.Deep;
const Options = state.Options;
const now = state.now;
const stats = &state.stats;
const upload = state.upload;
const build = experts.build;
const maskOwned = experts.maskOwned;
const slotsPass = experts.slotsPass;

/// The ring over the engine's decoder pools, made once; null (and the per-layer passes) if they cannot form one.
pub fn ringOf(dp: *Deep, e: *prompt.Engine) !?*ring_.Ring {
    if (dp.ring) |r| return r;
    if (dp.ring_failed) return null;
    var pools: [wt.layers_n]*pool_.Pool = undefined;
    for (e.layers[0..e.n_layers], 0..) |*l, i| {
        const x = l.exl3 orelse {
            dp.ring_failed = true;
            return null;
        };
        pools[i] = x.pool orelse {
            dp.ring_failed = true;
            return null;
        };
    }
    dp.ring = ring_.Ring.init(dp.gpa, e.driver, &pools, e.telemetry.enabled) catch |err| {
        std.log.warn("flashnext exl3 prefill: no line ring ({t}); each layer stages into its own lines", .{err});
        dp.ring_failed = true;
        return null;
    };
    if (e.layers[0].exl3.?.tp) |rk| {
        dp.ring.?.own_lo = rk.lo;
        dp.ring.?.own_hi = @min(rk.hi, routed_n_);
    }
    std.log.info("flashnext exl3 prefill: a ring of {d} expert slots over the {d} layers' pool lines", .{ dp.ring.?.slots(), wt.layers_n });
    return dp.ring;
}

/// Slots a prefetch leaves free: a layer's worth of misses and two passes.
pub fn keepFree(r: *const ring_.Ring, o: Options) usize {
    return @min(r.slots() / 2, routed_n_ + 2 * o.ring_pass);
}

/// The layer's experts over all `n` rows of the deep chunk, through the ring.
pub fn ringMoe(comptime Tri: type, dp: *Deep, r: *ring_.Ring, p: *batch.Pass(Tri), lw: *const @import("../../cuda_prompt_load.zig").Layer, layer: usize, n: usize, o: Options) !void {
    const x = lw.exl3.?;
    const pool = x.pool.?;
    const st = p.stream;
    const pairs = n * slots;
    const hl = dp.hl;
    const t0 = now(p.io);
    const host_picks: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.picks, pairs * 4)));
    try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(host_picks.ptr, dp.dptr(dp.lay.picks), pairs * 4, st.handle), "cuMemcpyDtoHAsync");
    try st.synchronize();
    maskOwned(host_picks, x.tp);
    const t1 = now(p.io);
    @memset(&dp.cnt, 0);
    for (host_picks) |pk_| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        dp.cnt[@intCast(pk_)] += 1;
    }
    r.drop(layer, dp.cnt[0..routed_n_]);
    // pass order: shared, then the held experts by load batch, then the misses hottest first
    var held: [routed_n_]u16 = undefined;
    var n_held: usize = 0;
    var miss: [routed_n_]u16 = undefined;
    var n_miss: usize = 0;
    var used: usize = 0;
    for (0..routed_n_) |id| {
        if (dp.cnt[id] == 0) continue;
        used += 1;
        r.counts[layer][id] +|= dp.cnt[id];
        const present = r.isLoaded(layer, id);
        r.stats.observeDemand(present, dp.cnt[id]);
        if (present) {
            held[n_held] = @intCast(id);
            n_held += 1;
        } else {
            miss[n_miss] = @intCast(id);
            n_miss += 1;
        }
    }
    dp.used[layer] = used;
    const ByBatch = struct {
        r: *const ring_.Ring,
        l: usize,
        fn lt(c: @This(), a: u16, b: u16) bool {
            return c.r.batch_of[c.l][a] < c.r.batch_of[c.l][b];
        }
    };
    std.mem.sort(u16, held[0..n_held], ByBatch{ .r = r, .l = layer }, ByBatch.lt);
    const ByCount = struct {
        c: *const [table_n]u32,
        fn gt(c: @This(), a: u16, b: u16) bool {
            if (c.c[a] != c.c[b]) return c.c[a] > c.c[b];
            return a < b;
        }
    };
    std.mem.sort(u16, miss[0..n_miss], ByCount{ .c = &dp.cnt }, ByCount.gt);
    // passes: at most ring_pass experts and member_cap members each; the shared expert opens pass 0
    var k: usize = 0;
    var lo: usize = 0;
    var members: usize = 0;
    var in_pass: usize = 0;
    dp.n_passes = 0;
    const cap = dp.lay.member_cap;
    const close = struct {
        fn f(d: *Deep, lo_: *usize, k_: usize, members_: *usize, in_pass_: *usize) void {
            if (k_ == lo_.*) return;
            d.passes[d.n_passes] = .{ .id_lo = @intCast(lo_.*), .id_hi = @intCast(k_), .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
            d.n_passes += 1;
            lo_.* = k_;
            members_.* = 0;
            in_pass_.* = 0;
        }
    }.f;
    if (dp.cnt[shared_id] > 0) {
        dp.order[k] = shared_id;
        k += 1;
        members += dp.cnt[shared_id];
    }
    for ([2][]const u16{ held[0..n_held], miss[0..n_miss] }) |list| for (list) |id| {
        if (in_pass == o.ring_pass or members + dp.cnt[id] > cap) close(dp, &lo, k, &members, &in_pass);
        dp.order[k] = id;
        k += 1;
        members += dp.cnt[id];
        in_pass += 1;
    };
    close(dp, &lo, k, &members, &in_pass);
    const staging0 = now(p.io);
    const copies0 = r.stats.copies;
    for (0..@min(2, dp.n_passes)) |j| try loadPass(dp, r, layer, j, false);
    const staging_ns = now(p.io) - staging0;
    const members_h: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.members, pairs * 4)));
    const items_h: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.items, 3 * pk.maxItems(pairs, table_n) * 4)));
    const counts = build(dp, host_picks, members_h, items_h);
    try upload(st, dp.dptr(dp.lay.members), std.mem.sliceAsBytes(members_h[0..counts.members]));
    try upload(st, dp.dptr(dp.lay.items), std.mem.sliceAsBytes(items_h[0 .. 3 * counts.items]));
    const t2 = now(p.io);
    stats.host_wait_ns += @intCast(t1 - t0);
    stats.plan_ns += @intCast(t2 - t1 - staging_ns);
    stats.members += counts.members;
    const l = pool.tables();
    // the first two passes' copies, then each pass: the next pass's copies, the gate, the kernel, the release
    for (dp.passes[0..dp.n_passes], 0..) |ps, j| {
        if (j + 2 < dp.n_passes) try loadPass(dp, r, layer, j + 2, true);
        try loadPass(dp, r, layer, j, false);
        const ids_ = dp.order[ps.id_lo..ps.id_hi];
        try r.gate(st, layer, ids_);
        if (o.slots_kernel) {
            try slotsPass(dp, x, st, l, host_picks, ps, n);
        } else {
            const dv: pk.Device = .{ .members = dp.dptr(dp.lay.members) + ps.m_lo * 4, .items = dp.dptr(dp.lay.items) + ps.i_lo * 12, .scratch = dp.dptr(dp.lay.kscratch) };
            try pk.run(&dp.kern, st, l, dp.dptr(dp.lay.mixed), dims, dp.dptr(dp.lay.picks), slots, ps.m_n, ps.i_n, dv, dp.dptr(dp.lay.y));
        }
        try r.release(st, layer, ids_);
        stats.passes += 1;
    }
    stats.copies += r.stats.copies - copies0;
    dp.history[layer] = .{ .rows = n, .used = used };
    const c1 = r.stats.copies;
    for (1..o.ring_ahead + 1) |ahead| {
        const nl = layer + ahead;
        if (nl >= wt.layers_n) break;
        _ = try r.prefetch(nl, history_.want(dp.history[nl], n, used, routed_n_), keepFree(r, o), .cross_layer_prefetch);
    }
    stats.prefetched += r.stats.copies - c1;
    stats.copies += r.stats.copies - c1;
    p.pending = layer;
}

/// Queues the copies of pass `j`'s routed experts the ring does not hold; `best_effort`: stop quietly when full.
fn loadPass(dp: *Deep, r: *ring_.Ring, layer: usize, j: usize, best_effort: bool) !void {
    const ps = dp.passes[j];
    for (dp.order[ps.id_lo..ps.id_hi]) |id| {
        if (id == shared_id) continue;
        if (try r.load(layer, id, .demand)) continue;
        if (best_effort) return;
        return error.Exl3PrefillRingFull;
    }
    try r.flush();
}
