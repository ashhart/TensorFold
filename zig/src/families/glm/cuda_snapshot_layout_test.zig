//! The live-half keep's plan on a real state of virtual addresses: the dead KDA half left out, the given parity's half.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const state = @import("cuda_state.zig");
const layout = @import("cuda_snapshot_layout.zig");
const t = std.testing;

const Fake = struct {
    var next: u64 = 0x200000000;
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = fill;
        return d;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += std.mem.alignForward(u64, n, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize) callconv(.c) c_int {
        return 0;
    }
};

/// TP=2's shapes on a KDA and a DSA layer, 4,096 rows: the arena head holds both halves of one KDA layer's state.
fn fixture(d: *cuda.Driver, w: *weights.Weights, layers: []weights.Layer) !state.State {
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    w.layers = layers;
    return state.State.init(t.allocator, d, w, 4096);
}

test "live half: the plan leaves the dead half out as one hole of the state's rec bytes, and nothing else changes" {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    var st = try fixture(&d, &w, &layers);
    defer st.deinit();
    const whole = try layout.Plan.withHalf(&st, &w.config, 2048, null, 0, null);
    const len = st.kda_layers * st.rec_layer;
    try t.expect(len > 0);
    try t.expectEqual(@as(?u1, null), whole.half);
    for ([_]u1{ 0, 1 }) |h| {
        const half = try layout.Plan.withHalf(&st, &w.config, 2048, null, 0, h);
        try t.expectEqual(@as(?u1, h), half.half);
        try t.expectEqual(whole.len - len, half.len);
        try t.expectEqual(whole.count + 1, half.count);
        const dead = st.rec + @as(u64, 1 - h) * len;
        // the head in two parts around the dead half: [start, dead) and [dead + len, the first cache's rows)
        try t.expectEqual(st.arena.buf.ptr, half.parts[0].live);
        try t.expectEqual(@as(usize, @intCast(dead - st.arena.buf.ptr)), half.parts[0].len);
        try t.expectEqual(dead + len, half.parts[1].live);
        try t.expectEqual(whole.parts[0].len - half.parts[0].len - len, half.parts[1].len);
        // every later part as in the whole plan
        for (whole.parts[1..whole.count], half.parts[2..half.count]) |a, b| try t.expectEqual(a, b);
    }
}

test "live half: the plan follows the parity it is given, not the slot's (a restore passes the kept state's)" {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    var st = try fixture(&d, &w, &layers);
    defer st.deinit();
    @memset(st.cur, 0); // the slot moved on since the keep: its live half is 0
    const len = st.kda_layers * st.rec_layer;
    const rec_at: usize = @intCast(st.rec - st.arena.buf.ptr);
    // the kept state lived on half 1: its plan leaves half 0 out, which starts where rec starts
    const kept = try layout.Plan.withHalf(&st, &w.config, 2048, null, 0, 1);
    try t.expectEqual(rec_at, kept.parts[0].len);
    // the slot's own parity would leave half 1 out instead: a different hole, a plan that restores the wrong bytes
    const slot = try layout.Plan.withHalf(&st, &w.config, 2048, null, 0, 0);
    try t.expectEqual(rec_at + len, slot.parts[0].len);
    try t.expect(kept.parts[0].len != slot.parts[0].len);
    try t.expectEqual(kept.len, slot.len);
}

test "live half: halfOf keeps both halves when the flag is off, when a layer is on the other half, or without KDA layers" {
    const same = [_]u1{ 1, 1, 1 };
    const mixed = [_]u1{ 1, 0, 1 };
    const zero = [_]u1{0};
    try t.expectEqual(@as(?u1, 1), layout.halfOf(&same, 3, true));
    try t.expectEqual(@as(?u1, null), layout.halfOf(&same, 3, false));
    try t.expectEqual(@as(?u1, null), layout.halfOf(&mixed, 3, true));
    try t.expectEqual(@as(?u1, null), layout.halfOf(&same, 0, true));
    try t.expectEqual(@as(?u1, null), layout.halfOf(&same, 2, true));
    try t.expectEqual(@as(?u1, 0), layout.halfOf(&zero, 1, true));
}

test "cache shape: GLM-5.3-Flash's pages keep their bytes, full GLM-5.3's 79 caches of 576-wide rows and pool 1 lay out" {
    const flash = state.Shape.of(&@import("config.zig").Config{}, 12);
    try t.expectEqual(@as(usize, 512), flash.row);
    try t.expectEqual(@as(usize, 557056), flash.cacheBytes());
    try t.expectEqual(@as(u64, 13056 * state.page_rows), flash.pageBytes());
    try t.expectEqual(@as(u64, 12 * 32768), flash.pkPageBytes());
    const full = state.Shape.of(&@import("config.zig").Config{ .rope = 64, .kpool = 1 }, 79);
    try t.expectEqual(@as(usize, 576), full.row);
    try t.expectEqual(@as(usize, 512 * 576 * 2), full.latentBytes());
    try t.expectEqual(@as(usize, 512 * 128 * 2), full.pkBytes());
    // 79 caches' 56,950,784 bytes, padded to whole 576-wide rows and 512 tokens (4,608 bytes): 4,096 more
    try t.expectEqual(@as(u64, 79 * (576 * 2 + 128 * 2) * state.page_rows + 4096), full.pageBytes());
    try t.expectEqual(@as(u64, 0), full.pageBytes() % (576 * 2));
    try t.expectEqual(@as(u64, 0), full.pageBytes() % state.page_rows);
    try t.expectEqual(@as(u64, 49_440), full.pageBytes() / (576 * 2)); // dsa_block's page_rows: a token list's rows
}

/// Full GLM-5.3's caches on a state: 78 DSA layers and no KDA, 576-wide latent rows, a pooled key a row.
fn fullFixture(d: *cuda.Driver, w: *weights.Weights, layers: []weights.Layer) !state.State {
    for (layers) |*l| l.mixer = .{ .mla = undefined };
    w.config = .{ .rope = 64, .kpool = 1 };
    w.share = try shard.Share.of(&w.config, 0, 1);
    w.layers = layers;
    return state.State.init(t.allocator, d, w, 4096);
}

test "cache shape: a full GLM-5.3 state's plan carries every cache's 576-wide rows and pooled keys, no KDA half" {
    var d = Fake.driver();
    var layers: [78]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    var st = try fullFixture(&d, &w, &layers);
    defer st.deinit();
    try t.expectEqual(@as(usize, 0), st.kda_layers);
    try t.expectEqual(@as(usize, 78), st.caches.len);
    try t.expectEqual(@as(usize, 4096 + 2), st.caches[0].pk_rows);
    const at = 2048;
    const plan = try layout.Plan.init(&st, &w.config, at, at, 0);
    try t.expectEqual(@as(?u1, null), plan.half);
    try t.expectEqual(@as(usize, 1 + 79 * 4 + 1), plan.count); // the head, each cache's latent, rings and pooled keys, the tail
    for (0..79) |j| {
        try t.expectEqual(@as(usize, at * 576 * 2), plan.parts[1 + 4 * j].len);
        try t.expectEqual(@as(usize, at * 128 * 2), plan.parts[2 + 4 * j].len);
        try t.expectEqual(@as(usize, (at + 2) * 128 * 2), plan.parts[4 + 4 * j].len);
    }
}

test "cache shape: on pages each cache's part is a 576-wide row's latent then its pooled keys" {
    var d = Fake.driver();
    var layers: [78]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    var st = try fullFixture(&d, &w, &layers);
    defer st.deinit();
    const s = state.shapeOf(&w);
    try t.expectEqual(@as(usize, 79), s.caches);
    const kv: u64 = 1 << 40;
    st.place(&w, .{ .kv = kv, .pages = 4, .page_bytes = state.pageBytes(&w) });
    for (st.caches, 0..) |x, j| {
        try t.expectEqual(kv + j * (512 * 576 * 2 + 512 * 128 * 2), x.lc);
        try t.expectEqual(x.lc + 512 * 576 * 2, x.pk);
    }
    try t.expectEqual(kv + 78 * s.cacheBytes(), st.mtp_caches.lc);
}

test "cache shape: full GLM-5.3's 57 shared layers keep latent rows alone, on pages and in a plan" {
    var d = Fake.driver();
    var layers: [78]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    var st = try fullFixture(&d, &w, &layers);
    st.deinit();
    for (3..78) |i| if (i < 6 or (i - 6) % 4 != 0) w.config.shared.set(i); // 0-2 and 6, 10, .., 74 index
    w.config.dsa = true;
    st = try state.State.init(t.allocator, &d, &w, 4096);
    defer st.deinit();
    const s = state.shapeOf(&w);
    const lb = 512 * 576 * 2;
    const pb = 512 * 128 * 2;
    try t.expectEqual(@as(usize, 22), s.pooledBefore(79)); // 21 layers and the MTP layer's
    try t.expectEqual(@as(u64, 22 * pb), s.pkPageBytes());
    try t.expectEqual(@as(u64, 79 * lb + 22 * pb + 1024), s.pageBytes()); // padded to 4,608-byte units
    try t.expectEqual(@as(usize, 0), st.caches[3].pk_rows);
    try t.expectEqual(@as(usize, 4096 + 2), st.caches[6].pk_rows);
    const kv: u64 = 1 << 40;
    st.place(&w, .{ .kv = kv, .pages = 4, .page_bytes = state.pageBytes(&w) });
    try t.expectEqual(kv + 7 * lb + 4 * pb, st.caches[7].lc); // after 0, 1, 2 and 6's keys
    try t.expectEqual(@as(u64, 0), st.caches[7].pk);
    try t.expectEqual(st.caches[6].lc + lb, st.caches[6].pk);
    try t.expectEqual(kv + 78 * lb + 21 * pb, st.mtp_caches.lc);
    var arena = try state.State.init(t.allocator, &d, &w, 4096);
    defer arena.deinit();
    const plan = try layout.Plan.init(&arena, &w.config, 2048, 2048, 0);
    try t.expectEqual(@as(usize, 1 + 22 * 4 + 57 + 1), plan.count); // a shared cache: its latent rows only
}

test "cache shape: index keys stay in GLM-5.3-Flash's ring of index_ring rows, a rebase keeping whole pools" {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    var st = try fixture(&d, &w, &layers);
    defer st.deinit();
    try t.expectEqual(@as(usize, state.index_ring), state.shapeOf(&w).keys);
    try t.expectEqual(@as(usize, state.index_ring), st.ring_rows);
    try t.expectEqual(@as(usize, 4), st.pool);
}
