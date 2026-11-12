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
