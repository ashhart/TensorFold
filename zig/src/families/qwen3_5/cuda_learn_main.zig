//! Qwen MoE qualification against independent serving passes and directional finite differences.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("cuda.zig");
const train = qwen.train;
const Trainer = train.Trainer;

const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };
fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}
fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    defer a.free(bytes);
    const doc = try std.json.parseFromSlice([]u32, a, bytes, .{ .allocate = .alloc_always });
    defer doc.deinit();
    return a.dupe(u32, doc.value);
}
/// A separate serving pass computes the loss with an FP64 host log-sum-exp and no trainer hooks.
fn servingLoss(a: std.mem.Allocator, e: *qwen.Engine, sq: train.Seq) !f64 {
    var cache = try qwen.engine.Cache.init(a, e, sq.ids.len + 1);
    defer cache.deinit(a);
    var total: f64 = 0;
    for (sq.ids[0 .. sq.ids.len - 1], 0..) |id, row| {
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{id}, .logits);
        if (row + 1 < sq.start) continue;
        const logits = e.rowLogits(0);
        var top = -std.math.inf(f64);
        for (logits) |v| top = @max(top, bf(v));
        var sum: f64 = 0;
        for (logits) |v| sum += @exp(bf(v) - top);
        total += top + @log(sum) - bf(logits[sq.ids[row + 1]]);
    }
    return total / @as(f64, @floatFromInt(sq.ids.len - sq.start));
}
pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: tf-qwen-moe-learn MODEL IDS_JSON_ARRAY\n", .{});
        return 2;
    }
    const ids = try readIds(a, io, args[2]);
    defer a.free(ids);
    if (ids.len < 24 or ids.len > train.max_rows + 1) return error.BadExample;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try qwen.Engine.init(a, io, &ctx, args[1], false, null);
    defer e.deinit();
    if (!e.g.moe()) return error.MoeCheckpointRequired;
    const D = e.g.hidden;
    const t = try Trainer.init(a, io, e);
    defer t.deinit();
    var sq: train.Seq = .{ .ids = ids, .start = ids.len - 6, .decide = ids.len - 7 };
    try identity(a, t, sq);
    const serving = try servingLoss(a, e, sq);
    var result: [1]train.Result = undefined;
    try t.pass(&.{sq}, .loss, turn, &result);
    if (!std.math.isFinite(result[0].loss) or @abs(serving - result[0].loss) > 1e-3)
        return error.TrainerServingMismatch;
    var together: [2]train.Result = undefined;
    try t.pass(&.{ sq, sq }, .loss, turn, &together);
    for (together) |p| if (@abs(p.loss - serving) > 1e-3) return error.PackedLossMismatch;
    std.debug.print("PASS Qwen MoE serving/trainer/packed loss {d:.8}\n", .{serving});
    const rows = [_]usize{ 0, sq.start - 5, sq.start - 1 };
    const low = train.keepLow(e.g.layers);
    const layers = [_]usize{ low, low + 1, low + 2, e.g.layers - 2 };
    const push = try a.alloc(f32, D);
    defer a.free(push);
    const dir = try a.alloc(f32, D);
    defer a.free(dir);
    var check: qwen.learn_direction.Case = .{ .t = t, .sq = sq, .push = push, .dir = dir, .label = "Qwen MoE" };
    try qwen.learn_direction.run(&check, &rows, &layers);
    // Exercise the same forward and adjoint with a live gated pocket, leaving the checkpoint on disk unchanged.
    const pocket = try a.alloc(f32, 3 * 64 * D);
    defer a.free(pocket);
    @memset(pocket, 0);
    var rng = std.Random.DefaultPrng.init(35);
    for (0..3) |p| {
        for (pocket[p * 64 * D ..][0..D]) |*v| v.* = rng.random().floatNorm(f32) * 0.05;
    }
    const layer = e.g.layers - 2;
    sq.nudge = .{};
    try t.pass(&.{sq}, .loss, turn, &result);
    const before = result[0].loss;
    try e.pockets.set(e.stream, layer, pocket[0 .. 64 * D], pocket[64 * D ..][0 .. 64 * D], pocket[128 * D ..], 1);
    sq.nudge = .{};
    try identity(a, t, sq);
    try t.pass(&.{sq}, .loss, turn, &result);
    if (result[0].loss == before) return error.TaughtTrainerForwardUnchanged;
    try @import("cuda_learn_moe_check.zig").run(a, t, layer);
    return 0;
}

/// Snapshot a separate serving forward, then compare the learner's inputs, picks, gates and raw logits by bytes.
fn identity(a: std.mem.Allocator, t: *Trainer, sq: train.Seq) !void {
    const e = t.e;
    const g = e.g;
    const n = sq.ids.len - 1;
    const slots = g.slots();
    const inputs = try a.alloc(u16, g.layers * n * g.hidden);
    defer a.free(inputs);
    const ids = try a.alloc(i32, g.layers * n * slots);
    defer a.free(ids);
    const wts = try a.alloc(f32, g.layers * n * slots);
    defer a.free(wts);
    const logits = try a.alloc(u16, n * g.vocab);
    defer a.free(logits);
    var cache = try qwen.engine.Cache.init(a, e, n + 1);
    defer cache.deinit(a);
    var keep = t.keep;
    keep.layers = t.light;
    const previous = e.learning;
    const mlp = e.mlp;
    defer {
        e.learning = previous;
        e.mlp = mlp;
    }
    e.learning = &keep;
    e.mlp = null;
    for (sq.ids[0..n], 0..) |id, row| {
        keep.at = row;
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{id}, .logits);
    }
    try e.stream.synchronize();
    for (0..g.layers) |l| {
        for (0..n) |row| @memcpy(inputs[(l * n + row) * g.hidden ..][0..g.hidden], t.normed(l, row));
        try fetch(e, t.light[l].ids, std.mem.sliceAsBytes(ids[l * n * slots ..][0 .. n * slots]));
        try fetch(e, t.light[l].wts, std.mem.sliceAsBytes(wts[l * n * slots ..][0 .. n * slots]));
    }
    try fetch(e, t.logits, std.mem.sliceAsBytes(logits));
    e.learning = previous;
    try t.scores(sq);
    for (0..g.layers) |l| {
        for (0..n) |row| {
            if (!std.mem.eql(u16, inputs[(l * n + row) * g.hidden ..][0..g.hidden], t.normed(l, row))) return error.TrainerInputsDiffer;
            const gate = wts[(l * n + row) * slots + slots - 1];
            if (@as(u32, @bitCast(gate)) != @as(u32, @bitCast(t.gateAt(l, row)))) return error.TrainerSharedGateDiffers;
        }
        const now = try a.alloc(i32, n * slots);
        defer a.free(now);
        try fetch(e, t.light[l].ids, std.mem.sliceAsBytes(now));
        if (!std.mem.eql(i32, ids[l * n * slots ..][0 .. n * slots], now)) return error.TrainerPicksDiffer;
        const weights = try a.alloc(f32, n * slots);
        defer a.free(weights);
        try fetch(e, t.light[l].wts, std.mem.sliceAsBytes(weights));
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(wts[l * n * slots ..][0 .. n * slots]), std.mem.sliceAsBytes(weights))) return error.TrainerWeightsDiffer;
    }
    const got = try a.alloc(u16, logits.len);
    defer a.free(got);
    try fetch(e, t.logits, std.mem.sliceAsBytes(got));
    if (!std.mem.eql(u16, logits, got)) return error.TrainerScoresDiffer;
    std.debug.print("PASS serving/trainer inputs, picks, routing weights, gates and logits byte-equal at every layer\n", .{});
}
fn fetch(e: *qwen.Engine, src: u64, bytes: []u8) !void {
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(bytes.ptr, src, bytes.len), "qualification read");
}
