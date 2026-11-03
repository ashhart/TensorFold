//! Synthetic serving/trainer identity, packed-sequence and nudged-gradient smoke checks. No checkpoint downloads.
const std = @import("std");
const tf = @import("tensorfold");
const mtl = @import("metal");
const q = tf.qwen27;
const identity = @import("learning_identity.zig");
const graph = @import("synthetic_graph.zig");

fn write(a: std.mem.Allocator, io: std.Io, dir: []const u8, m: *q.model.Model) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var folder = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer folder.close(io);
    var tensors: std.ArrayList(tf.shard_edit.New) = .empty;
    defer tensors.deinit(a);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"weight_map\":{");
    var it = m.checkpoint.tensors.iterator();
    while (it.next()) |entry| {
        const t = entry.value_ptr.*;
        try tensors.append(a, .{ .name = entry.key_ptr.*, .dtype = t.dtype, .shape = try a.dupe(usize, t.shape[0..t.rank]), .bytes = t.buffer.contents()[t.offset..][0..t.bytes] });
        try index.print(a, "{s}\"{s}\":\"model.safetensors\"", .{ if (tensors.items.len == 1) "" else ",", entry.key_ptr.* });
    }
    try index.appendSlice(a, "}}");
    try tf.shard_edit.create(a, try std.fs.path.join(a, &.{ dir, "model.safetensors" }), &.{.{ "format", "mlx" }}, tensors.items);
    try folder.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}
pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.Usage;
    try run(a, init.io, args[1]);
}

/// Called by the Metal contract runner as well as the bounded standalone gate.
pub fn run(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try graph.create(a);
    defer m.deinit();
    try write(a, io, dir, m);
    const t = try q.train.Trainer.init(a, io, m, dir, null);
    defer t.deinit();
    var runner = try q.decode_round.Runner.init(a, m, 1, 32);
    defer runner.deinit();
    const ids = [_]u32{ 7, 13, 29, 31 };
    const seq: q.train.Seq = .{ .ids = &ids, .start = 2, .decide = 1 };
    const turn = q.train.Turn{ .decide = 1, .whole = true };
    var results: [2]q.train.Result = undefined;
    var baseline: [3][96]u16 = undefined;
    for (0..3) |row| {
        try t.forwardRow(&runner, 0, row, ids[row..][0..1], true);
        @memcpy(&baseline[row], q.train.slice(u16, m.frame.get(.logits), 96));
    }
    try t.pass(&.{seq}, .loss, turn, &results);
    for (seq.start - 1..3) |row| if (!std.mem.eql(u16, &baseline[row], q.train.slice(u16, @import("tensorfold").qwen27.projection.Ref{ .buffer = t.logits.buffer, .offset = row * 96 * 2 }, 96))) return error.TrainerForwardDiffers;
    const solo = results[0].loss;
    const other_ids = [_]u32{ 17, 19, 31, 37, 41 };
    const other: q.train.Seq = .{ .ids = &other_ids, .start = 3, .decide = 1 };
    try t.pass(&.{other}, .loss, turn, &results);
    const other_solo = results[0].loss;
    try t.pass(&.{ seq, other }, .loss, turn, &results);
    if (results[0].loss != solo or results[1].loss != other_solo) return error.PackedLossDiffers;
    const nudged: q.train.Seq = .{ .ids = &ids, .start = 2, .decide = 1, .nudge = .{ .layers = .{ 1, 2, 4 }, .rows = .{ 0, 1, 2 } } };
    try t.pass(&.{nudged}, .target, turn, &results);
    var norm: f64 = 0;
    for (0..3) |i| for (t.pushAt(0, i, i)) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteLearningGradient;
        norm += @as(f64, value) * value;
    };
    if (norm == 0) return error.MissingLearningGradient;
    const fingerprint = std.crypto.hash.sha2.Sha256.hash;
    var solo_gradients: [3][32]u8 = undefined;
    for (0..3) |i| fingerprint(std.mem.sliceAsBytes(t.pushAt(0, i, i)), &solo_gradients[i], .{});
    try t.pass(&.{ nudged, other }, .target, turn, &results);
    for (0..3) |i| {
        var packed_hash: [32]u8 = undefined;
        fingerprint(std.mem.sliceAsBytes(t.pushAt(0, i, i)), &packed_hash, .{});
        if (!std.mem.eql(u8, &solo_gradients[i], &packed_hash)) return error.PackedGradientDiffers;
    }
    var first: [32]u8 = undefined;
    fingerprint(std.mem.sliceAsBytes(t.pushAt(0, 0, 0)), &first, .{});
    for (0..4) |_| {
        try t.pass(&.{nudged}, .target, turn, &results);
        var next: [32]u8 = undefined;
        fingerprint(std.mem.sliceAsBytes(t.pushAt(0, 0, 0)), &next, .{});
        if (!std.mem.eql(u8, &first, &next)) return error.LearningRepeatDiffers;
    }
    // A no-nudge target still keeps every residual read by the lesson's pocket validation.
    try t.pass(&.{seq}, .target, turn, &results);
    if (std.mem.allEqual(u16, t.residual(1, 0), 0)) return error.MissingLearningResidual;
    var same: [1]bool = undefined;
    try t.greedy(&.{seq}, &same);
    // A scaled open pocket must disappear at scale zero, while an untaught stream must skip every saved neuron.
    const neurons = 64;
    const n = neurons * m.config.hidden;
    const gate = try a.alloc(f32, n);
    const up = try a.alloc(f32, n);
    const down = try a.alloc(f32, n);
    @memset(gate, 0);
    @memset(up, 0);
    @memset(down, 0);
    gate[0] = 4;
    up[0] = 1;
    for (down[0..m.config.hidden], 0..) |*v, i| v.* = if (i % 2 == 0) 0.5 else -0.5;
    try q.slide.Learner.Host.pocketSwiglu(.{ .m = m, .dir = dir }, 2, gate, up, down, 1);
    t.from[2] = 0;
    var scaled = seq;
    scaled.scale = 0;
    try t.pass(&.{scaled}, .loss, turn, &results);
    if (results[0].loss != solo) return error.ScaledPocketNotZero;
    try runner.reset(0);
    runner.untaught[0] = true;
    for (0..3) |row| {
        try t.forwardRow(&runner, 0, row, ids[row..][0..1], true);
        if (!std.mem.eql(u16, &baseline[row], q.train.slice(u16, m.frame.get(.logits), 96))) return error.UntaughtPocketDiffers;
    }
    // Poison skipped prompt-head storage so a stale value can never accidentally satisfy the identity gate.
    @memset(q.train.slice(u16, t.logits, (seq.start - 1) * m.config.vocab), 0x7fc1);
    // The pocket input backward and the scaled entry run through the production trainer, too.
    try t.pass(&.{nudged}, .target, turn, &results);
    for (t.pushAt(0, 0, 0)) |value| if (!std.math.isFinite(value)) return error.NonFinitePocketGradient;
    // The serving entry and the trainer's scale-one entry must agree with live neurons, too.
    runner.untaught[0] = false;
    try runner.reset(0);
    for (0..3) |row| {
        try t.forwardRow(&runner, 0, row, ids[row..][0..1], true);
        const kept = q.train.slice(u16, .{ .buffer = t.logits.buffer, .offset = row * 96 * 2 }, 96);
        const hidden = q.train.slice(u16, .{ .buffer = t.final_h.buffer, .offset = row * m.config.hidden * 2 }, m.config.hidden);
        // The native layer program finishes in frame.hidden after its two residual swaps per block.
        try identity.check(row, seq.start, hidden, q.train.slice(u16, m.frame.get(.hidden), m.config.hidden), kept, q.train.slice(u16, m.frame.get(.logits), 96));
    }
    std.debug.print("Q27_TRAIN native_logits_equal=true packed_loss_equal=true repeat=5 nonzero_gradient_norm={d:.9} target_residuals=true greedy_checked=true scaled_zero_equal=true untaught_equal=true pocket_backward=true taught_logits_equal=true prompt_hidden_equal=true\n", .{@sqrt(norm)});
}
