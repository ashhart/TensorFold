//! Greedy lesson checks and conversation keys use the serving runner with fresh native states.
const std = @import("std");
const train = @import("train.zig");
const Runner = @import("decode_round.zig").Runner;
const to = @import("train_ops.zig");
const mtl = @import("metal");

pub fn residual(t: *train.Trainer, l: usize, row: usize) []const u16 {
    if (l == 0 or l > t.g.layers or (l < t.g.layers and (t.mode != .target or l < t.low)))
        return train.slice(u16, t.zero_history, t.g.hidden);
    const ref = if (l == t.g.layers) t.final_h else t.tape[l].h_in;
    return train.slice(u16, to.plus(ref, row * t.g.hidden * 2), t.g.hidden);
}

pub fn greedy(t: *train.Trainer, seqs: []const train.Seq, same: []bool) !void {
    if (seqs.len > train.max_seqs or same.len < seqs.len or t.m.learning != null) return error.BadPass;
    const prompt = t.m.kernels.prompt;
    t.m.kernels.prompt = false;
    defer t.m.kernels.prompt = prompt;
    for (seqs, 0..) |sq, si| {
        if (sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len > train.max_rows + 1) return error.BadExample;
        try t.runner.reset(@intCast(si));
        same[si] = true;
        for (0..sq.ids.len - 1) |row| {
            try t.forwardRow(&t.runner, @intCast(si), row, sq.ids[row..][0..1], row + 1 >= sq.start);
            if (row + 1 < sq.start) continue;
            const logits = train.slice(u16, t.m.frame.get(.logits), t.g.vocab);
            var top: u32 = 0;
            var best = -std.math.inf(f32);
            for (logits, 0..) |bits, id| {
                const v: f32 = @bitCast(@as(u32, bits) << 16);
                if (!std.math.isFinite(v)) return error.NonFiniteLearningLogit;
                if (v > best) {
                    best = v;
                    top = @intCast(id);
                }
            }
            same[si] = same[si] and top == sq.ids[row + 1];
        }
    }
}

pub fn tapRows(t: *train.Trainer, ids: []const u32, context: usize, rows: []const u32, layers: []const usize, out: []u16) !void {
    const D = t.g.hidden;
    if (ids.len == 0 or ids.len >= t.g.max_position or context > ids.len or out.len != rows.len * layers.len * D or t.m.learning != null) return error.BadTap;
    for (rows, 0..) |row, i| if (row >= ids.len or (i > 0 and row <= rows[i - 1])) return error.BadTap;
    for (layers) |l| if (l >= t.g.layers) return error.BadTap;
    // Long-context taps are admitted beside the actual trainer and serving allocations, before a second cache exists.
    const need = try @import("prompt_memory.zig").runnerBytes(t.g, @intCast(ids.len + 1), 1, t.m.frame.capacity, t.m.device.tensorUnits(), 0);
    const free = t.m.device.maxWorkingSet() -| t.m.device.allocated();
    if (need + (8 << 30) > free) return error.LearningMemory;
    var runner = try Runner.init(t.gpa, t.m, 1, @intCast(ids.len + 1));
    defer runner.deinit();
    runner.learning_rows = true;
    runner.learning_io = t.io;
    runner.setResident(t.follow);
    const old_mode = t.mode;
    const old_at = t.hook_at;
    const prompt = t.m.kernels.prompt;
    t.mode = .keys;
    t.hook_at = 0;
    t.m.kernels.prompt = false;
    defer {
        t.mode = old_mode;
        t.hook_at = old_at;
        t.m.kernels.prompt = prompt;
        t.m.learning = null;
    }
    t.m.learning = .{ .ctx = t, .save = @import("train_tape.zig").save, .add = plainPocket };
    var ri: usize = 0;
    for (ids, 0..) |_, row| {
        try t.forwardRow(&runner, 0, row, ids[row..][0..1], false);
        if (ri == rows.len or row != rows[ri]) continue;
        for (layers, 0..) |l, j| @memcpy(out[(ri * layers.len + j) * D ..][0..D], t.normed(l, 0));
        ri += 1;
    }
}
fn plainPocket(ctx: *anyopaque, e: mtl.ComputeEncoder, l: usize, x: to.Ref, y: to.Ref, rows: u32) !void {
    const t: *train.Trainer = @ptrCast(@alignCast(ctx));
    if (t.m.pockets) |*p| try p.add(e, l, x, y, rows);
}
