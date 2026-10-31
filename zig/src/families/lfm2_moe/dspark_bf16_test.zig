//! The DSpark drafter's bf16 kernels on the GPU against host references: the linears and the Markov scan over bf16 W2.
const std = @import("std");
const mtl = @import("metal");
const quant = @import("dspark_quant.zig");
const base = @import("dspark_test.zig");

const Args = extern struct { rows: u32, n: u32, k: u32 };

/// y = x W^T through `key` on random bf16 data, each output within a bf16 step of the fp64 product.
fn linear(comptime key: []const u8, rows: usize, n: usize, k: usize, groups: [2]usize, threads: usize) !void {
    var r = try base.Rig.init();
    defer r.deinit();
    const x = try r.buffer(u16, rows * k);
    defer x.deinit();
    const w = try r.buffer(u16, n * k);
    defer w.deinit();
    const y = try r.buffer(u16, rows * n);
    defer y.deinit();
    base.noise(31, x.slice(u16, rows * k), 2);
    base.noise(32, w.slice(u16, n * k), 0.5);
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(r.k.get(key));
    e.setBuffer(x, 0, 0);
    e.setBuffer(w, 0, 1);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k) }, 2);
    e.setBuffer(y, 0, 3);
    e.dispatchGroups(mtl.Size.of(groups[0], groups[1], 1), mtl.Size.of(threads, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    const xs = x.slice(u16, rows * k);
    const ws = w.slice(u16, n * k);
    const ys = y.slice(u16, rows * n);
    for (0..rows) |i| for (0..n) |j| {
        var dot: f64 = 0;
        for (0..k) |t| dot += @as(f64, quant.toF32(xs[i * k + t])) * @as(f64, quant.toF32(ws[j * k + t]));
        const got: f64 = quant.toF32(ys[i * n + j]);
        try std.testing.expect(@abs(got - dot) <= @abs(dot) * 0x1p-7 + 1e-3);
    };
}

test "lfm2 dspark: the bf16 rows kernel is x W^T for one row, a full 16 and a ragged 19 (two row groups)" {
    for ([_]usize{ 1, 16, 19 }) |rows| try linear("lfm2_bf16_rows", rows, 100, 288, .{ 13, (rows + 15) / 16 }, 256);
}

test "lfm2 dspark: the bf16 tile kernel is x W^T for a ragged 70 rows and 100 columns, k a multiple of 32" {
    try linear("lfm2_bf16_mma", 70, 100, 288, .{ 2, 2 }, 128);
}

test "lfm2 dspark: the bf16 tile kernel at the drafter's widths" {
    try linear("lfm2_bf16_mma", 33, 512, 2048, .{ 8, 1 }, 128);
}

test "lfm2 dspark: a Markov step over bf16 W2 picks the argmax of bf16(base + bf16(W2 . W1[prev])), a NaN row flagged" {
    var r = try base.Rig.init();
    defer r.deinit();
    const rank = 256;
    const vocab = 1300;
    const streams = 4;
    const groups = (vocab + 511) / 512;
    const w1 = try r.buffer(u16, vocab * rank);
    defer w1.deinit();
    base.noise(11, w1.slice(u16, vocab * rank), 1);
    const w2 = try r.buffer(u16, vocab * rank);
    defer w2.deinit();
    base.noise(12, w2.slice(u16, vocab * rank), 0.5);
    const ids = try r.buffer(u32, 2 * streams);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 2 * streams), &[_]u32{ 17, 9, vocab - 1, 9, vocab + 5, 0, 3, 3 });
    const logits = try r.buffer(u16, streams * 2 * vocab);
    defer logits.deinit();
    base.noise(14, logits.slice(u16, streams * 2 * vocab), 4);
    logits.slice(u16, streams * 2 * vocab)[3 * 2 * vocab + 700] = 0x7fc0;
    const best = try r.buffer(f32, streams * groups);
    defer best.deinit();
    const at = try r.buffer(u32, streams * groups);
    defer at.deinit();
    const picks = try r.buffer(u32, 2 * streams);
    defer picks.deinit();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(r.k.get("lfm2_markov_scan_bf16"));
    for ([_]mtl.Buffer{ logits, w1, w2, ids }, 0..) |b, i| e.setBuffer(b, 0, i);
    e.setValue([4]u32{ 2 * vocab, rank, vocab, 2 }, 4);
    for ([_]mtl.Buffer{ best, at }, 5..) |b, i| e.setBuffer(b, 0, i);
    e.dispatchGroups(mtl.Size.of(groups, streams, 1), mtl.Size.of(256, 1, 1));
    e.setPipeline(r.k.get("lfm2_markov_pick"));
    e.setBuffer(best, 0, 0);
    e.setBuffer(at, 0, 1);
    e.setValue(@as(u32, groups), 2);
    e.setBuffer(picks, 0, 3);
    e.dispatchGroups(mtl.Size.of(streams, 1, 1), mtl.Size.of(256, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    const got = picks.slice(u32, 2 * streams);
    const logit = logits.slice(u16, streams * 2 * vocab);
    for (0..streams) |s| {
        if (s == 3) {
            try std.testing.expectEqual(@as(u32, 1), got[2 * s + 1]);
            continue;
        }
        try std.testing.expectEqual(@as(u32, 0), got[2 * s + 1]);
        const id = ids.slice(u32, 2 * streams)[2 * s];
        var top: f64 = -std.math.inf(f64);
        var mine: f64 = 0;
        for (0..vocab) |v| {
            var dot: f64 = 0;
            for (0..rank) |i| {
                const e1: f64 = if (id < vocab) quant.toF32(w1.slice(u16, vocab * rank)[id * rank + i]) else 0;
                dot += @as(f64, quant.toF32(w2.slice(u16, vocab * rank)[v * rank + i])) * e1;
            }
            const score: f64 = quant.toF32(quant.toBf16(quant.toF32(logit[s * 2 * vocab + v]) + quant.toF32(quant.toBf16(@floatCast(dot)))));
            top = @max(top, score);
            if (v == got[2 * s]) mine = score;
        }
        try std.testing.expect(got[2 * s] < vocab and mine >= top - @abs(top) * 0x1p-7);
    }
}
