const std = @import("std");
const gpu = @import("tensor_metal");

fn call(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, split: bool) !gpu.Tensor {
    return if (split) @import("attention_split32.zig").run(g, q, k, v, shape) else @import("attention_device.zig").run(g, q, k, v, shape);
}

// Times the served FP32 attention shapes and samples their FP64 error; prints an output hash for bit comparisons.
pub fn run(g: *gpu.Gpu) !void {
    const Case = struct { shape: [5]usize, split: bool };
    for ([_]Case{
        .{ .shape = .{ 1, 8, 5184, 36544, 32 }, .split = true },
        .{ .shape = .{ 1, 16, 5184, 5184, 64 }, .split = true },
        .{ .shape = .{ 9, 16, 576, 576, 64 }, .split = false },
        .{ .shape = .{ 1, 8, 5184, 5184, 32 }, .split = true },
    }) |case| {
        const batch, const heads, const rows, const keys, const dim = case.shape;
        const q = try g.alloc(batch * rows * heads * dim, .f32);
        defer q.deinit();
        const k = try g.alloc(batch * keys * heads * dim, .f32);
        defer k.deinit();
        const v = try g.alloc(k.count, .f32);
        defer v.deinit();
        var state: u64 = 0x2545f4914f6cdd1d;
        for ([_]gpu.Tensor{ q, k, v }, 0..) |t, which| for (t.slice(f32, t.count)) |*x| {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            const unit = @as(f32, @floatFromInt(state >> 40)) / (1 << 24) - 0.5;
            x.* = unit * @as(f32, if (which == 2) 2 else 3);
        };
        try g.beginBatch();
        const out = try call(g, q, k, v, case.shape, case.split);
        defer out.deinit();
        try g.endBatch();
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(out.slice(u8, out.count * 4), &digest, .{});
        var worst: f64 = 0;
        const scores = try std.heap.page_allocator.alloc(f64, keys);
        defer std.heap.page_allocator.free(scores);
        for (0..8) |sample| {
            const b = sample % batch;
            const row = (sample * 7919) % rows;
            const h = (sample * 3) % heads;
            var maximum: f64 = -std.math.inf(f64);
            for (scores, 0..) |*s, j| {
                var dot: f64 = 0;
                for (0..dim) |d| dot += @as(f64, q.get(((b * rows + row) * heads + h) * dim + d)) * k.get(((b * keys + j) * heads + h) * dim + d);
                s.* = dot / @sqrt(@as(f64, @floatFromInt(dim)));
                maximum = @max(maximum, s.*);
            }
            var total: f64 = 0;
            for (scores) |*s| {
                s.* = @exp(s.* - maximum);
                total += s.*;
            }
            for (0..dim) |d| {
                var sum: f64 = 0;
                for (scores, 0..) |s, j| sum += s * v.get(((b * keys + j) * heads + h) * dim + d);
                worst = @max(worst, @abs(sum / total - out.get(((b * rows + row) * heads + h) * dim + d)));
            }
        }
        var best: f64 = std.math.inf(f64);
        for (0..4) |_| {
            const before = g.gpu_seconds;
            try g.beginBatch();
            var outputs: [2]gpu.Tensor = undefined;
            for (&outputs) |*o| o.* = try call(g, q, k, v, case.shape, case.split);
            try g.endBatch();
            for (outputs) |o| o.deinit();
            best = @min(best, (g.gpu_seconds - before) / 2);
        }
        std.debug.print("attention {any}: {d:.3} ms, sampled FP64 max {e}, output {x}\n", .{ case.shape, best * 1e3, worst, digest[0..8] });
    }
}
