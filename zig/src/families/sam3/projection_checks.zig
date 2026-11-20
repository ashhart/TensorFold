const std = @import("std");
const gpu = @import("tensor_metal");

fn repeated(g: *gpu.Gpu, x: gpu.Tensor, w: gpu.Tensor, rows: usize, cols: usize, depth: usize) !struct { tensor: gpu.Tensor, seconds: f64 } {
    const out = try g.matmul(x, w, rows, cols, depth);
    errdefer out.deinit();
    const before = g.gpu_seconds;
    for (0..5) |_| {
        const next = try g.matmul(x, w, rows, cols, depth);
        defer next.deinit();
        if (!std.mem.eql(u8, out.slice(u8, out.count * 4), next.slice(u8, next.count * 4))) return error.ProjectionNotRepeatable;
    }
    return .{ .tensor = out, .seconds = (g.gpu_seconds - before) / 5 };
}

const dense_planes = @import("dense_planes.zig");
const Candidate = enum { half_weight, compensated, expansion_gelu, packing, plain, plain_residual, plain_tiled, plain_gelu, exact, exact_residual, exact_gelu };

fn candidate(g: *gpu.Gpu, which: Candidate, x: gpu.Tensor, planes: gpu.Tensor, w: gpu.Tensor, wh: gpu.Tensor, bias: gpu.Tensor, residual: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
    return switch (which) {
        .half_weight => @import("dense_device.zig").run(g, x, wh, bias, rows, cols, depth, .nk_half),
        .compensated => @import("dense_device.zig").run(g, x, w, bias, rows, cols, depth, .nk_compensated),
        .expansion_gelu => @import("dense_expansion.zig").run(g, x, w, bias, rows, cols, depth, true),
        .packing => dense_planes.run(g, x, wh, bias, rows, cols, depth, .plain, .none),
        .plain => dense_planes.run(g, planes, wh, bias, rows, cols, depth, .plain, .none),
        .plain_residual => dense_planes.run(g, planes, wh, bias, rows, cols, depth, .plain, .{ .residual = residual }),
        .plain_tiled => dense_planes.run(g, planes, wh, bias, rows, cols, depth, .plain, .{ .tile_residual = residual }),
        .plain_gelu => dense_planes.run(g, x, wh, bias, rows, cols, depth, .plain, .gelu_planes),
        .exact => dense_planes.run(g, planes, wh, bias, rows, cols, depth, .compensated, .none),
        .exact_residual => dense_planes.run(g, planes, wh, bias, rows, cols, depth, .compensated, .{ .residual = residual }),
        .exact_gelu => dense_planes.run(g, x, wh, bias, rows, cols, depth, .compensated, .gelu_planes),
    };
}

// Times exact FP16-weight projections at the served trunk shapes and samples their FP64 error.
pub fn benchmarkPlanes(g: *gpu.Gpu) !void {
    for ([_][3]usize{ .{ 5184, 1024, 4736 }, .{ 5184, 3072, 1024 }, .{ 5184, 1024, 1024 }, .{ 5184, 4736, 1024 } }) |shape| {
        const rows, const cols, const depth = shape;
        const x = try g.alloc(rows * depth, .f32);
        defer x.deinit();
        var w = try g.alloc(cols * depth, .f32);
        defer w.deinit();
        const wh = try g.alloc(cols * depth, .f16);
        defer wh.deinit();
        const bias = try g.alloc(cols, .f32);
        defer bias.deinit();
        var state: u64 = 0x9e3779b97f4a7c15;
        for (x.slice(f32, x.count), 0..) |*value, i| {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            const unit = @as(f32, @floatFromInt(state >> 40)) / (1 << 24) - 0.5;
            value.* = if (i % 61 == 0) unit * 0x1p-20 else unit * 4;
        }
        for (w.slice(f32, w.count), wh.slice(f16, wh.count)) |*value, *half| {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            half.* = @floatCast((@as(f32, @floatFromInt(state >> 40)) / (1 << 24) - 0.5) / 16);
            value.* = half.*;
        }
        for (bias.slice(f32, bias.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 7)) / 64;
        w.tensor_operand_exact = true;
        const residual = try g.alloc(rows * cols, .f32);
        defer residual.deinit();
        @memset(residual.slice(f32, residual.count), 0);
        try g.beginBatch();
        const planes = try dense_planes.pack(g, x, rows, depth);
        defer planes.deinit();
        try g.endBatch();
        for (std.enums.values(Candidate)) |which| {
            try g.beginBatch();
            const out = try candidate(g, which, x, planes, w, wh, bias, residual, rows, cols, depth);
            defer out.deinit();
            try g.endBatch();
            var worst: f64 = 0;
            var total: f64 = 0;
            const checked = out.dtype == .f32 and which != .plain_tiled and which != .expansion_gelu;
            for (0..if (checked) 48 else 0) |sample| {
                const row = (sample * 7919) % rows;
                for (0..cols) |col| {
                    var expected: f64 = bias.get(col);
                    for (0..depth) |k| expected += @as(f64, x.get(row * depth + k)) * w.get(col * depth + k);
                    const err = @abs(expected - out.get(row * cols + col));
                    worst = @max(worst, err);
                    total += err;
                }
            }
            var best: f64 = std.math.inf(f64);
            for (0..5) |_| {
                const before = g.gpu_seconds;
                try g.beginBatch();
                var outputs: [8]gpu.Tensor = undefined;
                for (&outputs) |*o| o.* = try candidate(g, which, x, planes, w, wh, bias, residual, rows, cols, depth);
                try g.endBatch();
                for (outputs) |o| o.deinit();
                best = @min(best, (g.gpu_seconds - before) / 8);
            }
            std.debug.print("{d}x{d}x{d} {s}: {d:.3} ms, sampled FP64 max {e}, mean {e}\n", .{ rows, cols, depth, @tagName(which), best * 1e3, worst, total / @as(f64, @floatFromInt(48 * cols)) });
        }
    }
}

pub fn run(g: *gpu.Gpu) !void {
    for ([_][3]usize{ .{ 65, 129, 257 }, .{ 65, 128, 256 }, .{ 32, 129, 257 }, .{ 32, 128, 256 }, .{ 5184, 3072, 1024 }, .{ 5184, 4736, 1024 }, .{ 5184, 1024, 4736 } }) |shape| {
        const rows, const cols, const depth = shape;
        const x = try g.alloc(rows * depth, .f32);
        defer x.deinit();
        var w = try g.alloc(cols * depth, .f32);
        defer w.deinit();
        for (x.slice(f32, x.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i % 997)) * 0.037);
        for ([_]bool{ false, true }) |exact| {
            for (w.slice(f32, w.count), 0..) |*value, i| {
                const full = @cos(@as(f32, @floatFromInt(i % 991)) * 0.017) / 32;
                value.* = if (exact) @as(f16, @floatCast(full)) else full;
            }
            w.tensor_operand_exact = exact;
            g.compact_gemm = false;
            const baseline = try repeated(g, x, w, rows, cols, depth);
            defer baseline.tensor.deinit();
            g.compact_gemm = true;
            const compact = try repeated(g, x, w, rows, cols, depth);
            defer compact.tensor.deinit();
            const equal = std.mem.eql(u8, baseline.tensor.slice(u8, baseline.tensor.count * 4), compact.tensor.slice(u8, compact.tensor.count * 4));
            std.debug.print("FP32 projection {d}x{d}x{d}, exact weights {}: tile32 {d:.3} ms, tile16 {d:.3} ms, equal {}\n", .{ rows, cols, depth, exact, baseline.seconds * 1000, compact.seconds * 1000, equal });
            if (!equal) return error.ProjectionReplacementChanged;
        }
    }
    g.compact_gemm = false;
}
