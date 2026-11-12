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
