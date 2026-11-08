//! Actual checkpoint projections on identical deterministic BF16 rows, against independent FP64 dot products.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const projection = @import("projection.zig");
const math = @import("math.zig");
const Model = tf.deepseek_v41.Model;
const ex = tf.exl3_projection;
const Case = struct { name: []const u8, matrix: ex.Matrix };

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, dir: []const u8) !void {
    const first = &model.weights.layers[0];
    const cases = [_]Case{
        .{ .name = "wq_a", .matrix = first.attn.wq_a },
        .{ .name = "wq_b", .matrix = first.attn.wq_b },
        .{ .name = "wkv", .matrix = first.attn.wkv },
        .{ .name = "wo_a_group0", .matrix = first.attn.wo_a[0] },
        .{ .name = "wo_b", .matrix = first.attn.wo_b },
        .{ .name = "engram_wkv", .matrix = model.weights.layers[1].engram.?.wkv },
        .{ .name = "routed_gate", .matrix = try tf.deepseek_v41_weights.lookup(&model.checkpoint, "layers.0.ffn.experts.0.w1", 5120, 2304) },
        .{ .name = "routed_up", .matrix = try tf.deepseek_v41_weights.lookup(&model.checkpoint, "layers.0.ffn.experts.0.w3", 5120, 2304) },
        .{ .name = "routed_down", .matrix = try tf.deepseek_v41_weights.lookup(&model.checkpoint, "layers.0.ffn.experts.0.w2", 2304, 5120) },
        .{ .name = "head", .matrix = model.weights.head },
    };
    const x = try model.device.buffer(8192 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    const y = try model.device.buffer(2 * 129280 * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try model.device.buffer(try ex.scratchBytes(8192, 129280, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    for (x.slice(u16, 8192), 0..) |*value, i|
        value.* = math.b(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 97)) - 48)) / 32);
    for ([_]mtl.Buffer{ x, y, scratch }) |buffer| model.residency.add(buffer);
    model.residency.commit();
    model.residency.requestResidency();
    defer {
        for ([_]mtl.Buffer{ x, y, scratch }) |buffer| model.residency.remove(buffer);
        model.residency.commit();
    }
    var reference = try projection.Scratch.init(gpa, 8192, 129280);
    defer reference.deinit();
    const expected = try gpa.alloc(f32, 129280);
    defer gpa.free(expected);
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "{\"input\":\"bf16((17*i%97-48)/32)\",\"reference_dot\":\"FP64\",\"cases\":[");
    var all_pass = true;
    for (cases, 0..) |case, index| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const matrix = case.matrix;
        const cb = model.queue.commandBufferUnretained();
        const enc = cb.compute(.serial);
        var encoding = true;
        errdefer if (encoding) enc.end();
        ex.useMatrix(enc, matrix);
        try ex.encode(&model.pipelines.projection, enc, matrix, .{ .buffer = x }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = matrix.k, .y_stride = matrix.n, .output = .f32 });
        enc.end();
        encoding = false;
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.OperatorGpuFailure;
        const actual = y.slice(f32, 129280);
        try projection.applyF32(.{ .trellis = matrix }, x.slice(u16, matrix.k), expected[0..matrix.n], &reference);
        var max_abs: f64 = 0;
        var worst: usize = 0;
        var squared: f64 = 0;
        var magnitude: f64 = 0;
        for (actual[0..matrix.n], expected[0..matrix.n], 0..) |a, b, column| {
            if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonfiniteOperatorOutput;
            const delta: f64 = @as(f64, a) - b;
            if (@abs(delta) > max_abs) {
                max_abs = @abs(delta);
                worst = column;
            }
            squared += delta * delta;
            magnitude += @as(f64, b) * b;
        }
        const nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
        all_pass = all_pass and nrms <= 1e-4;
        const line = try std.fmt.allocPrint(gpa, "{{\"name\":\"{s}\",\"k\":{d},\"n\":{d},\"normalized_rms_error\":{d},\"max_absolute_error\":{d},\"worst_column\":{d},\"worst_gpu\":{d},\"worst_reference\":{d}}}", .{ case.name, matrix.k, matrix.n, nrms, max_abs, worst, actual[worst], expected[worst] });
        defer gpa.free(line);
        if (index > 0) try report.appendSlice(gpa, ",");
        try report.appendSlice(gpa, line);
        std.debug.print("{s}\n", .{line});
    }
    try report.appendSlice(gpa, "]}\n");
    const path = try std.fs.path.join(gpa, &.{ dir, "operators.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
    if (!all_pass) return error.OperatorReferenceMismatch;
}
