//! The sibling's int8 affine kernel as ground truth: its compiled fragment layout and epilogue against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const am = tf.affine_mm;
const h = @import("operator_csa2_fp64.zig");
const math = @import("math.zig");

fn f32of(v: u16) f64 {
    const bits: u32 = @as(u32, v) << 16;
    return @floatCast(@as(f32, @bitCast(bits)));
}

fn bf16of(v: f64) u16 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// The int8 path end to end on the sibling's own compiled code: its row-sums kernel writes x's int8 codes,
// their per-TF_QG scales and the group sums, then its int8 MMA kernel dots them against the int8 codes.
// Compared against float64 of the formula the kernel implements, and against the exact affine product for
// the quantization's cost. TF_I8_GT=1.
test "int8 affine ground truth against float64 (TF_I8_GT=1)" {
    if (std.c.getenv("TF_I8_GT") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const rows: usize = 8;
    const k: usize = 256;
    const n: usize = 128;
    const group: usize = 64;
    const qg: usize = 32;
    const groups = k / group;
    var rng = h.Lcg{ .state = 0x00a8f00d };
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const x_gpu = try b.get(u16, rows * k);
    const codes = try b.get(u8, n * k);
    const scales = try b.get(u16, n * groups);
    const biases = try b.get(u16, n * groups);
    const xq = try b.get(i8, rows * k);
    const xa = try b.get(f32, rows * k / qg);
    const xs = try b.get(f32, rows * groups);
    const y = try b.get(u16, rows * n);
    for (x_gpu.slice(u16, rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    for (codes.slice(u8, n * k)) |*c| c.* = @truncate(rng.next());
    for (scales.slice(u16, n * groups)) |*s| s.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    for (biases.slice(u16, n * groups)) |*v| v.* = @bitCast(@as(f16, @floatCast(0.02 * @as(f64, rng.scale(1.0)))));

    const src = try am.source(device, a, .{ .bits = 8, .group = @intCast(group), .i8 = true, .qg = @intCast(qg) });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();

    var pipes: am.Pipes = undefined;
    for (am.names, 0..) |name, i| {
        pipes[i] = try mtl.Pipeline.init(device, lib, name, false);
    }
    defer for (pipes) |p| p.deinit();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    errdefer enc.end();
    enc.setBuffer(xq, 0, 7);
    enc.setBuffer(xa, 0, 8);
    am.rowSums(enc, pipes, @intCast(group), .{ .buf = x_gpu, .off = 0 }, .{ .buf = xs, .off = 0 }, @intCast(rows), @intCast(k));
    enc.setBuffer(xa, 0, 14);
    am.dense(enc, pipes, .{ .buf = xq, .off = 0 }, .{ .buf = xs, .off = 0 }, .{ .w = .{ .buf = codes, .off = 0 }, .s = .{ .buf = scales, .off = 0 }, .b = .{ .buf = biases, .off = 0 }, .k = @as(u32, @intCast(k)), .n = @as(u32, @intCast(n)) }, .{ .buf = y, .off = 0 }, @intCast(rows));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) {
        std.debug.print("INT8GT gpu failure\n", .{});
        return error.Int8GroundTruthGpuFailure;
    }

    const xq_v = xq.slice(i8, rows * k);
    const xa_v = xa.slice(f32, rows * k / qg);
    const xs_v = xs.slice(f32, rows * groups);
    const y_v = y.slice(u16, rows * n);
    const codes_v = codes.slice(u8, n * k);
    const sc_v = scales.slice(u16, n * groups);
    const bi_v = biases.slice(u16, n * groups);
    var unequal: usize = 0;
    var worst: f64 = 0;
    var energy: f64 = 0;
    var magnitude: f64 = 0;
    for (0..rows) |r| {
        for (0..n) |c| {
            var sum: f64 = 0;
            var ideal: f64 = 0;
            for (0..groups) |g| {
                var part: f64 = 0;
                var part_ideal: f64 = 0;
                for (0..group) |j| {
                    const i = g * group + j;
                    const x_val = f32of(x_gpu.slice(u16, rows * k)[r * k + i]);
                    part += @as(f64, xa_v[r * k / qg + i / qg]) * @as(f64, @floatFromInt(@as(i32, codes_v[c * k + i]) - 128)) * @as(f64, @floatFromInt(xq_v[r * k + i]));
                    part_ideal += @as(f64, @floatFromInt(codes_v[c * k + i])) * x_val;
                }
                const s_v = f32of(sc_v[c * groups + g]);
                const b_v = f32of(bi_v[c * groups + g]);
                sum += s_v * part + (b_v + 128.0 * s_v) * @as(f64, xs_v[r * groups + g]);
                ideal += s_v * part_ideal + b_v * @as(f64, xs_v[r * groups + g]);
            }
            const got = f32of(y_v[r * n + c]);
            if (bf16of(sum) != y_v[r * n + c]) unequal += 1;
            const scale = @max(@abs(sum), 0x1p-126);
            worst = @max(worst, @abs(got - sum) / scale);
            energy += (got - ideal) * (got - ideal);
            magnitude += ideal * ideal;
        }
    }
    std.debug.print("INT8GT rows={d} k={d} n={d} group={d} qg={d} cells={d} unequal={d} worst={d} nrms={d}\n", .{ rows, k, n, group, qg, rows * n, unequal, worst, @sqrt(energy / magnitude) });
    if (unequal != 0) return error.Int8GroundTruthMismatch;
}
