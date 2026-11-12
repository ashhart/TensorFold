const std = @import("std");
const gpu = @import("tensor_metal");
const ops = gpu.ops;
const Tensor = gpu.Tensor;
const Size = gpu.mtl.Size;
const Runtime = @import("ops.zig");
const Rope = Runtime.Rope;
const Conv = Runtime.Conv;
const flash = Runtime.flash;

pub fn check(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print("tf-sam3-kernels [--benchmark]\ntf-sam3-kernels --replay-attention Q.safetensors K.safetensors V.safetensors [ROW]\n", .{});
        return;
    }
    const replay = args.len >= 2 and std.mem.eql(u8, args[1], "--replay-attention");
    if (!replay and args.len != 1 and !(args.len == 2 and std.mem.eql(u8, args[1], "--benchmark"))) return error.UnexpectedSamKernelArgument;
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var g = try gpu.Gpu.init(init.gpa, @import("sam3_kernels").source);
    defer g.deinit();
    g.guards = true;
    g.buffer_cache_limit = 32 << 20;
    if (replay) return @import("attention_replay.zig").run(init, &g, args[2..]);
    if (args.len == 2) return benchmark(&g);
    try checkExactWeightProducts(&g);
    for ([_]usize{ 32, 256, 4736 }) |depth| {
        const x = try g.alloc(64 * depth, .f16);
        defer x.deinit();
        const w = try g.alloc(128 * depth, .f16);
        defer w.deinit();
        const bias = try g.alloc(128, .f32);
        defer bias.deinit();
        for (x.slice(f16, x.count), 0..) |*v, i| v.* = @floatCast((@as(f32, @floatFromInt(i % 97)) - 48) / 128);
        for (w.slice(f16, w.count), 0..) |*v, i| v.* = @floatCast((@as(f32, @floatFromInt(i % 89)) - 44) / 256);
        for (bias.slice(f32, bias.count), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 256;
        const out = try g.allocDevice(64 * 128, .f32);
        defer out.deinit();
        const active = try g.allocDevice(out.count, .f16);
        defer active.deinit();
        try g.beginBatch();
        errdefer if (g.batch != null) g.endBatch() catch {};
        for ([_]Tensor{ out, active }, 0..) |dst, activation| try g.dispatch(if (activation == 0) "im_sam_vision_linear" else "im_sam_vision_gelu", &.{ x, w, bias, dst }, [4]u32{ 64, 128, @intCast(depth), @intCast(activation) }, Size.of(32, 4, 2), Size.of(32, 4, 2));
        try g.endBatch();
        const expected_active = try ops.gelu(&g, out, true);
        defer expected_active.deinit();
        var maximum: f64 = 0;
        for (0..64) |row| for (0..128) |col| {
            var expected: f64 = bias.get(col);
            for (0..depth) |k| expected += @as(f64, x.get(row * depth + k)) * w.get(col * depth + k);
            maximum = @max(maximum, @abs(expected - out.get(row * 128 + col)));
            if (active.get(row * 128 + col) != @as(f32, @floatCast(@as(f16, @floatCast(expected_active.get(row * 128 + col)))))) return error.SamGeluParity;
        };
        std.debug.print("SAM FP16 projection K={d}: maximum float64 oracle error {e}\n", .{ depth, maximum });
        if (maximum > 1e-4) return error.SamProjectionParity;
    }
    const q = try g.alloc(65 * 2 * 32, .f32);
    defer q.deinit();
    const k = try g.alloc(97 * 2 * 32, .f32);
    defer k.deinit();
    const v = try g.alloc(k.count, .f32);
    defer v.deinit();
    for (q.slice(f32, q.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.03);
    for (k.slice(f32, k.count), v.slice(f32, v.count), 0..) |*key, *value, i| {
        key.* = @cos(@as(f32, @floatFromInt(i)) * 0.07);
        value.* = @sin(@as(f32, @floatFromInt(i)) * 0.13);
    }
    const qp = try ops.headsCopy(&g, q, 65, 2, 32, 64, 64, 0);
    defer qp.deinit();
    const kp = try ops.headsCopy(&g, k, 97, 2, 32, 64, 64, 0);
    defer kp.deinit();
    const vp = try ops.headsCopy(&g, v, 97, 2, 32, 64, 64, 0);
    defer vp.deinit();
    const actual = try ops.attentionCross(&g, q, k, v, 65, 97, 2, 1 / @sqrt(@as(f32, 32)));
    defer actual.deinit();
    const padded = try ops.attentionCross(&g, qp, kp, vp, 65, 97, 2, 1 / @sqrt(@as(f32, 32)));
    defer padded.deinit();
    const expected = try ops.headsCopy(&g, padded, 65, 2, 64, 32, 128, 0);
    defer expected.deinit();
    var maximum: f32 = 0;
    for (actual.slice(f32, actual.count), expected.slice(f32, expected.count)) |x, y| maximum = @max(maximum, @abs(x - y));
    std.debug.print("SAM 32-wide attention: maximum padded-reference error {e}\n", .{maximum});
    if (maximum > 1e-5) return error.SamAttentionParity;
    const table = try g.allocDevice(72 * 72 * 32, .f32);
    defer table.deinit();
    const rotated = try g.allocDevice(q.count, .f32);
    defer rotated.deinit();
    const original = try g.allocDevice(q.count, .f32);
    defer original.deinit();
    const rope_ = Rope{ .rows = 65, .heads = 2, .dim = 32, .spatial = 64, .side = 72, .scale = 1 };
    try g.dispatch("im_sam_rope_table", &.{table}, rope_, Size.of(table.count / 2, 1, 1), Size.of(256, 1, 1));
    try g.dispatch("im_sam_rope", &.{ q, original }, rope_, Size.of(q.count, 1, 1), Size.of(256, 1, 1));
    try g.dispatch("im_sam_rope_cached", &.{ q, table, rotated }, rope_, Size.of(q.count, 1, 1), Size.of(256, 1, 1));
    if (!std.mem.eql(u8, original.slice(u8, q.count * 4), rotated.slice(u8, q.count * 4))) return error.SamRotaryParity;
    const input = try g.alloc(3 * 5 * 32, .f32);
    defer input.deinit();
    const raw = try g.alloc(32 * 16 * 4, .f32);
    defer raw.deinit();
    const bias = try g.alloc(16, .f32);
    defer bias.deinit();
    for (input.slice(f32, input.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.03);
    for (raw.slice(f32, raw.count), 0..) |*value, i| value.* = @cos(@as(f32, @floatFromInt(i)) * 0.07) * 0.03;
    for (bias.slice(f32, bias.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 256;
    const matrix = try g.allocDevice(raw.count, .f32);
    defer matrix.deinit();
    const product = try g.allocDevice(6 * 10 * 16, .f32);
    defer product.deinit();
    const shuffled = try g.allocDevice(product.count, .f32);
    defer shuffled.deinit();
    const reference = try g.allocDevice(product.count, .f32);
    defer reference.deinit();
    const conv_ = Conv{ .h = 3, .w = 5, .c = 32, .n = 16, .k = 2, .oh = 6, .ow = 10, .stride = 2, .pad = 0, .first = 0, .rows = 0 };
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    try g.dispatch("im_sam_deconv_weight", &.{ raw, matrix }, [3]u32{ 16, 32, 4 }, Size.of(raw.count, 1, 1), Size.of(256, 1, 1));
    try g.matmulInto(input, matrix, product, 15, 64, 32);
    try g.dispatch("im_sam_deconv_shuffle", &.{ product, bias, shuffled }, conv_, Size.of(shuffled.count, 1, 1), Size.of(256, 1, 1));
    try g.dispatch("im_sam_deconv", &.{ input, raw, bias, reference }, conv_, Size.of(reference.count / 4, 1, 1), Size.of(256, 1, 1));
    try g.endBatch();
    maximum = 0;
    for (shuffled.slice(f32, shuffled.count), reference.slice(f32, reference.count)) |x, y| maximum = @max(maximum, @abs(x - y));
    std.debug.print("SAM cached rotary is byte-identical; transposed convolution maximum error {e}\n", .{maximum});
    if (maximum > 1e-4) return error.SamDeconvolutionParity;
    try checkFlash(&g);
    try @import("attention_replay.zig").checkUniform(&g);
    try checkSparseMask(&g);
    try checkVisionResidual(&g);
    try checkMemoryMasks(&g);
}

fn checkExactWeightProducts(g: *gpu.Gpu) !void {
    for ([_][3]usize{ .{ 65, 129, 257 }, .{ 65, 128, 256 }, .{ 64, 129, 257 }, .{ 64, 128, 4736 } }) |shape| {
        const rows, const cols, const depth = shape;
        const x = try g.alloc(rows * depth, .f32);
        defer x.deinit();
        const w = try g.alloc(cols * depth, .f32);
        defer w.deinit();
        for (x.slice(f32, x.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i % 997)) * 0.037);
        for (w.slice(f32, w.count), 0..) |*value, i| value.* = @as(f16, @floatCast(@cos(@as(f32, @floatFromInt(i % 991)) * 0.017)));
        const baseline = try g.matmul(x, w, rows, cols, depth);
        defer baseline.deinit();
        var exact = w;
        exact.tensor_operand_exact = true;
        const actual = try g.matmul(x, exact, rows, cols, depth);
        defer actual.deinit();
        if (!std.mem.eql(u8, baseline.slice(u8, baseline.count * 4), actual.slice(u8, actual.count * 4))) return error.SamExactWeightParity;
        std.debug.print("SAM exact-weight FP32 projection {d}x{d}x{d}: byte-identical to full reconstruction\n", .{ rows, cols, depth });
    }
}

fn checkMemoryMasks(g: *gpu.Gpu) !void {
    const Params = @import("memory.zig").MaskParams;
    for ([_]usize{ 1, 32 }) |channels| {
        const input_channels: usize = if (channels == 1) 1 else 16;
        const x = try g.alloc(7 * input_channels, .f32);
        defer x.deinit();
        const out = try g.allocDevice(7 * channels, .f32);
        defer out.deinit();
        for (x.slice(f32, x.count), 0..) |*value, i|
            value.* = (@as(f32, @floatFromInt(i % 29)) - 14) / 3;
        for ([_]u16{ 1, 0x8002, 0xffff }) |active| for (0..2) |hard| {
            if (channels == 1 and active != 1) continue;
            const scale: f32 = if (channels == 1) 20 else 2;
            const bias: f32 = if (channels == 1) -10 else -1;
            try g.dispatch("im_sam_memory_masks", &.{ x, out }, Params{
                .pixels = 7,
                .channels = @intCast(channels),
                .active = active,
                .conditioning = active & 0x8001,
                .hard = @intCast(hard),
                .scale = scale,
                .bias = bias,
            }, Size.of(out.count, 1, 1), Size.of(256, 1, 1));
            for (0..out.count) |i| {
                const c = i % channels;
                const slot = c % 16;
                const bit = @as(u16, 1) << @intCast(slot);
                const value: f64 = x.get(i / channels * input_channels + slot);
                const probability: f64 = if (hard != 0) @floatFromInt(@intFromBool(value > 0)) else 1 / (1 + @exp(-value));
                const expected: f64 = if (active & bit == 0) 0 else if (c >= 16)
                    @floatFromInt(@intFromBool(active & 0x8001 & bit != 0))
                else
                    probability * scale + bias;
                if (@abs(expected - out.get(i)) > 4e-6) return error.SamMemoryMaskParity;
            }
        };
    }
    std.debug.print("SAM memory masks: every slot, inactive channels, conditioning and hard masks match float64\n", .{});
}

fn checkVisionResidual(g: *gpu.Gpu) !void {
    const rows = 5184;
    const cols = 128;
    const depth = 32;
    const x = try g.alloc(rows * depth, .f16);
    defer x.deinit();
    const w = try g.alloc(cols * depth, .f16);
    defer w.deinit();
    const bias = try g.alloc(cols, .f32);
    defer bias.deinit();
    const residual = try g.alloc(rows * cols, .f32);
    defer residual.deinit();
    for (x.slice(f16, x.count), 0..) |*value, i| value.* = @floatCast(@sin(@as(f32, @floatFromInt(i)) * 0.017));
    for (w.slice(f16, w.count), 0..) |*value, i| value.* = @floatCast(@cos(@as(f32, @floatFromInt(i)) * 0.037) * 0.1);
    for (bias.slice(f32, bias.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 256;
    for (residual.slice(f32, residual.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.013);
    const projected = try g.allocDevice(residual.count, .f32);
    defer projected.deinit();
    const actual = try g.allocDevice(residual.count, .f32);
    defer actual.deinit();
    try g.dispatch("im_sam_vision_linear", &.{ x, w, bias, projected }, [4]u32{ rows, cols, depth, 0 }, Size.of(rows / 64 * 32, 4, 2), Size.of(32, 4, 2));
    for (0..2) |windowed| {
        try g.dispatch("im_sam_vision_residual", &.{ x, w, bias, actual, residual }, [4]u32{ rows, cols, depth, @intCast(windowed) }, Size.of(rows / 64 * 32, 4, 2), Size.of(32, 4, 2));
        for (0..rows) |row| {
            const win = row / 576;
            const local = row % 576;
            const dest = if (windowed == 1) (win / 3 * 24 + local / 24) * 72 + win % 3 * 24 + local % 24 else row;
            for (0..cols) |col| {
                const expected = residual.get(dest * cols + col) + projected.get(row * cols + col);
                if (actual.get(dest * cols + col) != expected) return error.SamResidualParity;
            }
        }
    }
    std.debug.print("SAM fused residual and all nine window tiles are byte-identical\n", .{});
}

fn checkFlash(g: *gpu.Gpu) !void {
    const rows = 65;
    const keys = 97;
    const heads = 2;
    for ([_]usize{ 32, 64, 128, 256 }) |dim| for ([_]gpu.DType{ .f32, .f16 }) |dtype| {
        if (dim > 64 and dtype == .f16) continue;
        const batch: usize = if (dim > 64) 1 else 2;
        const q = try g.alloc(rows * heads * dim * batch, dtype);
        defer q.deinit();
        const k = try g.alloc(keys * heads * dim * batch, dtype);
        defer k.deinit();
        const v = try g.alloc(k.count, dtype);
        defer v.deinit();
        for ([_]Tensor{ q, k, v }, 0..) |tensor, index| for (0..tensor.count) |i| {
            const value = @sin(@as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(index + 3)) * 0.017);
            if (dtype == .f32) tensor.slice(f32, tensor.count)[i] = value else tensor.slice(f16, tensor.count)[i] = @floatCast(value);
        };
        const actual = if (dim > 64)
            try ops.attentionCross(g, q, k, v, rows, keys, heads, 1 / @sqrt(@as(f32, @floatFromInt(dim))))
        else
            try flash(g, q, k, v, rows, keys, heads, batch);
        defer actual.deinit();
        try actual.checkFinite("SAM attention");
        var maximum: f64 = 0;
        for (0..batch) |b| for (0..heads) |h| for (0..rows) |row| {
            var scores: [keys]f64 = undefined;
            var top = -std.math.inf(f64);
            for (&scores, 0..) |*score, key| {
                score.* = 0;
                for (0..dim) |d| score.* += @as(f64, q.get(((b * rows + row) * heads + h) * dim + d)) * k.get(((b * keys + key) * heads + h) * dim + d);
                score.* /= @sqrt(@as(f64, @floatFromInt(dim)));
                top = @max(top, score.*);
            }
            var total: f64 = 0;
            for (&scores) |*score| {
                score.* = @exp(score.* - top);
                total += score.*;
            }
            for (0..dim) |d| {
                var value: f64 = 0;
                for (scores, 0..) |score, key| value += score * v.get(((b * keys + key) * heads + h) * dim + d);
                maximum = @max(maximum, @abs(value / total - actual.get(((b * rows + row) * heads + h) * dim + d)));
            }
        };
        std.debug.print("SAM {s} attention D={d}: maximum float64 oracle error {e}\n", .{ @tagName(dtype), dim, maximum });
        if (maximum > (if (dtype == .f32) @as(f64, 3e-4) else @as(f64, 1e-3))) return error.SamAttentionParity;
    };
}

fn checkSparseMask(g: *gpu.Gpu) !void {
    const x = try g.alloc(1152 * 1152, .f32);
    defer x.deinit();
    const w = try g.alloc(16 * 32 * 9, .f32);
    defer w.deinit();
    const bias = try g.alloc(16, .f32);
    defer bias.deinit();
    const out = try g.allocDevice(576 * 576 * 16, .f32);
    defer out.deinit();
    for (x.slice(f32, x.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.017);
    for (w.slice(f32, w.count), 0..) |*value, i| value.* = @as(f32, @floatCast(@as(f16, @floatCast(@cos(@as(f32, @floatFromInt(i)) * 0.037) * 0.1))));
    for (bias.slice(f32, bias.count), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 256;
    var maximum: f64 = 0;
    for (0..2) |conditioning| {
        try g.dispatch("im_sam_mask_downsample", &.{ x, w, bias, out }, @as(u32, @intCast(conditioning)), Size.of(out.count, 1, 1), Size.of(256, 1, 1));
        for ([_]usize{ 0, 1, 575, 576, 577, 1151, 576 * 288 + 288, 576 * 576 - 1 }) |pixel| for (0..16) |c| {
            var expected: f64 = bias.get(c);
            for (0..3) |ky| for (0..3) |kx| {
                const y = @as(isize, @intCast(pixel / 576 * 2 + ky)) - 1;
                const z = @as(isize, @intCast(pixel % 576 * 2 + kx)) - 1;
                if (y < 0 or z < 0 or y >= 1152 or z >= 1152) continue;
                expected += @as(f64, x.get(@intCast(y * 1152 + z))) * w.get(c * 32 * 9 + ky * 3 + kx);
                if (conditioning == 1) expected += w.get((c * 32 + 16) * 9 + ky * 3 + kx);
            };
            maximum = @max(maximum, @abs(expected - out.get(pixel * 16 + c)));
        };
    }
    std.debug.print("SAM sparse mask convolution: maximum float64 oracle error {e}\n", .{maximum});
    if (maximum > 1e-6) return error.SamConvolutionParity;
}

fn benchmark(g: *gpu.Gpu) !void {
    for ([_][5]usize{ .{ 65, 97, 2, 32, 1 }, .{ 576, 576, 16, 64, 9 }, .{ 5184, 5184, 16, 64, 1 }, .{ 5184, 36544, 8, 32, 1 } }) |shape| {
        const rows, const keys, const heads, const dim, const batch = shape;
        const q = try g.alloc(rows * heads * dim * batch, .f32);
        defer q.deinit();
        const k = try g.alloc(keys * heads * dim * batch, .f32);
        defer k.deinit();
        const v = try g.alloc(k.count, .f32);
        defer v.deinit();
        for (q.slice(f32, q.count), 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i % 10007)) * 0.03);
        for (k.slice(f32, k.count), v.slice(f32, v.count), 0..) |*x, *y, i| {
            x.* = @cos(@as(f32, @floatFromInt(i % 10009)) * 0.07);
            y.* = @sin(@as(f32, @floatFromInt(i % 10037)) * 0.13);
        }
        for ([_]gpu.DType{ .f32, .f16 }) |dtype| {
            const aq = if (dtype == .f32) q else try ops.cast(g, q, dtype);
            defer if (dtype != .f32) aq.deinit();
            const ak = if (dtype == .f32) k else try ops.cast(g, k, dtype);
            defer if (dtype != .f32) ak.deinit();
            const av = if (dtype == .f32) v else try ops.cast(g, v, dtype);
            defer if (dtype != .f32) av.deinit();
            var reference: ?Tensor = null;
            defer if (reference) |t| t.deinit();
            for ([_]bool{ false, true }) |specialized| {
                var fastest = std.math.inf(f64);
                for (0..4) |run| {
                    const before = g.gpu_seconds;
                    try g.beginBatch();
                    errdefer if (g.batch != null) g.endBatch() catch {};
                    var actual: ?Tensor = null;
                    for (0..4) |_| {
                        if (actual) |t| t.deinit();
                        const scale_ = 1 / @sqrt(@as(f32, @floatFromInt(dim)));
                        actual = if (specialized) try flash(g, aq, ak, av, rows, keys, heads, batch) else if (batch == 1) try ops.attentionCross(g, aq, ak, av, rows, keys, heads, scale_) else try ops.attentionBatched(g, aq, ak, av, rows, heads, heads, false, scale_, batch);
                    }
                    try g.endBatch();
                    if (run != 0) fastest = @min(fastest, (g.gpu_seconds - before) / 4);
                    if (run == 3) {
                        var maximum: f32 = 0;
                        if (!specialized) {
                            reference = actual;
                        } else {
                            for (0..q.count) |i| maximum = @max(maximum, @abs(actual.?.get(i) - reference.?.get(i)));
                            actual.?.deinit();
                        }
                        std.debug.print("SAM attention {s} {d}x{d} h{d} d{d} b{d}, specialized {}: {d:.3} ms, error {e}\n", .{ @tagName(dtype), rows, keys, heads, dim, batch, specialized, fastest * 1000, maximum });
                        if (maximum > (if (dtype == .f32) @as(f32, 1e-4) else @as(f32, 1e-3))) return error.SamAttentionParity;
                    } else actual.?.deinit();
                }
            }
        }
    }
}
