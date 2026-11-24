const std = @import("std");
extern "c" fn erfc(f64) f64;
const gpu = @import("tensor_metal");
const ops = gpu.ops;
const Tensor = gpu.Tensor;
const Size = gpu.mtl.Size;
const abi = @import("kernel_abi.zig");
const flash = @import("attention.zig").flash;
const gridRow = @import("spatial.zig").gridRow;
const operator_checks = @import("operator_checks.zig");

fn checkGeluTails(g: *gpu.Gpu) !void {
    const values = [_]f32{ -8, -7, -6.5, -6, -5.5, -5.25, -5, -4.5, -4, -3, -2, -1, -0.5 };
    const input = try g.alloc(values.len, .f32);
    defer input.deinit();
    @memcpy(input.slice(f32, values.len), &values);
    const output = try g.allocDevice(values.len, .f32);
    defer output.deinit();
    try g.dispatch("im_sam_gelu", &.{ input, output }, @as(u32, @intCast(values.len)), Size.of(values.len, 1, 1), Size.of(256, 1, 1));
    var maximum: f64 = 0;
    for (values, 0..) |value, index| {
        const x: f64 = value;
        const reference = 0.5 * x * erfc(-x / @sqrt(@as(f64, 2)));
        const error_ = @abs(reference - output.get(index));
        const bound = 4 * std.math.floatEps(f32) * @abs(reference);
        if (!std.math.isFinite(output.get(index)) or error_ > bound) {
            std.debug.print("SAM GELU at {d}: got {e}, FP64 {e}, error {e}, bound {e}\n", .{ value, output.get(index), reference, error_, bound });
            return error.SamGeluTailPrecision;
        }
        maximum = @max(maximum, error_);
    }
    std.debug.print("SAM GELU negative tails: {d} FP64 witnesses pass, maximum absolute error {e}\n", .{ values.len, maximum });
}

fn checkProjectionDomains(g: *gpu.Gpu) !void {
    const x = try g.alloc(16, .f32);
    defer x.deinit();
    const bias = try g.alloc(16, .f32);
    defer bias.deinit();
    const domain = try g.alloc(7, .f32);
    defer domain.deinit();
    @memset(bias.slice(f32, bias.count), 0);
    for ([_]gpu.DType{ .f16, .f32 }) |dtype| {
        const weight = try g.alloc(16 * 16, dtype);
        defer weight.deinit();
        for (0..weight.count) |i| {
            const value: f32 = if (i % 3 == 0) 0x1p-24 else 1;
            if (dtype == .f16) weight.slice(f16, weight.count)[i] = @floatCast(value) else weight.slice(f32, weight.count)[i] = value;
        }
        for ([_]f32{ 0x1p-79, 0x1p-80, 0x1p-56, std.math.inf(f32) }, 0..) |value, index| {
            @memset(x.slice(f32, x.count), value);
            @memset(domain.slice(u32, domain.count), 0);
            try g.dispatch(if (dtype == .f16) "sam_dense_half_domain" else "sam_dense_device_domain", &.{ x, weight, bias, domain }, [3]u32{ 1, 16, 16 }, Size.of(512, 1, 1), Size.of(128, 1, 1));
            try g.dispatch("sam_dense_device_plan", &.{domain}, [3]u32{ 1, 16, 16 }, Size.of(1, 1, 1), Size.of(128, 1, 1));
            const fallback = domain.slice(u32, domain.count)[0] != 0;
            if (fallback != (index == 1 or index == 3)) {
                std.debug.print("SAM {s} projection domain at {e}: fallback {}\n", .{ @tagName(dtype), value, fallback });
                return error.SamProjectionDomain;
            }
        }
        if (dtype == .f32) {
            @memset(x.slice(f32, x.count), 0x1p-56);
            weight.slice(f32, weight.count)[0] = 1 + 0x1p-20;
            @memset(domain.slice(u32, domain.count), 0);
            try g.dispatch("sam_dense_device_domain", &.{ x, weight, bias, domain }, [3]u32{ 1, 16, 16 }, Size.of(512, 1, 1), Size.of(128, 1, 1));
            try g.dispatch("sam_dense_device_plan", &.{domain}, [3]u32{ 1, 16, 16 }, Size.of(1, 1, 1), Size.of(128, 1, 1));
            if (domain.slice(u32, domain.count)[0] != 1) return error.SamProjectionDomain;
        }
    }
    std.debug.print("SAM projection domains: exact half coefficients admit normal-product tails and retain fallbacks\n", .{});
}

fn checkExpansionTinyInput(g: *gpu.Gpu) !void {
    const rows = 64;
    const cols = 128;
    const depth = 256;
    const x = try g.alloc(rows * depth, .f32);
    defer x.deinit();
    const w = try g.alloc(cols * depth, .f32);
    defer w.deinit();
    const bias = try g.alloc(cols, .f32);
    defer bias.deinit();
    for (x.slice(f32, x.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i % 997)) * 0.037);
    for (w.slice(f32, w.count), 0..) |*value, i| value.* = @as(f16, @floatCast(@cos(@as(f32, @floatFromInt(i % 991)) * 0.017)));
    for (bias.slice(f32, cols), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 256;
    x.slice(f32, x.count)[5 * depth + 17] = 0x1p-40;
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const actual = try @import("dense_expansion.zig").run(g, x, w, bias, rows, cols, depth, false);
    defer actual.deinit();
    try g.endBatch();
    var worst: f64 = 0;
    for (0..rows) |row| for (0..cols) |col| {
        var expected: f64 = bias.get(col);
        var magnitude: f64 = @abs(expected);
        for (0..depth) |k| {
            const term = @as(f64, x.get(row * depth + k)) * w.get(col * depth + k);
            expected += term;
            magnitude += @abs(term);
        }
        worst = @max(worst, @abs(expected - actual.get(row * cols + col)) / (8 * std.math.floatEps(f32) * magnitude));
    };
    std.debug.print("SAM exact expansion with one input below 2^-36: worst float64 oracle error {d:.3} of its 8-epsilon magnitude bound\n", .{worst});
    if (!(worst <= 1)) return error.SamExpansionTinyInput;
}

pub fn check(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print("tf-sam3-kernels [--benchmark | --benchmark-projections]\ntf-sam3-kernels --replay-attention[-two-pass] Q.safetensors K.safetensors V.safetensors [ROW]\ntf-sam3-kernels --replay-attention-operator CAPTURE.safetensors RECEIPT.json\ntf-sam3-kernels --replay-projection CAPTURE.safetensors RECEIPT.json [--baseline-gemm | --mps-ndarray | --compensated | --compensated-baseline]\ntf-sam3-kernels --check-mps-ndarray RECEIPT.json\n", .{});
        std.debug.print("tf-sam3-kernels --replay-text CHECKPOINT REFERENCE.safetensors RECEIPT.json [--compensated-mlp]\n", .{});
        std.debug.print("tf-sam3-kernels --replay-decoder CHECKPOINT REFERENCE.safetensors RECEIPT.json\ntf-sam3-kernels --replay-normalization CAPTURE.safetensors RECEIPT.json [--compensated]\nProjection backends also accept --tensor-device[-compensated], --tensor-device-nn[-compensated], --tensor-device-gelu[-baseline], --tensor-planes[-gelu][-compensated] and --tensor-wide[-compensated | -gelu].\ntf-sam3-kernels --replay-trunk CHECKPOINT TRACE_DIR RECEIPT.json [ROLE=LAYOUT | only=TEXT]... | --calibrate-trunk CAPTURE.safetensors\ntf-sam3-kernels --benchmark-planes | --benchmark-device-attention\nAttention operator backends accept --split-keys, --tensor32, --tensor64, --split32 and --split64.\n", .{});
        return;
    }
    const text_replay = (args.len == 5 or args.len == 6) and std.mem.eql(u8, args[1], "--replay-text");
    const decoder_replay = args.len == 5 and std.mem.eql(u8, args[1], "--replay-decoder");
    const two_pass = args.len >= 2 and std.mem.eql(u8, args[1], "--replay-attention-two-pass");
    const replay = two_pass or (args.len >= 2 and std.mem.eql(u8, args[1], "--replay-attention"));
    const projections = args.len == 2 and std.mem.eql(u8, args[1], "--benchmark-projections");
    const planes_benchmark = args.len == 2 and std.mem.eql(u8, args[1], "--benchmark-planes");
    const attention_benchmark = args.len == 2 and std.mem.eql(u8, args[1], "--benchmark-device-attention");
    const int8_benchmark = args.len == 2 and std.mem.eql(u8, args[1], "--benchmark-int8-nax");
    const projection_replay = args.len >= 2 and std.mem.eql(u8, args[1], "--replay-projection");
    const operator_replay = args.len >= 2 and std.mem.eql(u8, args[1], "--replay-attention-operator");
    const mps_check = args.len == 3 and std.mem.eql(u8, args[1], "--check-mps-ndarray");
    const trunk_replay = args.len >= 5 and std.mem.eql(u8, args[1], "--replay-trunk");
    const trunk_calibration = args.len == 3 and std.mem.eql(u8, args[1], "--calibrate-trunk");
    const normalization_replay = (args.len == 4 or args.len == 5) and std.mem.eql(u8, args[1], "--replay-normalization");
    if (!trunk_replay and !trunk_calibration and !normalization_replay and !decoder_replay and !text_replay and !replay and !projections and !planes_benchmark and !attention_benchmark and !int8_benchmark and !projection_replay and !operator_replay and !mps_check and args.len != 1 and !(args.len == 2 and std.mem.eql(u8, args[1], "--benchmark"))) return error.UnexpectedSamKernelArgument;
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const small_baseline = projection_replay and args.len == 5 and std.mem.eql(u8, args[4], "--compensated-baseline");
    var g = try gpu.Gpu.init(init.gpa, if (small_baseline) @import("sam3_kernels").baseline_source else @import("sam3_kernels").source);
    defer g.deinit();
    g.guards = true;
    g.buffer_cache_limit = 32 << 20;
    if (trunk_replay) return @import("trunk_replay.zig").run(init, &g, args[2..]);
    if (trunk_calibration) return @import("trunk_replay.zig").calibrate(init, &g, args[2]);
    if (normalization_replay) return @import("normalization_replay.zig").run(init, &g, args[2..]);
    if (decoder_replay) return @import("decoder_replay.zig").run(init, &g, args[2..]);
    if (text_replay) return @import("text_replay.zig").run(init, &g, args[2..]);
    if (projections) return @import("projection_checks.zig").run(&g);
    if (planes_benchmark) return @import("projection_checks.zig").benchmarkPlanes(&g);
    if (attention_benchmark) return @import("attention_bench.zig").run(&g);
    if (int8_benchmark) return @import("wide_checks.zig").benchmarkInt8Nax(&g);
    if (mps_check) return @import("mps_ndarray_probe.zig").check(init, &g, args[2]);
    if (projection_replay) return @import("projection_replay.zig").run(init, &g, args[2..]);
    if (operator_replay) return @import("attention_operator_replay.zig").run(init, &g, args[2..]);
    if (replay) g.attention_two_pass = two_pass;
    if (replay) return @import("attention_replay.zig").run(init, &g, args[2..]);
    if (args.len == 2) return benchmark(&g);
    try checkGeluTails(&g);
    try checkProjectionDomains(&g);
    try @import("wide_checks.zig").checkFlushedProducts(&g);
    try checkExpansionTinyInput(&g);
    try @import("planes_checks.zig").run(&g);
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
        const tile = abi.TileProject{ .rows = 64, .cols = 128, .depth = @intCast(depth), .scatter = 0, .grid = 72, .tile = 24 };
        for ([_]Tensor{ out, active }, 0..) |dst, activation| try g.dispatch(if (activation == 0) abi.tile_project else abi.tile_project_gelu, &.{ x, w, bias, dst }, tile, Size.of(32, 4, 2), Size.of(32, 4, 2));
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
    try checkFlash(&g);
    try @import("attention_replay.zig").checkUniform(&g);
    try checkTileResidual(&g);
    try checkTileNorm(&g);
    try checkRowsAttention(&g);
    try operator_checks.run(&g);
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

fn checkTileResidual(g: *gpu.Gpu) !void {
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
    const plain = abi.TileProject{ .rows = rows, .cols = cols, .depth = depth, .scatter = 0, .grid = 72, .tile = 24 };
    try g.dispatch(abi.tile_project, &.{ x, w, bias, projected }, plain, Size.of(rows / 64 * 32, cols / 128 * 4, 2), Size.of(32, 4, 2));
    for (0..2) |scatter| {
        var tiled = plain;
        tiled.scatter = @intCast(scatter);
        try g.dispatch(abi.tile_project_residual, &.{ x, w, bias, actual, residual }, tiled, Size.of(rows / 64 * 32, cols / 128 * 4, 2), Size.of(32, 4, 2));
        for (0..rows) |row| {
            const dest = if (scatter == 1) gridRow(72, 24, row) else row;
            for (0..cols) |col| {
                const expected = residual.get(dest * cols + col) + projected.get(row * cols + col);
                if (actual.get(dest * cols + col) != expected) return error.SamResidualParity;
            }
        }
    }
    std.debug.print("SAM fused residual and all nine tiles are byte-identical\n", .{});
}

fn fp16Ulp(value: f64) f64 {
    const magnitude = @abs(value);
    if (magnitude < 0x1p-14) return 0x1p-24;
    return std.math.pow(f64, 2, @floor(@log2(magnitude)) - 10);
}

fn checkTileNorm(g: *gpu.Gpu) !void {
    const rows = 5184;
    const width = 96;
    const x = try g.alloc(rows * width, .f32);
    defer x.deinit();
    const weight = try g.alloc(width, .f32);
    defer weight.deinit();
    const bias = try g.alloc(width, .f32);
    defer bias.deinit();
    for (x.slice(f32, x.count), 0..) |*value, i| value.* = @sin(@as(f32, @floatFromInt(i)) * 0.011) * 2 + 0.25;
    for (weight.slice(f32, width), bias.slice(f32, width), 0..) |*scale, *shift, i| {
        scale.* = @cos(@as(f32, @floatFromInt(i)) * 0.3);
        shift.* = @sin(@as(f32, @floatFromInt(i)) * 0.7) * 0.5;
    }
    const out = try g.allocDevice(rows * width, .f16);
    defer out.deinit();
    const core = try ops.norm(g, x, weight, bias, width, false, 1e-5);
    defer core.deinit();
    var worst: f64 = 0;
    for (0..2) |gather| {
        const params = abi.TileNorm{ .width = width, .gather = @intCast(gather), .grid = 72, .tile = 24, .eps = 1e-5 };
        try g.dispatch(abi.tile_norm_f16, &.{ x, weight, bias, out }, params, Size.of(rows * 256, 1, 1), Size.of(256, 1, 1));
        for (0..rows) |row| {
            const source = if (gather == 1) gridRow(72, 24, row) else row;
            var mean: f64 = 0;
            var magnitude: f64 = 0;
            for (0..width) |c| mean += x.get(source * width + c);
            for (0..width) |c| magnitude += @abs(x.get(source * width + c));
            mean /= width;
            magnitude /= width;
            var variance: f64 = 0;
            for (0..width) |c| variance += (x.get(source * width + c) - mean) * (x.get(source * width + c) - mean);
            variance /= width;
            const inverse = 1 / @sqrt(variance + 1e-5);
            for (0..width) |c| {
                const actual = out.get(row * width + c);
                if (actual != @as(f32, @floatCast(@as(f16, @floatCast(core.get(source * width + c)))))) return error.SamTileNormParity;
                const centered = x.get(source * width + c) - mean;
                const expected = centered * inverse * weight.get(c) + bias.get(c);
                // FP32 statistics carry about eight roundings of the row magnitude through the scale before the bias.
                const statistics = 16 * std.math.floatEps(f32) * ((magnitude + @abs(centered)) * inverse * @abs(weight.get(c)) + @abs(bias.get(c)));
                const err = @abs(expected - actual);
                if (err > fp16Ulp(expected) + statistics) return error.SamTileNormPrecision;
                worst = @max(worst, err / (fp16Ulp(expected) + statistics));
            }
        }
    }
    std.debug.print("SAM tile norm with and without gather: byte-identical to the core norm, worst float64 error {d:.3} of bound\n", .{worst});
}

fn checkRowsAttention(g: *gpu.Gpu) !void {
    const rows = 7;
    const keys = 13;
    const heads = 2;
    var maximum: f64 = 0;
    for ([_]usize{ 16, 48 }) |dim| for (0..2) |has_bias| {
        const q = try g.alloc(rows * heads * dim, .f32);
        defer q.deinit();
        const k = try g.alloc(keys * heads * dim, .f32);
        defer k.deinit();
        const v = try g.alloc(k.count, .f32);
        defer v.deinit();
        const bias = try g.alloc(heads * rows * keys, .f32);
        defer bias.deinit();
        for ([_]Tensor{ q, k, v, bias }, 0..) |tensor, index| for (tensor.slice(f32, tensor.count), 0..) |*value, i| {
            value.* = @sin(@as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(index + 3)) * 0.021) * 1.5;
        };
        const out = try g.allocDevice(q.count, .f32);
        defer out.deinit();
        const params = abi.RowsAttention{ .rows = rows, .keys = keys, .heads = heads, .head_width = @intCast(dim), .has_bias = @intCast(has_bias) };
        try g.dispatch(abi.attention_rows, &.{ q, k, v, if (has_bias == 1) bias else q, out }, params, Size.of(rows * 32, heads, 1), Size.of(32, 1, 1));
        for (0..heads) |h| for (0..rows) |row| {
            var scores: [keys]f64 = undefined;
            var top = -std.math.inf(f64);
            for (&scores, 0..) |*score, key| {
                score.* = 0;
                for (0..dim) |d| score.* += @as(f64, q.get((row * heads + h) * dim + d)) * k.get((key * heads + h) * dim + d);
                score.* /= @sqrt(@as(f64, @floatFromInt(dim)));
                if (has_bias == 1) score.* += bias.get((h * rows + row) * keys + key);
                top = @max(top, score.*);
            }
            var total: f64 = 0;
            for (&scores) |*score| {
                score.* = @exp(score.* - top);
                total += score.*;
            }
            for (0..dim) |d| {
                var value: f64 = 0;
                for (scores, 0..) |score, key| value += score * v.get((key * heads + h) * dim + d);
                maximum = @max(maximum, @abs(value / total - out.get((row * heads + h) * dim + d)));
            }
        };
    };
    std.debug.print("SAM attention rows with and without bias: maximum float64 oracle error {e}\n", .{maximum});
    if (!(maximum <= 2e-5)) return error.SamAttentionRowsParity;
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

test {
    _ = operator_checks;
}
