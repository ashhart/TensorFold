const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");
const Mps = @import("mps_ndarray_probe.zig").Backend;
const timed_batch = 64;
const batch_output_target = 32 << 20;
const NativeBackend = enum { tensor, small, tensor_device, tensor_device_compensated, tensor_device_nn, tensor_device_nn_compensated, tensor_device_gelu, tensor_device_gelu_baseline, expansion, expansion_gelu };

fn project(g: *gpu.Gpu, mps: ?*Mps, native: NativeBackend, x: gpu.Tensor, w: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
    return switch (native) {
        .small => @import("dense_small.zig").run(g, x, w, bias, rows, cols, depth),
        .expansion, .expansion_gelu => @import("dense_expansion.zig").run(g, x, w, bias, rows, cols, depth, native == .expansion_gelu),
        .tensor_device_gelu_baseline => activated: {
            const projected = try @import("dense_device.zig").run(g, x, w, bias, rows, cols, depth, .nk_gelu_linear);
            defer projected.deinit();
            const output = try g.allocDevice(projected.count, .f32);
            errdefer output.deinit();
            try g.dispatch("im_sam_gelu", &.{ projected, output }, @as(u32, @intCast(projected.count)), gpu.mtl.Size.of(projected.count, 1, 1), gpu.mtl.Size.of(256, 1, 1));
            break :activated output;
        },
        .tensor_device, .tensor_device_compensated, .tensor_device_nn, .tensor_device_nn_compensated, .tensor_device_gelu => @import("dense_device.zig").run(g, x, w, bias, rows, cols, depth, if (native == .tensor_device_gelu) .nk_gelu else if (native == .tensor_device_nn_compensated) .kn_compensated else if (native == .tensor_device_nn) .kn else if (native == .tensor_device_compensated) .nk_compensated else .nk),
        .tensor => if (mps) |backend| backend.run(g, x, w, bias, rows, cols, depth) else g.linear(x, w, bias, rows, cols, depth),
    };
}

fn batchCount(rows: usize, cols: usize) !usize {
    const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, cols), 4);
    if (bytes == 0) return error.ProjectionReplayShape;
    return @max(1, @min(timed_batch, batch_output_target / bytes));
}

fn upload(g: *gpu.Gpu, value: st.Tensor) !gpu.Tensor {
    if (value.dtype != .f32) return error.ExpectedFloat32Projection;
    const tensor = try g.alloc(value.numel(), .f32);
    errdefer tensor.deinit();
    @memcpy(tensor.slice(u8, value.bytes.len), value.bytes);
    try tensor.checkFinite("projection input");
    return tensor;
}

fn linear(g: *gpu.Gpu, mps: ?*Mps, native: NativeBackend, x: gpu.Tensor, w: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const output = try project(g, mps, native, x, w, bias, rows, cols, depth);
    errdefer output.deinit();
    try g.endBatch();
    return output;
}

fn timedBatch(g: *gpu.Gpu, mps: ?*Mps, native: NativeBackend, x: gpu.Tensor, w: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, expected: ?[]const u8) !void {
    var outputs: [timed_batch]gpu.Tensor = undefined;
    var made: usize = 0;
    defer for (outputs[0..made]) |output| output.deinit();
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    for (outputs[0..try batchCount(rows, cols)]) |*output| {
        output.* = try project(g, mps, native, x, w, bias, rows, cols, depth);
        made += 1;
    }
    try g.endBatch();
    if (expected) |bytes| for (outputs[0..made]) |output| {
        if (!std.mem.eql(u8, bytes, output.slice(u8, bytes.len))) return error.ProjectionBatchChanged;
    };
}

pub fn run(init: std.process.Init, g: *gpu.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 2 and args.len != 3) return error.ExpectedProjectionCaptureAndReceipt;
    const use_mps = args.len == 3 and std.mem.eql(u8, args[2], "--mps-ndarray");
    const baseline_gemm = args.len == 3 and std.mem.eql(u8, args[2], "--baseline-gemm");
    const small_baseline = args.len == 3 and std.mem.eql(u8, args[2], "--compensated-baseline");
    const use_small = small_baseline or (args.len == 3 and std.mem.eql(u8, args[2], "--compensated"));
    const compensated = args.len == 3 and std.mem.eql(u8, args[2], "--tensor-device-compensated");
    const gelu_baseline = args.len == 3 and std.mem.eql(u8, args[2], "--tensor-device-gelu-baseline");
    const expansion_gelu = args.len == 3 and std.mem.eql(u8, args[2], "--tensor-expansion-gelu");
    const expansion = expansion_gelu or (args.len == 3 and std.mem.eql(u8, args[2], "--tensor-expansion"));
    const activated = expansion_gelu or gelu_baseline or (args.len == 3 and std.mem.eql(u8, args[2], "--tensor-device-gelu"));
    const tensor_device = expansion or activated or compensated or (args.len == 3 and std.mem.eql(u8, args[2], "--tensor-device"));
    const compensated_nn = args.len == 3 and std.mem.eql(u8, args[2], "--tensor-device-nn-compensated");
    const tensor_device_nn = compensated_nn or (args.len == 3 and std.mem.eql(u8, args[2], "--tensor-device-nn"));
    const native_backend: NativeBackend = if (expansion_gelu) .expansion_gelu else if (expansion) .expansion else if (gelu_baseline) .tensor_device_gelu_baseline else if (activated) .tensor_device_gelu else if (compensated_nn) .tensor_device_nn_compensated else if (tensor_device_nn) .tensor_device_nn else if (compensated) .tensor_device_compensated else if (tensor_device) .tensor_device else if (use_small) .small else .tensor;
    if (args.len == 3 and !use_mps and !use_small and !baseline_gemm and !tensor_device and !tensor_device_nn) return error.UnexpectedProjectionBackend;
    var mps = if (use_mps) try Mps.init(g.device) else null;
    defer if (mps) |*backend| backend.deinit();
    const backend: ?*Mps = if (mps) |*value| value else null;
    g.buffer_cache_limit = 1 << 30;
    const a = init.arena.allocator();
    var file = try st.File.open(init.gpa, init.io, args[0]);
    defer file.close(init.io);
    const input = file.get("input") orelse return error.MissingProjectionInput;
    const weight = file.get("weight") orelse return error.MissingProjectionWeight;
    const bias_source = file.get("bias") orelse return error.MissingProjectionBias;
    const truth = file.get("output") orelse return error.MissingProjectionReference;
    const oracle = file.get("reference64") orelse return error.MissingFloat64Reference;
    const indices = file.get("row_indices") orelse return error.MissingReferenceRows;
    if (weight.rank != 2 or indices.dtype != .i64 or indices.rank != 1) return error.ProjectionReplayShape;
    const cols = weight.dim(0);
    const depth = weight.dim(1);
    if (cols == 0 or depth == 0 or input.numel() % depth != 0) return error.ProjectionReplayShape;
    const rows = input.numel() / depth;
    const batch_count = try batchCount(rows, cols);
    const samples = indices.dim(0);
    if (!bias_source.is(.f32, &.{cols}) or truth.dtype != .f32 or truth.numel() != rows * cols or
        !oracle.is(.f64, &.{ samples, cols }) or samples == 0) return error.ProjectionReplayShape;
    const x = try upload(g, input);
    defer x.deinit();
    const w = try upload(g, weight);
    defer w.deinit();
    const bias = try upload(g, bias_source);
    defer bias.deinit();
    const expansion_packing: ?@import("dense_expansion.zig").Packing = if (expansion) try @import("dense_expansion.zig").inspect(g, x, w, bias, rows, cols, depth, expansion_gelu) else null;
    g.compact_gemm = false;
    const baseline = try linear(g, null, if (activated) .tensor_device_gelu_baseline else .tensor, x, w, bias, rows, cols, depth);
    defer baseline.deinit();
    const candidate_weight = if (tensor_device_nn) transposed: {
        const target = try g.alloc(w.count, .f32);
        const values = target.slice(f32, target.count);
        for (0..cols) |col| {
            for (0..depth) |k| values[k * cols + col] = w.get(col * depth + k);
        }
        break :transposed target;
    } else w;
    defer if (tensor_device_nn) candidate_weight.deinit();
    g.compact_gemm = !baseline_gemm;
    const actual = try linear(g, backend, native_backend, x, candidate_weight, bias, rows, cols, depth);
    defer actual.deinit();
    const bytes = actual.slice(u8, actual.count * 4);
    const replacement_equal = std.mem.eql(u8, baseline.slice(u8, bytes.len), bytes);
    if (native_backend == .tensor and !use_mps and !replacement_equal) return error.ProjectionReplacementChanged;
    if (activated and !expansion and !replacement_equal) return error.ProjectionGeluReplacementChanged;
    const reference_equal = std.mem.eql(u8, truth.bytes, bytes);
    for (0..5) |_| {
        const repeated = try linear(g, backend, native_backend, x, candidate_weight, bias, rows, cols, depth);
        defer repeated.deinit();
        if (!std.mem.eql(u8, bytes, repeated.slice(u8, bytes.len))) return error.ProjectionNotRepeatable;
    }
    try timedBatch(g, backend, native_backend, x, candidate_weight, bias, rows, cols, depth, bytes);
    const warm_start = gpu.mtl.clock.seconds();
    while (gpu.mtl.clock.seconds() - warm_start < 1) {
        try timedBatch(g, backend, native_backend, x, candidate_weight, bias, rows, cols, depth, null);
    }
    var gpu_times: [10]f64 = undefined;
    var wall_times: [10]f64 = undefined;
    for (&gpu_times, &wall_times) |*gpu_time, *wall_time| {
        const before = g.gpu_seconds;
        const wall = gpu.mtl.clock.seconds();
        try timedBatch(g, backend, native_backend, x, candidate_weight, bias, rows, cols, depth, null);
        wall_time.* = (gpu.mtl.clock.seconds() - wall) / @as(f64, @floatFromInt(batch_count));
        gpu_time.* = (g.gpu_seconds - before) / @as(f64, @floatFromInt(batch_count));
    }
    const native_columns = try a.alloc(f64, cols);
    const torch_columns = try a.alloc(f64, cols);
    @memset(native_columns, 0);
    @memset(torch_columns, 0);
    for (0..samples) |sample| {
        const row = std.math.cast(usize, std.mem.readInt(i64, indices.bytes[sample * 8 ..][0..8], .little)) orelse return error.ReferenceRowOutOfRange;
        if (row >= rows) return error.ReferenceRowOutOfRange;
        for (0..cols) |col| {
            const reference: f64 = @bitCast(std.mem.readInt(u64, oracle.bytes[(sample * cols + col) * 8 ..][0..8], .little));
            const pytorch: f32 = @bitCast(std.mem.readInt(u32, truth.bytes[(row * cols + col) * 4 ..][0..4], .little));
            const native = actual.get(row * cols + col);
            if (!std.math.isFinite(reference) or !std.math.isFinite(pytorch) or !std.math.isFinite(native)) return error.NonFiniteProjectionReference;
            native_columns[col] = @max(native_columns[col], @abs(reference - native));
            torch_columns[col] = @max(torch_columns[col], @abs(reference - pytorch));
        }
    }
    var worse: usize = 0;
    for (native_columns, torch_columns) |native, pytorch| worse += @intFromBool(native > pytorch);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(file.map.memory, &digest, .{});
    const source_hash = std.fmt.bytesToHex(digest, .lower);
    std.crypto.hash.sha2.Sha256.hash(if (small_baseline) @import("sam3_kernels").baseline_source else @import("sam3_kernels").source, &digest, .{});
    const shader_hash = std.fmt.bytesToHex(digest, .lower);
    const backend_name = if (expansion_gelu) "tensor_exact_expansion_gelu_guarded" else if (expansion) "tensor_exact_expansion_guarded" else if (gelu_baseline) "tensor_device_separate_gelu_guarded" else if (activated) "tensor_device_fused_gelu_guarded" else if (compensated_nn) "tensor_device_nn_compensated_guarded" else if (tensor_device_nn) "tensor_device_nn_full_precision_guarded" else if (compensated) "tensor_device_compensated_guarded" else if (tensor_device) "tensor_device_full_precision_guarded" else if (small_baseline) "compensated_single_row_baseline" else if (use_small) "compensated_small_rows" else if (use_mps) "mps_ndarray" else if (baseline_gemm) "tensor_64_rows" else "tensor_32_rows";
    const json = try std.json.Stringify.valueAlloc(a, .{
        .shape = .{ rows, cols, depth },
        .capture_sha256 = source_hash[0..],
        .sam3_metal_source_sha256 = shader_hash[0..],
        .backend = backend_name,
        .expansion_packing = expansion_packing,
        .replacement_equal = replacement_equal,
        .pytorch_equal = reference_equal,
        .buffer_cache_limit = g.buffer_cache_limit,
        .repeat_runs = 5,
        .verified_batch_outputs = batch_count,
        .warmup_min_seconds = 1,
        .timing_batch = batch_count,
        .batch_output_target_bytes = batch_output_target,
        .gpu_seconds = gpu_times,
        .wall_seconds = wall_times,
        .reference_rows = samples,
        .native_max_error = std.mem.max(f64, native_columns),
        .pytorch_max_error = std.mem.max(f64, torch_columns),
        .worse_columns = worse,
        .native_column_errors = native_columns,
        .pytorch_column_errors = torch_columns,
    }, .{ .whitespace = .indent_2 });
    const receipt = try std.Io.Dir.cwd().createFile(init.io, args[1], .{ .exclusive = true });
    defer receipt.close(init.io);
    try receipt.writeStreamingAll(init.io, json);
    try receipt.writeStreamingAll(init.io, "\n");
    std.debug.print("Projection {d}x{d}x{d}, {s}: replacement equal {}, PyTorch equal {}, five repeat runs; FP64 max native {e}, PyTorch {e}; {d}/{d} columns worse\n", .{ rows, cols, depth, backend_name, replacement_equal, reference_equal, std.mem.max(f64, native_columns), std.mem.max(f64, torch_columns), worse, cols });
    if (worse != 0) return error.ProjectionReferenceWorse;
}
