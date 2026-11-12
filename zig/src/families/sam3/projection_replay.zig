const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");
const Mps = @import("mps_ndarray_probe.zig").Backend;

fn upload(g: *gpu.Gpu, value: st.Tensor) !gpu.Tensor {
    if (value.dtype != .f32) return error.ExpectedFloat32Projection;
    const tensor = try g.alloc(value.numel(), .f32);
    errdefer tensor.deinit();
    @memcpy(tensor.slice(u8, value.bytes.len), value.bytes);
    try tensor.checkFinite("projection input");
    return tensor;
}

fn linear(g: *gpu.Gpu, mps: ?*Mps, small: bool, x: gpu.Tensor, w: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const output = if (small) try @import("dense_small.zig").run(g, x, w, bias, rows, cols, depth) else if (mps) |backend| try backend.run(g, x, w, bias, rows, cols, depth) else try g.linear(x, w, bias, rows, cols, depth);
    errdefer output.deinit();
    try g.endBatch();
    return output;
}

pub fn run(init: std.process.Init, g: *gpu.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 2 and args.len != 3) return error.ExpectedProjectionCaptureAndReceipt;
    const use_mps = args.len == 3 and std.mem.eql(u8, args[2], "--mps-ndarray");
    const use_small = args.len == 3 and std.mem.eql(u8, args[2], "--compensated");
    if (args.len == 3 and !use_mps and !use_small) return error.UnexpectedProjectionBackend;
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
    const samples = indices.dim(0);
    if (!bias_source.is(.f32, &.{cols}) or truth.dtype != .f32 or truth.numel() != rows * cols or
        !oracle.is(.f64, &.{ samples, cols }) or samples == 0) return error.ProjectionReplayShape;
    const x = try upload(g, input);
    defer x.deinit();
    const w = try upload(g, weight);
    defer w.deinit();
    const bias = try upload(g, bias_source);
    defer bias.deinit();
    g.compact_gemm = false;
    const baseline = try linear(g, null, false, x, w, bias, rows, cols, depth);
    defer baseline.deinit();
    g.compact_gemm = true;
    const actual = try linear(g, backend, use_small, x, w, bias, rows, cols, depth);
    defer actual.deinit();
    const bytes = actual.slice(u8, actual.count * 4);
    const replacement_equal = std.mem.eql(u8, baseline.slice(u8, bytes.len), bytes);
    if (!use_mps and !use_small and !replacement_equal) return error.ProjectionReplacementChanged;
    const reference_equal = std.mem.eql(u8, truth.bytes, bytes);
    for (0..5) |_| {
        const repeated = try linear(g, backend, use_small, x, w, bias, rows, cols, depth);
        defer repeated.deinit();
        if (!std.mem.eql(u8, bytes, repeated.slice(u8, bytes.len))) return error.ProjectionNotRepeatable;
    }
    const warm_start = gpu.mtl.clock.seconds();
    while (gpu.mtl.clock.seconds() - warm_start < 1) {
        const warm = try linear(g, backend, use_small, x, w, bias, rows, cols, depth);
        warm.deinit();
    }
    var gpu_times: [10]f64 = undefined;
    var wall_times: [10]f64 = undefined;
    for (&gpu_times, &wall_times) |*gpu_time, *wall_time| {
        const before = g.gpu_seconds;
        const wall = gpu.mtl.clock.seconds();
        const timed = try linear(g, backend, use_small, x, w, bias, rows, cols, depth);
        wall_time.* = gpu.mtl.clock.seconds() - wall;
        gpu_time.* = g.gpu_seconds - before;
        timed.deinit();
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
    const json = try std.json.Stringify.valueAlloc(a, .{
        .shape = .{ rows, cols, depth },
        .capture_sha256 = source_hash[0..],
        .backend = if (use_small) "compensated_small_rows" else if (use_mps) "mps_ndarray" else "tensor_16_rows",
        .replacement_equal = replacement_equal,
        .pytorch_equal = reference_equal,
        .buffer_cache_limit = g.buffer_cache_limit,
        .repeat_runs = 5,
        .warmup_min_seconds = 1,
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
    std.debug.print("Projection {d}x{d}x{d}, {s}: replacement equal {}, PyTorch equal {}, five repeat runs; FP64 max native {e}, PyTorch {e}; {d}/{d} columns worse\n", .{ rows, cols, depth, if (use_small) "compensated" else if (use_mps) "MPS" else "tensor", replacement_equal, reference_equal, std.mem.max(f64, native_columns), std.mem.max(f64, torch_columns), worse, cols });
    if (worse != 0) return error.ProjectionReferenceWorse;
}
