const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");

fn upload(g: *gpu.Gpu, tensor: st.Tensor) !gpu.Tensor {
    if (tensor.dtype != .f32) return error.ExpectedFloat32NormalizationCapture;
    const out = try g.alloc(tensor.numel(), .f32);
    @memcpy(out.slice(u8, tensor.bytes.len), tensor.bytes);
    return out;
}

pub fn run(init: std.process.Init, g: *gpu.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 2 and args.len != 3) return error.ExpectedNormalizationCaptureAndReceipt;
    const compensated = args.len == 3 and std.mem.eql(u8, args[2], "--compensated");
    if (args.len == 3 and !compensated) return error.UnexpectedNormalizationBackend;
    const a = init.arena.allocator();
    var file = try st.File.open(init.gpa, init.io, args[0]);
    defer file.close(init.io);
    const input = file.get("input") orelse return error.MissingNormalizationInput;
    const weight = file.get("weight") orelse return error.MissingNormalizationWeight;
    const bias_source = file.get("bias") orelse return error.MissingNormalizationBias;
    const truth = file.get("output") orelse return error.MissingNormalizationOutput;
    const oracle = file.get("reference64") orelse return error.MissingFloat64Reference;
    const epsilon = file.get("eps") orelse return error.MissingNormalizationEpsilon;
    if (input.rank != 2 or weight.rank != 1 or weight.dim(0) == 0 or
        !input.is(.f32, &.{ input.dim(0), weight.dim(0) }) or
        !truth.is(.f32, input.shape[0..input.rank]) or !oracle.is(.f64, input.shape[0..input.rank]) or
        !bias_source.is(.f32, &.{weight.dim(0)}) or !epsilon.is(.f32, &.{1})) return error.NormalizationCaptureShape;
    const eps: f32 = @bitCast(std.mem.readInt(u32, epsilon.bytes[0..4], .little));
    if (!std.math.isFinite(eps) or eps <= 0) return error.NormalizationCaptureEpsilon;
    const x = try upload(g, input);
    defer x.deinit();
    const w = try upload(g, weight);
    defer w.deinit();
    const bias = try upload(g, bias_source);
    defer bias.deinit();
    const width = weight.dim(0);
    const actual = try normalize(g, x, w, bias, width, eps, compensated);
    defer actual.deinit();
    const bytes = actual.slice(u8, actual.count * 4);
    for (0..5) |_| {
        const repeated = try normalize(g, x, w, bias, width, eps, compensated);
        defer repeated.deinit();
        if (!std.mem.eql(u8, bytes, repeated.slice(u8, bytes.len))) return error.NormalizationNotRepeatable;
    }
    var gpu_times: [10]f64 = undefined;
    for (&gpu_times) |*elapsed| {
        const before = g.gpu_seconds;
        var outputs: [8]gpu.Tensor = undefined;
        var used: usize = 0;
        defer for (outputs[0..used]) |value| value.deinit();
        try g.beginBatch();
        errdefer if (g.batch != null) g.endBatch() catch {};
        for (&outputs) |*value| {
            value.* = try normalize(g, x, w, bias, width, eps, compensated);
            used += 1;
        }
        try g.endBatch();
        elapsed.* = (g.gpu_seconds - before) / outputs.len;
        for (outputs) |value| if (!std.mem.eql(u8, bytes, value.slice(u8, bytes.len))) return error.NormalizationBatchChanged;
    }
    const native_columns = try a.alloc(f64, width);
    const torch_columns = try a.alloc(f64, width);
    @memset(native_columns, 0);
    @memset(torch_columns, 0);
    for (0..actual.count) |i| {
        const reference: f64 = @bitCast(std.mem.readInt(u64, oracle.bytes[i * 8 ..][0..8], .little));
        const pytorch: f32 = @bitCast(std.mem.readInt(u32, truth.bytes[i * 4 ..][0..4], .little));
        const native = actual.get(i);
        if (!std.math.isFinite(reference) or !std.math.isFinite(pytorch) or !std.math.isFinite(native)) return error.NonFiniteNormalizationCapture;
        native_columns[i % width] = @max(native_columns[i % width], @abs(reference - native));
        torch_columns[i % width] = @max(torch_columns[i % width], @abs(reference - pytorch));
    }
    var worse: usize = 0;
    for (native_columns, torch_columns) |native, pytorch| worse += @intFromBool(native > pytorch);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(file.map.memory, &digest, .{});
    const capture_hash = std.fmt.bytesToHex(digest, .lower);
    std.crypto.hash.sha2.Sha256.hash(@import("sam3_kernels").source, &digest, .{});
    const source_hash = std.fmt.bytesToHex(digest, .lower);
    const json = try std.json.Stringify.valueAlloc(a, .{
        .shape = input.shape[0..input.rank],
        .eps = eps,
        .compensated = compensated,
        .capture_sha256 = capture_hash[0..],
        .sam3_metal_source_sha256 = source_hash[0..],
        .pytorch_equal = std.mem.eql(u8, truth.bytes, bytes),
        .repeat_runs = 5,
        .gpu_seconds = gpu_times,
        .timing_batch = 8,
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
    std.debug.print("Normalization {d}x{d}, eps {e}: five equal repeats, FP64 max native {e}, PyTorch {e}; {d}/{d} columns worse\n", .{
        input.dim(0), width, eps, std.mem.max(f64, native_columns), std.mem.max(f64, torch_columns), worse, width,
    });
    if (worse != 0) return error.NormalizationReferenceWorse;
}

fn normalize(g: *gpu.Gpu, x: gpu.Tensor, w: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32, compensated: bool) !gpu.Tensor {
    if (!compensated) return gpu.ops.norm(g, x, w, bias, width, false, eps);
    return @import("normalization.zig").run(g, x, w, bias, width, eps);
}
