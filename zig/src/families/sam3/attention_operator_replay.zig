const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");
const Backend = enum { tensor, split, device, split32 };
const PhaseProfile = struct {
    const Entry = struct { name: []const u8, gpu_seconds: f64 };
    entries: [8]Entry = undefined,
    count: usize = 0,

    fn record(context: *anyopaque, name: []const u8, seconds: f64) !void {
        const self: *PhaseProfile = @ptrCast(@alignCast(context));
        if (self.count == self.entries.len) return error.AttentionProfileOverflow;
        self.entries[self.count] = .{ .name = name, .gpu_seconds = seconds };
        self.count += 1;
    }
};

fn upload(g: *gpu.Gpu, value: st.Tensor) !gpu.Tensor {
    if (value.dtype != .f32 or value.rank != 4) return error.ExpectedFloat32Attention;
    const batch, const heads, const rows, const dim = value.shape[0..4].*;
    const tensor = try g.alloc(value.numel(), .f32);
    errdefer tensor.deinit();
    for (0..batch) |b| for (0..heads) |h| for (0..rows) |row| {
        const source = ((b * heads + h) * rows + row) * dim * 4;
        const destination = ((b * rows + row) * heads + h) * dim * 4;
        @memcpy(tensor.slice(u8, tensor.count * 4)[destination..][0 .. dim * 4], value.bytes[source..][0 .. dim * 4]);
    };
    try tensor.checkFinite("captured attention input");
    return tensor;
}

fn attention(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, backend: Backend) !gpu.Tensor {
    const batch, const heads, const rows, const keys, const dim = shape;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(dim)));
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const output = if (backend == .device)
        try @import("attention_device.zig").run(g, q, k, v, shape)
    else if (backend == .split32)
        try @import("attention_split32.zig").run(g, q, k, v, shape)
    else if (backend == .split)
        try @import("attention_split.zig").run(g, q, k, v, shape)
    else if (batch == 1)
        try gpu.ops.attentionCross(g, q, k, v, rows, keys, heads, scale)
    else if (rows == keys)
        try gpu.ops.attentionBatched(g, q, k, v, rows, heads, heads, false, scale, batch)
    else
        return error.UnsupportedBatchedAttentionReplay;
    errdefer output.deinit();
    try g.endBatch();
    return output;
}

pub fn run(init: std.process.Init, g: *gpu.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 2 and args.len != 3) return error.ExpectedAttentionCaptureAndReceipt;
    const split = args.len == 3 and std.mem.eql(u8, args[2], "--split-keys");
    const split32 = args.len == 3 and std.mem.eql(u8, args[2], "--split32");
    const split64 = args.len == 3 and std.mem.eql(u8, args[2], "--split64");
    const device32 = args.len == 3 and std.mem.eql(u8, args[2], "--tensor32");
    const device = device32 or (args.len == 3 and std.mem.eql(u8, args[2], "--tensor64"));
    const backend: Backend = if (device) .device else if (split32 or split64) .split32 else if (split) .split else .tensor;
    if (args.len == 3 and !split and !split32 and !split64 and !device) return error.UnexpectedAttentionReplayBackend;
    const a = init.arena.allocator();
    var file = try st.File.open(init.gpa, init.io, args[0]);
    defer file.close(init.io);
    const query = file.get("q") orelse return error.MissingAttentionQuery;
    const key = file.get("k") orelse return error.MissingAttentionKey;
    const value = file.get("v") orelse return error.MissingAttentionValue;
    const truth = file.get("output") orelse return error.MissingAttentionOutput;
    const oracle = file.get("reference64") orelse return error.MissingFloat64Reference;
    const indices = file.get("row_indices") orelse return error.MissingReferenceRows;
    if (query.dtype != .f32 or query.rank != 4) return error.AttentionReplayShape;
    const batch, const heads, const rows, const dim = query.shape[0..4].*;
    if (batch == 0 or heads == 0 or rows == 0 or (dim != 32 and dim != 64 and dim != 256) or key.rank != 4 or
        (device and dim != (if (device32) @as(usize, 32) else 64)) or (split32 and dim != 32) or (split64 and dim != 64)) return error.AttentionReplayShape;
    const keys = key.dim(2);
    if (keys == 0 or !key.is(.f32, &.{ batch, heads, keys, dim }) or !value.is(.f32, &.{ batch, heads, keys, dim }) or
        !truth.is(.f32, &.{ batch, heads, rows, dim }) or indices.dtype != .i64 or indices.rank != 1 or indices.dim(0) == 0 or
        !oracle.is(.f64, &.{ batch, heads, indices.dim(0), dim })) return error.AttentionReplayShape;
    g.buffer_cache_limit = 1 << 30;
    const q = try upload(g, query);
    defer q.deinit();
    const k = try upload(g, key);
    defer k.deinit();
    const v = try upload(g, value);
    defer v.deinit();
    const shape = [5]usize{ batch, heads, rows, keys, dim };
    const actual = try attention(g, q, k, v, shape, backend);
    defer actual.deinit();
    try actual.checkFinite("captured attention replay");
    for (0..5) |_| {
        const repeated = try attention(g, q, k, v, shape, backend);
        defer repeated.deinit();
        if (!std.mem.eql(u8, actual.slice(u8, actual.count * 4), repeated.slice(u8, repeated.count * 4))) return error.AttentionNotRepeatable;
    }
    if (split or split32 or split64) try checkWorkspace(g, q, k, v, shape, actual, backend);
    const warm_start = gpu.mtl.clock.seconds();
    while (gpu.mtl.clock.seconds() - warm_start < 1) {
        const warm = try attention(g, q, k, v, shape, backend);
        warm.deinit();
    }
    var gpu_times: [10]f64 = undefined;
    var wall_times: [10]f64 = undefined;
    for (&gpu_times, &wall_times) |*gpu_time, *wall_time| {
        const before = g.gpu_seconds;
        const wall = gpu.mtl.clock.seconds();
        const timed = try attention(g, q, k, v, shape, backend);
        wall_time.* = gpu.mtl.clock.seconds() - wall;
        gpu_time.* = g.gpu_seconds - before;
        timed.deinit();
    }
    var profile: PhaseProfile = .{};
    if (backend == .split32) {
        g.operator_observer = .{ .context = &profile, .write = PhaseProfile.record };
        defer g.operator_observer = null;
        const profiled = try attention(g, q, k, v, shape, backend);
        defer profiled.deinit();
        if (!std.mem.eql(u8, actual.slice(u8, actual.count * 4), profiled.slice(u8, profiled.count * 4))) return error.AttentionProfileChanged;
    }
    const native_columns = try a.alloc(f64, heads * dim);
    const torch_columns = try a.alloc(f64, heads * dim);
    const Worst = struct { batch: usize, head: usize, row: usize, column: usize, native: f32, pytorch: f32, reference: f64 };
    const worst_outputs = try a.alloc(Worst, heads * dim);
    @memset(worst_outputs, .{ .batch = 0, .head = 0, .row = 0, .column = 0, .native = 0, .pytorch = 0, .reference = 0 });
    @memset(native_columns, 0);
    @memset(torch_columns, 0);
    var different: usize = 0;
    for (0..batch) |b| for (0..heads) |h| for (0..rows) |row| for (0..dim) |d| {
        const source = ((b * heads + h) * rows + row) * dim + d;
        const destination = ((b * rows + row) * heads + h) * dim + d;
        different += @intFromBool(std.mem.readInt(u32, truth.bytes[source * 4 ..][0..4], .little) != actual.slice(u32, actual.count)[destination]);
    };
    const samples = indices.dim(0);
    for (0..batch) |b| for (0..heads) |h| for (0..samples) |sample| {
        const row = std.math.cast(usize, std.mem.readInt(i64, indices.bytes[sample * 8 ..][0..8], .little)) orelse return error.ReferenceRowOutOfRange;
        if (row >= rows) return error.ReferenceRowOutOfRange;
        for (0..dim) |d| {
            const at = ((b * heads + h) * samples + sample) * dim + d;
            const reference: f64 = @bitCast(std.mem.readInt(u64, oracle.bytes[at * 8 ..][0..8], .little));
            const source = ((b * heads + h) * rows + row) * dim + d;
            const pytorch: f32 = @bitCast(std.mem.readInt(u32, truth.bytes[source * 4 ..][0..4], .little));
            const native = actual.get(((b * rows + row) * heads + h) * dim + d);
            if (!std.math.isFinite(reference) or !std.math.isFinite(pytorch) or !std.math.isFinite(native)) return error.NonFiniteAttentionReference;
            if (@abs(reference - native) > native_columns[h * dim + d]) {
                native_columns[h * dim + d] = @abs(reference - native);
                worst_outputs[h * dim + d] = .{ .batch = b, .head = h, .row = row, .column = d, .native = native, .pytorch = pytorch, .reference = reference };
            }
            torch_columns[h * dim + d] = @max(torch_columns[h * dim + d], @abs(reference - pytorch));
        }
    };
    var worse: usize = 0;
    var worse_outputs: std.ArrayList(Worst) = .empty;
    for (native_columns, torch_columns, worst_outputs) |native, pytorch, worst| if (native > pytorch) {
        worse += 1;
        try worse_outputs.append(a, worst);
    };
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(file.map.memory, &digest, .{});
    const source_hash = std.fmt.bytesToHex(digest, .lower);
    std.crypto.hash.sha2.Sha256.hash(actual.slice(u8, actual.count * 4), &digest, .{});
    const output_hash = std.fmt.bytesToHex(digest, .lower);
    std.crypto.hash.sha2.Sha256.hash(@import("sam3_kernels").source, &digest, .{});
    const shader_hash = std.fmt.bytesToHex(digest, .lower);
    const json = try std.json.Stringify.valueAlloc(a, .{
        .shape = shape,
        .capture_sha256 = source_hash[0..],
        .sam3_metal_source_sha256 = shader_hash[0..],
        .output_sha256 = output_hash[0..],
        .repeat_runs = 5,
        .warmup_min_seconds = 1,
        .backend = if (split64) "tensor64_split_keys_guarded" else if (split32) "tensor32_split_keys_guarded" else if (device32) "tensor32_guarded_full_precision" else if (device) "tensor64_guarded_full_precision" else if (split) "tensor_split_keys" else if (dim == 256) "tensor_two_pass" else "tensor_fused",
        .split_chunk_blocks = if (split32 or split64) @import("sam3_kernels").attention_split32.chunk_blocks else if (split) @import("sam3_kernels").attention_split.chunk_blocks else @as(usize, 0),
        .split_value_parts = if (split) @import("sam3_kernels").attention_split.value_parts else @as(usize, 0),
        .interleaved_workspace_verified = split or split32 or split64,
        .different_outputs = different,
        .gpu_seconds = gpu_times,
        .wall_seconds = wall_times,
        .operator_profile = profile.entries[0..profile.count],
        .profile_scope = "one synchronized pass after timing; excluded from timing samples",
        .reference_rows_per_head = samples,
        .native_max_error = std.mem.max(f64, native_columns),
        .pytorch_max_error = std.mem.max(f64, torch_columns),
        .worse_columns = worse,
        .worse_outputs = worse_outputs.items,
        .native_column_errors = native_columns,
        .pytorch_column_errors = torch_columns,
    }, .{ .whitespace = .indent_2 });
    const receipt = try std.Io.Dir.cwd().createFile(init.io, args[1], .{ .exclusive = true });
    defer receipt.close(init.io);
    try receipt.writeStreamingAll(init.io, json);
    try receipt.writeStreamingAll(init.io, "\n");
    std.debug.print("Attention b{d} h{d} {d}x{d} d{d}: five equal repeats; {d} outputs differ from PyTorch; FP64 max native {e}, PyTorch {e}; {d}/{d} columns worse\n", .{ batch, heads, rows, keys, dim, different, std.mem.max(f64, native_columns), std.mem.max(f64, torch_columns), worse, heads * dim });
    if (worse != 0) return error.AttentionReferenceWorse;
}

fn splitRun(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, backend: Backend) !gpu.Tensor {
    return if (backend == .split32) @import("attention_split32.zig").run(g, q, k, v, shape) else @import("attention_split.zig").run(g, q, k, v, shape);
}

fn checkWorkspace(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, expected: gpu.Tensor, backend: Backend) !void {
    const batch, const heads, const rows, const keys, const dim = shape;
    const width = heads * dim;
    const single_query = try g.alloc(batch * width, .f32);
    defer single_query.deinit();
    for (0..batch) |b|
        @memcpy(single_query.slice(f32, single_query.count)[b * width ..][0..width], q.slice(f32, q.count)[b * rows * width ..][0..width]);
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const first = try splitRun(g, q, k, v, shape, backend);
    defer first.deinit();
    const middle = try splitRun(g, single_query, k, v, .{ batch, heads, 1, keys, dim }, backend);
    defer middle.deinit();
    const last = try splitRun(g, q, k, v, shape, backend);
    defer last.deinit();
    try g.endBatch();
    const bytes = expected.slice(u8, expected.count * 4);
    if (!std.mem.eql(u8, bytes, first.slice(u8, bytes.len)) or !std.mem.eql(u8, bytes, last.slice(u8, bytes.len))) return error.AttentionWorkspaceChanged;
    for (0..batch) |b| {
        const target = expected.slice(f32, expected.count)[b * rows * width ..][0..width];
        const actual = middle.slice(f32, middle.count)[b * width ..][0..width];
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(target), std.mem.sliceAsBytes(actual))) return error.AttentionWorkspaceRowsDiffer;
    }
}
