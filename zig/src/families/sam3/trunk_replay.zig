const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");
const Store = @import("checkpoint.zig").Store;
const dense_device = @import("dense_device.zig");
const projection = @import("projection.zig");
extern "c" fn erf(f64) f64;

const Route = dense_device.Layout;

const sample_rows = 64;
const Kind = enum { qkv, attention_out, expand, contract, deconv };
const roles = [_]struct { suffix: []const u8, role: Kind }{
    .{ .suffix = ".attn.qkv.input", .role = .qkv },
    .{ .suffix = ".attn.proj.input", .role = .attention_out },
    .{ .suffix = ".mlp.fc1.input", .role = .expand },
    .{ .suffix = ".mlp.fc2.input", .role = .contract },
    .{ .suffix = ".deconv.input", .role = .deconv },
};

fn productionRoute(kind: Kind) Route {
    return switch (kind) {
        .deconv => .wide,
        inline else => |k| projection.fp32WeightLayout(@field(projection.TrunkRole, @tagName(k))),
    };
}

// The packed deconvolution matrix `[4 out, in]` (row tap * out + output) and its zero bias, as the route uses them.
fn packDeconv(a: std.mem.Allocator, raw: st.Tensor) ![2]st.Tensor {
    if (raw.dtype != .f32 or raw.rank != 4 or raw.dim(2) != 2 or raw.dim(3) != 2) return error.TrunkReplayShape;
    const inputs = raw.dim(0);
    const outputs = raw.dim(1);
    const matrix = try a.alloc(u8, raw.bytes.len);
    for (0..4) |tap| for (0..outputs) |o| for (0..inputs) |c| {
        const from = ((c * outputs + o) * 4 + tap) * 4;
        @memcpy(matrix[((tap * outputs + o) * inputs + c) * 4 ..][0..4], raw.bytes[from..][0..4]);
    };
    const zeros = try a.alloc(u8, 16 * outputs);
    @memset(zeros, 0);
    var weight = st.Tensor{ .dtype = .f32, .rank = 2, .shape = @splat(1), .bytes = matrix };
    weight.shape[0] = 4 * outputs;
    weight.shape[1] = inputs;
    var bias = st.Tensor{ .dtype = .f32, .rank = 1, .shape = @splat(1), .bytes = zeros };
    bias.shape[0] = 4 * outputs;
    return .{ weight, bias };
}

const Result = struct {
    key: []const u8,
    role: Kind,
    layout: Route,
    shape: [3]usize,
    tiny_inputs: usize,
    gpu_ms: f64,
    native_max: f64,
    model_max: f64,
    native_mean: f64,
    model_mean: f64,
    worse_columns: usize,
    worst_column: usize,
    worst_column_native: f64,
    worst_column_model: f64,
};

fn gelu(value: f64) f64 {
    return 0.5 * value * (1 + erf(value * @sqrt(@as(f64, 0.5))));
}

fn upload(g: *gpu.Gpu, value: st.Tensor) !gpu.Tensor {
    if (value.dtype != .f32) return error.ExpectedFloat32Operand;
    const tensor = try g.alloc(value.numel(), .f32);
    @memcpy(tensor.slice(u8, value.bytes.len), value.bytes);
    return tensor;
}

fn read(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

fn project(g: *gpu.Gpu, x: gpu.Tensor, w: gpu.Tensor, b: gpu.Tensor, shape: [3]usize, layout: Route) !gpu.Tensor {
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const out = try dense_device.run(g, x, w, b, shape[0], shape[1], shape[2], layout);
    errdefer out.deinit();
    try g.endBatch();
    return out;
}

// One projection: FP64 oracle and a sequential FMA FP32 model on sampled rows, against the native layout.
fn replay(g: *gpu.Gpu, a: std.mem.Allocator, key: []const u8, role: Kind, layout: Route, input: st.Tensor, weight: st.Tensor, bias: st.Tensor) !Result {
    @setFloatMode(.strict);
    const cols = weight.dim(0);
    const depth = weight.dim(1);
    if (input.numel() % depth != 0 or bias.numel() != cols) return error.TrunkReplayShape;
    const rows = input.numel() / depth;
    const x = try upload(g, input);
    defer x.deinit();
    const w = try upload(g, weight);
    defer w.deinit();
    const b = try upload(g, bias);
    defer b.deinit();
    const shape = [3]usize{ rows, cols, depth };
    const actual = try project(g, x, w, b, shape, layout);
    defer actual.deinit();
    var fastest = std.math.inf(f64);
    for (0..3) |_| {
        const before = g.gpu_seconds;
        const again = try project(g, x, w, b, shape, layout);
        defer again.deinit();
        fastest = @min(fastest, g.gpu_seconds - before);
        if (!std.mem.eql(u8, actual.slice(u8, actual.count * 4), again.slice(u8, again.count * 4))) return error.TrunkReplayNotRepeatable;
    }
    var tiny: usize = 0;
    for (0..x.count) |i| {
        const bits = @as(u32, @bitCast(x.get(i))) & 0x7fffffff;
        tiny += @intFromBool(bits != 0 and bits < 0x2d800000);
    }
    const native_columns = try a.alloc(f64, cols);
    defer a.free(native_columns);
    const model_columns = try a.alloc(f64, cols);
    defer a.free(model_columns);
    @memset(native_columns, 0);
    @memset(model_columns, 0);
    var native_sum: f64 = 0;
    var model_sum: f64 = 0;
    const exact = try a.alloc(f64, sample_rows * cols);
    defer a.free(exact);
    const model = try g.alloc(sample_rows * cols, .f32);
    defer model.deinit();
    const models = model.slice(f32, model.count);
    for (0..sample_rows) |sample| {
        const row = (sample * 7919 + 13) % rows;
        for (0..cols) |col| {
            var sum: f64 = 0;
            var chain: f32 = 0;
            for (0..depth) |k| {
                const xv = read(input.bytes, row * depth + k);
                const wv = read(weight.bytes, col * depth + k);
                sum += @as(f64, xv) * wv;
                chain = @mulAdd(f32, xv, wv, chain);
            }
            exact[sample * cols + col] = sum + read(bias.bytes, col);
            models[sample * cols + col] = chain + read(bias.bytes, col);
        }
    }
    // The model takes the same kept GELU kernel as the native epilogue, so only the sums differ.
    const activated = if (role == .expand) try g.allocDevice(model.count, .f32) else model;
    defer if (role == .expand) activated.deinit();
    if (role == .expand) try g.dispatch("im_sam_gelu", &.{ model, activated }, @as(u32, @intCast(model.count)), gpu.mtl.Size.of(model.count, 1, 1), gpu.mtl.Size.of(256, 1, 1));
    for (0..sample_rows) |sample| {
        const row = (sample * 7919 + 13) % rows;
        for (0..cols) |col| {
            const want = if (role == .expand) gelu(exact[sample * cols + col]) else exact[sample * cols + col];
            const modelled: f64 = activated.get(sample * cols + col);
            const native_error = @abs(want - actual.get(row * cols + col));
            const model_error = @abs(want - modelled);
            native_columns[col] = @max(native_columns[col], native_error);
            model_columns[col] = @max(model_columns[col], model_error);
            native_sum += native_error;
            model_sum += model_error;
        }
    }
    var result = Result{ .key = key, .role = role, .layout = layout, .shape = shape, .tiny_inputs = tiny, .gpu_ms = fastest * 1e3, .native_max = std.mem.max(f64, native_columns), .model_max = std.mem.max(f64, model_columns), .native_mean = native_sum / @as(f64, @floatFromInt(sample_rows * cols)), .model_mean = model_sum / @as(f64, @floatFromInt(sample_rows * cols)), .worse_columns = 0, .worst_column = 0, .worst_column_native = 0, .worst_column_model = 0 };
    var worst_ratio: f64 = -1;
    for (native_columns, model_columns, 0..) |native, reference, col| {
        result.worse_columns += @intFromBool(native > reference);
        const ratio = if (reference == 0) (if (native == 0) 0 else std.math.inf(f64)) else native / reference;
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            result.worst_column = col;
            result.worst_column_native = native;
            result.worst_column_model = reference;
        }
    }
    return result;
}

/// `--calibrate-trunk CAPTURE`: the replay's FP32 model against PyTorch's own output on a PyTorch projection capture.
pub fn calibrate(init: std.process.Init, g: *gpu.Gpu, path: []const u8) !void {
    @setFloatMode(.strict);
    const a = init.arena.allocator();
    var file = try st.File.open(init.gpa, init.io, path);
    defer file.close(init.io);
    const input = file.get("input") orelse return error.MissingProjectionInput;
    const weight = file.get("weight") orelse return error.MissingProjectionWeight;
    const bias = file.get("bias") orelse return error.MissingProjectionBias;
    const truth = file.get("output") orelse return error.MissingProjectionReference;
    const oracle = file.get("reference64") orelse return error.MissingFloat64Reference;
    const indices = file.get("row_indices") orelse return error.MissingReferenceRows;
    const activated = file.get("linear_output") != null;
    const cols = weight.dim(0);
    const depth = weight.dim(1);
    const samples = indices.dim(0);
    const model = try g.alloc(samples * cols, .f32);
    defer model.deinit();
    for (0..samples) |sample| {
        const row: usize = @intCast(std.mem.readInt(i64, indices.bytes[sample * 8 ..][0..8], .little));
        for (0..cols) |col| {
            var chain: f32 = 0;
            for (0..depth) |k| chain = @mulAdd(f32, read(input.bytes, row * depth + k), read(weight.bytes, col * depth + k), chain);
            model.slice(f32, model.count)[sample * cols + col] = chain + read(bias.bytes, col);
        }
    }
    const modelled = if (activated) try g.allocDevice(model.count, .f32) else model;
    defer if (activated) modelled.deinit();
    if (activated) try g.dispatch("im_sam_gelu", &.{ model, modelled }, @as(u32, @intCast(model.count)), gpu.mtl.Size.of(model.count, 1, 1), gpu.mtl.Size.of(256, 1, 1));
    const columns = try a.alloc(f64, 2 * cols);
    @memset(columns, 0);
    var sums = [2]f64{ 0, 0 };
    for (0..samples) |sample| {
        const row: usize = @intCast(std.mem.readInt(i64, indices.bytes[sample * 8 ..][0..8], .little));
        for (0..cols) |col| {
            const want: f64 = @bitCast(std.mem.readInt(u64, oracle.bytes[(sample * cols + col) * 8 ..][0..8], .little));
            const errors = [2]f64{ @abs(want - modelled.get(sample * cols + col)), @abs(want - read(truth.bytes, row * cols + col)) };
            for (errors, 0..) |e, i| {
                columns[i * cols + col] = @max(columns[i * cols + col], e);
                sums[i] += e;
            }
        }
    }
    var looser: usize = 0;
    for (0..cols) |col| looser += @intFromBool(columns[col] > columns[cols + col]);
    const count: f64 = @floatFromInt(samples * cols);
    std.debug.print("{s}: FP64 max model {e:.3}, PyTorch {e:.3}; mean model {e:.3}, PyTorch {e:.3}; {d}/{d} columns where the model is worse than PyTorch\n", .{ path, std.mem.max(f64, columns[0..cols]), std.mem.max(f64, columns[cols..]), sums[0] / count, sums[1] / count, looser, cols });
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

/// `--replay-trunk CHECKPOINT TRACE_DIR RECEIPT [ROLE=LAYOUT | only=TEXT]...`: every traced FP32-weight input.
pub fn run(init: std.process.Init, g: *gpu.Gpu, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.ExpectedCheckpointTraceAndReceipt;
    const a = init.arena.allocator();
    var layouts: [5]Route = undefined;
    for (std.enums.values(Kind)) |role| layouts[@backingInt(role)] = productionRoute(role);
    var only: []const u8 = "";
    for (args[3..]) |arg| {
        const split = std.mem.indexOfScalar(u8, arg, '=') orelse return error.ExpectedRoleLayout;
        if (std.mem.eql(u8, arg[0..split], "only")) {
            only = arg[split + 1 ..];
            continue;
        }
        const role = std.meta.stringToEnum(Kind, arg[0..split]) orelse return error.UnknownTrunkRole;
        layouts[@backingInt(role)] = std.meta.stringToEnum(Route, arg[split + 1 ..]) orelse return error.UnknownLayout;
    }
    var store = try Store.open(init.gpa, init.io, args[0]);
    defer store.deinit();
    var files: std.ArrayList([]const u8) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(init.io, args[1], .{ .iterate = true });
    defer dir.close(init.io);
    var it = dir.iterate();
    while (try it.next(init.io)) |entry| if (std.mem.endsWith(u8, entry.name, ".input.safetensors") and std.mem.indexOf(u8, entry.name, only) != null)
        try files.append(a, try std.fmt.allocPrint(a, "{s}/{s}", .{ args[1], entry.name }));
    std.mem.sort([]const u8, files.items, {}, lessThan);
    g.buffer_cache_limit = 1 << 30;
    var results: std.ArrayList(Result) = .empty;
    var worse: usize = 0;
    for (files.items) |path| {
        var trace = try st.File.open(init.gpa, init.io, path);
        defer trace.close(init.io);
        if (trace.names.count() != 1) return error.ExpectedOneTracedInput;
        const name = trace.names.keys()[0];
        const entry = for (roles) |r| {
            if (std.mem.endsWith(u8, name, r.suffix)) break r;
        } else continue;
        const deconv = entry.role == .deconv;
        const suffix: []const u8 = if (deconv) ".deconv.input" else ".input";
        const key = try a.dupe(u8, name[0 .. name.len - suffix.len]);
        const raw = try store.get(try std.fmt.allocPrint(a, "{s}.weight", .{key}));
        const weight, const bias = if (deconv) try packDeconv(a, raw) else .{ raw, try store.get(try std.fmt.allocPrint(a, "{s}.bias", .{key})) };
        const layout = layouts[@backingInt(entry.role)];
        const result = try replay(g, a, key, entry.role, layout, trace.get(name).?, weight, bias);
        worse += result.worse_columns + @intFromBool(result.native_mean > result.model_mean);
        std.debug.print("{s} {t}: {d:.3} ms, FP64 max {e:.3} (model {e:.3}), mean {e:.3} (model {e:.3}), {d}/{d} columns worse, {d} tiny inputs\n", .{ key, result.layout, result.gpu_ms, result.native_max, result.model_max, result.native_mean, result.model_mean, result.worse_columns, result.shape[1], result.tiny_inputs });
        try results.append(a, result);
    }
    const json = try std.json.Stringify.valueAlloc(a, .{ .sample_rows = sample_rows, .model = "sequential FP32 FMA over the depth, then the bias; GELU by the kept im_sam_gelu kernel", .results = results.items }, .{ .whitespace = .indent_2 });
    const receipt = try std.Io.Dir.cwd().createFile(init.io, args[2], .{ .exclusive = true });
    defer receipt.close(init.io);
    try receipt.writeStreamingAll(init.io, json);
    try receipt.writeStreamingAll(init.io, "\n");
    std.debug.print("trunk replay: {d} projections, {d} worse columns or means\n", .{ results.items.len, worse });
}
