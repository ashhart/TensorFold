//! Shared retained-input diagnostic capture and metrics.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const projection = @import("projection.zig");
const math = @import("math.zig");
const attention = tf.deepseek_v41.attention;
const Model = tf.deepseek_v41.Model;
const ex = tf.exl3_projection;

pub const AttentionStage = attention.DiagnosticStage;
pub const attention_widths = [_]usize{ 1280, 1280, 32768, 32768, 512, 512, 512, 512, 32768, 32768, 8192, 5120 };
pub const attention_elements = blk: {
    var total: usize = 0;
    for (attention_widths) |width| total += width;
    break :blk total;
};
pub fn attentionOffset(stage: AttentionStage) usize {
    var offset: usize = 0;
    for (attention_widths[0..@backingInt(stage)]) |width| offset += width;
    return offset;
}
pub const AttentionCapture = struct {
    model: *Model,
    native: mtl.Buffer,
    reference: []u16,

    pub fn view(self: *const AttentionCapture, stage: AttentionStage) attention.View {
        return .{ .buffer = self.native, .offset = attentionOffset(stage) * 2 };
    }
    pub fn row(self: *const AttentionCapture, stage: AttentionStage) []u16 {
        return self.native.slice(u16, attention_elements)[attentionOffset(stage)..][0..attention_widths[@backingInt(stage)]];
    }
    pub fn cpuRow(self: *const AttentionCapture, stage: AttentionStage) []u16 {
        return self.reference[attentionOffset(stage)..][0..attention_widths[@backingInt(stage)]];
    }
    pub fn gpu(ctx: *anyopaque, enc: mtl.ComputeEncoder, stage: AttentionStage, source: attention.View) !void {
        const self: *AttentionCapture = @ptrCast(@alignCast(ctx));
        const cache = &self.model.pipelines.attention.cache;
        if (stage == .kv_quantized) {
            try cache.encodeUnpack(enc, source, self.view(stage), .{ .rows = 1, .width = 512, .stride = 512, .packed_stride = 528, .block = 32, .kind = .window_fp8 });
        } else {
            cache.encodeCopy(enc, source, self.view(stage), @intCast(attention_widths[@backingInt(stage)] * 2));
        }
    }
    pub fn cpu(ctx: *anyopaque, stage: AttentionStage, values: []const u16) void {
        const self: *AttentionCapture = @ptrCast(@alignCast(ctx));
        @memcpy(self.cpuRow(stage), values);
    }
};

pub fn attentionMetric(comptime T: type, gpa: std.mem.Allocator, report: *std.ArrayList(u8), count: *usize, name: []const u8, comparison: []const u8, actual: []const T, expected: []const T) !void {
    if (actual.len == 0 or actual.len != expected.len) return error.InvalidAttentionMetricShape;
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    var max_abs: f64 = 0;
    var worst: usize = 0;
    var unequal: usize = 0;
    var first: ?usize = null;
    for (actual, expected, 0..) |a, r, i| {
        const x = if (T == u16) math.f(a) else a;
        const y = if (T == u16) math.f(r) else r;
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonfiniteAttentionDiagnostic;
        const delta = @as(f64, x) - y;
        squared += delta * delta;
        magnitude += @as(f64, y) * y;
        if (@abs(delta) > max_abs) {
            max_abs = @abs(delta);
            worst = i;
        }
        const ab = if (T == u16) a else math.b(a);
        const rb = if (T == u16) r else math.b(r);
        if (ab != rb) {
            unequal += 1;
            if (first == null) first = i;
        }
    }
    const line = try std.json.Stringify.valueAlloc(gpa, .{
        .name = name,
        .comparison = comparison,
        .elements = actual.len,
        .output_dtype = if (T == u16) "bf16" else "f32",
        .normalized_rms_error = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude),
        .max_absolute_error = max_abs,
        .worst_index = worst,
        .bf16_unequal_count = unequal,
        .bf16_first_unequal_index = first,
        .worst_native = if (T == u16) math.f(actual[worst]) else actual[worst],
        .worst_reference = if (T == u16) math.f(expected[worst]) else expected[worst],
    }, .{});
    defer gpa.free(line);
    if (count.* != 0) try report.appendSlice(gpa, ",\n");
    try report.appendSlice(gpa, line);
    count.* += 1;
    std.debug.print("{s}\n", .{line});
}

pub fn attentionProjection(gpa: std.mem.Allocator, model: *Model, matrix: ex.Matrix, input: attention.View, name: []const u8, scratch: *projection.Scratch, expected: []f32, report: *std.ArrayList(u8), count: *usize) !void {
    const input_row = @as([*]const u16, @ptrCast(@alignCast(input.buffer.contents() + input.offset)))[0..matrix.k];
    try projection.applyF32(.{ .trellis = matrix }, input_row, expected[0..matrix.n], scratch);
    const output = model.workspace.view(.logits);
    const cb = model.queue.commandBufferUnretained();
    const enc = cb.compute(.serial);
    var encoding = true;
    errdefer if (encoding) enc.end();
    ex.useMatrix(enc, matrix);
    try ex.encode(&model.pipelines.projection, enc, matrix, .{ .buffer = input.buffer, .offset = input.offset }, .{ .buffer = output.buffer, .offset = output.offset }, model.workspace.projection(), .{ .count = 1, .x_stride = matrix.k, .y_stride = matrix.n, .output = .f32 });
    enc.end();
    encoding = false;
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.OperatorGpuFailure;
    const actual = @as([*]const f32, @ptrCast(@alignCast(output.buffer.contents() + output.offset)))[0..matrix.n];
    try attentionMetric(f32, gpa, report, count, name, "same_native_input_fp64_dot", actual, expected[0..matrix.n]);
}

pub fn diagnosticRow(comptime T: type, view: attention.View, len: usize) []T {
    return @as([*]T, @ptrCast(@alignCast(view.buffer.contents() + view.offset)))[0..len];
}
pub fn attentionWrite(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, data: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}
