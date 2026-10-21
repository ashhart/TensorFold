//! Gate-deciding sensitivity: FP32 and FP64 run the same equations on identical tokens and checkpoint bytes.

const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const reference = @import("reference.zig");

/// The caller writes the logits after its residency counters: artifact metadata reads are not execution.
pub const Logits = struct {
    fp32: []f32,
    fp64: []f64,
    /// Per-token (FP32, FP64) argmax. The caller prints it after its residency counters.
    argmax: [][2]u32,
    pub fn deinit(self: Logits, gpa: std.mem.Allocator) void {
        gpa.free(self.fp32);
        gpa.free(self.fp64);
        gpa.free(self.argmax);
    }
};

pub const Report = struct {
    result: Result,
    logits: Logits,
};

pub const Result = struct {
    tokens: usize,
    max_absolute_error: f64,
    rms_error: f64,
    normalized_rms_error: f64, // sqrt(sum((fp32-fp64)^2)/sum(fp64^2)), the same shape as the whole-model gate
    normalized_rms_error_against_fp32: f64,
    greedy_disagreements: usize,
    fp32_seconds: f64,
    fp64_seconds: f64,
    evidence_origin: []const u8 = "same_checkpoint_same_equations_two_widths_no_gpu",
};

fn argmaxF32(row: []const f32) usize {
    var best: usize = 0;
    var score: f32 = -std.math.inf(f32);
    for (row, 0..) |v, i|
        if (v > score) {
            best = i;
            score = v;
        };
    return best;
}

fn argmaxF64(row: []const f64) usize {
    var best: usize = 0;
    var score: f64 = -std.math.inf(f64);
    for (row, 0..) |v, i|
        if (v > score) {
            best = i;
            score = v;
        };
    return best;
}

pub fn run(gpa: std.mem.Allocator, model: *Model, ids: []const u32, token_map: []const u8) !Report {
    if (ids.len == 0 or ids.len > 16) return error.InvalidOracleWindow;
    const F32 = reference.f32_reference.Reference;
    const F64 = reference.Module(f64).Reference;
    var fp32_ref = try F32.init(gpa, model.cfg.text_config, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    defer fp32_ref.deinit();
    var fp64_ref = try F64.init(gpa, model.cfg.text_config, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    defer fp64_ref.deinit();
    const fp32_logits = try gpa.alloc(f32, ids.len * 129280);
    errdefer gpa.free(fp32_logits);
    const fp64_logits = try gpa.alloc(f64, ids.len * 129280);
    errdefer gpa.free(fp64_logits);
    const argmax = try gpa.alloc([2]u32, ids.len);
    errdefer gpa.free(argmax);
    var result = Result{ .tokens = ids.len, .max_absolute_error = 0, .rms_error = 0, .normalized_rms_error = 0, .normalized_rms_error_against_fp32 = 0, .greedy_disagreements = 0, .fp32_seconds = 0, .fp64_seconds = 0 };
    var squared: f64 = 0;
    var reference_f64_squared: f64 = 0;
    var reference_f32_squared: f64 = 0;
    for (ids, 0..) |id, i| {
        const f32_start = mtl.clock.seconds();
        const expected32 = try fp32_ref.step(id);
        result.fp32_seconds += mtl.clock.seconds() - f32_start;
        const f64_start = mtl.clock.seconds();
        const expected64 = try fp64_ref.step(id);
        result.fp64_seconds += mtl.clock.seconds() - f64_start;
        const wide32 = fp32_logits[i * 129280 ..][0..129280];
        const wide64 = fp64_logits[i * 129280 ..][0..129280];
        @memcpy(wide32, expected32);
        @memcpy(wide64, expected64);
        for (expected32, expected64) |a, b| {
            if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonFiniteReferenceLogits;
            const delta: f64 = @as(f64, a) - b;
            result.max_absolute_error = @max(result.max_absolute_error, @abs(delta));
            squared += delta * delta;
            reference_f64_squared += @as(f64, b) * b;
        }
        argmax[i] = .{ @intCast(argmaxF32(wide32)), @intCast(argmaxF64(wide64)) };
        if (argmax[i][0] != argmax[i][1]) result.greedy_disagreements += 1;
    }
    const count: f64 = @floatFromInt(fp32_logits.len);
    result.rms_error = @sqrt(squared / count);
    result.normalized_rms_error = if (reference_f64_squared == 0) result.rms_error else @sqrt(squared / reference_f64_squared);
    for (fp32_logits) |v| reference_f32_squared += @as(f64, v) * v;
    result.normalized_rms_error_against_fp32 = if (reference_f32_squared == 0) result.rms_error else @sqrt(squared / reference_f32_squared);
    return .{ .result = result, .logits = .{ .fp32 = fp32_logits, .fp64 = fp64_logits, .argmax = argmax } };
}
