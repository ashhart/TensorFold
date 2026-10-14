//! Actual checkpoint teacher comparison: independent CPU arithmetic versus the resident GPU decode path.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("tensorfold").deepseek_v41.Model;
const Reference = @import("reference.zig").Reference;
const nrms_limit: f64 = 0.01;

pub const Result = struct {
    tokens: usize,
    max_absolute_error: f64,
    rms_error: f64,
    normalized_rms_error: f64,
    greedy_disagreements: usize,
    gpu_seconds: f64,
    cpu_seconds: f64,
    normalized_rms_limit: f64 = nrms_limit,
    numeric_gate_passed: bool = false,
    greedy_gate_passed: bool = false,
};

fn maximum(row: []const f32) !u32 {
    var id: u32 = 0;
    var best = -std.math.inf(f32);
    for (row, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonfiniteTeacherLogits;
        if (v > best) {
            best = v;
            id = @intCast(i);
        }
    }
    return id;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !Result {
    if (ids.len == 0 or ids.len > 16) return error.InvalidOracleWindow;
    var reference = try Reference.init(gpa, model.cfg.text_config, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    defer reference.deinit();
    const cpu = try gpa.alloc(f32, ids.len * 129280);
    defer gpa.free(cpu);
    const gpu = try gpa.alloc(f32, cpu.len);
    defer gpa.free(gpu);
    try model.reset();
    var result = Result{ .tokens = ids.len, .max_absolute_error = 0, .rms_error = 0, .normalized_rms_error = 0, .greedy_disagreements = 0, .gpu_seconds = 0, .cpu_seconds = 0 };
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    for (ids, 0..) |id, i| {
        const actual = try model.forward(&.{id}, .all);
        result.gpu_seconds += model.last_gpu_seconds;
        @memcpy(gpu[i * 129280 ..][0..129280], actual);
        const start = mtl.clock.seconds();
        const expected = try reference.step(id);
        result.cpu_seconds += mtl.clock.seconds() - start;
        @memcpy(cpu[i * 129280 ..][0..129280], expected);
        if (try maximum(actual) != try maximum(expected)) result.greedy_disagreements += 1;
        for (actual, expected) |a, b| {
            if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonfiniteTeacherLogits;
            const error_: f64 = @as(f64, a) - b;
            result.max_absolute_error = @max(result.max_absolute_error, @abs(error_));
            squared_error += error_ * error_;
            squared_reference += @as(f64, b) * b;
        }
        std.debug.print("oracle teacher {d}/{d}: GPU token={d}, CPU token={d}\n", .{ i + 1, ids.len, try maximum(actual), try maximum(expected) });
    }
    result.rms_error = @sqrt(squared_error / @as(f64, @floatFromInt(cpu.len)));
    result.normalized_rms_error = if (squared_reference == 0) result.rms_error else @sqrt(squared_error / squared_reference);
    result.numeric_gate_passed = result.normalized_rms_error <= result.normalized_rms_limit;
    result.greedy_gate_passed = result.greedy_disagreements == 0;
    const cp = try std.fs.path.join(gpa, &.{ dir, "oracle-logits.f32" });
    defer gpa.free(cp);
    const gp = try std.fs.path.join(gpa, &.{ dir, "gpu-logits.f32" });
    defer gpa.free(gp);
    const report = try std.fs.path.join(gpa, &.{ dir, "oracle.json" });
    defer gpa.free(report);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cp, .data = std.mem.sliceAsBytes(cpu) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = gp, .data = std.mem.sliceAsBytes(gpu) });
    const json = try std.json.Stringify.valueAlloc(gpa, result, .{});
    defer gpa.free(json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report, .data = json });
    std.debug.print("{s}\n", .{json});
    try enforceGate(result.normalized_rms_error, result.greedy_disagreements);
    return result;
}

fn enforceGate(normalized_rms_error: f64, greedy_disagreements: usize) !void {
    if (!std.math.isFinite(normalized_rms_error)) return error.NonfiniteTeacherLogits;
    if (normalized_rms_error > nrms_limit) return error.TeacherReferenceMismatch;
    if (greedy_disagreements != 0) return error.TeacherGreedyMismatch;
}

test "teacher numeric gate remains independent of greedy agreement" {
    try std.testing.expectError(error.TeacherReferenceMismatch, enforceGate(0.04, 0));
    try std.testing.expectError(error.TeacherReferenceMismatch, enforceGate(0.010000001, 1));
    try std.testing.expectError(error.TeacherGreedyMismatch, enforceGate(0.01, 1));
    try enforceGate(0.01, 0);
    try std.testing.expectError(error.NonfiniteTeacherLogits, enforceGate(std.math.nan(f64), 0));
    try std.testing.expectError(error.NonfiniteTeacherLogits, enforceGate(std.math.inf(f64), 0));
}
