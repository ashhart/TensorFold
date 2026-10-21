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

/// Indexer-selection boundary per window: contract margin (BF16-rounded FP32 scores), FP64-ideal margin, support flips.
const MarginLog = struct {
    records: [1024]MarginRecord = undefined,
    len: usize = 0,

    const MarginRecord = struct {
        layer: u32,
        position: u32,
        count: u32,
        selected: u32,
        margin: f32,
        margin_ideal: f32,
        flips: u32,
        lowest_selected: f32,
        highest_unselected: f32,
    };

    fn capture(ctx: *anyopaque, layer: usize, position: usize, count: usize, selected_len: usize, margin: f32, margin_ideal: f32, flips: usize, lowest: f32, highest_unselected: f32) void {
        const self: *MarginLog = @ptrCast(@alignCast(ctx));
        if (self.len == self.records.len) return;
        self.records[self.len] = .{
            .layer = @intCast(layer),
            .position = @intCast(position),
            .count = @intCast(count),
            .selected = @intCast(selected_len),
            .margin = margin,
            .margin_ideal = margin_ideal,
            .flips = @intCast(flips),
            .lowest_selected = lowest,
            .highest_unselected = highest_unselected,
        };
        self.len += 1;
    }

    fn write(self: *MarginLog, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var flips: usize = 0;
        var tight: usize = 0;
        var total: usize = 0;
        for (self.records[0..self.len]) |r| {
            total += 1;
            flips += r.flips;
            if (r.margin <= 0) tight += 1;
            var buf: [320]u8 = undefined;
            const line = try std.fmt.bufPrint(&buf, "{{\"layer\":{d},\"position\":{d},\"count\":{d},\"selected\":{d},\"margin\":{d},\"margin_ideal\":{d},\"flips\":{d},\"lowest_selected\":{d},\"highest_unselected\":{d}}}\n", .{ r.layer, r.position, r.count, r.selected, r.margin, r.margin_ideal, r.flips, r.lowest_selected, r.highest_unselected });
            try text.appendSlice(gpa, line);
        }
        var head_buf: [192]u8 = undefined;
        const head = try std.fmt.bufPrint(&head_buf, "{{\"records\":{d},\"total_flips\":{d},\"zero_or_negative_margin\":{d}}}\n", .{ total, flips, tight });
        const path = try std.fs.path.join(gpa, &.{ dir, "indexer-margins.jsonl" });
        defer gpa.free(path);
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, head);
        try file.writeStreamingAll(io, text.items);
        std.debug.print("indexer margins: {d} windows, {d} flips contract-vs-ideal, {d} zero-or-negative margins\n", .{ total, flips, tight });
    }
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
    var margins = MarginLog{};
    reference.attention_state.score_probe = .{ .ctx = &margins, .record = &MarginLog.capture };
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
    try margins.write(gpa, io, dir);
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
