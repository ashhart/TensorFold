//! Actual checkpoint teacher comparison: independent CPU arithmetic versus the resident GPU decode path.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const reference_module = @import("reference.zig");
const Reference = reference_module.Reference;
const engram_rows = tf.deepseek_v41_engram_rows;
const nrms_limit: f64 = 0.01;

pub const Result = struct {
    tokens: usize,
    max_absolute_error: f64,
    rms_error: f64,
    normalized_rms_error: f64,
    /// KL divergence of the teacher's next-token distribution from the engine's, per row: mean and worst.
    mean_kl: f64 = 0,
    max_kl: f64 = 0,
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

/// The same argmax over the FP64 chain's own logits.
fn maximumWide(row: []const f64) !u32 {
    var id: u32 = 0;
    var best = -std.math.inf(f64);
    for (row, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonfiniteTeacherLogits;
        if (v > best) {
            best = v;
            id = @intCast(i);
        }
    }
    return id;
}

fn streamRow(p: *anyopaque, layer: usize, row: u32, out: []u16) anyerror!void {
    const r: *engram_rows.Rows = @ptrCast(@alignCast(p));
    // `layer` arrives as the reference's own index (0 for layer 1, 1 for layer 14), which is the reader's order.
    try r.rowsU32(layer, &.{row}, 256, out);
    if (std.c.getenv("TF_DSV41_DUMP_ROW") != null) {
        // One line per row fetched: enough to check these exact bytes against a direct read of the shard.
        var fnv: u64 = 0xcbf29ce484222325;
        for (std.mem.sliceAsBytes(out)) |b| {
            fnv ^= b;
            fnv *%= 0x100000001b3;
        }
        var hex: [32]u8 = undefined;
        std.debug.print("ROWDUMP layer={d} row={d} fnv={s} first={x},{x},{x},{x}\n", .{ layer, row, std.fmt.bufPrint(&hex, "{x}", .{fnv}) catch "?", out[0], out[1], out[2], out[3] });
    }
}
pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !Result {
    return runOf(f32, gpa, io, model, ids, token_map, dir);
}

/// The same oracle against the FP64 chain -- the reference chain at its ideal contract (every reduction class
/// widened). The goal's band is written against that chain, not against the FP32 source contract.
pub fn runF64(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !Result {
    return runOf(f64, gpa, io, model, ids, token_map, dir);
}

fn runOf(comptime T: type, gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !Result {
    if (ids.len == 0 or ids.len > 256) return error.InvalidOracleWindow;
    var reference = try reference_module.ReferenceOf(T).init(gpa, model.cfg, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    defer reference.deinit();
    // A streamed model keeps its Engram tables on disk; give the reference a row source so the comparison can happen.
    var stream_rows: ?*engram_rows.Rows = null;
    if (std.c.getenv("TF_DSV41_ENGRAM_STREAM_DIR")) |d| {
        stream_rows = try engram_rows.Rows.open(gpa, io, std.mem.span(d), 0, 1, &.{ 1, 14 }, 4096);
        reference.attachRows(.{ .ptr = stream_rows.?, .rowFn = streamRow });
    }
    defer if (stream_rows) |r| r.close();
    var margins = MarginLog{};
    reference.attention_state.score_probe = .{ .ctx = &margins, .record = &MarginLog.capture };
    const cpu = try gpa.alloc(T, ids.len * 129280);
    defer gpa.free(cpu);
    const gpu = try gpa.alloc(f32, cpu.len);
    defer gpa.free(gpu);
    try model.reset();
    var result = Result{ .tokens = ids.len, .max_absolute_error = 0, .rms_error = 0, .normalized_rms_error = 0, .greedy_disagreements = 0, .gpu_seconds = 0, .cpu_seconds = 0 };
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    // The engine's logits only. The FP64 teacher is deterministic and stored; the band reads this file against it,
    // so an arm does not pay the CPU chain again.
    if (std.c.getenv("TF_DSV41_ORACLE_GPU_ONLY") != null) {
        for (ids, 0..) |id, i| {
            const actual = try model.forward(&.{id}, .all);
            result.gpu_seconds += model.last_gpu_seconds;
            @memcpy(gpu[i * 129280 ..][0..129280], actual);
        }
        const gp = try std.fs.path.join(gpa, &.{ dir, "gpu-logits.f32" });
        defer gpa.free(gp);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = gp, .data = std.mem.sliceAsBytes(gpu) });
        const report = try std.fs.path.join(gpa, &.{ dir, "oracle.json" });
        defer gpa.free(report);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report, .data = "{\"mode\":\"oracle-gpu-only\"}" });
        std.debug.print("oracle-gpu-only: {d} tokens, {d:.3} gpu seconds\n", .{ ids.len, result.gpu_seconds });
        return result;
    }
    for (ids, 0..) |id, i| {
        const actual = try model.forward(&.{id}, .all);
        result.gpu_seconds += model.last_gpu_seconds;
        @memcpy(gpu[i * 129280 ..][0..129280], actual);
        const start = mtl.clock.seconds();
        const expected = try reference.step(id);
        result.cpu_seconds += mtl.clock.seconds() - start;
        @memcpy(cpu[i * 129280 ..][0..129280], expected);
        const kl = try klRow(actual, expected);
        result.mean_kl += kl;
        result.max_kl = @max(result.max_kl, kl);
        const expected_token = if (T == f32) try maximum(expected) else try maximumWide(expected);
        if (try maximum(actual) != expected_token) result.greedy_disagreements += 1;
        for (actual, expected) |a, b| {
            if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonfiniteTeacherLogits;
            const wide: f64 = if (T == f32) @as(f64, b) else b;
            const error_: f64 = @as(f64, a) - wide;
            result.max_absolute_error = @max(result.max_absolute_error, @abs(error_));
            squared_error += error_ * error_;
            squared_reference += wide * wide;
        }
        std.debug.print("oracle teacher {d}/{d}: GPU token={d}, CPU token={d}\n", .{ i + 1, ids.len, try maximum(actual), expected_token });
    }
    if (ids.len > 0) result.mean_kl /= @as(f64, @floatFromInt(ids.len));
    result.rms_error = @sqrt(squared_error / @as(f64, @floatFromInt(cpu.len)));
    result.normalized_rms_error = if (squared_reference == 0) result.rms_error else @sqrt(squared_error / squared_reference);
    result.numeric_gate_passed = result.normalized_rms_error <= result.normalized_rms_limit;
    result.greedy_gate_passed = result.greedy_disagreements == 0;
    // The FP32 chain's teacher keeps the shipped file name; the FP64 chain's teacher writes its own FP64 file,
    // so the two chains' logits are read on their own rulers and never confused.
    const cp = try std.fs.path.join(gpa, &.{ dir, if (T == f32) "oracle-logits.f32" else "oracle-logits.f64" });
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

/// The teacher's distribution against the engine's on one logits row: KL(teacher || engine), f64.
fn klRow(actual: []const f32, expected: anytype) !f64 {
    var max_a: f64 = -std.math.inf(f64);
    var max_e: f64 = -std.math.inf(f64);
    for (actual, expected) |a, b| {
        if (!std.math.isFinite(a)) return error.NonfiniteTeacherLogits;
        const w: f64 = if (@TypeOf(expected[0]) == f32) @as(f64, b) else b;
        if (!std.math.isFinite(w)) return error.NonfiniteTeacherLogits;
        max_a = @max(max_a, @as(f64, a));
        max_e = @max(max_e, w);
    }
    var sum_a: f64 = 0;
    var sum_e: f64 = 0;
    for (actual, expected) |a, b| {
        const w: f64 = if (@TypeOf(expected[0]) == f32) @as(f64, b) else b;
        sum_a += @exp(@as(f64, a) - max_a);
        sum_e += @exp(w - max_e);
    }
    const lse_a = max_a + @log(sum_a);
    const lse_e = max_e + @log(sum_e);
    var kl: f64 = 0;
    for (actual, expected) |a, b| {
        const w: f64 = if (@TypeOf(expected[0]) == f32) @as(f64, b) else b;
        const p = @exp(w - lse_e);
        kl += p * ((w - lse_e) - (@as(f64, a) - lse_a));
    }
    return @max(kl, 0);
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
