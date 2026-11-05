//! The real case CLI against small recorded teachers, with a deliberately wrong but repeatable backend.
const std = @import("std");
const fixture = @import("case_fixture");
const t = std.testing;
const Case = enum { full, partial, absent, missing, single, one_branch, start_only, wrong_shape, partial_two, zero };

fn check(kind: Case, backend: []const u8, expected: u8, needle: ?[]const u8) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "case.safetensors" });
    const steps: usize = if (kind == .single) 1 else 3;
    var exact = [_]f32{ 1.001, -2.002, 4.004, -8.008, 0.5005, -1.001, 2.002, -4.004, 0.25025, -0.5005, 1.001, -2.002 };
    var rounded = [_]f32{ 1, -2, 4, -8, 0.5, -1, 2, -4, 0.25, -0.5, 1, -2 };
    if (kind == .zero) {
        @memset(&exact, 0);
        @memset(&rounded, 0);
    }
    const prompt = [_]u16{ 0x3f80, 0x3f80 };
    const times = [_]f32{ 900, 600, 300 };
    const sigmas = [_]f32{ 0.9, 0.6, 0.3, 0 };
    const coverage: usize = if (kind == .partial) 1 else if (kind == .partial_two) 2 else steps;
    var entries: std.ArrayList(fixture.Entry) = .empty;
    try entries.appendSlice(a, &.{
        .{ .name = "latents", .dtype = "F32", .shape = &.{ 1, 1, 4 }, .bytes = std.mem.sliceAsBytes(exact[0..4]) },
        .{ .name = "prompt", .dtype = "BF16", .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&prompt) },
        .{ .name = "negative", .dtype = "BF16", .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&prompt) },
        .{ .name = "timesteps", .dtype = "F32", .shape = &.{steps}, .bytes = std.mem.sliceAsBytes(times[0..steps]) },
        .{ .name = "sigmas", .dtype = "F32", .shape = &.{steps + 1}, .bytes = std.mem.sliceAsBytes(sigmas[0 .. steps + 1]) },
        .{ .name = "inputs", .dtype = "F32", .shape = &.{ steps, 1, 1, 4 }, .bytes = std.mem.sliceAsBytes(exact[0 .. steps * 4]) },
        .{ .name = "velocity", .dtype = "F32", .shape = &.{ steps, 1, 1, 4 }, .bytes = std.mem.sliceAsBytes(rounded[0 .. steps * 4]) },
        .{ .name = "velocity_negative", .dtype = "F32", .shape = &.{ steps, 1, 1, 4 }, .bytes = std.mem.sliceAsBytes(rounded[0 .. steps * 4]) },
    });
    if (kind == .start_only) entries.shrinkRetainingCapacity(5);
    if (kind != .absent and kind != .start_only) try entries.appendSlice(a, &.{
        .{ .name = "velocity_f32", .dtype = "F32", .shape = if (kind == .wrong_shape) &.{ coverage, 1, 2, 2 } else &.{ coverage, 1, 1, 4 }, .bytes = std.mem.sliceAsBytes(exact[0 .. coverage * 4]) },
        .{ .name = "velocity_negative_f32", .dtype = "F32", .shape = &.{ coverage, 1, 1, 4 }, .bytes = std.mem.sliceAsBytes(exact[0 .. coverage * 4]) },
    });
    if (kind == .one_branch) _ = entries.pop();
    if (kind != .missing) try fixture.write(a, t.io, tmp.dir, "case.safetensors", entries.items);
    const result = try std.process.run(a, t.io, .{ .argv = &.{ @import("teacher_host_options").probe, backend, path, dir }, .stdout_limit = .limited(4096), .stderr_limit = .limited(65536) });
    if (result.term != .exited or result.term.exited != expected) {
        std.debug.print("teacher child expected exit {d}: {any}\n{s}\n", .{ expected, result.term, result.stderr });
        return error.WrongTeacherExit;
    }
    if (needle) |s| try t.expect(std.mem.indexOf(u8, result.stderr, s) != null);
    if (kind == .missing or kind == .wrong_shape) return;
    const raw = try tmp.dir.readFileAlloc(t.io, "run.json", a, .limited(65536));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{ .allocate = .alloc_always });
    const root = parsed.value.object;
    const status = if (expected == 0) "PASS" else if (expected == 1) "FAIL" else "INCONCLUSIVE";
    try t.expectEqualStrings(status, root.get("status").?.string);
    const repeat = root.get("repeat_identical").?;
    if (kind == .start_only) {
        try t.expect(repeat == .null);
        return;
    }
    try t.expectEqual(!std.mem.eql(u8, backend, "repeat-fault"), repeat.bool);
    const records = root.get("teacher_forced").?.array.items;
    try t.expectEqual(steps, records.len);
    if (std.mem.eql(u8, backend, "element-fault") or std.mem.eql(u8, backend, "negative-fault")) {
        const key = if (std.mem.eql(u8, backend, "element-fault")) "prompt_gate" else "negative_gate";
        for (records, 0..) |record, step| {
            const verdict = record.object.get(key).?.object;
            try t.expectEqualStrings(if (step >= coverage) "INCONCLUSIVE" else if (step == 1) "FAIL" else "PASS", verdict.get("status").?.string);
            if (step == 1) try t.expectEqual(@as(i64, 3), verdict.get("worst_element").?.integer);
        }
    }
    if (std.mem.eql(u8, backend, "nonfinite-fault")) {
        const verdict = records[1].object.get("prompt_gate").?.object;
        try t.expect(!verdict.get("finite").?.bool);
        try t.expect(verdict.get("error_abs").? == .null);
        try t.expectEqual(@as(i64, 2), verdict.get("worst_element").?.integer);
    }
    if (std.mem.eql(u8, backend, "repeat-fault")) try t.expectEqualStrings(if (kind == .partial) "INCONCLUSIVE" else "PASS", root.get("teacher_status").?.string);
}

test "MOSS teacher CLI: all steps inside their bf16 envelope pass" {
    try check(.full, "normal", 0, null);
}
test "MOSS teacher CLI: one prompt element fails at its step despite identical repeats" {
    try check(.full, "element-fault", 1, "step 2 prompt");
}
test "MOSS teacher CLI: the negative branch is also gated" {
    try check(.full, "negative-fault", 1, "step 2 negative");
}
test "MOSS teacher CLI: a missing case file is INCONCLUSIVE" {
    try check(.missing, "normal", 2, "INCONCLUSIVE");
}
test "MOSS teacher CLI: absent float32 tensors are INCONCLUSIVE" {
    try check(.absent, "normal", 2, "INCONCLUSIVE");
}
test "MOSS teacher CLI: partial float32 coverage is INCONCLUSIVE" {
    try check(.partial, "normal", 2, "INCONCLUSIVE");
}
test "MOSS teacher CLI: a one-step case can repeat without indexing a second timestep" {
    try check(.single, "normal", 0, null);
}
test "MOSS teacher CLI: changed repeats fail separately from teacher agreement" {
    try check(.full, "repeat-fault", 1, "DIFFERENT bytes");
}

test "MOSS teacher CLI: one missing branch cannot pass" {
    try check(.one_branch, "normal", 2, "INCONCLUSIVE");
}
test "MOSS teacher CLI: generation without teachers is INCONCLUSIVE" {
    try check(.start_only, "normal", 2, "INCONCLUSIVE");
}
test "MOSS teacher CLI: malformed teacher geometry fails instead of flattening a different grid" {
    try check(.wrong_shape, "normal", 1, "BadCaseTensor");
}
test "MOSS teacher CLI: nonfinite failure remains valid JSON" {
    try check(.full, "nonfinite-fault", 1, "FAIL");
}
test "MOSS teacher CLI: repeat failure takes precedence over incomplete coverage" {
    try check(.partial, "repeat-fault", 1, "DIFFERENT bytes");
}

test "MOSS teacher CLI: a numerical miss takes precedence over later missing coverage" {
    try check(.partial_two, "element-fault", 1, "FAIL");
}
test "MOSS teacher CLI: zero reference rows retain valid diagnostics" {
    try check(.zero, "normal", 0, null);
}
