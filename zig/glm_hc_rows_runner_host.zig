//! HC-only runner admission and actual legacy filter selection without loading CUDA.
const std = @import("std");
const plan = @import("tests/cuda/contract_glm_hc_rows_plan.zig");
const after = @import("tests/cuda/contract_after.zig");
const fixture = @import("tests/cuda/contract_glm_hc_rows.zig");
const t = std.testing;
test "GLM HC runner: explicit commands reach numerical and embedded limits modes" {
    try t.expectEqual(@as(?plan.Mode, .numerical), plan.mode("glm-hc-rows-contract"));
    try t.expectEqual(@as(?plan.Mode, .limits), plan.mode("glm-hc-rows-limits"));
    try t.expect(plan.mode("glm-hc-rows-contract-extra") == null);
}
test "GLM HC runner: original filter already selects replacement and wrong-row control" {
    var f: fixture.Fixture = undefined;
    const words = [_][]const u8{ "glm HC", "hc_partial8_rows" };
    for ([_]bool{ false, true }, plan.names) |fault, name| {
        f.fault = fault;
        const k = f.kernel();
        try t.expectEqualStrings(name, k.spec.name);
        try t.expect(after.wanted(k.spec.name, &words));
        try t.expectEqual(fault, k.control != null);
    }
}

test "GLM HC runner: measured limits refuse spills, zero occupancy and missing or mismatched usage" {
    const limits = @import("core").contract.limits;
    const target = limits.target(121).?;
    const good: limits.Usage = .{ .regs = 32, .static = 256, .bound = 1024 };
    var notes: std.ArrayList(u8) = .empty;
    defer notes.deinit(t.allocator);
    for (plan.launches) |launch| {
        try t.expectEqual(@as(usize, 0), try plan.audit(t.allocator, launch, good, good, 1, target, &notes));
        var bad = good;
        bad.local = 4;
        try t.expect(try plan.audit(t.allocator, launch, good, bad, 1, target, &notes) > 0);
        bad = good;
        bad.regs += 1;
        try t.expect(try plan.audit(t.allocator, launch, good, bad, 1, target, &notes) > 0);
        bad = good;
        bad.regs = 0;
        try t.expect(try plan.audit(t.allocator, launch, bad, bad, 1, target, &notes) > 0);
        try t.expect(try plan.audit(t.allocator, launch, good, good, 0, target, &notes) > 0);
        bad = good;
        bad.bound = 128;
        try t.expect(try plan.audit(t.allocator, launch, good, bad, 1, target, &notes) > 0);
    }
}
test "GLM HC runner: summary names both fixtures and records a nonvacuous result" {
    var bytes: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&bytes);
    try writer.print(plan.summary, .{ plan.names[0], plan.names[1], 1, 1, 0, 1, 1, 0 });
    for (plan.names) |name| try t.expect(std.mem.indexOf(u8, writer.buffered(), name) != null);
    try t.expect(std.mem.indexOf(u8, writer.buffered(), "1 kernels, 1 pass, 0 fail; 1 controls, 1 red, 0 missed") != null);
}

test "GLM HC runner: production dispatcher rejects extra arguments before touching CUDA" {
    const run = @import("tests/cuda/contract_glm_hc_rows_run.zig");
    try t.expect(!try run.dispatch(undefined, "unrelated", &.{}, &.{}));
    try t.expectError(error.HcRowsUnexpectedArguments, run.dispatch(undefined, plan.numerical_command, &.{"unused"}, &.{}));
    try t.expectError(error.HcRowsUnexpectedArguments, run.dispatch(undefined, plan.limits_command, &.{"unused"}, &.{}));
}
