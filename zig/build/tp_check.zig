//! tf-tp-check (zig/tests/tp_check_main.zig), the mixed-TP gate over rank dumps, and its host tests.

const std = @import("std");
const tp = @import("tp.zig");

fn module(b: *std.Build, root: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("tp", tp.module(b, target, optimize, tp.runtime(b, target, optimize)));
    return m;
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const gate_tests = b.step("test-tp-check", "Mixed-TP gate: dump format, host reference, run-to-run and one-box agreement checks");
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = module(b, "zig/tests/tp_check_test.zig", host, .debug) }));
    gate_tests.dependOn(&tests.step);
    test_step.dependOn(&tests.step);
    const files = b.addRunArtifact(b.addTest(.{ .root_module = module(b, "zig/tests/tp_check_file_test.zig", host, .ReleaseSafe) }));
    gate_tests.dependOn(&files.step); // about 2 GiB while it runs, so not in every `zig build test`
    // ReleaseSafe whatever -Doptimize says: a gate keeps its safety checks
    const exe = b.addExecutable(.{ .name = "tf-tp-check", .root_module = module(b, "zig/tests/tp_check_main.zig", target, .ReleaseSafe) });
    b.step("tf-tp-check", "The mixed-TP gate (#1236 rules 1, 2, 4) over each rank's collective and logits dumps").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
