//! Local backward references and counted real-source faults use the same served shader strings.
const std = @import("std");
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: ?std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, existing: ?*std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/qwen27_learning_checks.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "learning_sources", .module = sources }, .{ .name = "contract", .module = existing orelse @import("kernel_contract.zig").module(b, target, .debug) } } });
}
pub fn steps(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const checks = module(b, b.graph.host, .Debug, metal, engine.import_table.get("qwen_runtime_sources").?, null);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = checks }));
    b.step("test-qwen27-learning-reference-host", "Independent binary64 local adjoints and source fault witnesses, no devices").dependOn(&run.step);
    const exe = b.addExecutable(.{ .name = "tf-qwen27-learning-local", .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen27_learning_main.zig"), .target = b.graph.host, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "checks", .module = checks } } }) });
    b.step("tf-qwen27-learning-local", "Build the bounded four-kernel backward panel without running it").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
