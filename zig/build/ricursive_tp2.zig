//! Ripple's two-GB10 CUDA family in the build: its module, its follower and forward tool, and its host tests.
const std = @import("std");

/// Ripple at two ranks (families/ricursive/tp2.zig) over the runtime, the core and the lane core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: ?std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/ricursive/tp2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

/// tf-ricursive-tp2: rank 1's follower, and both ranks' forward rows (prompt and decode) for a pack's logits.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/ricursive_tp2_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "ricursive_tp2", .module = module(b, target, optimize, cuda, core, lanes) } } });
    const exe = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-ricursive-tp2", .root_module = root }), .{});
    b.getInstallStep().dependOn(&exe.step);
    b.step("ricursive-tp2-cuda", "Build Ripple's two-GB10 follower and forward rows tool").dependOn(&exe.step);
}

/// The family's host tests (config, admission, cuts, plan words) in `all`; Linux only, as the family imports NCCL.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    if (b.graph.host.result.os.tag != .linux) return;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, null, cuda, core, lanes) }));
    all.dependOn(&run.step);
    b.step("test-ricursive-tp2-host", "Ripple's two-GB10 CUDA family on the host: config, admission, TP cuts").dependOn(&run.step);
}
