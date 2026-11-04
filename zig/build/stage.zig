//! The pipeline stage link (zig/src/fabric/stage): its module, host tests, and compile checks for Linux and macOS.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/fabric/stage/stage.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.graph.host;
    if (host.result.os.tag != .linux and host.result.os.tag != .macos) return;
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .Debug) }));
    run.has_side_effects = true; // loopback sockets
    const step = b.step("test-stage", "Pipeline stage link: rings of two and three ranks on loopback, refusals and failures");
    step.dependOn(&run.step);
    all.dependOn(step);
    const compile = b.step("test-stage-compile", "Compile the stage link's tests for Linux and macOS on aarch64");
    for ([_]std.Target.Os.Tag{ .linux, .macos }) |os| {
        const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = os });
        compile.dependOn(&b.addTest(.{ .root_module = module(b, target, .Debug) }).step);
    }
    all.dependOn(compile);
}
