//! Optional runtime refusal and discovery regressions with absent files and a fake loader.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const experimental = b.createModule(.{ .root_source_file = b.path("zig/experimental_family_test.zig"), .target = b.graph.host });
    const policy = b.addRunArtifact(b.addTest(.{ .root_module = experimental }));
    b.step("test-experimental-family-host", "Exact release opt-in policy without model or GPU work").dependOn(&policy.step);
    all.dependOn(&policy.step);
    const fixture_module = b.createModule(.{ .target = b.graph.host, .link_libc = true, .link_libcpp = true });
    fixture_module.addCSourceFile(.{ .file = b.path("zig/tests/runtime_discovery_mock.cpp"), .flags = &.{"-std=c++17"} });
    const fixture_library = b.addLibrary(.{ .name = "runtime-discovery-mock", .linkage = .dynamic, .root_module = fixture_module });
    const fixture = b.addOptions();
    fixture.addOptionPath("library", fixture_library.getEmittedBin());
    const exit_module = b.createModule(.{ .root_source_file = b.path("zig/runtime_discovery_exit.zig"), .target = b.graph.host, .link_libc = true });
    exit_module.addOptions("discovery_fixture", fixture);
    const exit_run = b.addRunArtifact(b.addExecutable(.{ .name = "runtime-discovery-exit", .root_module = exit_module }));
    exit_run.has_side_effects = true;
    b.step("test-runtime-discovery-exit-host", "Discovery preserves loaded exit callbacks in separate host processes").dependOn(&exit_run.step);
    all.dependOn(&exit_run.step);
    const root = b.createModule(.{ .root_source_file = b.path("zig/runtime_library_host_test.zig"), .target = b.graph.host, .link_libc = true });
    const fixtures = b.addOptions();
    for ([_][]const u8{ "complete", "abi", "wait", "store", "wrong" }) |kind| {
        const m = b.createModule(.{ .target = b.graph.host, .link_libc = true });
        m.addCSourceFile(.{ .file = b.path("zig/tests/rpc_library_mock.c"), .flags = &.{b.fmt("-DMOCK_{s}=1", .{kind})} });
        const lib = b.addLibrary(.{ .name = b.fmt("rpc-mock-{s}", .{kind}), .linkage = .dynamic, .root_module = m });
        fixtures.addOptionPath(kind, lib.getEmittedBin());
    }
    root.addOptions("rpc_fixtures", fixtures);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"runtime library:"} }));
    const step = b.step("test-runtime-libraries-host", "Optional library refusals and discovery without a GPU");
    step.dependOn(&run.step);
    step.dependOn(&exit_run.step);
    const warnings = b.createModule(.{ .root_source_file = b.path("zig/rpc_warning_check.zig"), .target = b.graph.host, .link_libc = true });
    warnings.addOptions("rpc_fixtures", fixtures);
    const check = b.addRunArtifact(b.addExecutable(.{ .name = "rpc-warning-check", .root_module = warnings }));
    step.dependOn(&check.step);
    all.dependOn(&check.step);
    all.dependOn(&run.step);
}
