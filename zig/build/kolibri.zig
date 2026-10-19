//! Kolibri family construction and CPU metadata controls on the maintained CUDA runtime.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/kolibri1/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn host(b: *std.Build, m: *std.Build.Module, api: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-kolibri-host", "Kolibri config, ownership and copy-draft metadata without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    b.step("test-kolibri-context-host", "Logical Kolibri context and physical verification reserve").dependOn(
        &b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"Kolibri context:"} })).step,
    );
    const options = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_options.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "engine_api", .module = api }} });
    b.step("test-native-options-host", "Generic family option forwarding through declared fields only").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = options })).step);
}
