//! Serving admission controls call the real startup before any checkpoint or device work.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step, api: *std.Build.Module, nemotron: *std.Build.Module) void {
    const refusal = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_refusal.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "nemotron", .module = nemotron } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/cuda_qualification_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "qualified_refusal", .module = refusal }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"CUDA qualification:"} }));
    b.step("test-cuda-qualification-host", "Default and recorded profile admission plus experimental refusal before devices").dependOn(&run.step);
    all.dependOn(&run.step);
}
