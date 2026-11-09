//! Actual Metal/CUDA causal forwards and complete pocket entry points, compiled with no device work.
const std = @import("std");
pub fn compile(b: *std.Build, family: *std.Build.Module, backend: []const u8) void {
    const obj = b.addObject(.{ .name = b.fmt("nemotron-{s}-locate-host", .{backend}), .root_module = b.createModule(.{
        .root_source_file = b.path("zig/locate_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "family", .module = family }},
    }) });
    _ = obj.getEmittedBin();
    b.step(b.fmt("compile-nemotron-{s}-locate-host", .{backend}), "Compile actual causal forward and learner entries without shaders or GPU execution").dependOn(&obj.step);
}
