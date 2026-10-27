//! The native CUDA host check also compiles the EXL3 prefill body object.
const std = @import("std");

pub fn check(b: *std.Build, target: std.Build.ResolvedTarget, check_native: *std.Build.Step.Compile, flashnext: *std.Build.Module) void {
    const native_check = b.step("check-native-cuda-host", "Compile the real native CUDA server without kernels or GPU execution");
    native_check.dependOn(&check_native.step);
    const prefill_body = b.addObject(.{ .name = "exl3-prefill-body", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exl3_prefill_body_compile.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "flashnext", .module = flashnext }},
    }) });
    _ = prefill_body.getEmittedBin();
    native_check.dependOn(&prefill_body.step);
}
