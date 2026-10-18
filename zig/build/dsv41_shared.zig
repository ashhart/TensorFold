const std = @import("std");
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, file: []const u8) *std.Build.Module {
    const tok = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tok);
    return b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/families/deepseek_v41/{s}", .{file})), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
}
pub fn steps(b: *std.Build) void {
    const host = b.addTest(.{ .root_module = module(b, b.graph.host, "shared_test.zig") });
    const run = b.addRunArtifact(host);
    run.setCwd(b.path("."));
    b.step("test-dsv41-shared", "Shared DeepSeek checkpoint host tests, no GPU dependencies").dependOn(&run.step);
    const linux_target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
    const cross = b.addTest(.{ .root_module = module(b, linux_target, "shared_test.zig") });
    _ = cross.getEmittedBin();
    b.step("compile-dsv41-shared", "Compile shared tests for Linux without executing").dependOn(&cross.step);
    const builder = b.createModule(.{ .root_source_file = b.path("zig/dsv41_token_map.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const token_tests = b.addRunArtifact(b.addTest(.{ .root_module = builder }));
    b.step("test-dsv41-token-map", "PCG64 and Unicode normalization without Metal").dependOn(&token_tests.step);
    const tool = b.addExecutable(.{ .name = "tf-dsv41-token-map", .root_module = builder });
    b.step("compile-dsv41-token-map", "Compile actual token-map CLI without running or Metal").dependOn(&tool.step);
}
