//! The checkpoint I/O regressions alone, Debug and ReleaseFast, for -Dtarget; `test` runs them in their full binaries.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const name = if (mode == .fast) "test-checkpoint-io-fast" else "test-checkpoint-io-host";
        const step = b.step(name, "Header-only checkpoint admission, Qwen shard release and header checks without a GPU");
        const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = mode, .link_libc = true });
        core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
        core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = mode, .link_libc = true }));
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"checkpoint admission:"} })).step);
        const qwen = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/loader.zig"), .target = target, .optimize = mode, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen, .filters = &.{"Qwen shard integrity:"} })).step);
    }
}
