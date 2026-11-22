//! tf-h3-image-prompt-host: the prompt stage on the CPU, Mac or the Spark alike — no Metal.
const std = @import("std");

/// `zig build tf-h3-image-prompt-host` (any -Dtarget, among them aarch64-linux-gnu for the Spark) and
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const tool_step = b.step("tf-h3-image-prompt-host", "MiniMax-H3-Image's prompt stage on the CPU: Qwen3-VL to layer 50, the token refiner, the picture case");
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt-host", .root_module = tool(b, target, optimize) });
    tool_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/h3/prompt_case_test.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = @import("h3_host.zig").coreModule(b, target) },
            .{ .name = "moss_noise", .module = noise(b, target, optimize) },
        },
    }) }));
    const tests = b.step("test-h3-image-prompt-host", "The prompt stage's case writer: byte for byte the Metal tool's layout, and torch's CPU randn");
    tests.dependOn(&run.step);
    test_step.dependOn(&run.step);
}

/// The tool's module: the prompt path's host pieces (its Metal-only parts stay unanalyzed), the host checkpoint
fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const core = @import("h3_host.zig").coreModule(b, target);
    const host_prompt = b.createModule(.{
        .root_source_file = b.path("zig/src/families/h3/prompt.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    const checkpoint = b.createModule(.{
        .root_source_file = b.path("zig/src/families/h3/checkpoint.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    return b.createModule(.{
        .root_source_file = b.path("zig/tests/h3_image_prompt_host.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "h3_prompt", .module = host_prompt },
            .{ .name = "h3_checkpoint", .module = checkpoint },
            .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }) },
            .{ .name = "moss_noise", .module = noise(b, target, optimize) },
        },
    });
}

fn noise(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize });
}
