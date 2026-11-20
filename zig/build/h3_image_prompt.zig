//! MiniMax-H3-Image's prompt path: its host rules' tests, a host compile of tf-h3-image-prompt, and the Metal tool.
const std = @import("std");

/// `test-h3-image-prompt` (the rules, no GPU) and on macOS `compile-h3-image-prompt` (the tool, no shader compiled).
pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .link_libc = true });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/picture_prompt_test.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-h3-image-prompt", "MiniMax-H3-Image's prompt rules: ids, tags, positions, rotary, Qwen copy sizes, configs").dependOn(&run.step);
    test_step.dependOn(&run.step);
    if (target.result.os.tag != .macos) return;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const weights = b.addTest(.{ .root_module = tool(b, target, .Debug, metal) });
    const weights_step = b.step("test-h3-image-prompt-weights", "The prompt path's weights opened from a fixture, only the accepted tensors read ahead, prefetch on and off");
    for ([_][]const u8{ "16", "0" }) |count| {
        const r = b.addRunArtifact(weights);
        r.setEnvironmentVariable("TF_LOAD_PREFETCH", count);
        weights_step.dependOn(&r.step);
    }
    test_step.dependOn(weights_step);
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt-host-compile", .root_module = tool(b, target, .Debug, metal) });
    b.step("compile-h3-image-prompt", "Compile tf-h3-image-prompt without compiling or executing shaders").dependOn(&exe.step);
}

/// `zig build tf-h3-image-prompt`: a prompt to the refiner's text rows on Metal, and its synthetic selftest.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt", .root_module = tool(b, target, optimize, metal) });
    b.step("tf-h3-image-prompt", "MiniMax-H3-Image's prompt to its text rows on Metal: Qwen3-VL to layer 50, the token refiner").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) *std.Build.Module {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/prompt_kernels.zig") });
    const prompt = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/prompt.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "metal", .module = metal },
        .{ .name = "core", .module = core },
        .{ .name = "h3_prompt_kernels", .module = kernels },
        .{ .name = "gpu_turn", .module = b.createModule(.{ .root_source_file = b.path("zig/src/server/gpu_turn.zig"), .target = target, .optimize = optimize }) },
    } });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const fixture = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_prompt_fixture.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
    return b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_prompt.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3_prompt", .module = prompt }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "core", .module = core }, .{ .name = "checkpoint_fixture", .module = fixture }, .{ .name = "moss_noise", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize }) } } });
}
