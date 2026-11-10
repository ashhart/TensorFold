//! MiniMax-H3-Image's prompt path: its host rules' tests, a host compile of tf-h3-image-prompt, and the Metal tool.
const std = @import("std");

/// `test-h3-image-prompt` (the rules, no GPU) and on macOS `compile-h3-image-prompt` (the tool, no shader compiled).
pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/picture_prompt_test.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "qwen_vl_rope", .module = rope(b, target) }, .{ .name = "picture", .module = picture(b, target) } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-h3-image-prompt", "MiniMax-H3-Image's prompt rules: ids, tags, positions, rotary, Qwen copy sizes, configs").dependOn(&run.step);
    test_step.dependOn(&run.step);
    if (target.result.os.tag != .macos) return;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt-host-compile", .root_module = tool(b, target, .Debug, metal) });
    b.step("compile-h3-image-prompt", "Compile tf-h3-image-prompt without compiling or executing shaders").dependOn(&exe.step);
}

/// `zig build tf-h3-image-prompt`: a prompt to the refiner's text rows on Metal, and its synthetic selftest.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-h3-image-prompt", .root_module = tool(b, target, optimize, metal) });
    b.step("tf-h3-image-prompt", "MiniMax-H3-Image's prompt to its text rows on Metal: Qwen3-VL to layer 50, the token refiner").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) *std.Build.Module {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/prompt_kernels.zig") });
    const prompt = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/prompt.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = st }, .{ .name = "h3_prompt_kernels", .module = kernels }, .{ .name = "qwen_vl_rope", .module = rope(b, target) }, .{ .name = "picture", .module = picture(b, target) } } });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const fixture = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_fixture.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_prompt.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3_prompt", .module = prompt }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "checkpoint_fixture", .module = fixture }, .{ .name = "moss_noise", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/noise.zig"), .target = target, .optimize = optimize }) } } });
}

fn rope(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/qwen_vl/rope.zig"), .target = target });
}

fn picture(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/picture/picture.zig"), .target = target, .link_libc = true });
}
