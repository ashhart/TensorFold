//! MiniMax-H3-Image's vision tool: tf-h3-image-vision (Qwen3-VL's tower with DeepStack), its host compile and tests.
const std = @import("std");

fn metal(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.linkFramework("Metal", .{});
    m.linkFramework("Foundation", .{});
    m.linkFramework("IOSurface", .{});
    m.linkSystemLibrary("objc", .{});
    return m;
}

/// The tower, its settings, shards, reference and core's pictures: one core root, so no file sits in two modules.
fn tower(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    // the engine's kernel sources, without the packed library only its own builds embed
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/qwen_vl_tools.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal(b, target, optimize) }, .{ .name = "kernel_sources", .module = sources } } });
}

/// The tool: the tower module, picture_prompt's Qwen copy rule reading that module as its `picture`, the writer.
fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const vl = tower(b, target, optimize);
    const copy = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/qwen_copy.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "picture", .module = vl }} });
    const fixture = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_fixture.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{ .root_source_file = b.path("zig/tests/h3_image_vision.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = vl.import_table.get("metal").? }, .{ .name = "qwen_vl", .module = vl }, .{ .name = "h3_qwen_copy", .module = copy }, .{ .name = "checkpoint_fixture", .module = fixture } } });
}

/// On macOS: `tf-h3-image-vision`, `compile-h3-image-vision` (no shader compiled), the tower's host tests.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    if (target.result.os.tag != .macos) return;
    const exe = b.addExecutable(.{ .name = "tf-h3-image-vision", .root_module = tool(b, target, optimize) });
    b.step("tf-h3-image-vision", "MiniMax-H3-Image's reference picture to its Qwen3-VL vision and DeepStack rows on Metal").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const host = b.addExecutable(.{ .name = "tf-h3-image-vision-host-compile", .root_module = tool(b, target, .Debug) });
    b.step("compile-h3-image-vision", "Compile tf-h3-image-vision without compiling or executing shaders").dependOn(&host.step);
    // settings and DeepStack's parse, tensor plans, shard prefixes, scratch and the reference's pieces: no GPU
    const filters = &.{ "the 27B's files parse", "a tower these kernels do not read", "DeepStack", "the buffer holds the 27B tower", "Qwen-VL shards:", "a 4,096-token tower's scratch", "the reference's LayerNorm" };
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tower(b, target, .Debug), .filters = filters }));
    if (b.top_level_steps.get("test-h3-image-prompt")) |s| s.step.dependOn(&run.step);
    test_step.dependOn(&run.step);
}
