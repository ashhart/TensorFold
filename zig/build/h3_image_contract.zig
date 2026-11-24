//! MiniMax-H3-Image's kernels under the contract: tf-h3-image-contract compiles them at load, as its tools do.
const std = @import("std");

/// The edit encoder's, prompt path's and image decoder's kernel sources (each module embeds one .metal file).
pub fn sources(b: *std.Build) [3]std.Build.Module.Import {
    return .{
        .{ .name = "h3_image_encoder_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_encoder.zig") }) },
        .{ .name = "h3_prompt_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/prompt_kernels.zig") }) },
        .{ .name = "h3_image_decoder_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/image_decoder.zig") }) },
    };
}

/// macOS: `tf-h3-image-contract` (gates 1-4, controls, this GPU's limits) and `h3-image-contract`, which runs it.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    if (target.result.os.tag != .macos) return;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/h3_image_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, optimize) } } });
    for (sources(b)) |s| root.addImport(s.name, s.module);
    const exe = b.addExecutable(.{ .name = "tf-h3-image-contract", .root_module = root });
    b.step("tf-h3-image-contract", "Build MiniMax-H3-Image's kernel contract runner from runtime sources only").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const run = b.addRunArtifact(exe);
    run.has_side_effects = true;
    b.step("h3-image-contract", "MiniMax-H3-Image's Metal kernels through the contract's gates on this GPU, each control red").dependOn(&run.step);
}
