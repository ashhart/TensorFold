//! Host or image-backed CUDA runtime module construction.
const std = @import("std");

pub fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// `zig/src/cuda/kernels.zig`'s options: whether images are embedded, and the one SM a -Dgpu server's images are for.
pub fn kernelOptions(b: *std.Build, with_kernels: bool, built_sm: ?u32) *std.Build.Step.Options {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", with_kernels);
    options.addOption(?u32, "built_sm", built_sm);
    return options;
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
pub fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    return runtimeFor(b, target, optimize, images, null);
}

/// `runtime` whose every image is for `built_sm` alone (-Dgpu): a context on a GPU of another SM is refused.
pub fn runtimeFor(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath, built_sm: ?u32) *std.Build.Module {
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    const options = kernelOptions(b, with, built_sm);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (@import("cuda_kernels.zig").kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}
