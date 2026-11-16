//! What the model-free kernel tests need to reach the Flash Next HIP kernels they check against its own plan.
const std = @import("std");

/// `kernel_module` gains `core`, the memory sizing, and the Flash Next kernel reach built over both.
pub fn wire(b: *std.Build, target: std.Build.ResolvedTarget, kernel_module: *std.Build.Module, core: *std.Build.Module) void {
    kernel_module.addImport("core", core);
    kernel_module.addImport("cuda_memory", b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_memory.zig"), .target = target }));
    const reach = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/kernel_reach.zig"), .target = target, .link_libc = true, .imports = &.{
        .{ .name = "hip", .module = kernel_module },
        .{ .name = "core", .module = core },
    } });
    kernel_module.addImport("flashnext_kernel", reach);
}
