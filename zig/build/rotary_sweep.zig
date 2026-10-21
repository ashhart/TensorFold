//! The runtime-compiled sweep uses embedded source and no offline shader producer.
const std = @import("std");
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module) void {
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const family = b.createModule(.{ .root_source_file = b.path("zig/flashnext_bound_family.zig"), .target = target, .imports = &.{ .{ .name = "kernel_sources", .module = sources }, .{ .name = "metal", .module = metal } } });
    const exe = b.addExecutable(.{ .name = "tf-rotary-sweep", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/rotary_sweep_main.zig"),
        .target = target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "tensorfold", .module = family },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .safe) },
        },
    }) });
    b.step("tf-rotary-sweep", "Build the exhaustive rotary sweep without offline shader compilation; no run").dependOn(&b.addInstallArtifact(exe, .{}).step);
    b.step("compile-rotary-sweep-host", "Compile-analysis of the real sweep without device execution").dependOn(&exe.step);
    const cache = b.addExecutable(.{ .name = "tf-rotary-cache-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/rotary_cache_main.zig"),
        .target = target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = family }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .safe) } },
    }) });
    b.step("compile-rotary-cache-host", "Compile the real cache bit-hash gate without shaders or GPU execution").dependOn(&cache.step);
    b.step("tf-rotary-cache-check", "Build the cache bit-hash gate without offline shader compilation; no run").dependOn(&b.addInstallArtifact(cache, .{}).step);
}
