//! Prompt-mask contract fixture without offline Metal compilation or model weights.
const std = @import("std");
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const frags = b.createModule(.{ .root_source_file = b.path("zig/src/core/frags.zig"), .target = target, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const files = b.addWriteFiles();
    const exports = b.createModule(.{ .root_source_file = files.add("masked-exports.zig", "pub const frags = @import(\"frags\");\n"), .imports = &.{.{ .name = "frags", .module = frags }} });
    const contract = @import("kernel_contract.zig").module(b, target, optimize);
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/masked_values_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "contract", .module = contract }, .{ .name = "tensorfold", .module = exports }, .{ .name = "kernel_sources", .module = sources } } });
    const exe = b.addExecutable(.{ .name = "tf-masked-values", .root_module = root });
    root.addImport("masked_sources", b.createModule(.{ .root_source_file = b.path("zig/masked_value_sources.zig"), .imports = &.{.{ .name = "kernel_sources", .module = sources }} }));
    root.addImport("h3_kernels", b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") }));
    @import("qwen4.zig").metalNative(b, target, root);
    const install = b.addInstallArtifact(exe, .{});
    b.step("tf-masked-values", "Build the isolated prompt-mask fixtures from runtime sources only").dependOn(&install.step);
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/masked_value_data.zig"), .target = target, .imports = &.{.{ .name = "contract", .module = contract }} }) });
    const host = b.step("test-masked-values-host", "The hidden/visible exceptional-value fixtures and contract host gates");
    host.dependOn(&b.addRunArtifact(tests).step);
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract })).step);
}
