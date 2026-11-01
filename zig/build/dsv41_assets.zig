const std = @import("std");

pub fn steps(b: *std.Build) void {
    const step = b.step("dsv41-assets", "Install CPU-only DeepSeek token-map and AOT-plan generators");
    const asset_module = @import("dsv41_cuda.zig").module(b, b.graph.host, "asset_main.zig", false);
    const assets = b.addExecutable(.{ .name = "tf-dsv41-assets", .root_module = asset_module });
    step.dependOn(&b.addInstallArtifact(assets, .{}).step);
    const token_map = b.addExecutable(.{ .name = "tf-dsv41-token-map", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/dsv41_token_map.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    }) });
    step.dependOn(&b.addInstallArtifact(token_map, .{}).step);
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = asset_module }));
    b.step("test-dsv41-assets-host", "DeepSeek asset commands refuse GPU operations and malformed arguments").dependOn(&tests.step);
}
