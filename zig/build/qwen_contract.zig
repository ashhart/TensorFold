//! Shader-free tests of the served Qwen contract fixtures and their rejection controls.
const std = @import("std");
pub fn host(b: *std.Build, core: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_qwen_served_ref.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "core", .module = core }},
    }) }));
    b.step("test-qwen-served-contract-host", "FP64 Qwen contract references and corruption controls without GPU code").dependOn(&run.step);
    all.dependOn(&run.step);
}

pub fn images(b: *std.Build, runner: *std.Build.Step.Compile, controls: ?std.Build.LazyPath, h3: ?std.Build.LazyPath, kernels: anytype, files: []const ?std.Build.LazyPath) void {
    const install = @import("kernel_contract.zig").cuda(b, runner, controls, h3);
    for (kernels, files) |k, file| {
        if (std.mem.eql(u8, k.name, "moe") or std.mem.eql(u8, k.name, "qwen36_mtp")) {
            if (file) |image| install.dependOn(&b.addInstallFile(image, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
        }
    }
}
