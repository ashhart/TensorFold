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
    const page_math = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_qwen_pages_numerical.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "core", .module = core }},
    }) }));
    b.step("test-qwen-pages-contract-host", "Paged Qwen FP32 stress and non-finite references without GPU code").dependOn(&page_math.step);
    all.dependOn(&page_math.step);
    imageTests(b, core, all);
}

pub fn images(b: *std.Build, runner: *std.Build.Step.Compile, controls: ?std.Build.LazyPath, h3: ?std.Build.LazyPath, kernels: anytype, files: []const ?std.Build.LazyPath) *std.Build.Step {
    const install = @import("kernel_contract.zig").cuda(b, runner, controls, h3);
    for (kernels, files) |k, file| {
        if (std.mem.eql(u8, k.name, "moe") or std.mem.eql(u8, k.name, "experts") or std.mem.eql(u8, k.name, "qwen36_mtp") or std.mem.eql(u8, k.name, "sample") or std.mem.eql(u8, k.name, "glm_glue")) {
            if (file) |image| install.dependOn(&b.addInstallFile(image, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
        }
    }
    return install;
}

fn imageTests(b: *std.Build, core: *std.Build.Module, all: *std.Build.Step) void {
    const files = b.addWriteFiles();
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/kernels.zig"), .target = b.graph.host });
    const kernel_options = b.addOptions();
    kernel_options.addOption(bool, "with_kernels", true);
    kernels.addOptions("kernel_options", kernel_options);
    for (@import("cuda_kernels.zig").kernels) |k| kernels.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{
        .root_source_file = files.add(b.fmt("{s}.fatbin", .{k.name}), k.name),
    });
    const facade = b.createModule(.{ .root_source_file = files.add("image_cuda.zig", "pub const kernels = @import(\"kernels\");"), .target = b.graph.host });
    facade.addImport("kernels", kernels);
    const tests = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_images_test.zig"),
        .target = b.graph.host,
        .imports = &.{ .{ .name = "cuda", .module = facade }, .{ .name = "core", .module = core } },
    });
    @import("cuda_contract_deepseek.zig").inert(b, tests);
    const opts = b.addOptions();
    opts.addOption(bool, "controls", true);
    tests.addOptions("contract_options", opts);
    tests.addAnonymousImport("fatbin_contract_controls", .{ .root_source_file = files.add("contract_controls.fatbin", "contract_controls") });
    tests.addAnonymousImport("fatbin_h3", .{ .root_source_file = files.add("h3.fatbin", "h3") });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests, .filters = &.{"contract image:"} }));
    b.step("test-contract-images-host", "Live contract image routing and unknown-name refusal with inert bytes").dependOn(&run.step);
    all.dependOn(&run.step);
}
