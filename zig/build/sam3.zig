const std = @import("std");

pub const Targets = struct { tests: *std.Build.Step, tools: *std.Build.Step, kernels: *std.Build.Step };

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Targets {
    const safetensors = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target });
    const sys = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = target, .link_libc = true });
    safetensors.addImport("sys", sys);
    const torch = b.createModule(.{ .root_source_file = b.path("zig/src/core/torch/archive.zig"), .target = target });
    torch.addImport("sys", sys);
    torch.link_libc = true;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    metal.addImport("sys", sys);
    for ([_][]const u8{ "Metal", "Foundation", "IOSurface" }) |framework| metal.linkFramework(framework, .{});
    metal.linkSystemLibrary("objc", .{});
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const shaders = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sam3/kernels.zig") });
    const rounding = b.createModule(.{ .root_source_file = b.path("zig/src/core/bf16_round.zig"), .target = target });
    const tensors = b.createModule(.{ .root_source_file = b.path("zig/src/core/tensor_metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "safetensors", .module = safetensors }, .{ .name = "bf16_round", .module = rounding } } });
    const module = b.addModule("sam3", .{
        .root_source_file = b.path("zig/src/families/sam3/sam3.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "safetensors", .module = safetensors }, .{ .name = "torch_checkpoint", .module = torch }, .{ .name = "tensor_metal", .module = tensors }, .{ .name = "sam3_kernels", .module = shaders } },
    });
    module.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    module.import_table.get("tokenizer").?.addImport("sys", sys);
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    const test_step = b.step("test-sam3", "SAM3 and SAM3.1 host validation and session contracts");
    test_step.dependOn(&tests.step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = torch })).step);
    const inspect = b.addExecutable(.{ .name = "tf-sam3-checkpoint", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/sam3_checkpoint.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "sam3", .module = module }},
    }) });
    const tools = b.step("sam3", "Build native SAM3/SAM3.1 image, video, checkpoint and kernel tools");
    install(b, tools, inspect);
    const kernels = b.addExecutable(.{ .name = "tf-sam3-kernels", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/sam3_kernels.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "sam3", .module = module }},
    }) });
    install(b, tools, kernels);
    const image = b.addExecutable(.{ .name = "tf-sam3-image", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/sam3_image.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sam3", .module = module }, .{ .name = "safetensors", .module = safetensors }, .{ .name = "metal", .module = metal } },
    }) });
    install(b, tools, image);
    const video = b.addExecutable(.{ .name = "tf-sam3-video", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/sam3_video.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sam3", .module = module }, .{ .name = "safetensors", .module = safetensors }, .{ .name = "metal", .module = metal } },
    }) });
    video.root_module.addImport("process_memory", b.createModule(.{ .root_source_file = b.path("zig/src/server/process_memory.zig"), .target = target, .link_libc = true }));
    video.root_module.addImport("checkpoint_fixture", b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_fixture.zig"), .target = target }));
    install(b, tools, video);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = video.root_module })).step);
    const compare = b.addExecutable(.{ .name = "tf-sam3-compare", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/sam3_compare.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sam3", .module = module }, .{ .name = "safetensors", .module = safetensors } },
    }) });
    install(b, tools, compare);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = compare.root_module })).step);
    const tensor_compare = b.addExecutable(.{ .name = "tf-sam3-tensors", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/sam3_tensors.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "safetensors", .module = safetensors }},
    }) });
    install(b, tools, tensor_compare);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tensor_compare.root_module })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module.import_table.get("tokenizer").? })).step);
    const run = b.addRunArtifact(kernels);
    if (b.option(bool, "benchmark", "Time the SAM attention shapes") orelse false) run.addArg("--benchmark");
    const checks = b.step("sam3-kernels", "Run SAM Metal numerical checks without loading a model");
    checks.dependOn(&run.step);
    return .{ .tests = test_step, .tools = tools, .kernels = checks };
}

fn install(b: *std.Build, aggregate: *std.Build.Step, executable: *std.Build.Step.Compile) void {
    const artifact = b.addInstallArtifact(executable, .{});
    aggregate.dependOn(&artifact.step);
    b.step(executable.name, "Build the native SAM developer tool").dependOn(&artifact.step);
}
