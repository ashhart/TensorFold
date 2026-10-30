const std = @import("std");
const shared = @import("dsv41_shared.zig");
const units = @import("dsv41_units.zig");
pub fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, images: bool) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", images);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    cuda.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = .Debug }));
    if (images) {
        const empty = b.addWriteFiles().add("compile-only.fatbin", "SYNTHETIC-COMPILE-ONLY");
        const all = @import("cuda.zig").kernels;
        for (all) |k| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = empty });
    }
    return cuda;
}
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, file: []const u8, images: bool) *std.Build.Module {
    const cuda = runtime(b, target, images);
    const opts = b.addOptions();
    opts.addOption(bool, "with_kernels", images);
    opts.addOption(u32, "family_scope", 41);
    const dk = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kernels.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    dk.addImport("cuda", cuda);
    dk.addOptions("dsv41_kernel_options", opts);
    const empty = b.addWriteFiles().add("compile-only-dsv41.fatbin", "SYNTHETIC-COMPILE-ONLY");
    for (units.units ++ units.optional) |k| dk.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{k.name}), .{ .root_source_file = empty });
    return assemble(b, target, .Debug, file, cuda, dk);
}
/// The family's root `file` over the runtime `cuda` and kernel module `dk`, every family module at `optimize`.
pub fn assemble(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, file: []const u8, cuda: *std.Build.Module, dk: *std.Build.Module) *std.Build.Module {
    const m = shared.moduleAt(b, target, optimize, file);
    m.addImport("cuda", cuda);
    m.addImport("dsv41_kernels", dk);
    const tp = @import("tp.zig").module(b, target, optimize, cuda);
    m.addImport("tp", tp);
    m.addImport("kv", @import("dsv41_state.zig").kvModuleAt(b, target, optimize, cuda, tp));
    @import("dsv41_draft.zig").addImportsAt(b, m, target, optimize);
    return m;
}
pub fn steps(b: *std.Build) void {
    _ = @import("dsv41_images.zig").images(b, b.option([]const u8, "dsv41-nvcc", "Optional family nvcc compiler"), b.option([]const u8, "dsv41-fatbins", "Optional complete family prebuilt image directory"));
    _ = @import("dsv41_images.zig").module(b, b.graph.host, .Debug, runtime(b, b.graph.host, false), null);
    const check_target = b.resolveTargetQuery(.{ .cpu_arch = b.graph.host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
    const heads_check = b.addExecutable(.{ .name = "tf-dsv41-prefill-heads", .root_module = module(b, check_target, "prefill_heads_check.zig", false) });
    b.step("dsv41-prefill-heads", "Build the AOT-only padded attention comparison for Linux, never run").dependOn(&b.addInstallArtifact(heads_check, .{}).step);
    const planner = module(b, b.graph.host, "cuda_test.zig", false);
    b.step("test-dsv41-bounds-host", "DeepSeek resource bounds and real wrapper controls without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = planner, .filters = &.{"DeepSeek bounds:"} })).step);
    const binding_tests = b.addRunArtifact(b.addTest(.{ .root_module = planner.import_table.get("dsv41_kernels").? }));
    const common_tests = b.addRunArtifact(b.addTest(.{ .root_module = planner.import_table.get("cuda").? }));
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, "cuda_test.zig", false) }));
    const host_step = b.step("test-dsv41-cuda-host", "DeepSeek launch plans and binding contracts without device calls");
    host_step.dependOn(&host.step);
    host_step.dependOn(&binding_tests.step);
    host_step.dependOn(&common_tests.step);
    const sm_module = module(b, b.graph.host, "cuda_test.zig", true).import_table.get("dsv41_kernels").?;
    const sm_test = b.addTest(.{ .root_module = sm_module, .filters = &.{"DeepSeek refuses pre-cluster GPUs"} });
    const sm_run = b.addRunArtifact(sm_test);
    b.step("test-dsv41-sm-host", "Check SM refusal before image load with a fake driver and inert images").dependOn(&sm_run.step);
    const forms = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/exl3_prepare.zig"), .target = b.graph.host, .optimize = .Debug }) }));
    host_step.dependOn(&forms.step);
    const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
    const cross = b.addTest(.{ .root_module = module(b, target, "cuda_test.zig", true) });
    _ = cross.getEmittedBin();
    b.step("compile-dsv41-cuda", "Emit Linux CUDA binding bodies with synthetic nonexecuted images").dependOn(&cross.step);
    const consumers_base = module(b, target, "cuda_test.zig", true);
    const consumers = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_compile.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    consumers.addImport("cuda", consumers_base.import_table.get("cuda").?);
    consumers.addImport("core", consumers_base.import_table.get("core").?);
    consumers.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    const controls = b.addOptions();
    controls.addOption(bool, "controls", true);
    consumers.addOptions("contract_options", controls);
    consumers.addAnonymousImport("fatbin_h3", .{ .root_source_file = b.addWriteFiles().add("compile-h3", "SYNTHETIC-COMPILE-ONLY") });
    consumers.addAnonymousImport("fatbin_contract_controls", .{ .root_source_file = b.addWriteFiles().add("compile-controls", "SYNTHETIC-COMPILE-ONLY") });
    @import("cuda_contract_deepseek.zig").inert(b, consumers);
    const consumer_object = b.addObject(.{ .name = "contract-consumers", .root_module = consumers });
    _ = consumer_object.getEmittedBin();
    b.step("compile-cuda-contract-consumers", "Emit both real resource consumers with synthetic nonexecuted images").dependOn(&consumer_object.step);
    const catalog = b.addExecutable(.{ .name = "dsv41-catalog", .root_module = module(b, b.graph.host, "catalog.zig", false) });
    const record = b.addRunArtifact(catalog);
    const output = record.captureStdOut(.{});
    b.step("emit-dsv41-catalog", "Record symbol resolution without CUDA device work").dependOn(&b.addInstallFile(output, "dsv41-launches.txt").step);
    const runner_base = module(b, target, "cuda_test.zig", true);
    const runner = b.addExecutable(.{ .name = "tf-dsv41-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/dsv41_main.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = runner_base.import_table.get("cuda").? }, .{ .name = "dsv41_kernels", .module = runner_base.import_table.get("dsv41_kernels").? } },
    }) });
    _ = runner.getEmittedBin();
    b.step("compile-dsv41-runner", "Emit the supplied device fixture runner with nonexecuted synthetic images").dependOn(&runner.step);
}
