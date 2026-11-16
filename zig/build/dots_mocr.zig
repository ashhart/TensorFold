//! dots.mocr checkpoint and processor host checks, without loading a GPU runtime.
const std = @import("std");

pub fn hostTests(b: *std.Build, test_step: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = b.graph.host, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, b.graph.host));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .link_libc = true }));
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const family = b.createModule(.{
            .root_source_file = b.path("zig/src/families/dots_mocr/dots_mocr.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api } },
        });
        const run = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{"dots:"} }));
        run.has_side_effects = true;
        const name = if (mode == .debug) "test-dots-mocr-host" else "test-dots-mocr-fast";
        b.step(name, "dots.mocr config, tensor inventory and image processor checks; TF_DOTS_DIR checks local shards").dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}

pub fn hipTests(b: *std.Build, hip: *std.Build.Module, test_step: *std.Build.Step) void {
    const host = b.addRunArtifact(b.addTest(.{ .root_module = hip, .filters = &.{ "dots HIP columns:", "dots HIP vision:", "dots HIP text:" } }));
    test_step.dependOn(&host.step);
    b.step("test-dots-hip-host", "dots.mocr HIP column, vision and text operator admission without GPU execution").dependOn(&host.step);
    const family = hipModule(b, hip);
    const weights_host = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "dots HIP weights:", "dots HIP graph:", "dots HIP plan:" } }));
    b.step("test-dots-hip-weights-host", "HIP weight ownership, layout and budget refusal without GPU calls").dependOn(&weights_host.step);
    test_step.dependOn(&weights_host.step);
    const fast_family = hipModule(b, hip);
    fast_family.optimize = .fast;
    const plan_fast = b.addRunArtifact(b.addTest(.{ .root_module = fast_family, .filters = &.{"dots HIP plan:"} }));
    b.step("test-dots-hip-plan-fast", "Check serving memory budgets with release arithmetic and no GPU").dependOn(&plan_fast.step);
    test_step.dependOn(&plan_fast.step);
    const upload_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/dots_weights_test.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "dots_mocr", .module = family }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
    const upload = b.addTest(.{ .name = "tf-dots-hip-weights-test", .root_module = upload_module, .filters = &.{"dots HIP device weights:"} });
    b.step("build-dots-hip-weights-test", "Build the real checkpoint upload check without running a GPU").dependOn(&b.addInstallArtifact(upload, .{}).step);
    const upload_run = b.addRunArtifact(upload);
    upload_run.has_side_effects = true;
    b.step("test-dots-hip-weights", "Upload TF_DOTS_DIR on the selected HIP GPU after memory admission").dependOn(&upload_run.step);
    const cache_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/dots_cache_test.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "dots_mocr", .module = family } },
    });
    const cache = b.addTest(.{ .name = "tf-dots-hip-cache-test", .root_module = cache_module, .filters = &.{"dots HIP device cache:"} });
    b.step("build-dots-hip-cache-test", "Build dense KV prefix-copy checks without running a GPU").dependOn(&b.addInstallArtifact(cache, .{}).step);
}

pub fn hipModule(b: *std.Build, hip: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/dots_mocr/hip.zig"),
        .target = hip.resolved_target,
        .optimize = hip.optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
}

pub fn columnObjects(b: *std.Build, cc: @import("hip.zig").Hipcc) void {
    const step = b.step("hip-columns-objects", "Compile BF16 column operators and their controls for the four recorded HIP chips");
    const vision = b.step("hip-vision-objects", "Compile BF16 vision operators and full attention for all recorded HIP chips");
    for (@import("../src/core/contract/limits.zig").rdna) |chip| {
        const ops = @import("hip.zig").compile(b, cc, chip.name, b.path("zig/kernels/hip/ops/ops.hip"), "group0.hsaco");
        const ctl = @import("hip.zig").compile(b, cc, chip.name, b.path("zig/kernels/hip/contract_controls.hip"), "contract_controls.hsaco");
        step.dependOn(&b.addInstallFileWithDir(ops, .prefix, b.fmt("columns-objects/{s}/group0.hsaco", .{chip.name})).step);
        step.dependOn(&b.addInstallFileWithDir(ctl, .prefix, b.fmt("columns-objects/{s}/contract_controls.hsaco", .{chip.name})).step);
        const pre = @import("hip.zig").compile(b, cc, chip.name, b.path("zig/kernels/hip/attention/prefill.hip"), "group4.hsaco");
        vision.dependOn(&b.addInstallFileWithDir(ops, .prefix, b.fmt("vision-objects/{s}/ops.hsaco", .{chip.name})).step);
        vision.dependOn(&b.addInstallFileWithDir(pre, .prefix, b.fmt("vision-objects/{s}/prefill.hsaco", .{chip.name})).step);
    }
}

pub fn modelChecks(b: *std.Build, hip: *std.Build.Module) void {
    const module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/dots_model.zig"),
        .target = hip.resolved_target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "dots_mocr", .module = hipModule(b, hip) }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
    const exe = b.addExecutable(.{ .name = "tf-dots-hip-check", .root_module = module });
    b.step("tf-dots-hip-check", "Build native dots.mocr intermediate checks without running the model").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
