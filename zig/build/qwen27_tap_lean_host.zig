//! Prompt tap liveness, publication and byte identity without a Metal compiler or GPU.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    if (b.graph.host.result.os.tag == .linux) {
        const data = b.addWriteFiles();
        _ = data.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
        const ids = b.createModule(.{ .root_source_file = data.add("tap-compile-draft-ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
        const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos });
        const m = @import("metal_targets.zig").metalEngineModules(b, target, .debug, ids);
        // Objects target macOS from Linux; no Apple framework or Objective-C library is linked.
        m.metal.frameworks.clearRetainingCapacity();
        m.metal.link_objects.clearRetainingCapacity();
        m.sources.addImport("nemotron_packed_metallib", @import("metal_targets.zig").embedded(b, b.addWriteFiles().add("tap-compile-only.metallib", ""), "nemotron_packed.metallib"));
        @import("qwen27_host.zig").compileHost(b, m.metal, m.engine, m.lanes);
    }
    const host = b.step("test-qwen27-tap-lean-host", "Actual prompt tap plan and publication on poisoned fake buffers");
    const logic = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/tap_prompt_test.zig"), .target = b.graph.host });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = logic })).step);
    all.dependOn(host);
    const contracts = b.step("test-qwen27-tap-lean-contract-host", "Strict tap flag and original independent gather regressions without shaders");
    const env = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host });
    contracts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = env, .filters = &.{ "tap lean environment:", "environment: the Metal Qwen" } })).step);
    const gather = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/tap_gather.zig"), .target = b.graph.host });
    contracts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gather })).step);
    all.dependOn(contracts);
}
pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, compile_step: *std.Build.Step) void {
    const target = metal.resolved_target.?;
    const object = b.addObject(.{ .name = "qwen27-tap-contract-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/q27_tap_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) } } }) });
    _ = object.getEmittedBin();
    compile_step.dependOn(&object.step);
}
