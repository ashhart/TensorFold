//! CUDA Qwen preview host gates share the production family modules and dependency graph.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    const host = b.graph.host;
    @import("qwen_before_host.zig").tests(b, step);
    const reuse_owned = b.addRunArtifact(b.addTest(.{ .root_module = mods.qwen35, .filters = &.{ "a kept state holds", "every geometry's pieces", "fp8 pieces follow", "a state resumes only", "a restore refuses", "a failed restore's host copy", "a device save pins nothing", "retained reuse features:" } }));
    b.step("test-qwen35-reuse-owned-host", "Kept layouts, image identity, copy failures and device-buffer cleanup without a device").dependOn(&reuse_owned.step);
    step.dependOn(&reuse_owned.step);
    const serving_guard = b.addRunArtifact(b.addTest(.{ .root_module = mods.qwen35, .filters = &.{"Qwen served qualification:"} }));
    b.step("test-qwen35-qualified-serving-host", "Direct Qwen pre-device qualification refusal and upstream pass-through").dependOn(&serving_guard.step);
    step.dependOn(&serving_guard.step);
    const cache_policy_step = b.step("test-prompt-cache-policy-host", "Cache features and retained cuts follow the serving options");
    for ([_][]const u8{ "zig/src/native/cuda_snaps.zig", "zig/src/native/cuda_options.zig" }) |file| {
        const module = b.createModule(.{ .root_source_file = b.path(file), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = mods.core }, .{ .name = "engine_api", .module = mods.qwen35.import_table.get("engine_api").? } } });
        cache_policy_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{"prompt cache policy:"} })).step);
    }

    const feature_copy = b.createModule(.{ .root_source_file = b.path("zig/qwen35_feature_snapshot_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = mods.qwen35 } } });
    b.step("test-qwen35-feature-snapshot-host", "Byte-exact target and feature snapshots through their real adapters").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = feature_copy, .filters = &.{"cached feature snapshots:"} })).step);
    b.step("test-qwen35-cache-grid-host", "Native prefix planning follows the runtime cold-prefill grid").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.qwen35, .filters = &.{"Qwen cache grid:"} })).step);
    b.step("test-draft-cache-history-host", "The real wrapper replays cached feature history before drafting").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.lanes, .filters = &.{"cached draft history:"} })).step);
    var restored_imports: std.array_list.Managed(std.Build.Module.Import) = .init(b.allocator);
    var imported = mods.qwen35.import_table.iterator();
    while (imported.next()) |entry| restored_imports.append(.{ .name = entry.key_ptr.*, .module = entry.value_ptr.* }) catch @panic("OOM");
    const restored_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_prefill.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = restored_imports.items });
    const restored_step = b.addRunArtifact(b.addTest(.{ .root_module = restored_module, .filters = &.{"Qwen restored history:"} }));
    b.step("test-qwen35-restored-step-host", "Restored draft history precedes the suffix forward without a device").dependOn(&restored_step.step);
    step.dependOn(&restored_step.step);
    const run_admission = b.createModule(.{ .root_source_file = b.path("zig/qwen35_cache_run_admission_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "qwen35", .module = mods.qwen35 } } });
    const run_admission_tests = b.addRunArtifact(b.addTest(.{ .root_module = run_admission, .filters = &.{"Qwen cached run:"} }));
    b.step("test-qwen35-cache-run-admission-host", "Real paged admission and cache eviction without a device").dependOn(&run_admission_tests.step);
    step.dependOn(&run_admission_tests.step);
    const prompt_cache = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_prompt_taps.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    b.step("test-qwen35-prompt-cache-host", "Restored CUDA prompt features through the real storage adapter").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = prompt_cache, .filters = &.{ "Qwen cache resume:", "Qwen shared prompt taps:" } })).step);
    const capacity = b.addRunArtifact(b.addTest(.{ .root_module = mods.qwen35, .filters = &.{ "Qwen prefill capacity:", "explicit prefill chunks" } }));
    b.step("test-qwen-prefill-capacity-host", "Full prompt metadata and shared wrapper capacity without a device").dependOn(&capacity.step);
    step.dependOn(&capacity.step);
    const memory = b.addRunArtifact(b.addTest(.{ .root_module = mods.qwen35, .filters = &.{"Qwen memory:"} }));
    b.step("test-qwen35-memory-host", "Per-stream storage bounds without a CUDA device").dependOn(&memory.step);
    step.dependOn(&memory.step);
    @import("nvfp4_contract.zig").host(b, cuda, mods.core, step);
}
