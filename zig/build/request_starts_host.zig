//! Exact message-start metadata and chat retention, without devices.
const std = @import("std");
pub fn steps(b: *std.Build, native: *std.Build.Module, server: *std.Build.Module) void {
    @import("server_seed_host.zig").steps(b, server);
    const api = native.import_table.get("engine_api").?;
    const core = native.import_table.get("core").?;
    const copies = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_snaps.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "core", .module = core } } });
    const budget = b.createModule(.{ .root_source_file = b.path("zig/cuda_cache_budget_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "copies", .module = copies }, .{ .name = "family", .module = native.import_table.get("lfm2_moe_cuda").? } } });
    b.step("test-lfm2-cache-budget-host", "Actual non-GLM CUDA family cut rules through the shared cache planner").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = budget, .filters = &.{"CUDA cache budget:"} })).step);
    b.step("test-cuda-message-starts-host", "Family-owned request-start metadata without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"request starts:"} })).step);
    b.step("test-message-starts-resume-host", "An edited tool result resumes at its message start through the real chat prepare path").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"tool edit starts:"} })).step);
    b.step("test-request-starts-turns-host", "Six chat histories keep their own exact start within one snapshot slot").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"turn starts:"} })).step);
}
