//! Host-only checks for route-tap records and the runtime/family session identity.
const std = @import("std");
pub fn tests(b: *std.Build, all: *std.Build.Step, draft_ids: *std.Build.Module) void {
    const mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/route_tap.zig"), .target = b.graph.host, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mod }));
    run.removeEnvironmentVariable("TF_ROUTE_TAP");
    b.step("test-route-tap-host", "Ring, record, drop and ownership regressions without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const cuda = @import("cuda_runtime.zig").runtime(b, b.graph.host, .debug, &.{});
    const f = @import("cuda.zig").family(b, b.graph.host, .debug, cuda, draft_ids);
    std.debug.assert(cuda.import_table.get("core").? == f.core);
    const shared = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/route_tap_core_test.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = f.core }, .{ .name = "cuda", .module = cuda } },
    }), .filters = &.{"route tap core:"} }));
    shared.removeEnvironmentVariable("TF_ROUTE_TAP");
    b.step("test-route-tap-core-host", "Engine core start is visible to the CUDA copy path, no device").dependOn(&shared.step);
    all.dependOn(&shared.step);
}
