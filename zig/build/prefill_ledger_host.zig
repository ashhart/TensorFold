//! Focused prompt attribution receipts, ordinary prompt regressions and real adapter compile coverage.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, family: *std.Build.Module, parent: *std.Build.Step) void {
    const runtime = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"CUDA prefill ledger:"} }));
    const env = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host, .link_libc = true }), .filters = &.{"CUDA prefill ledger:"} }));
    const adapter = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{"CUDA prefill adapter:"} }));
    const step = b.step("test-cuda-prefill-ledger-host", "Recorded prompt JSON, default-off callback identity, CUDA events and real adapter compile checks");
    b.step("test-cuda-prefill-ledger-runtime-host", "Recorded JSON, CUDA event ownership and driver forwarding").dependOn(&runtime.step);
    b.step("test-cuda-prefill-ledger-environment-host", "Exactly one default-off environment flag").dependOn(&env.step);
    b.step("test-cuda-prefill-adapter-host", "Actual native callbacks compile and off retains their identity").dependOn(&adapter.step);
    step.dependOn(&runtime.step);
    step.dependOn(&env.step);
    step.dependOn(&adapter.step);
    parent.dependOn(step);
}
