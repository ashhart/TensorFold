//! Focused prompt attribution receipts, ordinary prompt regressions and real adapter compile coverage.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, family: *std.Build.Module, parent: *std.Build.Step) void {
    const physical = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = b.graph.host, .link_libc = true }), .filters = &.{ "status lines parse", "64 KiB-page", "missing lines, other units", "a repeated line keeps", "this process's status reads" } }));
    const alias = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/server/process_resident.zig"), .target = b.graph.host, .link_libc = true }), .filters = &.{"process resident:"} }));
    const physical_step = b.step("test-process-resident-host", "Shared Linux status parser, live streaming read and server API compatibility");
    physical_step.dependOn(&physical.step);
    physical_step.dependOn(&alias.step);
    parent.dependOn(physical_step);
    const parser = b.addSystemCommand(&.{ "python3", "-m", "unittest", "discover", "-s", "tools/tests", "-p", "test_cuda_prefill_ledger.py", "-v" });
    parser.setEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1");
    b.step("test-cuda-prefill-ledger-parser-host", "Offline cold-stage parser controls without a server or device").dependOn(&parser.step);
    parent.dependOn(&parser.step);
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
