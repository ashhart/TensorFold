//! CUDA failure recovery and serving readiness exercised on host-only runtime/family modules.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, flashnext: *std.Build.Module, native: *std.Build.Module) void {
    b.step("test-flashnext-warm-host", "Actual CUDA warm image policy and lane admission without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{ "a reply's warm pass", "four concurrent streams all retain MTP", "eight concurrent streams all retain MTP" } })).step);
    const rule_tests = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_lanes_draft_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    var rule_imports = flashnext.import_table.iterator();
    while (rule_imports.next()) |entry| rule_tests.addImport(entry.key_ptr.*, entry.value_ptr.*);
    b.step("test-flashnext-draft-rule-host", "Shared greedy drafting keeps the configured confidence without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = rule_tests })).step);
    b.step("test-cuda-context-health-host", "Sticky CUDA context faults and real graph-instantiation refusal without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{ "CUDA context health:", "CUDA graph refusal:" } })).step);
    b.step("test-flashnext-loader-host", "The threaded NVFP4 expert loader: the loop's words, no pinned or device memory left after a failed load, geometry first").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{"Flash Next expert loader:"} })).step);
    b.step("test-flashnext-completion-host", "Actual CUDA sequence rollback drains asynchronous destinations before release").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"Flash Next completion:"} })).step);
    b.step("test-flashnext-window-host", "The served copy window and its buffers: a width the boot check proved, the shipped rows unless it runs").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{ "the served copy window widens", "the boot check's sweep", "a draft at 65,536 tokens", "the saved pieces follow", "the window's waits" } })).step);
    b.step("test-flashnext-prompt-failure-host", "A failed prompt-buffer fill frees the current and earlier buffers").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{"a failed fill frees the buffer it was filling and every one before it"} })).step);
    b.step("test-cuda-fault-admission-host", "Sticky context health reaches native status and queue admission").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"CUDA context health:"} })).step);
    b.step("test-cuda-readiness-host", "Real native readiness and synchronous startup preparation without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"family readiness reaches"} })).step);
    b.step("test-startup-log-host", "Ordered startup logs and early P2P facts without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"startup log:"} })).step);
}
