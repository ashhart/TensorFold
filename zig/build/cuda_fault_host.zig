//! CUDA failure recovery and serving readiness exercised on host-only runtime/family modules.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, flashnext: *std.Build.Module, native: *std.Build.Module) void {
    b.step("test-cuda-context-health-host", "Sticky CUDA context faults and real graph-instantiation refusal without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{ "CUDA context health:", "CUDA graph refusal:" } })).step);
    b.step("test-flashnext-completion-host", "Actual CUDA sequence rollback drains asynchronous destinations before release").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{"Flash Next completion:"} })).step);
    b.step("test-cuda-fault-admission-host", "Sticky context health reaches native status and queue admission").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"CUDA context health:"} })).step);
    b.step("test-cuda-readiness-host", "Real native readiness and synchronous startup preparation without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"family readiness reaches"} })).step);
}
