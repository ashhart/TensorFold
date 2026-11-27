const std = @import("std");

pub fn targets(b: *std.Build, runner_exe: *std.Build.Step.Compile) void {
    b.step("compile-cuda-tests", "Compile CUDA runner without executing a GPU").dependOn(&runner_exe.step);
    const shared_starts_run = b.addRunArtifact(runner_exe);
    shared_starts_run.addArg("glm-shared-starts-check");
    shared_starts_run.addPassthruArgs();
    b.step("glm-shared-starts-check", "Eight-layer GB10 state-path equality (TF_GLM_LAYERS=8), with tracked memory receipts").dependOn(&shared_starts_run.step);
}
