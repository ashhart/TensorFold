const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_prompt_gemm_profile_host.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    }) });
    const run = b.addRunArtifact(tests);
    b.step("test-glm-prompt-gemm-profile-host", "GLM NVFP4 prompt GEMM profiler shapes and launch geometry without a GPU").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
