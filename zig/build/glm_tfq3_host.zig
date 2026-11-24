//! GLM TFQ3's file checks, refusal fixtures and fusion/KDA predicates on any host (no Metal, no device).
const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/tfq3_check.zig"), .target = b.graph.host, .optimize = .debug });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-glm-tfq3-host", "TFQ3 header, section and record refusals, shared fusion and GLU contiguity rules").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
