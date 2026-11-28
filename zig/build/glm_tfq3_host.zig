//! GLM TFQ3's file checks, refusal fixtures and fusion/KDA predicates on any host (no Metal, no device).
const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/tfq3_check.zig"), .target = b.graph.host, .optimize = .debug });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-glm-tfq3-host", "TFQ3 header, section and record refusals, shared fusion and GLU contiguity rules").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const recording = b.createModule(.{ .root_source_file = b.path("zig/glm_rows_recording_metal.zig"), .target = b.graph.host });
    const sys = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = b.graph.host, .link_libc = true });
    const signs = b.createModule(.{ .root_source_file = b.path("zig/glm_tfq3_sign_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = recording }, .{ .name = "sys", .module = sys } } });
    const sign_run = b.addRunArtifact(b.addTest(.{ .root_module = signs, .filters = &.{"TFQ3 signs:"} }));
    b.step("test-glm-tfq3-sign-host", "Actual TFQ3 record boundaries through dense and KDA dispatch without Metal").dependOn(&sign_run.step);
    test_step.dependOn(&sign_run.step);
}
