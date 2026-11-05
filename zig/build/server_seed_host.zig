//! Request-local done times, request seeds and lesson wording seeds: host tests without shaders.
const std = @import("std");

pub fn steps(b: *std.Build, server_tests: *std.Build.Module) void {
    b.step("test-done-times-host", "Request-local done line times and lane admission on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"done times:"} })).step);
    b.step("test-request-seeds-host", "Fresh and explicit request seeds and completion logs without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"request seed:"} })).step);
    b.step("test-slide-ask-host", "Lesson wordings seeded from the lesson; --prompt-seed still wins").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"slide ask:"} })).step);
}
