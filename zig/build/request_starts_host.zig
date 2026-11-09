//! Exact message-start metadata and chat retention, without devices.
const std = @import("std");
pub fn steps(b: *std.Build, native: *std.Build.Module, server: *std.Build.Module) void {
    @import("server_seed_host.zig").steps(b, server);
    b.step("test-turn-header-cuts-host", "Warm and fresh template header cuts match without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"turn ends:"} })).step);
    b.step("test-cuda-message-starts-host", "Family-owned request-start metadata without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"request starts:"} })).step);
    b.step("test-request-starts-turns-host", "Six chat histories keep their own exact start within one snapshot slot").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"turn starts:"} })).step);
}
