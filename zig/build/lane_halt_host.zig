//! CPU checks for halt ordering, with the real lane host and fake target.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const halt = b.addTest(.{ .root_module = api, .filters = &.{ "halt (", "halt fails" } });
    b.step("test-lane-halt-host", "Halt ordering with a fake backend and no GPU dependencies").dependOn(&b.addRunArtifact(halt).step);
    b.step("test-lane-lone-host", "Existing lone-driver handoff and reclaim with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"lone driver"} })).step);
    b.step("test-lane-api-host", "Shared lane host and engine API CPU regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    b.step("test-embedding-lane-host", "Prompt-only embedding jobs on the actual lane worker without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"embedding lane:"} })).step);
    const copies = b.step("test-root-copy-host", "Root copy precedence, drafter absorption and real worker checks without GPU");
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"root copy:"} })).step);
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"root copy:"} })).step);
}
