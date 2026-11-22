//! Request-driven web access regressions use canned DNS and pure URL/address/redirect policies.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const one = b.createModule(.{ .root_source_file = b.path("zig/request_access_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const regression = b.addRunArtifact(b.addTest(.{ .root_module = one, .filters = &.{"request access audit:"} }));
    regression.has_side_effects = true;
    b.step("test-request-access-host", "Actual web client DNS deadline and queue admission without network").dependOn(&regression.step);
    all.dependOn(&regression.step);
    const policy = b.createModule(.{ .root_source_file = b.path("zig/request_access_policy_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const names = &.{ "public addresses only", "redirect locations resolve against the page", "a link whose ends are a newline", "a tab or newline wrapped inside", "a link that only replaces the query", "a relative link's dot segments", "request lines stay plain", "local and private names are refused", "security: cloud platform virtual", "security: request credentials", "security: an unschedulable fetch" };
    const rules = b.addRunArtifact(b.addTest(.{ .root_module = policy, .filters = names }));
    b.step("test-request-access-policy-host", "Existing web URL, address, origin and redirect policies without network").dependOn(&rules.step);
    all.dependOn(&rules.step);
}
