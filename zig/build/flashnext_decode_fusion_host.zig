//! Decode fusion admission, ABI, default-off environment and pending launch records without a GPU.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, family: *std.Build.Module, all: *std.Build.Step) void {
    const body = b.step("test-flashnext-decode-fusion-host", "Decode HC and MXFP8 direct-store admission and actual launch arguments");
    body.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "decode fusion:", "shared expert composition" } })).step);
    all.dependOn(body);
    const contracts = b.step("test-flashnext-decode-fusion-contract-host", "Decode fusion defaults, existing shared-stride refusals and pending resource records");
    contracts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"shared down stride:"} })).step);
    contracts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core, .filters = &.{"decode fusion records:"} })).step);
    const environment = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host, .link_libc = true });
    contracts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = environment, .filters = &.{"Flash Next strict env:"} })).step);
    all.dependOn(contracts);
}
