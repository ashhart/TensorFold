//! Focused launch recording for Nemotron's pre-M5 verify attention.
const std = @import("std");
pub fn steps(b: *std.Build) void {
    const metal = b.createModule(.{ .root_source_file = b.addWriteFiles().add("decode-record-metal.zig", "pub const Buffer = usize; pub const Pipeline = usize;\n"), .target = b.graph.host });
    const decode = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/decode_attention.zig"), .target = b.graph.host, .imports = &.{.{ .name = "metal", .module = metal }} });
    const root = b.createModule(.{ .root_source_file = b.path("zig/nemotron_decode_host.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{.{ .name = "decode", .module = decode }} });
    b.step("test-nemotron-decode-attention-host", "Record one-row verify attention extents, offsets and pages without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
    const cases = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/nemotron_decode_cases.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }} });
    b.step("test-nemotron-decode-cases-host", "Verify served-width coverage and reduced-precision stress controls without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cases })).step);
}
