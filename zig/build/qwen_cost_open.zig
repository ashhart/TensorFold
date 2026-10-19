//! Source-bound startup cache integration and descriptor identity checks, with no shader producer.
const std = @import("std");
pub fn tests(b: *std.Build, mods: anytype, native: anytype, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/qwen35_cost_open_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "native", .module = native.engines }, .{ .name = "engine_api", .module = native.api }, .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "lanes", .module = mods.lanes } } });
    const cost_run1 = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"window cost open:"} }));
    b.step("test-qwen35-cost-open-host", "Actual native opening consumes cached measured widths before its worker").dependOn(&cost_run1.step);
    all.dependOn(&cost_run1.step);
    const files = b.addWriteFiles();
    _ = files.add("target/config.json", "{}");
    _ = files.add("target/model.safetensors", "\x02\x00\x00\x00\x00\x00\x00\x00{}");
    _ = files.add("draft/model.safetensors", "\x02\x00\x00\x00\x00\x00\x00\x00{}");
    _ = files.addCopyFile(b.path("zig/tests/fixtures/qwen35_cost_dflash.json"), "draft/config.json");
    _ = files.addCopyFile(b.path("zig/tests/fixtures/qwen35_cost_dflash8.json"), "draft8/config.json");
    _ = files.add("draft8/model.safetensors", "\x02\x00\x00\x00\x00\x00\x00\x00{}");
    const process = b.createModule(.{ .root_source_file = b.path("zig/qwen35_cost_process.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "qwen35", .module = mods.qwen35 }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "core", .module = mods.core } } });
    const run = b.addRunArtifact(b.addExecutable(.{ .name = "qwen-cost-key-process", .root_module = process }));
    run.addDirectoryArg(files.getDirectory());
    run.has_side_effects = true;
    b.step("test-qwen35-cost-process-host", "Production cost keys across separate processes with inactive config tails").dependOn(&run.step);
    all.dependOn(&run.step);
    const identity = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_cost_identity.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "core", .module = mods.core }} });
    const cost_run2 = b.addRunArtifact(b.addTest(.{ .root_module = identity, .filters = &.{"cost identity:"} }));
    b.step("test-qwen35-cost-identity-host", "Drafter file identity invalidates stale timings without reading weight payloads").dependOn(&cost_run2.step);
    all.dependOn(&cost_run2.step);
}
