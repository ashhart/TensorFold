//! Disk spill lifecycle runs without shaders, checkpoints or devices.
const std = @import("std");
pub fn tests(b: *std.Build, api: *std.Build.Module, all: *std.Build.Step) void {
    const filter = b.option([]const u8, "rows-filter", "One spill lifecycle host regression") orelse "rows_";
    const run = b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{filter} }));
    b.step("test-rows-host", "Spill lifecycle and file exactness without a GPU").dependOn(&run.step);
    all.dependOn(&run.step);
    const scale = b.createModule(.{
        .root_source_file = b.path("zig/src/core/rows_startup_scale_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
    });
    var imports = api.import_table.iterator();
    while (imports.next()) |entry| scale.addImport(entry.key_ptr.*, entry.value_ptr.*);
    const measure = b.addRunArtifact(b.addTest(.{ .root_module = scale, .filters = &.{"restore scale: startup 1000"} }));
    measure.has_side_effects = true;
    b.step("test-rows-startup-scale", "Measure startup with 1K, 10K and 100K valid stored states without a GPU")
        .dependOn(&measure.step);
}
