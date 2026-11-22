//! Guarded native launch argument bytes and the strict opt-in, without a CUDA device.
const std = @import("std");
pub fn steps(b: *std.Build, family: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_pdl_test.zig"), .target = b.graph.host, .link_libc = true });
    var imports = family.import_table.iterator();
    while (imports.next()) |entry| root.addImport(entry.key_ptr.*, entry.value_ptr.*);
    const host = b.step("test-flashnext-pdl-host", "Flash Next guarded launch bytes and default-off capability policy without a GPU");
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"Flash Next PDL:"} })).step);
    const environment = b.createModule(.{ .root_source_file = b.path("zig/src/platform/environment.zig"), .target = b.graph.host, .link_libc = true });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = environment, .filters = &.{"Flash Next strict env:"} })).step);
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "DeltaNet steps:", "paged grouped prompts:" } })).step);
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family.import_table.get("cuda").?, .filters = &.{"a dependent lane launches its decode tile"} })).step);
    all.dependOn(host);
    const source = b.step("test-flashnext-pdl-source-host", "PDL entry coverage, untouched math and SASS gate refusal fixtures");
    source.dependOn(&b.addSystemCommand(&.{ "python3", "tools/test_flashnext_pdl.py" }).step);
    source.dependOn(&b.addSystemCommand(&.{ "python3", "tools/test_pdl_early_loads.py" }).step);
    all.dependOn(source);
}
