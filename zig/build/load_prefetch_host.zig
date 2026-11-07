const std = @import("std");
pub fn tests(b: *std.Build, core: *std.Build.Module, parent: *std.Build.Step) void {
    const module = b.createModule(.{ .root_source_file = b.path("zig/load_prefetch_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{"load prefetch:"} }));
    run.has_side_effects = true;
    b.step("test-load-prefetch-host", "Actual CUDA loader adapters, checked I/O, zero-reader off and allocation-failure cleanup").dependOn(&run.step);
    parent.dependOn(&run.step);
}
