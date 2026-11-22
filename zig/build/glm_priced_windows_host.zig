//! GLM publication and generic post-block allocation are two covering host steps, with no device images.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/glm_priced_windows_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = lanes } } });
    const first = b.step("test-glm-priced-windows-host", "Fake real-window timer through GLM publication, legacy cut and counted strict environment");
    first.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
    const environment = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{ "GLM strict env:", "GLM cost identity:" } }));
    environment.has_side_effects = true;
    first.dependOn(&environment.step);
    all.dependOn(first);
    const filters = &.{ "priced windows:", "allocate takes", "priced draft step", "zero draft step", "measuring round", "rows cheaper", "extendCosts", "lone price:" };
    const run = b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = filters }));
    b.step("test-priced-windows-core-host", "Real shared Engine windows, calibrated prefix surplus and legacy allocator/lone-price controls").dependOn(&run.step);
    all.dependOn(&run.step);
}
